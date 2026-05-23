#!/usr/bin/env python3
"""
scripts/kmidi_swarm.py — Hermes Tmux Matrix v3 (Hermes-as-conductor edition)

Architecture:
    Hermes Agent (`hermes -z`) is the BACKBONE BRAIN. It decomposes the user's
    feature request, generates per-stack subprompts, heals build failures with
    memory of past attempts, and synthesizes the final Active_Plan.md.

    The specialized subscription CLIs are EXECUTORS. Each owns one stack and
    writes code directly to the hidden git worktree:

        Rust intent_ir  ->  codex exec        (fallback: claude -p)
        C++ engine      ->  claude -p         (fallback: cursor-agent -p)
        Bindings        ->  claude -p         (fallback: cursor-agent -p)
        Python brain    ->  claude -p         (fallback: cursor-agent -p)
        React UI        ->  cursor-agent -p   (fallback: claude -p)
        Schema map      ->  gemini -p         (one-shot only)

    Hermes manages provider credentials internally via its `hermes auth` pool
    (OpenAI Codex OAuth, Nous, etc.). No API keys are read or stored by this
    script. Each subscription CLI uses its own session-based login.

Phoenix Protocol (auto-heal):
    Up to 4 build attempts per stack. After 3 failures the orchestrator runs
    AMNESIA RESET: swap to the fallback CLI, `git checkout -- .` in the pane,
    wipe the failure history, and try once more. Hermes regenerates the fix
    prompt at every retry, carrying memory of the full attempt history.

Usage:
    python3 scripts/kmidi_swarm.py "<feature request>"
    python3 scripts/kmidi_swarm.py --dry-run "<feature request>"   # phase 1 only
"""
from __future__ import annotations

import asyncio
import json
import os
import re
import subprocess
import sys
import time

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

SESSION = "kmidi-pipeline"

# Anchor every path against the repo root so the swarm can be launched from
# any pane, any cwd. WORKTREE_PATH is a *sibling* of the repo (not nested).
REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
WORKTREE_PATH = os.path.join(os.path.dirname(REPO_ROOT), "kmidi-agent-workspace")
WORKTREE_BRANCH = "feature/agent-swarm"

# Window/pane contract with scripts/kmidi-pipeline-tmux.sh. DO NOT renumber
# without updating that script in lockstep.
PANES = {
    "overview": (0, 0),  # Hermes status echoes here so the human sees phase progress
    "schemas":  (1, 0),
    "rust":     (2, 0),
    "cpp":      (3, 0),
    "bindings": (4, 0),
    "python":   (5, 0),  # top music_brain pane (5.1 is reserved for uvicorn)
    "react":    (6, 1),  # bottom pane of vertically-split window 6
}

# Build gates run *inside* the worktree (cd $WORKTREE) so that compile/lint
# sees the code the executors just wrote. The Phoenix poll waits for these
# commands to finish before scanning the buffer for errors.
BUILD_CMDS = {
    "rust":     "cd $WORKTREE/engine/intent_ir && cargo check --quiet 2>&1 | tail -120",
    "cpp":      "cd $WORKTREE && cmake --build build --target KellyCore 2>&1 | tail -120",
    "bindings": "cd $WORKTREE && cmake --build build --target penta_core_native 2>&1 | tail -120",
    "python":   ("cd $WORKTREE && python3 -m flake8 music_brain/ --max-line-length 100 "
                 "&& python3 -m pytest tests/unit -q --maxfail=3 2>&1 | tail -120"),
    "react":    "cd $WORKTREE && npm run build 2>&1 | tail -120",
}

# [primary, fallback]. Prompt is appended as final positional argv.
ROSTER = {
    "rust": [
        ["codex", "exec", "--skip-git-repo-check"],
        ["claude", "-p", "--permission-mode", "acceptEdits", "--output-format", "text"],
    ],
    "cpp": [
        ["claude", "-p", "--permission-mode", "acceptEdits", "--output-format", "text"],
        ["cursor-agent", "-p", "-f", "--output-format", "text"],
    ],
    "bindings": [
        ["claude", "-p", "--permission-mode", "acceptEdits", "--output-format", "text"],
        ["cursor-agent", "-p", "-f", "--output-format", "text"],
    ],
    "python": [
        ["claude", "-p", "--permission-mode", "acceptEdits", "--output-format", "text"],
        ["cursor-agent", "-p", "-f", "--output-format", "text"],
    ],
    "react": [
        ["cursor-agent", "-p", "-f", "--output-format", "text"],
        ["claude", "-p", "--permission-mode", "acceptEdits", "--output-format", "text"],
    ],
    "schemas": [
        ["gemini", "-p", "--yolo", "--output-format", "text"],
    ],
}

# Stacks driven by compile_and_heal (Phoenix loop). schemas is handled
# separately by map_schemas; overview is status echo only.
CODE_STACKS = ["rust", "cpp", "bindings", "python", "react"]

# Hermes Agent (backbone). Headless one-shot, auto-approve, suppresses prompts.
# Provider/model/credential routing is handled by the user's `hermes` config.
# `-z PROMPT` must be last so run_cli's appended prompt becomes its argument
# (hermes argparse refuses to consume the next token if it looks like a flag).
HERMES_CMD = ["hermes", "--yolo", "--accept-hooks", "-z"]
HERMES_TIMEOUT_SEC = 240

# Build poll: how long we wait for `cd && cmake/cargo/...` to finish before
# scanning the pane for errors. Tuned so slow C++/npm builds aren't truncated.
BUILD_POLL_INTERVAL_SEC = 2
BUILD_POLL_STABLE_SEC = 6     # buffer must be unchanged for this long
BUILD_POLL_MAX_SEC = 900      # 15 min cap per attempt

ANSI_RE = re.compile(r'\x1B(?:[@-Z\\-_]|\[[0-?]*[ -/]*[@-~])')
ERROR_RE = re.compile(r'(?i)(error:|FAILED|panic|fatal error|undefined reference)')
JSON_OBJ_RE = re.compile(r'\{[\s\S]*\}')


# ---------------------------------------------------------------------------
# Tmux interop
# ---------------------------------------------------------------------------

def tmux_send(window: int, pane: int, cmd: str) -> None:
    target = f"{SESSION}:{window}.{pane}"
    subprocess.run(
        ["tmux", "send-keys", "-t", target, cmd, "C-m"],
        check=False,
    )


def tmux_capture_and_clean(window: int, pane: int) -> str:
    target = f"{SESSION}:{window}.{pane}"
    result = subprocess.run(
        ["tmux", "capture-pane", "-p", "-t", target],
        capture_output=True,
        text=True,
        check=False,
    )
    return ANSI_RE.sub('', result.stdout)


def tmux_session_alive() -> bool:
    return subprocess.run(
        ["tmux", "has-session", "-t", SESSION],
        capture_output=True,
        check=False,
    ).returncode == 0


async def wait_for_pane_idle(window: int, pane: int,
                             *, max_wait: int = BUILD_POLL_MAX_SEC,
                             stable_for: int = BUILD_POLL_STABLE_SEC,
                             interval: int = BUILD_POLL_INTERVAL_SEC) -> str:
    """Poll the pane buffer until it has been unchanged for `stable_for`
    seconds, or until `max_wait` elapses. Returns the final captured buffer.

    This is the build-completion gate: long `cmake`/`npm` runs no longer
    produce a false "build CLEAN" because we don't sample mid-compile.
    """
    last = tmux_capture_and_clean(window, pane)
    stable = 0
    elapsed = 0
    while elapsed < max_wait:
        await asyncio.sleep(interval)
        elapsed += interval
        cur = tmux_capture_and_clean(window, pane)
        if cur == last:
            stable += interval
            if stable >= stable_for:
                return cur
        else:
            stable = 0
            last = cur
    return last


def hermes_status(msg: str) -> None:
    """Echo a Hermes phase marker into the overview pane so the human sees it."""
    safe = msg.replace("'", "'\\''")
    w, p = PANES["overview"]
    tmux_send(w, p, f"echo '[hermes] {safe}'")


# ---------------------------------------------------------------------------
# Subprocess driver (shared by Hermes and CLI executors)
# ---------------------------------------------------------------------------

def _agent_env() -> dict[str, str]:
    return {
        **os.environ,
        "CI": "true",
        "DEBIAN_FRONTEND": "noninteractive",
        "GIT_TERMINAL_PROMPT": "0",
        "GIT_EDITOR": "true",
        "GIT_PAGER": "cat",
        "PAGER": "cat",
    }


async def run_cli(argv: list[str], prompt: str, cwd: str,
                  timeout: int = 300) -> tuple[int, str, str]:
    """Spawn a CLI with `prompt` appended as the final positional argument."""
    try:
        proc = await asyncio.create_subprocess_exec(
            *argv, prompt,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=cwd,
            env=_agent_env(),
        )
    except FileNotFoundError as exc:
        return 127, "", f"CLI not installed: {exc}"

    try:
        stdout_b, stderr_b = await asyncio.wait_for(
            proc.communicate(), timeout=timeout,
        )
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return 124, "", f"timeout after {timeout}s"

    out = ANSI_RE.sub('', stdout_b.decode('utf-8', errors='replace'))
    err = ANSI_RE.sub('', stderr_b.decode('utf-8', errors='replace'))
    return proc.returncode or 0, out, err


# ---------------------------------------------------------------------------
# Hermes — backbone memory brain
# ---------------------------------------------------------------------------

def _extract_json_block(text: str) -> str:
    """Hermes's -z mode returns 'ONLY the final response text', but models
    occasionally still wrap JSON in ```fences```. Strip them and isolate the
    first {...} block."""
    text = text.strip()
    if text.startswith("```"):
        # drop opening fence (with or without language tag)
        text = text.split("\n", 1)[1] if "\n" in text else text
        text = text.rsplit("```", 1)[0]
    text = text.strip()
    match = JSON_OBJ_RE.search(text)
    return match.group(0) if match else text


async def hermes_think(prompt: str, cwd: str, *, expect_json: bool = False):
    """Send a one-shot prompt to Hermes Agent. Returns str (text mode) or
    dict (json mode). Raises RuntimeError on Hermes failure."""
    rc, out, err = await run_cli(
        HERMES_CMD, prompt, cwd, timeout=HERMES_TIMEOUT_SEC,
    )
    if rc != 0:
        tail = (err or out).strip().splitlines()[-3:]
        raise RuntimeError(f"hermes rc={rc}: {' | '.join(tail)}")

    text = out.strip()
    if not expect_json:
        return text

    raw = _extract_json_block(text)
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        raise RuntimeError(
            f"hermes returned non-JSON (parse: {exc}); first 400 chars:\n"
            f"{text[:400]}"
        ) from exc


# ---------------------------------------------------------------------------
# Phoenix Protocol (CLI executor with Hermes-driven heal)
# ---------------------------------------------------------------------------

async def compile_and_heal(stack: str, base_subprompt: str,
                           worktree: str) -> tuple[str, str, list[str]]:
    """Drive a stack-specific CLI, run the build in its tmux pane, and self-heal.

    `base_subprompt` is the per-stack instruction Hermes generated during the
    decompose phase. On build failure, Hermes is consulted again to produce a
    targeted fix prompt informed by the full attempt history.
    """
    window, pane = PANES[stack]
    # Expand the $WORKTREE placeholder so panes can be at any cwd.
    build_cmd = BUILD_CMDS[stack].replace("$WORKTREE", worktree)
    cli_idx = 0
    transcript: list[str] = []
    history: list[str] = []  # (cli, error_tail) pairs as text — the "memory"

    for attempt in range(4):
        argv = ROSTER[stack][cli_idx]
        cli_name = argv[0]

        # Build the prompt. First attempt uses Hermes's decomposed subprompt
        # verbatim. Subsequent attempts ask Hermes to synthesize a fix prompt
        # using the full attempt history.
        if attempt == 0 or not history:
            prompt = (
                f"You are working on the KmiDi monorepo at {worktree}, on "
                f"branch '{WORKTREE_BRANCH}'. You own the {stack.upper()} stack "
                f"only — do not touch other stacks.\n\n"
                f"TASK:\n{base_subprompt}\n\n"
                f"Use your edit tools to write files directly to disk. Do not "
                f"paste file contents into chat. Exit when complete; the build "
                f"is run separately."
            )
        else:
            heal_query = (
                f"You are the KmiDi swarm orchestrator. The {stack.upper()} "
                f"build keeps failing. Generate a focused fix prompt for the "
                f"`{cli_name}` CLI. The CLI will receive your prompt verbatim "
                f"and edit files in {worktree}. Stay within the {stack.upper()} "
                f"stack. Output ONLY the prompt text — no preamble, no fences, "
                f"no commentary.\n\n"
                f"ORIGINAL TASK:\n{base_subprompt}\n\n"
                f"ATTEMPT HISTORY ({len(history)} prior failures):\n"
                + "\n---\n".join(history)
            )
            try:
                prompt = await hermes_think(heal_query, worktree)
                transcript.append(f"  hermes regenerated fix prompt ({len(prompt)} chars)")
            except Exception as exc:
                transcript.append(f"  hermes heal failed: {exc} — using raw error context")
                prompt = (
                    f"Retry: {base_subprompt}\n\n"
                    f"PREVIOUS BUILD ERROR:\n{history[-1]}"
                )

        print(f"[{stack}] attempt {attempt + 1}/4 via {cli_name}", flush=True)
        transcript.append(f"attempt {attempt + 1}: {cli_name}")
        hermes_status(f"{stack}: attempt {attempt + 1} via {cli_name}")

        rc, out, err = await run_cli(argv, prompt, worktree)
        if rc != 0:
            tail = (err or out).strip().splitlines()[-1:] or ["(no output)"]
            transcript.append(f"  cli rc={rc}: {tail[0][:200]}")

        # Run the build in the assigned tmux pane (visible to the human),
        # then wait until the pane buffer stops changing so we don't sample
        # mid-compile and report a false "build CLEAN".
        tmux_send(window, pane, build_cmd)
        pane_buf = await wait_for_pane_idle(window, pane)

        if not ERROR_RE.search(pane_buf):
            transcript.append("  build CLEAN")
            return stack, "success", transcript

        error_tail = "\n".join(pane_buf.splitlines()[-40:])
        history.append(f"[attempt {attempt + 1} via {cli_name}]\n{error_tail}")
        transcript.append("  build FAILED")

        # 3rd failure -> AMNESIA RESET
        if attempt == 2:
            new_idx = (cli_idx + 1) % len(ROSTER[stack])
            transcript.append(
                f"  AMNESIA RESET: swap {ROSTER[stack][cli_idx][0]} -> "
                f"{ROSTER[stack][new_idx][0]}, revert worktree files, wipe history"
            )
            hermes_status(f"{stack}: AMNESIA RESET, swap to {ROSTER[stack][new_idx][0]}")
            cli_idx = new_idx
            tmux_send(
                window, pane,
                f"cd {worktree} && git restore --staged . && git checkout -- .",
            )
            await asyncio.sleep(1)
            history.clear()

    return stack, "failed", transcript


# ---------------------------------------------------------------------------
# Schema mapping (one-shot via gemini, prompt comes from Hermes decompose)
# ---------------------------------------------------------------------------

async def map_schemas(schemas_subprompt: str, worktree: str) -> bool:
    print("[schemas] mapping with gemini ...", flush=True)
    hermes_status("schemas: gemini mapping")
    argv = ROSTER["schemas"][0]
    prompt = (
        f"You are working in the KmiDi monorepo at {worktree}.\n\n"
        f"TASK:\n{schemas_subprompt}\n\n"
        f"Sources of truth for schemas:\n"
        f"  - shared_schemas/                  (canonical JSON)\n"
        f"  - src/types/                       (TypeScript)\n"
        f"  - engine/intent_ir/src/generated/  (Rust)\n"
        f"  - music_brain/api.py               (Python /generate)\n\n"
        f"Write a concise mapping document to "
        f"Obsidian_Vault/01_Context/schemas.md. Do not modify any source files."
    )
    os.makedirs(os.path.join(worktree, "Obsidian_Vault/01_Context"), exist_ok=True)
    rc, _out, err = await run_cli(argv, prompt, worktree, timeout=180)
    if rc != 0:
        tail = err.strip().splitlines()[-1:] or ["(no stderr)"]
        print(f"[schemas] gemini rc={rc}: {tail[0]}", flush=True)
    return rc == 0


# ---------------------------------------------------------------------------
# Worktree management
# ---------------------------------------------------------------------------

def ensure_worktree() -> str:
    """Create or reuse the sibling worktree on WORKTREE_BRANCH.

    All git operations run from REPO_ROOT so the swarm works regardless of
    the shell's cwd when it was launched.
    """
    if os.path.isdir(WORKTREE_PATH):
        print(f"reusing worktree: {WORKTREE_PATH}", flush=True)
        return WORKTREE_PATH

    print(f"creating worktree: {WORKTREE_PATH}", flush=True)
    branch_check = subprocess.run(
        ["git", "show-ref", "--verify", "--quiet", f"refs/heads/{WORKTREE_BRANCH}"],
        cwd=REPO_ROOT,
        check=False,
    )
    if branch_check.returncode != 0:
        subprocess.run(
            ["git", "branch", WORKTREE_BRANCH],
            cwd=REPO_ROOT,
            check=False,
        )

    res = subprocess.run(
        ["git", "worktree", "add", WORKTREE_PATH, WORKTREE_BRANCH],
        cwd=REPO_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    if res.returncode != 0:
        print(f"git worktree add failed: {res.stderr}", flush=True)
    return WORKTREE_PATH


# ---------------------------------------------------------------------------
# Preflight
# ---------------------------------------------------------------------------

def _which(cmd: str) -> bool:
    return subprocess.run(
        ["which", cmd], capture_output=True, check=False,
    ).returncode == 0


def _parse_args(argv: list[str]) -> tuple[bool, str]:
    """Returns (dry_run, user_prompt). Accepts `--dry-run` anywhere before
    the prompt. Skip-tmux check is implied by --dry-run."""
    args = list(argv[1:])
    dry_run = False
    if "--dry-run" in args:
        args.remove("--dry-run")
        dry_run = True
    if not args:
        print(
            'Usage: python3 scripts/kmidi_swarm.py [--dry-run] "<feature request>"',
            file=sys.stderr,
        )
        sys.exit(1)
    return dry_run, args[0]


def preflight(*, require_tmux: bool = True) -> None:
    if require_tmux and not tmux_session_alive():
        print(
            f"ERROR: tmux session '{SESSION}' is not running.\n"
            f"Start it first:  ./scripts/init_matrix.sh\n"
            f"(Pass --dry-run to skip this check and validate decomposition only.)",
            file=sys.stderr,
        )
        sys.exit(2)

    if not _which("hermes"):
        print(
            "ERROR: `hermes` CLI not found on PATH.\n"
            "Hermes is the swarm's backbone brain. Install it from "
            "https://hermesagent.dev or your existing distribution.\n"
            "Diagnose:  hermes doctor   |   hermes auth",
            file=sys.stderr,
        )
        sys.exit(3)

    missing_roles: list[str] = []
    for role, options in ROSTER.items():
        if not any(_which(argv[0]) for argv in options):
            missing_roles.append(role)
    if missing_roles:
        print(
            f"ERROR: no installed CLI for role(s) {missing_roles}.\n"
            f"Install one of: cursor-agent, claude, codex, gemini.\n"
            f"Check:   which codex claude cursor-agent gemini",
            file=sys.stderr,
        )
        sys.exit(4)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

async def main() -> None:
    dry_run, user_prompt = _parse_args(sys.argv)
    preflight(require_tmux=not dry_run)
    worktree = REPO_ROOT if dry_run else ensure_worktree()

    print(f"feature : {user_prompt}", flush=True)
    print(f"worktree: {worktree}{'  (dry-run)' if dry_run else ''}", flush=True)

    # ---- Phase 1: DECOMPOSE -----------------------------------------------
    if not dry_run:
        hermes_status("phase 1/3: decompose")
    decompose_prompt = (
        f"You are the KmiDi swarm conductor. Decompose the following feature "
        f"request into stack-specific subtasks. Each subtask will be handed to "
        f"a specialist CLI:\n"
        f"  rust     -> codex exec   (intent_ir crate, Rust 2021)\n"
        f"  cpp      -> claude -p    (KellyCore engine, JUCE 8, C++20)\n"
        f"  bindings -> claude -p    (pybind11 in bindings/, links KellyCore)\n"
        f"  python   -> claude -p    (music_brain/ FastAPI + helpers, flake8/pytest)\n"
        f"  react    -> cursor-agent (React 19 + Vite + TypeScript UI)\n"
        f"  schemas  -> gemini -p    (shared_schemas + sync to TS/Rust/Python)\n\n"
        f"Output STRICTLY a JSON object with exactly these keys:\n"
        f'  {{"rust": "...", "cpp": "...", "bindings": "...", "python": "...", '
        f'"react": "...", "schemas": "..."}}\n'
        f"Each value is a concrete coding instruction in plain prose, with "
        f"explicit file paths, function signatures, or component names where "
        f"possible. Scope rules: `python` must not edit bindings/*.cpp; "
        f"`bindings` must not edit music_brain/ unless cross-cutting. "
        f"If a stack has nothing to do for this feature, return the empty "
        f'string "" for that key. No markdown fences. No commentary. Only '
        f"the JSON object.\n\n"
        f"FEATURE REQUEST: {user_prompt}\n\n"
        f"REPO_ROOT: {REPO_ROOT}\n"
        f"WORKTREE: {worktree}"
    )
    try:
        plan = await hermes_think(decompose_prompt, worktree, expect_json=True)
    except Exception as exc:
        print(f"FATAL: hermes decompose failed: {exc}", file=sys.stderr)
        sys.exit(5)

    print(f"hermes plan keys: {list(plan.keys())}", flush=True)

    if dry_run:
        print("\n--- DRY RUN: decomposition only ---", flush=True)
        print(json.dumps(plan, indent=2), flush=True)
        return

    # ---- Phase 2: PARALLEL EXECUTE ---------------------------------------
    hermes_status("phase 2/3: parallel execute")
    schema_subprompt = plan.get("schemas") or user_prompt

    schemas_task = asyncio.create_task(map_schemas(schema_subprompt, worktree))
    code_results = await asyncio.gather(*[
        compile_and_heal(stack, plan.get(stack) or user_prompt, worktree)
        for stack in CODE_STACKS
    ])
    schemas_ok = await schemas_task

    # ---- Phase 3: SYNTHESIZE ---------------------------------------------
    hermes_status("phase 3/3: synthesize Active_Plan.md")
    synth_lines = []
    for stack, status, transcript in code_results:
        synth_lines.append(f"\n### {stack.upper()}: {status}")
        for line in transcript:
            synth_lines.append(f"  - {line}")
    synth_input = "\n".join(synth_lines)

    synthesize_prompt = (
        f"You are the KmiDi swarm conductor. The parallel agents have "
        f"completed their work. Write the contents of Active_Plan.md as "
        f"GitHub-flavored markdown summarizing this run. Sections:\n"
        f"  1. # Active Plan: <feature>\n"
        f"  2. ## Decomposition (one bullet per stack)\n"
        f"  3. ## Stack outcomes (status + key transcript moments per stack)\n"
        f"  4. ## Schema delta (point to Obsidian_Vault/01_Context/schemas.md "
        f"if {'OK' if schemas_ok else 'note that the gemini map FAILED'})\n"
        f"  5. ## Next actions for the human reviewer\n\n"
        f"Output ONLY the markdown body. No fences, no preamble, no explanation.\n\n"
        f"USER REQUEST: {user_prompt}\n\n"
        f"DECOMPOSITION:\n{json.dumps(plan, indent=2)}\n\n"
        f"BUILD TRANSCRIPTS:{synth_input}\n\n"
        f"SCHEMA STATUS: {'OK' if schemas_ok else 'FAILED'}"
    )
    try:
        plan_md = await hermes_think(synthesize_prompt, worktree)
    except Exception as exc:
        print(f"hermes synthesize failed: {exc} — falling back to raw transcript",
              flush=True)
        plan_md = (
            f"# Active Plan: {user_prompt}\n\n"
            f"_Generated by Hermes Tmux Matrix swarm at "
            f"{time.strftime('%Y-%m-%d %H:%M:%S')} (synth fallback)_\n\n"
            f"## Worktree\n\n`{worktree}` on `{WORKTREE_BRANCH}`\n\n"
            f"## Schema mapping\n\n"
            f"{'OK' if schemas_ok else 'FAILED'} — see "
            f"`Obsidian_Vault/01_Context/schemas.md`\n\n"
            f"## Stack results\n{synth_input}\n"
        )

    plan_path = os.path.join(worktree, "Obsidian_Vault/Master_Plans/Active_Plan.md")
    os.makedirs(os.path.dirname(plan_path), exist_ok=True)
    with open(plan_path, "w") as f:
        f.write(plan_md)

    summary = ", ".join(
        f"{stack}={status}" for stack, status, _ in code_results
    )
    print(f"\nswarm complete. plan: {plan_path}", flush=True)
    print(f"results: {summary}", flush=True)
    hermes_status(f"complete: {summary}")


if __name__ == "__main__":
    asyncio.run(main())
