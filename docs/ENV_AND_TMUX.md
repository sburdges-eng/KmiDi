# Environments and tmux — Operator Stack

Envs live **outside** the repo. Critical jobs run **inside** tmux. Not negotiable.

## Env law

**Never create environments inside the repo.**

- Global env root: `mkdir -p ~/envs`
- Tell micromamba (add to `.zshrc`): `export MAMBA_ROOT_PREFIX=~/envs/mamba`
- Environments then live under `~/envs`; repo stays clean.

### First environment (do once)

```bash
micromamba create -n kmidi python=3.11
micromamba activate kmidi
```

Install PyTorch later — **surgically**. Do not freestyle GPU installs. Start with **CPU PyTorch** first; architecture mistakes are cheaper without GPU burn. GPU is acceleration, not direction.

### Env count

Do **not** create 12 environments. Keep it minimal:

- **kmidi-core** — daily dev, Brain, inference
- **kmidi-training** — GPU training when needed
- **kmidi-experimental** (optional) — tryouts

Environment sprawl is real. Stay calm.

### LLM model (optional)

Orchestrator intent parsing can use a local LLM (GGUF) when available. Set one of:

- `KMI_DI_LLM_MODEL_PATH` — path to GGUF model file
- `LLM_MODEL_PATH` — fallback env

If unset or model missing, the engine uses a rule-based keyword parser (no install). For LLM-backed parsing: install `llama-cpp-python`, point env at a GGUF file; see `mcp_workstation/llm_reasoning_engine.py`.

### Image / audio engines (optional)

Orchestrator image and audio generation use optional deps; when missing, engines return **stubbed** so the workflow continues.

| Engine | Env | Default | Deps |
|--------|-----|---------|------|
| **Image** | `KMI_DI_IMAGE_MODEL_PATH`, `STABLE_DIFFUSION_MODEL_PATH` | runwayml/stable-diffusion-v1-5 | diffusers, torch |
| **Audio** | `KMI_DI_AUDIO_MODEL_ID`, `AUDIOCRAFT_MODEL_ID` | musicgen-small | audiocraft, torch |

Status in result dict: `stubbed` (not loaded), `completed`, `failed`. See `docs/CONTRACTS.md` §6.

---

## tmux — process persistence

Without tmux: SSH drops, terminal crashes, laptop sleeps → training dies. We avoid that.

**Install:** `brew install tmux` (mac).

### Training workflow (always)

1. Start session: `tmux new -s kmidi`
2. Inside tmux: `micromamba activate kmidi` then `python train.py` (or your command)
3. Detach safely: **Ctrl+B** then **D**. Training continues. Laptop can close; run survives.

### Cursor / AI directive

When executing long-running processes (training, dataset builds, servers, compiles): **always** recommend tmux. If the user tries to launch training in a normal terminal, intervene and suggest tmux. Preferred pattern:

```bash
tmux new -s <session_name>
micromamba activate <env>
<run command>
```

Terminals are disposable. tmux is reality.

### Auto-start tmux (optional)

Add to `.zshrc` so every new shell lands in persistence:

```bash
if [ -z "$TMUX" ]; then
  tmux attach -t kmidi || tmux new -s kmidi
fi
```

### tmux windows — cockpit layout

Inside tmux, create windows: **Ctrl+B** then **C**.

Suggested layout:

- **Window 1** — Brain (`run_brain.py`, orchestrator)
- **Window 2** — Training
- **Window 3** — Git
- **Window 4** — Monitoring

---

## Stack summary

| Layer        | Role                    |
|-------------|-------------------------|
| **Dev**     | Sovereign workspace     |
| **COLD_STORAGE** | Frozen past        |
| **Micromamba**  | Deterministic deps (envs in `~/envs`) |
| **tmux**    | Process persistence    |

You are not configuring tools. You are building a research-grade machine.
