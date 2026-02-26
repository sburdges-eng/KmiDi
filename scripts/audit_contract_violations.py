#!/usr/bin/env python3
"""
Parse implemented code snippets for violations of the LISTENING_SPEC contract.

Usage: python3 scripts/audit_contract_violations.py [--repo-root .]
Output: JSON or human-readable report of violations found.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Violation:
    rule_id: str
    rule_desc: str
    file: str
    line: int
    snippet: str
    severity: str = "error"  # error | warn
    expected: str = ""


@dataclass
class AuditResult:
    violations: list[Violation] = field(default_factory=list)
    files_scanned: int = 0

    def add(self, v: Violation) -> None:
        self.violations.append(v)

    def to_json(self) -> str:
        return json.dumps(
            {
                "violations": [
                    {
                        "rule_id": v.rule_id,
                        "rule_desc": v.rule_desc,
                        "file": v.file,
                        "line": v.line,
                        "snippet": v.snippet[:200],
                        "severity": v.severity,
                        "expected": v.expected,
                    }
                    for v in self.violations
                ],
                "files_scanned": self.files_scanned,
                "violation_count": len(self.violations),
            },
            indent=2,
        )


# ---------------------------------------------------------------------------
# Contract checks (from LISTENING_SPEC, INVARIANT_FILE_MAPPING)
# ---------------------------------------------------------------------------

def check_plugin_processor(path: Path, content: str, result: AuditResult) -> None:
    """P-1: No auto-playback from generation path."""
    lines = content.splitlines()
    in_generate = False
    generate_end_pattern = re.compile(r"^\s*(GeneratedMidi|void\s+\w+|bool\s+\w+|std::vector)")
    for i, line in enumerate(lines, 1):
        if "generateFromWound" in line or "generateFromJourney" in line:
            in_generate = True
            search_start = i - 1
        if in_generate:
            if "startPlayback" in line and "void PluginProcessor::startPlayback" not in line:
                # Extract surrounding context
                start = max(0, i - 2)
                end = min(len(lines), i + 2)
                snippet = "\n".join(lines[start:end])
                result.add(
                    Violation(
                        rule_id="P-1",
                        rule_desc="No auto-playback from generation",
                        file=str(path),
                        line=i,
                        snippet=snippet,
                        expected="startPlayback must not be called from generateFromWound/generateFromJourney",
                    )
                )
            if generate_end_pattern.match(line.strip()) and i > (search_start or 0) + 5:
                in_generate = False

    # P-1: startPlayback must have fromUserAction and jassert
    if "void PluginProcessor::startPlayback" in content or "startPlayback(" in content:
        if "fromUserAction" not in content:
            for i, line in enumerate(content.splitlines(), 1):
                if "startPlayback" in line and "fromUserAction" not in line and "void " not in line:
                    result.add(
                        Violation(
                            rule_id="P-1",
                            rule_desc="startPlayback must require fromUserAction",
                            file=str(path),
                            line=i,
                            snippet=line.strip(),
                            expected="startPlayback(bool fromUserAction); jassert(fromUserAction)",
                        )
                    )
        if "jassert(fromUserAction)" not in content and "startPlayback" in content:
            result.add(
                Violation(
                    rule_id="P-1",
                    rule_desc="startPlayback must jassert(fromUserAction)",
                    file=str(path),
                    line=1,
                    snippet="(missing jassert)",
                    severity="warn",
                )
            )


def check_intent_pipeline(path: Path, content: str, result: AuditResult) -> None:
    """I-3, R-1: ABSTAIN on empty. I-4: return invalid on ruleBreaks."""
    lines = content.splitlines()
    for i, line in enumerate(lines, 1):
        if "trimmed.empty()" in line or "trim(" in line:
            if "abstain" not in content and "ABSTAIN" not in content:
                result.add(
                    Violation(
                        rule_id="I-3/R-1",
                        rule_desc="Empty input must return ABSTAIN",
                        file=str(path),
                        line=i,
                        snippet=line.strip(),
                        expected="return IntentResult::abstain()",
                    )
                )
    if "abstain" not in content.lower():
        result.add(
            Violation(
                rule_id="I-3/R-1",
                rule_desc="IntentPipeline must reference ABSTAIN/abstain()",
                file=str(path),
                line=1,
                snippet="(no abstain found)",
            )
        )
    if "ruleBreaks" in content or "RuleBreak" in content:
        if "invalid" not in content.lower() or "return IntentResult::invalid()" not in content:
            result.add(
                Violation(
                    rule_id="I-4",
                    rule_desc="Constraint violations must return invalid()",
                    file=str(path),
                    line=1,
                    snippet="(ruleBreaks present but no return invalid)",
                    expected="return IntentResult::invalid()",
                )
            )
    if "jassert(!trimmed.empty())" not in content and "jassert" in content:
        if "infer" in content or "process(" in content:
            result.add(
                Violation(
                    rule_id="I-3",
                    rule_desc="Intent must not be inferred when input empty; jassert required",
                    file=str(path),
                    line=1,
                    snippet="(missing jassert before inference)",
                    severity="warn",
                )
            )


def check_chord_generator(path: Path, content: str, result: AuditResult) -> None:
    """I-2, M-1: No extension without escalationTokenPresent."""
    lines = content.splitlines()
    for i, line in enumerate(lines, 1):
        if "push_back(6)" in line or "scaleDegrees.push_back" in line:
            # Check context: must be guarded by escalationTokenPresent
            context_start = max(0, i - 8)
            context = "\n".join(lines[context_start:i + 2])
            if "escalationTokenPresent" not in context:
                result.add(
                    Violation(
                        rule_id="I-2/E-1",
                        rule_desc="7th/extension must be guarded by escalationTokenPresent",
                        file=str(path),
                        line=i,
                        snippet=line.strip(),
                        expected="if (escalationTokenPresent && ...) scaleDegrees.push_back(6)",
                    )
                )
    if "canProduceOutput" not in content:
        result.add(
            Violation(
                rule_id="Gate",
                rule_desc="ChordGenerator must call canProduceOutput before output",
                file=str(path),
                line=1,
                snippet="(missing canProduceOutput check)",
                expected="if (!canProduceOutput(intent)) return {}",
            )
        )
    if "jassert(escalationTokenPresent" not in content and "scaleDegrees" in content:
        result.add(
            Violation(
                rule_id="I-2",
                rule_desc="No extension without escalation token; jassert recommended",
                file=str(path),
                line=1,
                snippet="(missing jassert for extension guard)",
                severity="warn",
            )
        )


def check_plugin_editor(path: Path, content: str, result: AuditResult) -> None:
    """I-3, R-1: Empty Side B must not trigger generation."""
    if "onGenerate" in content or "generateFromJourney" in content:
        if "isEmpty" not in content and "empty" not in content.lower():
            result.add(
                Violation(
                    rule_id="R-1",
                    rule_desc="Must check for empty Side B before generation",
                    file=str(path),
                    line=1,
                    snippet="(no isEmpty/empty check)",
                    expected="if (sideBInput_.getText().trimStart().trimEnd().isEmpty()) return;",
                )
            )
        if "jassert(!input.text.empty())" not in content and "runTrustMVP" in content:
            result.add(
                Violation(
                    rule_id="I-3",
                    rule_desc="Empty input must not trigger generation; jassert required",
                    file=str(path),
                    line=1,
                    snippet="(missing jassert before runTrustMVP)",
                    severity="warn",
                )
            )


def check_midi_builder(path: Path, content: str, result: AuditResult) -> None:
    """I-1: melody/bass must check has_value() before adding."""
    if "generated.melody" in content or "generated.bass" in content:
        if "has_value()" not in content:
            result.add(
                Violation(
                    rule_id="I-1",
                    rule_desc="melody/bass must check has_value() before adding (no invention)",
                    file=str(path),
                    line=1,
                    snippet="(melody/bass used without has_value check)",
                    expected="if (generated.melody.has_value()) { ... }",
                )
            )


def check_types(path: Path, content: str, result: AuditResult) -> None:
    """canProduceOutput gate; I-1 optional melody/bass."""
    if "canProduceOutput" not in content:
        result.add(
            Violation(
                rule_id="Gate",
                rule_desc="canProduceOutput must be defined in Types.h",
                file=str(path),
                line=1,
                snippet="(missing canProduceOutput)",
            )
        )
    if "GeneratedMidi" in content and "std::optional" not in content and "optional" not in content:
        if "melody" in content or "bass" in content:
            result.add(
                Violation(
                    rule_id="I-1",
                    rule_desc="melody/bass must be std::optional (no invention)",
                    file=str(path),
                    line=1,
                    snippet="(melody/bass not optional)",
                    severity="warn",
                )
            )


def check_processor_can_produce(path: Path, content: str, result: AuditResult) -> None:
    """PluginProcessor must call canProduceOutput before producing output."""
    if "runTrustMVP" in content or "generateFromWound" in content:
        if "canProduceOutput" not in content:
            result.add(
                Violation(
                    rule_id="Gate",
                    rule_desc="runTrustMVP/generate must call canProduceOutput",
                    file=str(path),
                    line=1,
                    snippet="(missing canProduceOutput gate)",
                )
            )
        if "jassert(canProduceOutput" not in content:
            result.add(
                Violation(
                    rule_id="Gate",
                    rule_desc="Must jassert canProduceOutput before output",
                    file=str(path),
                    line=1,
                    snippet="(missing jassert canProduceOutput)",
                    severity="warn",
                )
            )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

CHECKERS = {
    "plugin/PluginProcessor.cpp": [check_plugin_processor, check_processor_can_produce],
    "engine/IntentPipeline.cpp": [check_intent_pipeline],
    "midi/ChordGenerator.cpp": [check_chord_generator],
    "midi/MidiBuilder.cpp": [check_midi_builder],
    "plugin/PluginEditor.cpp": [check_plugin_editor],
    "common/Types.h": [check_types],
}


def audit(repo_root: Path) -> AuditResult:
    result = AuditResult()
    for rel_path, checkers in CHECKERS.items():
        path = repo_root / rel_path
        if not path.exists():
            continue
        result.files_scanned += 1
        content = path.read_text()
        for fn in checkers:
            fn(path, content, result)
    return result


def main() -> None:
    ap = argparse.ArgumentParser(description="Audit code for LISTENING_SPEC violations")
    ap.add_argument("--repo-root", type=Path, default=Path(__file__).parent.parent)
    ap.add_argument("--json", action="store_true", help="Output JSON")
    args = ap.parse_args()
    result = audit(args.repo_root)
    if args.json:
        print(result.to_json())
    else:
        if result.violations:
            print("LISTENING CONTRACT VIOLATIONS\n")
            for v in result.violations:
                print(f"  [{v.rule_id}] {v.rule_desc}")
                print(f"    {v.file}:{v.line}")
                snippet_preview = v.snippet[:120] + ("..." if len(v.snippet) > 120 else "")
                print(f"    {snippet_preview}")
                if v.expected:
                    print(f"    Expected: {v.expected}")
                print()
            print(f"Total: {len(result.violations)} violation(s) in {result.files_scanned} file(s)")
            exit(1)
        else:
            print(f"PASS: No violations in {result.files_scanned} file(s)")


if __name__ == "__main__":
    main()
