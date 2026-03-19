"""
Unified MIDI tokenizer helpers for KellyBrain.

The LanguageModel path is offline-first by default. Maestro REMI-BPE assets
must already exist locally unless the caller explicitly opts into remote
loading.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union
import os


TokenSource = Union[str, Path, Sequence[int]]


class KmiDiTokenizer:
    """Base class for KmiDi tokenizers."""

    def encode(self, data: Any) -> List[int]:
        raise NotImplementedError

    def decode(self, tokens: List[int], output_path: Optional[Union[str, Path]] = None):
        raise NotImplementedError


class SimpleMIDITokenizer(KmiDiTokenizer):
    """
    Simple rule-based MIDI tokenizer.
    Notes (0-127), Velocity (128-159), Timing (160-287).
    """

    def __init__(self, vocab_size: int = 392):
        self.vocab_size = vocab_size
        self.note_range = (0, 128)
        self.velocity_range = (128, 160)
        self.timing_range = (160, 288)

        self.pad_token = vocab_size - 4
        self.bos_token = vocab_size - 3
        self.eos_token = vocab_size - 2
        self.bar_token = vocab_size - 1

    def encode(self, midi_events: List[Dict[str, Any]]) -> List[int]:
        tokens = [self.bos_token]
        for event in midi_events:
            if event.get("type") == "bar":
                tokens.append(self.bar_token)
            elif event.get("type") == "note_on":
                tokens.append(int(event["note"]) % 128)
                vel_bin = min(31, int(event.get("velocity", 64)) // 4)
                tokens.append(128 + vel_bin)
                time_pos = int(float(event.get("time_in_bar", 0.0)) * 127) % 128
                tokens.append(160 + time_pos)
        tokens.append(self.eos_token)
        return tokens

    def decode(
        self,
        tokens: List[int],
        output_path: Optional[Union[str, Path]] = None,
    ) -> List[Dict[str, Any]]:
        del output_path

        events: List[Dict[str, Any]] = []
        i = 0
        current_time = 0.0
        while i < len(tokens):
            token = int(tokens[i])
            if token == self.bar_token:
                events.append({"type": "bar", "time": current_time})
                current_time += 1.0
            elif self.note_range[0] <= token < self.note_range[1]:
                note = token
                velocity = 64
                timing = 0.0
                if i + 1 < len(tokens) and self.velocity_range[0] <= tokens[i + 1] < self.velocity_range[1]:
                    velocity = (int(tokens[i + 1]) - 128) * 4
                    i += 1
                if i + 1 < len(tokens) and self.timing_range[0] <= tokens[i + 1] < self.timing_range[1]:
                    timing = (int(tokens[i + 1]) - 160) / 127.0
                    i += 1
                events.append(
                    {
                        "type": "note_on",
                        "note": note,
                        "velocity": velocity,
                        "time": current_time + timing,
                    }
                )
            i += 1
        return events


class MidiTokWrapper(KmiDiTokenizer):
    """Offline-first wrapper for MidiTok REMI/BPE tokenizers."""

    def __init__(
        self,
        model_id: str = "NathanFradet/Maestro-REMI-bpe20k",
        tokenizer_path: Optional[Union[str, Path]] = None,
        allow_remote: Optional[bool] = None,
    ):
        self.model_id = model_id
        self.tokenizer_path = Path(tokenizer_path).expanduser() if tokenizer_path else None
        self.allow_remote = self._resolve_allow_remote(allow_remote)
        self.tokenizer = None
        self.load_error = ""
        self.pad_token: Optional[int] = None
        self.bos_token: Optional[int] = None
        self.eos_token: Optional[int] = None
        self.special_tokens: Dict[str, Optional[int]] = {}
        self._load_tokenizer()

    def _resolve_allow_remote(self, allow_remote: Optional[bool]) -> bool:
        if allow_remote is not None:
            return allow_remote

        env_value = os.getenv("KMIDI_TOKENIZER_ALLOW_REMOTE", "").strip().lower()
        return env_value in {"1", "true", "yes", "on"}

    def _candidate_local_paths(self) -> List[Path]:
        candidates: List[Path] = []
        if self.tokenizer_path is not None:
            candidates.append(self.tokenizer_path)

        model_path = Path(self.model_id).expanduser()
        if model_path.exists():
            candidates.append(model_path)

        env_path = os.getenv("MIDITOK_TOKENIZER_PATH", "").strip()
        if env_path:
            candidates.append(Path(env_path).expanduser())

        unique_candidates: List[Path] = []
        for candidate in candidates:
            if candidate.exists() and candidate not in unique_candidates:
                unique_candidates.append(candidate)
        return unique_candidates

    def _load_tokenizer(self) -> None:
        try:
            from miditok import REMI
        except ImportError:
            self.load_error = "miditok is not installed"
            self.tokenizer = None
            return

        source: Optional[str] = None
        local_candidates = self._candidate_local_paths()
        if local_candidates:
            source = str(local_candidates[0])
        elif self.allow_remote:
            source = self.model_id
        else:
            self.load_error = (
                "tokenizer assets must exist locally for offline mode; "
                "set tokenizer_path or KMIDI_TOKENIZER_ALLOW_REMOTE=1"
            )
            self.tokenizer = None
            return

        try:
            self.tokenizer = REMI.from_pretrained(source)
            self._sync_special_tokens()
        except Exception as exc:
            self.tokenizer = None
            self.load_error = f"failed to load tokenizer: {exc}"

    def _sync_special_tokens(self) -> None:
        if self.tokenizer is None:
            self.special_tokens = {"pad": None, "bos": None, "eos": None}
            return

        ids = getattr(self.tokenizer, "special_tokens_ids", {}) or {}

        def lookup(*names: str) -> Optional[int]:
            for name in names:
                if isinstance(ids, dict) and name in ids:
                    return int(ids[name])
                attr_names = (
                    f"{name.lower()}_token_id",
                    f"{name.lower()}_id",
                    f"{name.upper()}_TOKEN_ID",
                )
                for attr_name in attr_names:
                    if hasattr(self.tokenizer, attr_name):
                        return int(getattr(self.tokenizer, attr_name))
            return None

        self.pad_token = lookup("PAD")
        self.bos_token = lookup("BOS", "SOS")
        self.eos_token = lookup("EOS")
        self.special_tokens = {
            "pad": self.pad_token,
            "bos": self.bos_token,
            "eos": self.eos_token,
        }

    def _normalize_token_ids(self, tokens: Any) -> List[int]:
        if tokens is None:
            return []
        if hasattr(tokens, "ids"):
            return [int(token) for token in tokens.ids]
        if isinstance(tokens, (str, bytes, Path)):
            return []
        if isinstance(tokens, Sequence):
            if len(tokens) == 0:
                return []
            first = tokens[0]
            if hasattr(first, "ids"):
                return [int(token) for token in first.ids]
            if isinstance(first, Sequence) and not isinstance(first, (str, bytes)):
                return self._normalize_token_ids(first)
            return [int(token) for token in tokens]
        return list(tokens)

    def _ensure_special_tokens(self, tokens: Sequence[int]) -> List[int]:
        normalized = [int(token) for token in tokens]
        if self.bos_token is not None and (not normalized or normalized[0] != self.bos_token):
            normalized.insert(0, self.bos_token)
        if self.eos_token is not None and (not normalized or normalized[-1] != self.eos_token):
            normalized.append(self.eos_token)
        return normalized

    def encode(self, midi_source: TokenSource) -> List[int]:
        if self.tokenizer is None:
            return []

        if isinstance(midi_source, (list, tuple)):
            return self._normalize_token_ids(midi_source)

        tokenized = self.tokenizer(str(midi_source))
        return self._normalize_token_ids(tokenized)

    def encode_for_language_model(self, midi_source: TokenSource) -> List[int]:
        return self._ensure_special_tokens(self.encode(midi_source))

    def decode(
        self,
        tokens: List[int],
        output_path: Optional[Union[str, Path]] = None,
    ):
        if self.tokenizer is None:
            return None

        normalized = self._normalize_token_ids(tokens)
        if not normalized:
            return None

        score = self.tokenizer.decode([normalized])
        if output_path is not None:
            output = str(Path(output_path))
            if hasattr(score, "dump_midi"):
                score.dump_midi(output)
            elif hasattr(score, "dump"):
                score.dump(output)
        return score


def get_tokenizer(tokenizer_type: str = "simple", **kwargs: Any) -> KmiDiTokenizer:
    if tokenizer_type == "simple":
        return SimpleMIDITokenizer(**kwargs)
    if tokenizer_type == "miditok":
        return MidiTokWrapper(**kwargs)
    return SimpleMIDITokenizer()
