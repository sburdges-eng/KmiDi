#!/usr/bin/env python3
"""License allowlist gate for third-party model downloads."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable
from urllib import request

ALLOWED_LICENSES: frozenset[str] = frozenset({"mit", "apache-2.0", "bsd-3-clause"})


class LicenseGateError(RuntimeError):
    """Raised when a model's license metadata is disallowed or unavailable."""


@dataclass(frozen=True)
class ThirdPartyNotice:
    name: str
    source: str
    license_id: str


def normalize_license_id(raw: str) -> str:
    cleaned = raw.strip().lower()
    replacements = {
        "apache2": "apache-2.0",
        "apache-2": "apache-2.0",
        "apache 2.0": "apache-2.0",
        "bsd3": "bsd-3-clause",
        "bsd-3": "bsd-3-clause",
        "bsd 3-clause": "bsd-3-clause",
    }
    return replacements.get(cleaned, cleaned)


def extract_license_id(metadata: dict[str, Any]) -> str:
    direct = metadata.get("license")
    if isinstance(direct, str) and direct.strip():
        return normalize_license_id(direct)

    card_data = metadata.get("cardData")
    if isinstance(card_data, dict):
        card_license = card_data.get("license")
        if isinstance(card_license, str) and card_license.strip():
            return normalize_license_id(card_license)

    tags = metadata.get("tags")
    if isinstance(tags, list):
        for tag in tags:
            if not isinstance(tag, str):
                continue
            lowered = tag.strip().lower()
            if lowered.startswith("license:"):
                return normalize_license_id(lowered.split(":", 1)[1])

    return ""


def ensure_allowlisted_license(
    model_id: str,
    metadata: dict[str, Any],
    allowlist: frozenset[str] | set[str] = ALLOWED_LICENSES,
) -> str:
    license_id = extract_license_id(metadata)
    if not license_id:
        raise LicenseGateError(f"License metadata missing for model: {model_id}")

    normalized_allowlist = {normalize_license_id(item) for item in allowlist}
    if license_id not in normalized_allowlist:
        allowed = ", ".join(sorted(normalized_allowlist))
        raise LicenseGateError(
            f"Disallowed license '{license_id}' for {model_id}. Allowed licenses: {allowed}"
        )
    return license_id


def fetch_hf_model_metadata(model_id: str, token: str = "", timeout_sec: int = 30) -> dict[str, Any]:
    url = f"https://huggingface.co/api/models/{model_id}"
    req = request.Request(url)
    if token:
        req.add_header("Authorization", f"Bearer {token}")

    with request.urlopen(req, timeout=timeout_sec) as response:
        payload = response.read().decode("utf-8")
    data = json.loads(payload)
    if not isinstance(data, dict):
        raise LicenseGateError(f"Unexpected metadata payload for model {model_id}")
    return data


def validate_hf_model_license(
    model_id: str,
    token: str = "",
    allowlist: frozenset[str] | set[str] = ALLOWED_LICENSES,
) -> tuple[str, dict[str, Any]]:
    metadata = fetch_hf_model_metadata(model_id=model_id, token=token)
    license_id = ensure_allowlisted_license(model_id=model_id, metadata=metadata, allowlist=allowlist)
    return license_id, metadata


def download_hf_model_snapshot(
    model_id: str,
    destination_dir: Path,
    token: str = "",
    revision: str = "main",
) -> tuple[str, Path]:
    license_id, _ = validate_hf_model_license(model_id=model_id, token=token)

    try:
        from huggingface_hub import snapshot_download  # type: ignore
    except Exception as exc:
        raise RuntimeError(
            "huggingface_hub is required to download model snapshots on AWS training hosts"
        ) from exc

    destination_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = snapshot_download(
        repo_id=model_id,
        revision=revision,
        token=token or None,
        local_dir=str(destination_dir),
        local_dir_use_symlinks=False,
    )
    return license_id, Path(snapshot_path)


def generate_third_party_notices(notices: Iterable[ThirdPartyNotice], output_path: Path) -> None:
    rows = list(notices)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lines: list[str] = [
        "# THIRD_PARTY_NOTICES",
        "",
        "| Dependency | Source | License |",
        "|---|---|---|",
    ]
    for notice in rows:
        lines.append(f"| {notice.name} | {notice.source} | {notice.license_id} |")

    if not rows:
        lines.append("| (none) | n/a | n/a |")

    lines.append("")
    output_path.write_text("\n".join(lines), encoding="utf-8")
