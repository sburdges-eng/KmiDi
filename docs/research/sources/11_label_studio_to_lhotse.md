# Convert Label Studio exports to Lhotse manifests

**Item:** Convert Label Studio exports to Lhotse manifests  
**Task:** Plan conversion from Label Studio exports to Lhotse manifests.  
**Secondary tasks:** Reuse Lhotse format from make_jepa_manifest.py.  
**Context:** Label Studio → Lhotse; make_jepa_manifest produces RecordingSet/SupervisionSet/CutSet.  
**Verification status:** PARTIALLY_VERIFIED  
**Verification basis:** repo_scan_only + source_title_only; make_jepa_manifest outputs Lhotse JSONL; no Label Studio in repo; no pasted source.  
**Known facts:**
- make_jepa_manifest.py outputs Lhotse JSONL to manifests/
- No Label Studio in repo  
**Unknowns:** Label Studio export schema; mapping to Lhotse supervisions; audio paths; task type.  
**Assumptions:** None  
**Constraints:** UNKNOWN  
**Ambiguities:** Export format version; multi-task labels.  
**Source text / data:** UNKNOWN (full source block not present)  
**Output format:** Script under scripts/; doc; optional config for paths and mapping.
