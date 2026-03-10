# OneDrive KmiDi-remote import

One-time import of key consolidation/implementation docs from OneDrive (see [DISCOVERY_ONEDRIVE_KMIDI_REMOTE.md](../DISCOVERY_ONEDRIVE_KMIDI_REMOTE.md)).

**Sources (either):**

- OneDrive: `~/Library/CloudStorage/OneDrive-Personal/JUCE 2/Desktop/KmiDi-remote`
- SSD (when mounted): `/Volumes/Sean's SSD/KmiDi_MASTER_VAULT/KmiDi` — same four docs; import was run from here when OneDrive was unavailable.

## Required: copy these four into this directory

When OneDrive is synced and the path is available, run from repo root:

```bash
./docs/onedrive-import/import_from_onedrive.sh
```

Or set `ONEDRIVE_KMIDI_REMOTE` if your path differs. Manual copy:

```bash
SRC="$HOME/Library/CloudStorage/OneDrive-Personal/JUCE 2/Desktop/KmiDi-remote"
DEST="$(git rev-parse --show-toplevel)/docs/onedrive-import"
cp "$SRC/KMIDI_CONSOLIDATION_SUMMARY.md" "$DEST/"
cp "$SRC/IMPLEMENTATION_PLAN.md" "$DEST/"
cp "$SRC/DESIGN_Integration_Architecture.md" "$DEST/"
cp "$SRC/MERGER_INFRASTRUCTURE_COMPLETE.md" "$DEST/"
```

## Optional: additional root-level docs (from discovery table)

**Already imported from SSD:** KMIDI_README.md, KMIDI_STRUCTURE_PLAN.md, ARCHITECTURE_REVIEW_2025-12-30.md, INFRASTRUCTURE.md, ROADMAP_Implementation.md, IMPLEMENTATION_ALTERNATIVES.md.

To pull any others (e.g. from OneDrive when synced):

```bash
# Same SRC and DEST as above
for f in HOW_TO_DEV_OP_101.md OPTIMAL_WORKFLOW_SUMMARY.md PUSH_STRATEGY.md \
         RELEASE_NOTES_v1.0.0.md RECOMMENDATIONS_Improvements.md QUICKSTART_TIER123.md \
         COPILOT_INSTRUCTIONS.md; do
  [ -f "$SRC/$f" ] && cp "$SRC/$f" "$DEST/"
done
```

## Production_Workflows and Songwriting_Guides

These are full folder trees (30+ guides each). They are **not** copied into the repo. See [ONEDRIVE_REFERENCE_LINKS.md](ONEDRIVE_REFERENCE_LINKS.md) for their locations and contents; use the OneDrive path as reference while they stay on cloud.

Do not develop from OneDrive; treat as read-only source for a single import.
