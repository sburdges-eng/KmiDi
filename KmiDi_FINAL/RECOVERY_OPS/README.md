# RECOVERY_OPS

This folder contains recovery scripts, documentation, inventories, and logs for the KmiDi_FINAL consolidation.

## Scripts
- disconnect_test.sh: Rename original roots with _DISCONNECTED suffix
- restore_disconnected.sh: Restore disconnected roots
- analyze_file_access.sh: Analyze captured file access logs for external paths

## Docs
- PHASE1_ROLLBACK.md: How to rollback Phase 1
- PHASE3_ARCHIVE_RESTORE.md: Restore from archive
- PHASE4_QUARANTINE_RESTORE.md: Restore from quarantine
- PHASE6_POST_DELETE.md: Recovery after deletion
- ARCHIVE_LOCATIONS.md: Archive storage locations

## Inventories
- Archive, quarantine, and graveyard manifests

## Logs
- disconnection, file access, archive verification, quarantine verification
