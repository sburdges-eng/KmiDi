# PR Checklist

Remote target: `https://github.com/sburdges-eng/KmiDi-listening-core.git`

1. Ensure network access is restored and refresh baseline:
   - `git fetch origin main --prune`
   - `git checkout main && git reset --hard origin/main`
2. Re-run reconciliation script and verify report diffs are expected.
3. Apply Batch A and Batch B from `recovery_reports/patchset_plan.md`.
4. Re-run configure/build/tests.
5. Push branch and open PR:
   - `git push -u origin codex/recovery-reconcile-20260218-043343`
   - `gh pr create --base main --head codex/recovery-reconcile-20260218-043343 --title "Recovery reconcile reports and gated apply plan" --body-file recovery_reports/patchset_plan.md`
