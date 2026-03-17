# Cleaning up the git graph (fewer parallel lines)

The "20+ parallel" lines in the graph come from **merge commits** already on `origin/main` (PR merges like #137, #130, #101, #127). Your local 8 commits are linear; the spaghetti is in the history below that.

## Options (safest → nuclear)

### 1. Simplify going forward (no history rewrite)

- **GitHub:** For new PRs, use **Squash and merge** (or **Rebase and merge**) instead of **Create merge commit**. New history stays linear.
- **Drop unused remote:** Fewer refs = slightly simpler view.
  ```bash
  git remote remove local-recovery   # if you don't need reconcile-20260218-0329
  ```
- **Squash your next push:** Before pushing, collapse your 8 local commits into one so you add a single commit to origin:
  ```bash
  git reset --soft origin/main
  git commit -m "feat: headless engine, large-file cleanup, warning fixes, and doc updates"
  git push origin main
  ```
  That leaves existing origin history as-is but keeps the *new* tip linear.

### 2. Flatten only your local 8 commits (already linear)

Your current 8 commits are already a straight line. If you just want one commit on top of `origin/main` when you push:

```bash
git reset --soft origin/main
git commit -m "feat: headless engine, large-file cleanup, warning fixes, docs"
git push origin main
```

Again, this does **not** change the 20+ parallel lines already on origin; it only keeps your addition as one commit.

### 3. Nuclear: flatten entire history (rewrite origin)

**Warning:** Rewrites history. Force-push required. Anyone else with a clone must re-clone or `git fetch origin && git reset --hard origin/main` (and lose local commits on main).

One way to get a single linear history from the current tree:

```bash
# 1. Make sure main is up to date locally
git checkout main
git fetch origin

# 2. Create a new root commit with the current tree (all files as they are now)
git checkout --orphan main-flat
git add -A
git commit -m "chore: flatten main history (single commit)"

# 3. Replace main with the flat history (destructive)
git branch -D main
git branch -m main-flat main

# 4. Force-push (only if you own the repo and accept the consequences)
git push origin main --force
```

After this, the graph is a single line. All previous commit hashes are gone; links to old commits (e.g. in issues, PRs) will be broken.

---

## Squash and keep history

You can squash into one commit on `main` and still keep the full history:

1. **Tag the tip before squashing** (so that tip stays reachable):
   ```bash
   git tag archive/main-pre-squash-2026-03   # or: git tag archive/main-pre-squash-2026-03 main
   ```
2. **Then squash** (e.g. `git reset --soft origin/main` and `git commit -m "..."`).
3. **History is preserved:** `git log archive/main-pre-squash-2026-03` shows the full pre-squash history; `main` has the single squashed commit.

**Note:** If that history ever contained files over GitHub's 100 MB limit (e.g. the `kmidi_brain` binary), do not push the archive tag — GitHub will reject it. Keep the tag local only; you can still run `git log archive/main-pre-squash-2026-03` to see the full history.

On GitHub, **Squash and merge** keeps the PR’s commit list and discussion; the branch (or a tag) keeps the commits in the repo.

---

**Recommendation:** Use **1** (squash merges going forward + optional squash of your 8 before push) unless you specifically need a single-commit history for the whole repo, in which case use **3** with care.
