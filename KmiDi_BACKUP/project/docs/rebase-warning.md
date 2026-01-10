# Rebase Warning – 2026-01-02

- Attempted to rebase `origin/Test` onto `origin/main`.
- Rebase aborted after first commit due to hundreds of add/add conflicts across docs, templates, assets, C++/Python/TypeScript sources, JUCE vendor files, and knowledge base JSON.
- No changes applied; rebase is not in progress. Working branch is back on `main`.
- Local working tree note: `src/components/GuideViewer.tsx` is modified and unstaged on `main`.

Next steps:
- Decide whether to cherry-pick specific commits from `origin/Test` instead of rebasing the whole branch.
- If rebasing the full branch is still desired, prepare a conflict-resolution plan (ours/theirs/manual) before retrying.***
