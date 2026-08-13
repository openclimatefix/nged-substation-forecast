# Temporary implementation plans

This directory holds **at most one** file — `<branch-name>.md`, the implementation plan for the
work in flight on this branch, written by the `plan-issue` skill before any code is touched and
deleted when that work merges (paste the plan, or a summary, into the PR body first). An issue
`plan-issue` sizes as simple gets no plan at all, so a branch with an empty `plans/` is not
necessarily one where the step was skipped. One worktree per branch is what keeps it to a single
file, so parallel sessions never collide; on `main` the directory is empty. Everything durable
belongs elsewhere, and nothing may link here from code or `docs/`: see "How planning works" in
`docs/roadmap/index.md`.
