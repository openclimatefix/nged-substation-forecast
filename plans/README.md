# Temporary implementation plans

This directory holds **at most one** plan at a time: the mechanical checklist for the PR (or
short PR sequence) currently in flight. It is deleted when that work merges — paste the plan
(or a summary) into the PR body first. This directory is usually empty; that is the intended
state.

Everything durable belongs elsewhere (see the use-case table in `docs/roadmap/index.md`):
design and dependencies go in `docs/roadmap/` pages (step-by-step mechanics under an
"Implementation details (deleted when this ships)" section); methods go in `docs/techniques/`;
the complete ordered task list is the GitHub Project.

Never reference files in this directory from code or from `docs/` — they are deleted on merge,
so any such reference rots.
