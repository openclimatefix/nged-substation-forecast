---
name: implement-issue
description: >-
  The routine for implementing an approved plan for a GitHub issue in
  openclimatefix/nged-substation-forecast: isolated worktree, implement, the green-before-push
  verification set, PR with labels and the JackKelly assignee, a fresh sub-agent adversarially
  reviews the diff, triage, stop for Jack — never merge. Load before writing any code for an
  issue, and when dispatching a sub-agent or a fresh session to solve one. To decide *what* to
  build, use `plan-issue` first.
---

# Implement a GitHub issue

This routine starts from an **approved plan**. The `plan-issue` skill (invoked as
`/plan-issue <N>`) is how you get one: it reads the issue, decides whether it is worth
implementing at all, writes `plans/<branch-name>.md`, has two fresh sub-agents adversarially
review the plan — one for simplicity, one for correctness and testability — and stops for Jack. It
also does step 1 below, so when it hands over, the worktree and branch already exist and
implementation resumes at step 2.

When dispatching a sub-agent (or a fresh Claude Code/Desktop session), give it these steps up
front — a report back after step 1 is not finished work.

1. **Set up an isolated worktree** so concurrent sessions don't collide:

    ```bash
    git worktree add .claude/worktrees/<branch-name> -b <branch-name>
    cd .claude/worktrees/<branch-name>
    ln -s /home/jack/dev/python/nged-substation-forecast/.env .env   # if it exists and isn't already there
    ```

    If several sub-agents run concurrently, also give each its own scratchpad subdirectory —
    a shared scratchpad root means two agents writing e.g. `pr_body.md` can collide and one
    agent's output gets briefly published under another's PR.

2. **Implement**, following every convention in CLAUDE.md — including leaving every doc page
   that touches the change consistent with the code as it now stands, describing only how the
   code works *now* (CLAUDE.md, "Write about the present, not the past"). For a docs change that
   touches a link, run `uv run mkdocs build --strict` and read the rendered HTML under `site/`,
   not just the linter — Python-Markdown has rendering gotchas that neither `pymarkdown scan`
   nor a successful `mkdocs build` catches on their own (see the `mkdocs-authoring` skill).

3. **Verify, all green before pushing**: `uv run ruff check .`, `uv run ruff format .`,
   `uv run --all-packages ty check`, `uv run pytest`, plus (if docs were touched)
   `uv run pymarkdown scan -r docs README.md CLAUDE.md packages/*/README.md` and
   `uv run mkdocs build --strict`.

4. **Push and open the PR** against `main`, with labels and `JackKelly` as assignee (`gh pr
   create` can't set either — follow with `gh pr edit --add-label <label>` and `gh pr edit
   --add-assignee JackKelly`), linking the issue so it closes on merge. Commit messages end
   with `Co-Authored-By: Claude <noreply@anthropic.com>`. See the `github-issue-pr-workflow`
   skill for the full PR checklist and the never-squash-merge rule.

5. **Spawn a *new*, independent sub-agent to adversarially review the PR** — give it only the
   PR number, not the implementer's reasoning, so it isn't anchored by it. Tailor the reviewer
   brief to the issue: name the failure modes most worth attacking (the risky claim, a
   behaviour change hiding inside a refactor, whether a new test would actually have failed on
   `main`) rather than asking for a generic review.

6. **Triage the review's findings** — verify each against the code rather than accepting it,
   fix genuine defects, push, and record why any finding was rejected.

7. **Stop and wait for Jack's review. Never merge.**

Stay inside the issue's scope; report unrelated design mistakes rather than fixing them.

**Why:** Jack reviews diffs in GitHub's UI and wants a PR to already have survived an
adversarial pass by the time he looks at it, so his review is the last line of defence rather
than the first. The fresh-reviewer requirement exists so the reviewer cannot be anchored by the
implementer's rationale; the triage step exists because reviewer findings are often wrong and
must not be applied uncritically.
