---
name: implement-issue
description: >-
  The routine for implementing an approved plan for a GitHub issue in
  openclimatefix/nged-substation-forecast, in the worktree and draft PR `plan-issue` already
  opened: implement, the green-before-push verification set, mark the PR ready for review, then up
  to two fresh sub-agents adversarially review the diff in turn — one for correctness and for
  keeping the code, tests and prose as short as they can be, then one that mutation-tests the
  change — triaging and pushing after each, stop for human review, never merge. How many of those
  reviews run is set by the issue size: a simple issue arrives here with no plan and no draft PR,
  so it opens its own PR and gets no review at all. Load before writing any code for an issue, and
  when dispatching a sub-agent or a fresh session to solve one. To decide *what* to build, use
  `plan-issue` first.
---

# Implement a GitHub issue

This routine starts from an **approved plan**. `plan-issue` (`/plan-issue <N>`) produces one: it
reads the issue, decides whether it is worth implementing, sizes the process it needs, writes
`plans/<branch-name>.md`, has up to two adversarial sub-agents review the plan, and stops for human
review. It also does step 1 below and opens the PR as a draft, so implementation resumes at step 2.

The one issue that arrives **without** a plan is one `plan-issue` sized as simple: a mechanical
change with no design to approve. It runs steps 1 to 4, then the prose-review pass below if the diff
carries new prose, then stops at step 9 with no adversarial pass. If you reach this skill without
`plan-issue` — a direct instruction to fix something, say — size the issue first, using
`plan-issue` step 3, and say which size you picked before you start.

**How many of the two reviews below to run** comes from that sizing: both for a complex issue, none
for a simple one, and between zero and two for a medium one, under the rules in `plan-issue` step 3.
Running only one means running step 5, not step 7.

When dispatching a sub-agent (or a fresh Claude Code/Desktop session), give it these steps up front
— a report back after step 1 is not finished work.

1. **Set up an isolated worktree** so concurrent sessions don't collide:

    ```bash
    git worktree add .claude/worktrees/<branch-name> -b <branch-name>
    cd .claude/worktrees/<branch-name>
    ln -s /home/jack/dev/python/nged-substation-forecast/.env .env   # if it exists and isn't already there
    ```

    If several sub-agents run concurrently, also give each its own scratchpad subdirectory — two
    agents writing `pr_body.md` into a shared scratchpad root can publish one agent's output under
    another's PR.

2. **Implement**, following every convention in CLAUDE.md — including leaving every doc page that
   touches the change consistent with the code as it now stands, describing only how the code works
   *now* (CLAUDE.md, "Write about the present, not the past"). For a docs change that touches a
   link, run `uv run mkdocs build --strict` and read the rendered HTML under `site/`, not just the
   linter — see the `mkdocs-authoring` skill.

3. **Verify, all green before pushing**: `uv run ruff check .`, `uv run ruff format .`, `uv run
   --all-packages ty check`, `uv run pytest` (add `--run-studies` if the diff touches
   `packages/studies/` or `studies/`), plus (if docs were touched) `uv run pymarkdown scan -r docs
   README.md CLAUDE.md packages/*/README.md` and `uv run mkdocs build --strict`.

    For a medium or complex issue, also run `/code-review medium` in the session. This is the local
    review, which works on a personal Max plan; the managed GitHub service and its `REVIEW.md` do
    not. It runs parallel agents and verifies each candidate against the code before reporting it.
    Fix the genuine findings and re-run the verification set above, so the sub-agent in step 5 does
    not spend its pass on bugs a built-in review would have found.

4. **Commit, push, and mark the PR ready for review.** With a plan, `plan-issue` already opened the
   draft PR: commit, push, and run `gh pr ready <PR-number>`. Rewrite the body for the code that now
   exists: drop the line saying the plan is unapproved, describe the change, and update the size,
   trigger answers and reviews if they changed. Without a plan (the simple path), open the PR now:
   `gh pr create` against `main`, then `gh pr edit --add-label <label>` and `gh pr edit
   --add-assignee JackKelly` (`gh pr create` can't set either), linking the issue so it closes on
   merge. Commit messages end with `Co-Authored-By: Claude <noreply@anthropic.com>`. The
   `github-issue-pr-workflow` skill has the PR checklist and the never-squash-merge rule.

    The body gives the size as the five trigger answers from `plan-issue` step 3, not as the one
    trigger that decided it, and names the adversarial reviews the PR is getting — saying outright
    where that is none. For a PR no sub-agent has attacked, human review is the first line of
    defence, and the reviewer must know that before reading the diff.

5. **First adversarial review: correctness, and cutting the change down.** Run this if the sizing
   called for it. Spawn a *new*, independent sub-agent and give it only the PR number, not the
   implementer's reasoning, so it isn't anchored by it. Tell it to try to refute the PR's claims
   rather than confirm them, because reviewers left to agree tend to agree without evidence, and
   not to report anything `ruff`, `ty`, `pydoclint` or `pymarkdown` already enforce. It attacks four
   things:

    - **Correctness.** Tailor this part to the issue: name the failure modes most worth attacking —
      the risky claim, a behaviour change hiding inside a refactor, whether each new test would
      actually have failed on `main`.
    - **Simplicity of the code.** Which added lines could go without a user of the system telling
      the difference? Does an existing function, package or Patito model already do what a new one
      does? Would a plain function do what a new class, config object or strategy object was added
      for? Is the diff generalising for a second caller that does not exist? For every guard, branch
      and error path the diff adds, can the condition it handles actually arise — which caller,
      which input, which sequence of events? A branch nothing can reach is dead weight plus a test
      that proves nothing; delete both. The exception is the degradation paths
      `docs/design-philosophy/inherent-stability.md` requires in production: an absent or stale
      input is always reachable, because the outside world is not ours to constrain.
    - **Simplicity of the tests.** Is a new test asserting what an existing test already asserts?
      Would a parametrised case, a plain literal or an existing fixture replace a bespoke builder?
    - **Concision of the prose** — every added line of docs, docstring, comment and the PR body.
      Which whole sentences carry no information: restating the heading, summarising what the
      reader has just read, hedging, or narrating what the change replaced? Cut whole sentences, not
      words (CLAUDE.md, "Prose style").

    Ask for each finding as a concrete deletion or replacement. The standard is that the PR adds no
    more lines of code, tests or prose than the change absolutely needs.

6. **Triage, commit and push** — verify each finding against the code rather than accepting it, fix
   the genuine ones, re-run the step-3 verification set, commit and push. Put a one-line tally in
   the PR body for each review (findings raised, accepted, rejected) and record each rejected
   finding with its one-line reason; the tallies show over time which reviews earn their cost. Every
   review step ends with the branch pushed, so GitHub always shows the state the last review left.

7. **Second adversarial review: mutation testing.** Run this if the sizing called for it. Spawn
   *another* new sub-agent — not the one from step 5, and again with no account of your reasoning.
   Its job is to break the production code the PR touches and find out whether the tests notice.
   There is no mutation-testing tool in this repo, so it works by hand in its own detached worktree,
   which keeps a mutation off the branch:

    ```bash
    git worktree add --detach .claude/worktrees/mutate-<branch-name> <branch-name>
    ```

    Keep this worktree (and its `uv sync`) here, not under `/tmp` — see CLAUDE.md, "never create a
    `uv venv` ... under `/tmp`".

    Brief it to assume the tests are weak. For each behavioural claim the diff makes, it introduces
    the smallest bug that breaks the claim, runs the tests covering it, then reverts the bug before
    trying the next. Mutations worth naming: flip a comparison or a boolean, swap two join keys, drop
    a `.filter()` or a `.round()`, return an argument unchanged, shift a lag by one period, make a
    warning path raise. It reports every mutation the suite stays green on, with the file, the
    mutation, and the test that should have caught it, then removes its worktree (`git worktree
    remove`). It never commits or pushes. The table under "NWP grid → H3 orientation coverage" in
    `docs/architecture/testing.md` is what a finished pass looks like.

8. **Triage, commit and push** — on the same terms as step 6. A surviving mutation is a gap only
   where the behaviour it breaks is behaviour we rely on; where it is, tighten or add a test and
   confirm that test goes red against the mutation and green without it. Re-run the step-3
   verification set, commit and push.

9. **Stop for human review. Never merge.** Report the size the issue was given, which reviews ran,
   what each changed, and what each found that you rejected.

**Run a prose-review pass whenever the diff carries more than a few lines of new prose** — a
docstring, a README section, a skill update, a PR body, in a Python or markdown file. Steps 5 and 7
check only concision, not the sentence-level rule sweep the `prose-review` skill runs, so a diff
heavy in new prose can pass both with pronoun, superlative and umbrella-noun violations still in it.
Size the pass using `prose-review`'s own table, fix what it finds, and push again before stopping.

Stay inside the issue's scope; report unrelated design mistakes rather than fixing them.

**Why:** diffs are reviewed in GitHub's UI, and a PR should already have survived an adversarial
pass when a human is asked to review it, so that review is the last line of defence rather than the
first. The reviewer is fresh so the implementer's rationale cannot anchor it. Triage exists because
reviewer findings are often wrong. Mutation testing goes second so it targets the tests that
survive the first round, and it gets its own reviewer because a green suite proves nothing on its
own: only writing the bug and watching shows whether a test would catch it.

**Why the reviews are sized rather than always run:** an adversarial pass costs wall-clock time and
a round of triage, and on a change whose correctness is visible in the diff it finds nothing the
diff did not show. The call is asymmetric — a review too many wastes a sub-agent, a review too few
puts an unattacked design in `main` — so when it is close, run the review.
