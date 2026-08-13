---
name: implement-issue
description: >-
  The routine for implementing an approved plan for a GitHub issue in
  openclimatefix/nged-substation-forecast: isolated worktree, implement, the green-before-push
  verification set, PR with labels and the JackKelly assignee, then up to two fresh sub-agents
  adversarially review the diff in turn — one for correctness and for keeping the code, tests and
  prose as short as they can be, then one that mutation-tests the change — triaging and pushing
  after each, stop for human review, never merge. How many of those reviews run is set by the issue
  size: a simple issue arrives here with no plan and gets none. Load before writing any code for an
  issue, and when dispatching a sub-agent or a fresh session to solve one. To decide *what* to
  build, use `plan-issue` first.
---

# Implement a GitHub issue

This routine starts from an **approved plan**. The `plan-issue` skill (invoked as
`/plan-issue <N>`) is how you get one: it reads the issue, decides whether it is worth
implementing at all, sizes how much process the issue needs, writes `plans/<branch-name>.md`, puts
it through up to two adversarial sub-agent reviews — one for simplicity, one for correctness and
testability — and stops for human review. It also does step 1 below, so when it hands over, the
worktree and branch already exist and implementation resumes at step 2.

The one issue that arrives here **without** a plan is one `plan-issue` sized as simple: a
mechanical change with no design to approve, which runs steps 1 to 4 and then stops for human
review with no adversarial pass at all. If you reach this skill without having gone through
`plan-issue` — a direct instruction to fix something, say — make that sizing judgement first,
using the criteria in `plan-issue` step 3, and say which size you picked before you start.

**How many of the two reviews below to run** comes from that sizing: both for a complex issue,
none for a simple one, and between zero and two for a medium one — your choice, under the rules in
`plan-issue` step 3. Running only one means running step 5, not step 7.

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

4. **Commit, push and open the PR** against `main`, with labels and `JackKelly` as assignee (`gh
   pr create` can't set either — follow with `gh pr edit --add-label <label>` and `gh pr edit
   --add-assignee JackKelly`), linking the issue so it closes on merge. Commit messages end
   with `Co-Authored-By: Claude <noreply@anthropic.com>`. See the `github-issue-pr-workflow`
   skill for the full PR checklist and the never-squash-merge rule.

    The body says **how the issue was sized and which adversarial reviews it is getting** — and
    where that is none, says so outright. A PR that no sub-agent has attacked is one where human
    review is the first line of defence rather than the last, and the reviewer has to know that
    before reading the diff.

5. **First adversarial review: correctness, and cutting the change down.** Run this if the sizing
   called for it. Spawn a *new*, independent sub-agent and give it only the PR number, not the
   implementer's reasoning, so it isn't anchored by it. It attacks four things:

    - **Correctness.** Tailor this part to the issue rather than asking for a generic review:
      name the failure modes most worth attacking — the risky claim, a behaviour change hiding
      inside a refactor, whether each new test would actually have failed on `main`.
    - **Simplicity of the code.** Which added lines could go without a user of the system
      telling the difference? Does an existing function, package or Patito model already do what
      a new one does? Would a plain function do what a new class, config object or strategy
      object was added for? Is the diff generalising for a second caller that does not exist?
      For every guard, branch and error path the diff adds, can the condition it handles
      actually arise — which caller, which input, which sequence of events? A branch nothing can
      reach, because no caller passes that argument or the Patito schema already rejects that
      row, is dead weight plus a test that proves nothing; say so and delete both. The exception
      is the degradation paths `docs/design-philosophy/inherent-stability.md` requires in
      production: an absent or stale input is always reachable, because the outside world is not
      ours to constrain.
    - **Simplicity of the tests.** Is a new test asserting what an existing test already
      asserts? Would a parametrised case, a plain literal or an existing fixture replace a
      bespoke builder?
    - **Concision of the prose** — every added line of docs, docstring, comment and the PR body
      itself. Which whole sentences carry no information: restating the heading, summarising
      what the reader has just read, hedging, or narrating what the change replaced? Cut whole
      sentences, not words (CLAUDE.md, "Prose style").

    Ask for each finding as a concrete deletion or replacement. The standard is that the PR adds
    no more lines of code, tests or prose than the change absolutely needs.

6. **Triage, commit and push** — verify each finding against the code rather than accepting it,
   fix the genuine ones, re-run the step-3 verification set, commit and push, and record each
   rejected finding with its one-line reason. Every review step ends with the branch pushed, so
   what is on GitHub always shows the state the last review left behind.

7. **Second adversarial review: mutation testing.** Run this if the sizing called for it. Spawn
   *another* new sub-agent — not the one from step 5, and again with no account of your reasoning.
   Its job is to break the production code the PR touches and find out whether the tests notice.
   There is no mutation-testing tool in this repo, so it does this by hand, in its own detached
   worktree, which keeps a mutation off the branch:

    ```bash
    git worktree add --detach .claude/worktrees/mutate-<branch-name> <branch-name>
    ```

    Brief it to take each behavioural claim the diff makes, introduce the smallest bug that
    breaks that claim, run the tests covering it, then revert the bug before trying the next
    one. Mutations worth naming: flip a comparison or a boolean, swap two join keys, drop a
    `.filter()` or a `.round()`, return an argument unchanged, shift a lag by one period, make a
    warning path raise. It reports every mutation the suite stays green on, with the file, the
    mutation, and the test that should have caught it — then removes its worktree (`git worktree
    remove`). It never commits or pushes. The table under "NWP grid → H3 orientation coverage"
    in `docs/architecture/testing.md` is what a finished pass of this looks like.

8. **Triage, commit and push** — on the same terms as step 6. A surviving mutation is a gap only
   where the behaviour it breaks is behaviour we rely on; where it is, tighten or add a test and
   confirm that test goes red against the mutation and green without it. Re-run the step-3
   verification set, commit and push.

9. **Stop and wait for human review. Never merge.** Report the size the issue was given, which
   reviews ran, what each of them changed, and what each found that you rejected.

Stay inside the issue's scope; report unrelated design mistakes rather than fixing them.

**Why:** Jack reviews diffs in GitHub's UI and wants a PR to already have survived an
adversarial pass by the time he looks at it, so his review is the last line of defence rather
than the first. The fresh-reviewer requirement exists so the reviewer cannot be anchored by the
implementer's rationale; the triage step exists because reviewer findings are often wrong and
must not be applied uncritically. Mutation testing goes second because it should be aimed at the
tests that survive the first round, not at ones the first round deletes — and it gets its own
reviewer because a green suite proves nothing on its own: the only way to learn whether a test
would catch the bug it exists for is to write that bug and watch.

**Why the reviews are sized rather than always run:** an adversarial pass costs wall-clock time
and produces findings that have to be triaged, and on a change whose correctness is visible in the
diff it finds nothing that the diff did not already show. Spending it there delays the change and
buries the human reviewer in process for no gain. The number is a judgement call precisely because
the cost of getting it wrong is asymmetric — a review too many wastes a sub-agent, a review too
few puts an unattacked design in `main` — so when the call is close, run the review.
