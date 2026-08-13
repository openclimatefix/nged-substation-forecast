---
name: plan-issue
description: >-
  Turn a GitHub issue in openclimatefix/nged-substation-forecast into a reviewed implementation
  plan, writing no code: read the issue and its comments, decide whether it is worth implementing
  at all, size how much process it needs, write a plan that may overrule a stale issue body, put it
  through up to two fresh adversarial sub-agent reviews — one hunting for a simpler approach, then
  one checking correctness and testability — triage the findings, stop for human review. A very
  simple issue skips the plan entirely and goes straight to `implement-issue`. Load whenever the
  user asks for a plan for an issue, says "plan issue N" or "/plan-issue N", asks whether an issue
  is worth doing, or asks you to think through an implementation before touching code. To implement
  an approved plan, use `implement-issue` instead.
---

# Plan a GitHub issue

The output is a **plan file and a recommendation**, never a code change. A human approves the plan
before any code moves, so the review of the eventual diff checks execution rather than discovering
the design for the first time. The exception is an issue simple enough that there is no design to
approve: step 3 sizes the issue, and a simple one skips this skill entirely.

Take the issue number from the invocation (`/plan-issue 500`). If several are given, run the whole
procedure once per issue, each on its own branch — do not merge them into one plan unless the
issues turn out to be the same change, in which case say so explicitly in step 2.

## 1. Gather

Read all of this before forming an opinion:

```bash
gh issue view <N> --json number,title,body,labels,assignees,state,url
gh issue view <N> --comments
```

- **The comments matter as much as the body.** An issue body is written once; the comments are
  where it gets corrected. Where they disagree, the comments usually win.
- **Follow the issue's links.** Most issues in this repo link to a `docs/` page (rendered site
  links) and to the PR or review that spawned them. Read those, and read the code the issue
  names — the issue's description of current behaviour is a claim to verify, not a fact.
- **Check for work already in flight**, because sessions run in parallel:

  ```bash
  gh pr list --state open --search "<N>"
  git branch -a --list '*issue-<N>*'
  ```

  If something is already open against the issue, stop and report that instead of planning.

- **Read the relevant docs.** `docs/` holds prior discussion that predates the issue:
  `docs/roadmap/` for the milestone arc and any "Implementation details" section covering this
  work, `docs/architecture/` for why the current design is what it is, `docs/techniques/` for
  method write-ups, `docs/design-philosophy/` for the rules the plan has to obey.

## 1a. Name the session

As soon as you have the issue title, **open your very next reply with a session-title line and
nothing above it**:

```text
Planning: Cap late-series metadata table (#510)
```

An imperative phrase naming what the issue *changes*, about 50 characters, then the issue number
in parentheses. Not the issue's own title verbatim — those run long and bury the point (#512's is
"Reject unknown keys in config_overrides (extra=\"forbid\" on BaseForecasterConfig)", which wants
to become "Reject unknown config_overrides keys (#512)").

The user picks sessions out of the desktop app's session list, where the default title is derived
from the `/plan-issue <N>` invocation and reads "Plan issue 510" — which does not say what the
session is about once several are open at once.

**Do not reach for a tool to do this.** Every session-management tool refuses the session it is
running in, by contract, so there is nothing to call. The line above is a hint to whatever titles
the session, and failing that a title the user can copy in one gesture. If you find the session list
still showing "Plan issue \<N\>" after a run, say so rather than working around it — that is the
signal that this needs solving at the harness level instead.

## 2. Decide whether it is worth implementing

This is a real gate, not a formality. State a clear verdict before writing any plan, and be
willing to answer "no" or "not like this". Grounds for pushing back:

- The issue's premise is **factually wrong** about the current code — the bug it describes is
  already fixed, or never reproduced in the first place.
- The issue is **stale**: it was written against a design that has since changed, so its proposed
  fix targets code that no longer exists or no longer works that way.
- The change **conflicts with a design principle** in
  `docs/design-philosophy/design-principles.md`, and the trade is not worth it.
- It is **subsumed** by another open issue, or its two halves belong in different issues.
- The cost is out of proportion to the benefit at this project's stage.

If the verdict is "not worth it" or "worth it, but not as described", say so, give the reasoning,
and stop there rather than dutifully planning something you have just argued against. A short
"here is why this issue should be closed / re-scoped / merged into #M" is a legitimate and
valuable output of this skill.

If the verdict is "worth it, roughly as described", continue — but you are still free to overrule
any specific mechanism the issue proposes. Say plainly which parts of the issue body you are
departing from and why.

## 3. Size the process to the issue

Not every issue needs a plan, and not every plan needs two reviews. Pick one of three sizes,
**state it in your reply with a one-line reason before doing anything else**, and let the human
override it.

**Simple — no plan, and no agentic review at all.** Skip the rest of this skill and go straight to
`implement-issue` at its step 1: worktree, implement, the verification set, PR, stop for human
review. An issue is simple when all of these hold:

- The change is mechanical, and reading the diff is enough to see whether it is right — a rename or
  a find-and-replace across files, a prose or docs edit, deleting dead code, a dependency bump, a
  one-line fix whose failing test you can name before you start.
- There is one obvious way to do it, so there is no design to approve.
- It touches no Patito contract, no production degradation path, no asset graph, and nothing about
  what gets stored.
- The green-before-push verification set is the whole of the risk: if `ruff`, `ty`, `pytest` and
  the markdown lint pass, the change is right.

Issue #583 — replace one word across seven skill and docs files — is the shape to picture. Say in
the same reply that you are taking this path, and open it with `Implementing:` rather than the
`Planning:` prefix from step 1a.

**Complex — the full routine**: a plan, both plan reviews below, and both diff reviews in
`implement-issue`. An issue is complex when any of these hold:

- It adds or changes a Patito model, a Delta table, an asset, or anything else about what gets
  stored.
- It touches the production serving path, or a degradation rule in
  `docs/design-philosophy/inherent-stability.md`.
- More than one design would defensibly satisfy it, so the choice wants approving before code moves.
- It spans enough code that you could not name every caller of what it changes without searching.

**Medium — everything else.** Write the plan, then choose how much review to spend on it: between
zero and two of the plan reviews below (steps 5 and 7, each with its triage step), and between
zero and two of the diff reviews in `implement-issue`. The choice is yours, under four rules:

- **Run the earlier of a pair first.** One plan review means the simplicity review, not the
  correctness one, because a plan bigger than its issue survives a correctness review intact. One
  diff review means the correctness-and-cut-it-down review, not the mutation pass.
- **Spend a plan review** on a plan that adds an abstraction, a config field, a column or a new
  file (the simplicity review), or on a plan whose account of current behaviour was hard to pin
  down or whose tests are the risky part (the correctness review).
- **Spend a diff review** on a diff that came out larger than the plan implied or grew a lot of
  prose (the first review), or on a change whose whole value is a behaviour the tests are meant to
  pin down — a boundary, an ordering, a join key, a warning path that must not raise (the mutation
  pass).
- **When you cannot decide, run it.** A review costs one sub-agent; the other error puts a design
  nobody attacked into `main`.

Carry the size and the chosen counts forward: they go in the plan file (step 4), in the report
(step 9), and into the PR body when `implement-issue` opens it, so whoever reviews the diff knows
what scrutiny it has already had.

## 4. Write the plan

The plan is a committed file on the issue's own branch, so set up the worktree now rather than
leaving it to implementation — this is step 1 of the `implement-issue` skill, pulled forward:

```bash
git worktree add .claude/worktrees/<branch-name> -b <branch-name>
cd .claude/worktrees/<branch-name>
ln -s /home/jack/dev/python/nged-substation-forecast/.env .env   # if it exists and isn't already there
```

Then write the plan to **`plans/<branch-name>.md`** inside that worktree, commit it and push the
branch:

```bash
git push -u origin <branch-name>
```

**Then hand over the plan before anything else happens to it.** Say in your next reply that the
plan is ready and give a **clickable markdown link** to it, written as the path relative to the
worktree so the terminal turns it into a link: `[plans/<branch-name>.md](plans/<branch-name>.md)`.
Do this *before* launching any reviewer. The reviews take minutes, and whoever wants to read the
first draft — or to stop the work outright — should not have to wait for them, nor work out
afterwards which parts of the plan the reviews wrote.

One worktree per issue means `plans/` holds exactly one file on each branch, so the "at most one
plan" rule in `plans/README.md` holds with no coordination between parallel sessions. Committing
now rather than at implementation time makes the plan durable: it survives Claude Code shutting
down, and it is already in the diff when the PR opens, so the reviewer sees the plan next to the
code that claims to follow it. It is deleted at ship time into the PR body, per the existing rule.

Open the file with two brief summaries, before any of the sections below: what the problem or
feature is, then what the planned solution is. A reader who stops after the first two paragraphs
should already know what is broken (or missing) and roughly how the plan fixes it, without reading
the file-by-file detail.

The plan covers:

- **Verdict, size and departures** — the step-2 conclusion, the step-3 size with the reviews it
  buys, and every point where the plan differs from the issue body, each with its reason.
- **What changes, file by file** — named files and functions, with what happens to each. Prefer
  naming the actual symbols over describing the change in the abstract.
- **Design-philosophy check** — how the change sits against
  `docs/design-philosophy/inherent-stability.md`: does this code path run in production (degrade,
  widen bands, record the degradation on the row) or in R&D (fail fast)? If it adds or edits an
  asset check, confirm it is `WARN`/`blocking=False` and that its body cannot raise. Cite the
  hypotheses in `docs/design-philosophy/engineering-hypotheses.md` by label (`H1`, `T1.2`) where
  the change is meant to deliver one. If the plan trades away a principle from
  `design-principles.md`, name the principle and say what is bought in return.
- **Tests** — which tests get added or changed, and for each new test, *the assertion it makes
  that would fail on `main` today*. A test that passes before and after the change is not a test
  of this change.
- **Docs to update** — every page left inconsistent by the change, written to describe how the
  code works *now* (CLAUDE.md, "Write about the present, not the past"). Include ship-time
  triage if this issue completes a roadmap item: promote surviving design decisions to their
  permanent home, delete the "Implementation details" section, update the roadmap status banner.
- **Verification commands** — the green-before-push set from the `implement-issue` skill, plus
  anything this change specifically needs (for example `uv run pytest --run-network -m network` for
  convention-sensitive NWP conversion code, or `uv run mkdocs build --strict` *and reading the
  rendered HTML* for any change that touches links).
- **Risks and open questions** — the things the human reviewer should decide, stated as questions
  with your recommendation attached.

Do not paste large code blocks into the plan. Name the change; the implementer writes the code.

## 5. First adversarial review: is there a simpler way?

Run this if step 3 called for it — always for a complex issue, and for a medium one whenever it is
the review you chose. The two reviews go to two separate fresh sub-agents, in this order:
simplicity first, then correctness. Simplifying a plan can break it, so the correctness reviewer
has to see the plan that simplification left behind, not the one that went in.

Launch a **new** sub-agent for this first pass. Give it the issue number and the path to the plan
file, and **nothing about your reasoning** — it must not be anchored by the argument that produced
the plan. Its single job is to find a simpler way to satisfy the issue, and its default assumption
is that one exists.

Name the attacks that apply to this issue:

- **Can the problem actually happen?** For each failure the plan defends against, ask the reviewer
  to name the caller, the input and the sequence of events that produces it. A bug nobody can
  reach — because no caller passes that argument, or the Patito schema already rejects that row,
  or the data source cannot emit that value — is theoretical, and the plan should drop it rather
  than carry code and tests for it. The degradation paths
  `docs/design-philosophy/inherent-stability.md` requires in production are the exception: an
  absent or stale input is always reachable, because the outside world is not ours to constrain.
- Which parts of the plan solve a problem the issue did not ask about? Cut them and say what
  breaks.
- Is there an existing function, package or Patito model that already does what a proposed new
  one would? (`packages/` is small enough to check.)
- Would a plain function do what a new class, abstract base class, config object or strategy
  object is proposed for?
- Is a new column, table, asset or config field earning its place, or is the value already
  derivable from what is stored?
- Is the plan generalising for a second caller that does not exist? This project is young; a
  breaking change later is cheap.
- What is the smallest change to the existing code that a user of the system could not tell apart
  from the plan's version?

Ask for each simplification as a concrete alternative — what to do instead, which files it
touches, and what capability is given up by taking it.

**The reviewer is not confined to the plan's own scope.** Say so explicitly in the brief: if the
simplest way to satisfy the issue is a different architecture — collapsing two packages into one,
replacing an abstraction with a different one, moving a responsibility from the serving path into
the training loop, changing how something is stored — it should propose that, even though it is far
larger than the issue. Nobody else depends on this code yet, so a sweeping change costs a rewrite
rather than a migration path, and the goal is code that ends up elegant rather than code that
accreted around whatever was there first.

A proposal that big has to arrive with its case made, not as a suggestion:

- **What it buys** — which specific complexity disappears, and where the code gets shorter or the
  concepts get fewer.
- **What it costs** — which packages, assets, saved models, Delta tables or docs pages have to
  change, and roughly how much work that is.
- **What it gives up** — what the current design does that the new one cannot, and which principle
  in `docs/design-philosophy/design-principles.md` is being traded away.
- **Whether it has to happen now** — can the issue ship under the current architecture, with the
  rearchitecture as its own issue afterwards? That is usually the answer, and saying so is not a
  weaker recommendation.

## 6. Triage and revise

Verify each proposed simplification against the code rather than accepting it — **reviewer
findings are often wrong, and applying them uncritically makes the plan worse**. Take the genuine
ones into the plan file. Reject a simplification when it drops something the issue actually asked
for, or trades away a rule in `docs/design-philosophy/` — and say which, in one line.

For each finding you reject, record the finding and the one-line reason in the plan file, so the
human reviewer can see what was considered and dismissed. Commit the revised plan and push, so the
branch on GitHub is never behind what the reviews have already done.

**A proposed rearchitecture is the human reviewer's call, not yours** — it is bigger than the
issue, so neither adopting it nor dropping it silently is right. Put it in the plan's "Risks and
open questions" with the sub-agent's pros and cons, your view of whether it is genuinely simpler,
and a recommendation on whether it should become its own issue. Then plan the issue under the
current architecture unless the human reviewer says otherwise.

## 7. Second adversarial review: correctness and testability

Run this if step 3 called for it. Launch **another** new sub-agent — not the one from step 5, and
again with no account of your reasoning or of what the first review changed. Give it the issue number and the path to the
revised plan file. Its job is to establish whether the plan, as now written, is correct and whether
its tests would actually catch it being wrong.

Tailor the brief to this issue's specific failure modes rather than asking for a generic review.
The attacks worth naming, when they apply:

- Does the plan's description of *current* behaviour actually match the code on `main`?
- Would each proposed test really have failed before the change? A test that passes on `main`
  today tests nothing about this change.
- Is any part of the plan untestable as written — needing network, wall-clock time, or a whole
  trained model to exercise one branch? Say what would have to change to make it testable.
- Does the change fail closed anywhere production is supposed to degrade — in particular, can any
  warning path now raise?
- Is a refactor hiding a behaviour change inside it?
- Does the plan miss a caller, a doc page, or a schema that this change invalidates?
- Is the issue body stale in a way the plan inherited rather than caught?

Ask the reviewer for findings with file/line evidence, and for an explicit verdict on each: real
defect, or not.

## 8. Triage the findings

Verify each finding against the code, on the same terms as step 6: fix the genuine ones in the
plan file, and record each rejected finding with its one-line reason. Commit and push the updated
plan, so the branch carries the original plan and what each review did to it.

## 9. Stop

Report: the verdict from step 2, the size from step 3 and which reviews it bought, a short summary
of the plan, what each review that ran changed, and what each found that you rejected. Give the
branch name and, again, the clickable link to the plan file.

**Do not write any code, and do not open a PR.** Once a human approves the plan, implementation runs
under the `implement-issue` skill, resuming at its step 2 in the worktree this skill already
created — implement, verify, PR, then the diff reviews step 3 called for, each by a further
independent sub-agent, triaging and pushing after each, stop.
