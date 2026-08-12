---
name: plan-issue
description: >-
  Turn a GitHub issue in openclimatefix/nged-substation-forecast into a reviewed implementation
  plan, writing no code: read the issue and its comments, decide whether it is worth implementing
  at all, write a plan that may overrule a stale issue body, put it through two fresh adversarial
  sub-agent reviews — one hunting for a simpler approach, then one checking correctness and
  testability — triage the findings, stop for Jack. Load whenever Jack asks for a plan for an issue,
  says "plan issue N" or "/plan-issue N", asks whether an issue is worth doing, or asks you to
  think through an implementation before touching code. To implement an approved plan, use
  `implement-issue` instead.
---

# Plan a GitHub issue

The output is a **plan file and a recommendation**, never a code change. Jack approves a plan
before any code moves, so his review of the eventual diff checks execution rather than discovering
the design for the first time.

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

Jack picks sessions out of the desktop app's session list, where the default title is derived from
the `/plan-issue <N>` invocation and reads "Plan issue 510" — which does not say what the session
is about once several are open at once.

**Do not reach for a tool to do this.** Every session-management tool refuses the session it is
running in, by contract, so there is nothing to call. The line above is a hint to whatever titles
the session, and failing that a title Jack can copy in one gesture. If you find the session list
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

## 3. Write the plan

The plan is a committed file on the issue's own branch, so set up the worktree now rather than
leaving it to implementation — this is step 1 of the `implement-issue` skill, pulled forward:

```bash
git worktree add .claude/worktrees/<branch-name> -b <branch-name>
cd .claude/worktrees/<branch-name>
ln -s /home/jack/dev/python/nged-substation-forecast/.env .env   # if it exists and isn't already there
```

Then write the plan to **`plans/<branch-name>.md`** inside that worktree, and commit it.

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

- **Verdict and departures** — the step-2 conclusion, and every point where the plan differs
  from the issue body, each with its reason.
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
- **Risks and open questions** — the things Jack should decide, stated as questions with your
  recommendation attached.

Do not paste large code blocks into the plan. Name the change; the implementer writes the code.

## 4. First adversarial review: is there a simpler way?

The plan now goes through **two** reviews, by two separate fresh sub-agents, in this order:
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

## 5. Triage and revise

Verify each proposed simplification against the code rather than accepting it — **reviewer
findings are often wrong, and applying them uncritically makes the plan worse**. Take the genuine
ones into the plan file. Reject a simplification when it drops something the issue actually asked
for, or trades away a rule in `docs/design-philosophy/` — and say which, in one line.

For each finding you reject, record the finding and the one-line reason in the plan file, so Jack
can see what was considered and dismissed. Commit the revised plan.

**A proposed rearchitecture is Jack's call, not yours** — it is bigger than the issue, so neither
adopting it nor dropping it silently is right. Put it in the plan's "Risks and open questions" with
the reviewer's pros and cons, your view of whether it is genuinely simpler, and a recommendation on
whether it should become its own issue. Then plan the issue under the current architecture unless
Jack says otherwise.

## 6. Second adversarial review: correctness and testability

Launch **another** new sub-agent — not the one from step 4, and again with no account of your
reasoning or of what the first review changed. Give it the issue number and the path to the
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

## 7. Triage the findings

Verify each finding against the code, on the same terms as step 5: fix the genuine ones in the
plan file, and record each rejected finding with its one-line reason. Commit the updated plan, so
the branch carries the original plan and what both reviews did to it.

## 8. Stop

Report to Jack: the verdict from step 2, a short summary of the plan, what each of the two reviews
changed, and what each found that you rejected. Give the branch name and the path to the plan
file.

**Do not write any code, and do not open a PR.** Once Jack approves the plan, implementation runs
under the `implement-issue` skill, resuming at its step 2 in the worktree this skill already
created — implement, verify, PR, then two further independent adversarial reviews of the *diff*,
triaging and pushing after each, stop.
