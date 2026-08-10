---
name: plan-issue
description: >-
  Turn a GitHub issue in openclimatefix/nged-substation-forecast into a reviewed implementation
  plan, without writing any code. Reads the issue and its comments, decides whether the issue is
  worth implementing at all, writes a plan that may overrule a stale issue body, has a fresh
  sub-agent adversarially review that plan, triages the findings, and then stops for Jack's
  review. Use this whenever Jack asks for a plan for an issue, says "plan issue N" or
  "/plan-issue N", asks whether an issue is worth doing, or asks you to think through how to
  implement an issue before touching code. Do not use it to implement an issue — that is the
  `implement-issue` skill, which starts once a plan from this skill has been approved.
---

# Plan a GitHub issue

The output of this skill is a **plan file and a recommendation**, never a code change. Planning
and implementing are deliberately separate: Jack approves a plan before any code moves, so that
his review of the eventual diff is checking execution rather than discovering the design for the
first time.

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

One worktree per issue means one checkout per branch, so `plans/` holds exactly one file on each
branch and the "at most one plan" rule in `plans/README.md` holds without any coordination
between parallel sessions. Committing it now rather than at implementation time is what makes the
plan durable: it survives Claude Code shutting down, and it is already in the diff when the
implementation PR opens, so the reviewer sees the plan next to the code that claims to follow it.
It is deleted at ship time, with its content pasted into the PR body — the existing rule, not a
new one.

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
- **Verification commands** — the green-before-push set from CLAUDE.md, plus anything this
  change specifically needs (for example `uv run pytest --run-network -m network` for
  convention-sensitive NWP conversion code, or `uv run mkdocs build --strict` *and reading the
  rendered HTML* for any change that touches links).
- **Risks and open questions** — the things Jack should decide, stated as questions with your
  recommendation attached.

Do not paste large code blocks into the plan. Name the change; the implementer writes the code.

## 4. Adversarial review by a fresh sub-agent

Launch a **new** sub-agent. Give it the issue number and the path to the plan file, and **nothing
about your reasoning** — it must not be anchored by the argument that produced the plan.

Tailor the brief to this issue's specific failure modes rather than asking for a generic review.
The attacks worth naming, when they apply:

- Does the plan's description of *current* behaviour actually match the code on `main`?
- Would each proposed test really have failed before the change?
- Does the change fail closed anywhere production is supposed to degrade — in particular, can any
  warning path now raise?
- Is a refactor hiding a behaviour change inside it?
- Does the plan miss a caller, a doc page, or a schema that this change invalidates?
- Is the issue body stale in a way the plan inherited rather than caught?

Ask the reviewer for findings with file/line evidence, and for an explicit verdict on each: real
defect, or not.

## 5. Triage the findings

Verify each finding against the code rather than accepting it — **reviewer findings are often
wrong, and applying them uncritically makes the plan worse**. Fix the genuine ones in the plan
file. For each finding you reject, record the finding and the one-line reason it was rejected, in
the plan file, so Jack can see what was considered and dismissed. Commit the updated plan, so the
branch carries both the original plan and what the review did to it.

## 6. Stop

Report to Jack: the verdict from step 2, a short summary of the plan, what the review changed, and
what it found that you rejected. Give the branch name and the path to the plan file.

**Do not write any code, and do not open a PR.** Once Jack approves the plan, implementation runs
under the `implement-issue` skill, resuming at its step 2 in the worktree this skill already
created — implement, verify, PR, a second and independent adversarial review of the *diff*,
triage, stop.
