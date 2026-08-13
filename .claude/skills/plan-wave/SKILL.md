---
name: plan-wave
description: >-
  Work out the next wave of GitHub issues under an epic in
  openclimatefix/nged-substation-forecast that can safely be planned and implemented in parallel,
  then dispatch them as background-task chips Jack launches as separate Claude Code sessions.
  Reads the epic's open sub-issues, verifies each one's file surface against the code, groups
  between one and five that share no file, and records the wave on the epic so the next run knows
  its number. Load whenever Jack asks what to work on next under an epic, says "plan the next
  wave", "/plan-wave 138", or asks which issues can run in parallel. Plans exactly one wave and
  stops — it does not schedule the whole epic, and it writes no code and no per-issue plan.
---

# Plan the next wave of parallel work

The output is a **wave**: between one and five open issues under the epic that can be worked at
the same time by separate Claude Code sessions without colliding, plus one chip per issue for Jack
to launch, plus a comment on the epic recording what was dispatched.

Plan **one wave, then stop.** Do not sketch the waves after it. The epic changes while a wave is
in flight — under v0.2, planning issues #480 and #496 filed seven new sub-issues between them, and
those became most of the next two waves. A schedule written before that work happened would have
been wrong about everything past the wave in progress, and the effort spent writing it wasted.

Take the epic number from the invocation (`/plan-wave 138`). If Jack names the epic by milestone
("v0.2") instead, find it: epics map 1:1 to the roadmap milestones.

## 1. Establish the wave number

Jack uses it to tell this wave's sessions apart from the last one's in the desktop app's session
list, so it has to be right.

The ledger is a comment on the epic issue, written by step 7 of this skill at the end of every
run. Read the epic's comments and take the highest wave number you find, plus one:

```bash
gh issue view <EPIC> --comments
```

If there are no ledger comments and every sub-issue is open, this is wave 1. If there are no
ledger comments but sub-issues are already closed, waves have run before this skill existed —
**ask Jack which number to start from**, then seed the ledger in step 7. That is the one question
worth blocking on, and it is asked once per epic.

## 2. Check the previous wave has landed

A wave is finished when its issues are closed and its pull requests are merged:

```bash
gh pr list --state open --json number,title,headRefName
git worktree list
```

Cross-check against the previous ledger comment. If issues from the last wave are still open, say
so and ask Jack before dispatching — an unmerged branch still owns its files, so a new wave chosen
against `main` can collide with work that is nearly ready to land. Leftover worktrees in
`.claude/worktrees/` are the other tell.

## 3. Read the candidates

List the epic's open sub-issues:

```bash
gh api graphql -f query='{repository(owner:"openclimatefix",name:"nged-substation-forecast"){issue(number:<EPIC>){subIssues(first:100){nodes{number title state}}}}}' --jq '[.data.repository.issue.subIssues.nodes[] | select(.state=="OPEN") | "\(.number) \(.title)"] | .[]'
```

Then read every open one in full — body **and** comments — because the wave is chosen on what each
issue actually changes, which the title never tells you. Check the recorded dependencies too:

```bash
gh api repos/openclimatefix/nged-substation-forecast/issues/<N>/dependencies/blocked_by --jq '[.[] | "#\(.number)"] | join(", ")'
```

Most dependencies in this repo are *not* recorded that way. They are prose inside a body — "depends
on #496 landing", "this should be settled first", "not part of #228" — and those bind just as
tightly. Collect them as you read.

## 4. Map the file surface — by reading the code, not the issue

For each candidate, name the files and the functions it will edit. **Verify each one against the
code**, because an issue's account of where something lives is a claim, and claims go stale: #505
discussed the corruption signal entirely in terms of `dynamical_data`'s
`convert_to_polars.py`, but `assess_nwp_quality` and `NwpQualityReport` are in
`packages/contracts/src/contracts/weather_schemas.py`, which is where most of that diff had to
land. A wave built on the issue's version of that would have paired it with something that owned
`contracts`.

`grep -rn` for the symbols the issue names, and read the function the issue proposes to change.
The surface that matters is per-file, and for the crowded files in this repo (`defs/checks.py`,
`defs/assets.py`, `defs/production_assets.py`) per-function as well.

Watch for surfaces that are easy to miss:

- **`uv.lock`.** Two issues that add or drop a dependency edit different `pyproject.toml` sections,
  which merges cleanly, and the same lockfile, which does not. Serialise them, or tell whichever
  lands second to re-run `uv lock`.

- **Cross-cutting changes.** An issue that edits one thing in every capture site, decorator or
  docstring across the repo — Sentry event shape, asset tags, an error-message sweep — collides
  with everything. Give it a wave with few neighbours, and prefer to run it after the files it
  touches have settled rather than before.

- **Shared test files and fixtures**, which collide as readily as the module under test.

## 5. Build the wave

Pick the largest set, up to five, in which **no two issues edit the same file**, subject to:

- Every dependency from step 3 is respected — both the recorded `blocked by` links and the prose
  ones.

- The ship issues (bump the version, push the image) go last, alone, and in order. The version bump
  also carries the epic's ship-time triage, so it wants every other issue closed first.

- An issue whose own body says it may not be worth doing still gets dispatched, but say so in its
  chip prompt: `plan-issue` returning "close this, it is subsumed by #N" is a good outcome, and the
  session should not feel obliged to plan around it.

Two issues touching the same file in provably separate regions — one editing `@asset(...)`
decorators while another rewrites an asset body — may share a wave, but only if **both** chip
prompts name the other's territory and say to stop and ask Jack rather than edit it. Prefer
serialising over relying on that.

Prefer fewer, larger-value sessions to five thin ones. Five is a ceiling, not a target: the wave
costs Jack five plan reviews.

## 6. Present the wave, then drop the chips

Show Jack the wave as a table — issue, one-line change, file surface, and what it waits on — plus
the reason anything obvious was held back. Then drop one chip per issue with `spawn_task` in the
same reply. A chip is inert until Jack clicks it, so there is no need to ask first.

**Chip title**: `W<n>: <imperative phrase naming what the issue changes> (#<N>)` — for example
`W4: Tag assets R&D or production (#423)`. Under 60 characters. This deliberately departs from
`spawn_task`'s "start with a verb" convention, and from the `Planning:` prefix in `plan-issue`
step 1a: the chip title becomes the spawned session's title, the app's auto-titling already adds
"planning" to it, and the sidebar is too narrow to spend characters saying so twice.

Each chip prompt has to stand alone — the session cannot see this conversation — and carries:

1. The issue number, the wave number, and `/plan-wave`'s standing instruction: run
   `/plan-issue <N>` first, write no code until Jack approves the plan, then follow
   `implement-issue`.

2. An instruction to keep the `W<n>:` prefix when `plan-issue` step 1a asks it to state a session
   title, so the session does not retitle itself out of the wave.

3. **The file surface from step 4**, including any correction to what the issue body claims. This
   is the most valuable thing in the prompt: it saves the session the search, and it stops the
   session inheriting a wrong location from the issue.

4. **The other sessions' territory**, named file by file, with an instruction to stop and ask Jack
   rather than edit it. Say which issues own those files and that they are running concurrently.

5. **Any prerequisite to verify first** — `gh issue view <M> --json state` — with an instruction to
   ask Jack rather than plan on top of code that is about to move.

6. **The design questions the plan must settle**, especially where the issue leaves them open, and
   any repo rule the change implicates: a check staying `WARN`/`blocking=False` and unable to
   raise, production degrading where R&D fails fast, a hypothesis label the change delivers.

Set `cwd` to the repository root, and write the `tldr` in plain English with no file paths — it is
what Jack reads in the tooltip.

## 7. Record the wave on the epic

Post one comment on the epic. It is the ledger step 1 reads next time, and the record of which
issues belonged to which wave:

```bash
gh issue comment <EPIC> --body-file <path>
```

The comment states the wave number, the date, the issues dispatched with a phrase each, what each
one waits on, and which open sub-issues were held back and why. Open it with the Claude Code
attribution line every GitHub body written by Claude carries, and do not hard-wrap it — see the
`github-issue-pr-workflow` skill for both rules.

Then stop. Do not plan the wave after this one, and do not write per-issue plans — each session
does its own under `plan-issue`, in its own worktree, with its own adversarial reviews.

**Why the collision rule is the whole design:** these sessions branch from `main` and merge back
independently, with no coordination between them and no shared context. Everything else — the
worktree isolation, the one-plan-per-branch rule, the fresh-reviewer requirement — already holds
per session. The only failure this scheduling can prevent is two sessions rewriting the same lines
from different premises, so the file surface is what the wave is built on, and a wave of three
that cannot collide beats a wave of five that might.
