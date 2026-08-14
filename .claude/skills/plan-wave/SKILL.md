---
name: plan-wave
description: >-
  Work out the next wave of GitHub issues under an epic in
  openclimatefix/nged-substation-forecast that can safely be worked in parallel, then split them
  between two tracks: mechanical issues this skill implements itself end to end, and issues needing
  human judgement, which go out as background-task chips the user launches as separate Claude Code
  sessions. Reads the epic's open sub-issues, verifies each one's file surface against the code,
  spots issues that are duplicates or near-duplicates and should be folded together, sorts the rest
  by how much judgement each needs, groups them into slots that share no file, and records the wave
  on the epic so the next run knows its number. Load whenever the user asks what to work on next
  under an epic, says "plan the next wave", "/plan-wave 138", or asks which issues can run in
  parallel. Plans exactly one wave and stops — it does not schedule the whole epic, and it writes
  no per-issue plan for anything it hands to a chip.
---

# Plan the next wave of parallel work

The output is a **wave**: a set of slots, each a single open issue or a group of issues folded
together, that can be worked at the same time without colliding, plus a comment on the epic
recording what was dispatched.

Each slot goes to one of two tracks:

- **The agentic track** — mechanical issues, which this skill implements itself, orchestrating
  sub-agents through implement, review, triage and merge without leaving the session. There is no
  planning sub-agent: a slot only qualifies for this track because its design is already settled,
  so there is nothing left for a plan to decide.
- **The chip track** — issues needing human judgement, which go out as `spawn_task` chips the user
  launches as separate Claude Code sessions, one per slot.

Step 6 decides which track each slot takes, and getting that split right is what makes the wave
worth planning: putting a judgement call on the agentic track wastes a round of review discovering
the agent had no authority to make it, and putting a docstring correction on the chip track spends
a human session on something no human needs to read.

Plan **one wave, then stop.** Do not sketch the waves after it. The epic changes while a wave is
in flight — under v0.2, planning issues #480 and #496 filed seven new sub-issues between them, and
those became most of the next two waves. A schedule written before that work happened would have
been wrong about everything past the wave in progress, and the effort spent writing it wasted.

Take the epic number from the invocation (`/plan-wave 138`). If the epic is named by milestone
("v0.2") instead, find it: epics map 1:1 to the roadmap milestones.

## 1. Establish the wave number

The wave number is what tells this wave's sessions apart from the last one's in the desktop app's
session list, so it has to be right.

The ledger is a comment on the epic issue, written by step 10 of this skill at the end of every
run. Read the epic's comments and take the highest wave number you find, plus one:

```bash
gh issue view <EPIC> --comments
```

If there are no ledger comments and every sub-issue is open, this is wave 1. If there are no
ledger comments but sub-issues are already closed, waves have run before this skill existed —
**ask the human which number to start from**, then seed the ledger in step 10. That is the one
question worth blocking on, and it is asked once per epic.

## 2. Check the previous wave has landed

A wave is finished when its issues are closed and its pull requests are merged:

```bash
gh pr list --state open --json number,title,headRefName
git worktree list
```

Cross-check against the previous ledger comment. If issues from the last wave are still open, say
so and ask the human before dispatching — an unmerged branch still owns its files, so a new wave
chosen against `main` can collide with work that is nearly ready to land. Leftover worktrees in
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

## 4. Decide which issues belong in one session

Not every candidate is its own session. While reading in step 3, watch for issues that should be
implemented **together**, by a single agent, rather than dispatched as separate parallel chips:

- **Duplicates.** Two issues describing the same change, filed separately — often because a design
  discussion forked, or because whoever filed the second one did not find the first. Dispatching
  both wastes a wave slot and risks two competing PRs for the same diff.
- **Issues too entangled to plan apart.** One issue's design decision determines the other's — a
  schema change and every caller it breaks, a helper being introduced and its first real usage —
  so that planning them separately means re-deriving the same context twice, or risks one session's
  plan silently contradicting the other's while both are in flight.

Fold such issues into a single wave slot covering every issue number in the group — one chip, or one
agentic slot, depending on where step 6 puts it. Say in the wave table why they are grouped rather
than run apart. Do not group two issues merely because they touch the same file — that is the
collision case step 7 already handles by serialising or fencing territory. Grouping is for when the
issues are not actually separate pieces of work, whatever the tracker says.

A group counts as one slot against its track's ceiling, not one per issue. A group whose members
would land on different tracks belongs on the chip track: if any part of the work needs a human
judgement, the whole slot does.

## 5. Map the file surface — by reading the code, not the issue

For each wave slot (a single issue, or a group from step 4), name the files and the functions it
will edit — for a group, map the combined surface once. **Verify each one against the
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

## 6. Split the wave between the two tracks

Sort each slot into the agentic track or the chip track. The question is not how *large* the change
is but how much of it is already decided: an agent can execute a settled decision across fifty
files, and cannot make an unsettled one in three lines.

A slot goes on the **agentic track** only if every one of these holds:

- **There is one obvious way to do it.** The issue says what to change, not what to decide.
- **It touches no Patito contract**, and nothing about what gets stored.
- **It touches no production degradation path** — nothing in `defs/` that decides whether the
  service degrades or raises.
- **It needs no retrain, no leaderboard re-run, and invalidates no existing comparison.**
- **The verification set is the whole of the risk**, and where the slot touches behaviour, a test
  watched going red on the bug it exists for. A prose-only slot satisfies this on the verification
  set alone — there is nothing to mutate.

Typical agentic-track work: documentation that disagrees with the code, a docstring contradicting
its own function, dead code and dead configuration, renames, dependency declarations, a test that
cannot detect the regression it is named for, a coverage gap on an existing behaviour.

Everything else goes on the **chip track**, and these push a slot there on their own:

- More than one defensible design, or a design question the issue leaves open.
- A change to a contract, to what is stored, or to the serving path.
- The issue's own body doubting whether the work is worth doing — deciding to close an issue is a
  human call.
- A trade-off between the repo's own rules, where the change buys one principle by spending
  another.

**When the call is close, use a chip.** The costs are asymmetric: a chip spent on mechanical work
wastes a session, while a judgement call made agentically lands a decision nobody chose, in `main`,
with a green suite over it.

Record the track beside each slot in the step 8 table, with the reason in a few words.

## 7. Build the wave

Pick the largest set of slots in which **no two slots edit the same file** — a rule that spans both
tracks, because an agentic slot and a chip slot collide exactly as readily as two chips. Subject
to:

- Every dependency from step 3 is respected — both the recorded `blocked by` links and the prose
  ones.

- The ship issues (bump the version, push the image) go last, alone, and in order. The version bump
  also carries the epic's ship-time triage, so it wants every other issue closed first.

- An issue whose own body says it may not be worth doing still gets dispatched, but say so in its
  chip prompt: `plan-issue` returning "close this, it is subsumed by #N" is a good outcome, and the
  session should not feel obliged to plan around it.

Two issues touching the same file in provably separate regions — one editing `@asset(...)`
decorators while another rewrites an asset body — may share a wave as two separate slots, but only
if **both** prompts name the other's territory and say to stop and ask the human rather than edit
it. Prefer serialising over relying on that.

**Each track has its own ceiling, for different reasons.**

At most **five chips**, and prefer fewer, larger-value sessions to five thin ones. Five is a
ceiling rather than a target: each chip costs a human a plan review, and a group from step 4 buys
back a slot without losing coverage.

At most **five concurrent sub-agents** on the agentic track, which is a memory limit rather than an
attention one — running more than that has killed the machine outright. Queue the rest. Tell every
sub-agent not to retrain a model or materialise a large asset without asking first, because one
agent doing either while four others run is what exhausts the memory.

## 8. Present the wave, then drop the chips

Present the wave as a table — issue or group, one-line change, **track**, file surface, and what it
waits on — plus the reason anything obvious was held back. Then drop one chip per **chip-track**
slot with `spawn_task` in the same reply. A chip is inert until it is clicked, so there is no need
to ask first. Agentic-track slots get no chip; step 9 runs them.

**Chip title**: `W<n>: <imperative phrase naming what the slot changes> (#<N>)` — for example
`W4: Tag assets R&D or production (#423)`, or `W4: Merge the duplicate NWP-outage checks (#423,
#431)` for a group. Under 60 characters. This deliberately departs from `spawn_task`'s "start with
a verb" convention, and from the `Planning:` prefix in `plan-issue` step 1a: the chip title becomes
the spawned session's title, the app's auto-titling already adds "planning" to it, and the sidebar
is too narrow to spend characters saying so twice.

Each chip prompt has to stand alone — the session cannot see this conversation — and carries:

1. The issue number (or, for a group, every issue number in it and why they were folded together),
   the wave number, and `/plan-wave`'s standing instruction: run `/plan-issue <N>` first — for a
   group, starting from whichever issue is most complete and saying how the others in the group are
   being resolved (folded into the same change, or closed as a duplicate) — and let it size the
   work; if it writes a plan, write no code until a human approves that plan; if it sizes the work
   simple, go straight on to `implement-issue`.

2. An instruction to keep the `W<n>:` prefix when `plan-issue` step 1a asks it to state a session
   title, so the session does not retitle itself out of the wave.

3. **The file surface from step 5**, including any correction to what the issue body claims. This
   is the most valuable thing in the prompt: it saves the session the search, and it stops the
   session inheriting a wrong location from the issue.

4. **The other sessions' territory**, named file by file, with an instruction to stop and ask the
   human rather than edit it. Say which issues own those files and that they are running
   concurrently.

5. **Any prerequisite to verify first** — `gh issue view <M> --json state` — with an instruction to
   ask the human rather than plan on top of code that is about to move.

6. **The design questions the plan must settle**, especially where the issue leaves them open, and
   any repo rule the change implicates: a check staying `WARN`/`blocking=False` and unable to
   raise, production degrading where R&D fails fast, a hypothesis label the change delivers.

Set `cwd` to the repository root, and write the `tldr` in plain English with no file paths — it is
what the user reads in the tooltip.

## 9. Run the agentic track

Stay in this session and orchestrate sub-agents. One issue, one PR. Per slot:

1. **Implement.** One sub-agent, given the issue, the file surface from step 5, the other slots'
   territory, and the repo's standing rules — the verification set, one line per paragraph in every
   GitHub body, and the closing-link rule below. It opens the PR with labels and the `JackKelly`
   assignee, and does not merge.

2. **Review.** A *fresh* sub-agent, given the PR number and none of the implementer's reasoning, so
   it cannot inherit a wrong premise. Point it at the specific claim most damaging if wrong, and
   require it to re-run every mutation the PR reports rather than trusting the result. A test the
   reviewer did not watch go red is a test nobody has shown to work.

3. **Triage yourself.** Reviewer findings are often wrong, and acting on one uncritically is how a
   correct number gets "corrected" into a wrong one. Verify each finding against the code before
   acting; reject with a reason in a sentence. Send genuine ones back to the implementer, which
   still holds the context, rather than starting a new agent.

4. **Merge**, after the guard below.

**End of the wave**, once every agentic slot has merged: two adversarial reviewers over the whole
wave's combined diff, **in parallel, in separate worktrees**. Parallel is right here because no
fixes land between them, so both read the same tree and neither anchors the other. They need
separate worktrees because both mutate code to prove findings, and one reverting the other's
mutation manufactures fabricated results. Then one sub-agent implements the triaged fixes in a
single follow-up PR, reviewed like any other before it merges.

This is the opposite of `implement-issue`'s two reviews, which are deliberately **sequential**
because a triage-and-push step sits between them: its second reviewer must read the fixed tree, or
the fixes ship unread by anyone.

**The closing-link guard, before every merge.** GitHub registers a closing link whether the keyword
falls before or after the reference, ignores negations, and the link is sticky once registered —
editing the text afterwards does not remove it. `closingIssuesReferences` stays empty until the
merge lands, so it is no evidence at all beforehand. Grep the body and the commits yourself:

```bash
KW='\b(clos|fix|resolv)[a-z]*\b.{0,60}#[0-9]+|#[0-9]+.{0,60}\b(clos|fix|resolv)[a-z]*\b'
gh pr view <N> --json body --jq .body | grep -inE "$KW"
gh pr view <N> --json commits --jq '.commits[]|.messageHeadline+" "+.messageBody' | grep -inE "$KW"
```

The word boundaries matter: without them `openclimatefix` matches on its own trailing "fix", and a
guard that fires on every PR stops being read. After merging, check each referenced issue's state.

**Tell every implementing sub-agent to merge `main` into its branch as its last act before
reporting**, then re-run the full verification set on the merged tree. Slots run concurrently
against a moving base, and a branch that was current when its agent started routinely is not by the
time it finishes.

## 10. Record the wave on the epic

Post one comment on the epic. It is the ledger step 1 reads next time, and the record of which
issues belonged to which wave:

```bash
gh issue comment <EPIC> --body-file <path>
```

The comment states the wave number, the date, the issues dispatched with a phrase each — noting
which were grouped into one slot and why — what each slot waits on, and which open sub-issues were
held back and why. Open it with the Claude Code attribution line every GitHub body written by
Claude carries, and do not hard-wrap it — see the `github-issue-pr-workflow` skill for both rules.

The comment also records which track each slot took, so the next run can see what the split looked
like and whether it held.

Then stop. Do not plan the wave after this one, and do not write per-issue plans for chip slots —
each of those sessions writes its own under `plan-issue`, in its own worktree, with whatever
adversarial reviews that issue's size warrants.

**Why the collision rule is the whole design:** every slot branches from `main` and merges back
independently, with no coordination and no shared context — chip sessions cannot see each other,
and sub-agents on the agentic track cannot see each other either. Everything else — the worktree
isolation, the one-plan-per-branch rule, the fresh-reviewer requirement — already holds per slot.
The only failure this scheduling can prevent is two slots rewriting the same lines from different
premises, so the file surface is what the wave is built on, and a wave of three that cannot collide
beats a wave of five that might.

**Why the track split earns its step:** the two tracks fail in opposite directions, so the cost of
a misplacement depends entirely on which way it goes. A mechanical issue on the chip track is
merely wasteful — a human reads a plan for a docstring fix. A judgement call on the agentic track
is worse than wasteful: the agent will make the call, defend it fluently, pass every check, and
land a decision nobody chose. That asymmetry is why the close calls go to chips.
