---
name: simplicity-clean-room
description: >-
  A three-stage sub-agent exercise that tests whether a module is more complicated than its problem
  requires: extract a black-box requirements spec, have an isolated agent design the module afresh
  from that spec alone, then compare the two under a filter that treats only *unjustified*
  divergence as a finding. Produces a report, never a code change. Load when Jack suspects a module
  is over-built, asks for a simplicity review of a specific file or package, or says "clean room" —
  and read the "What it actually yields" section before promising him anything.
---

# The code-structure clean room

Three sub-agents in sequence, each seeing less or more than the last:

1. **Requirements** — extract what the module must do, judged from outside, with evidence.
2. **Design** — a *different* agent, which never sees the implementation, designs the module from
   that spec alone.
3. **Comparison** — a third agent compares the two and asks which of the real implementation's
   complexity nothing justifies.

The output is a report. **Change no repo files in any stage.** Acting on the findings is a separate
piece of work, and Jack decides which findings are worth it.

## What it actually yields

Say this to Jack up front, because it sets the right expectation and it is what two runs both
showed:

- **The structure usually comes back justified.** Run 1 (`defs/cv_assets.py`, 1,074 lines) and run 2
  (`packages/ml_core/src/ml_core/features/`, 1,062 lines) both concluded that the architecture was
  earned — every substantial piece of machinery the fresh design lacked had a commit, a test or a
  measurement behind it. In run 2 the fresh design was the *more* complex of the two in six places.
- **The findings that were worth acting on were mostly not complexity.** Run 1's three were a latent
  crash, an internal inconsistency and a performance defect. Run 2's top four were a `collect()`
  guard measured O(n) where the docs promise O(1), an unconditional sort of the largest frame in the
  system, three dead branches, and a `validate()` override that two commits reason about as a safety
  net and that nothing calls. The exercise finds these because building a spec forces someone to
  read every line and ask what it is for — not because it was designed to find them.
- **The "considered and rejected" table is the larger half of the report, in both runs.** It is the
  part that answers "is this over-built?" with evidence. Treat it as a first-class output, not an
  appendix.

Budget roughly **500–600k tokens** of sub-agent context and **35 minutes** of wall clock for a
~1,000-line target. Pick a target in the 500–1,500-line range: below that the spec costs more than
the code, above it stage 2 cannot hold the problem.

## Set up

```bash
mkdir -p <scratchpad>/cleanroom_<target>/inputs
```

Everything lives in the session scratchpad. Nothing this exercise produces belongs in the repo.

## Stage 1 — black-box requirements

Launch a fresh sub-agent. Tell it the target, and that stages 2 and 3 exist and what each does with
its output — an agent that knows its spec is the *only* thing stage 2 will see writes a better one.

Point it at the code, its tests, its callers, the docs, and `git log`/`git blame` on the target. A
constraint that arrived in a named commit with a message explaining itself is the strongest evidence
available, and this repo has a lot of them.

### The black-box rule

The spec **may** name the public API surface, Patito schema classes and their columns, config
fields, Delta table names, Dagster asset and partition-set names, string grammars, and functions in
*other* packages that form part of the contract.

The spec **may not** name any private helper, module-private function, local variable, loop
structure, batch or chunk size, intermediate frame, or file split inside the target. Give the agent
the rewrite rule in the brief, because it is the fastest way to convey what "black box" means here:

> If you catch yourself writing "the pipeline calls `_nullify_leaky_lags` after joining", rewrite it
> as the observable property: "a lag feature shorter than or equal to the forecast lead time must be
> null in the output, because otherwise it encodes the target".

Private names may appear **only** as citations — a test name, a `path:line`. Test names unavoidably
contain private identifiers, and that is fine.

### The three-way split, which is where this stage fails

Requirements, constraints, and incidental current behaviour go in three separate sections:

- **Requirements** — observable behaviour the module must have, each with evidence.
- **Constraints** — facts about the world that bound the solution space: data volumes, memory
  ceilings, dtype and null semantics, a lookahead-bias rule, a measured figure. **Each stated as a
  fact with its reason and its number, never as an instruction.** "The NWP table holds ~5.9 billion
  rows, so a `pl.len()` over it wraps a u32 index (issue #293)" — not "you must use the streaming
  engine". Stage 2 has to be free to satisfy a constraint by a different route.
- **Incidental current behaviour** — true of the code today, required by nothing: a particular batch
  size, an ordering no test pins, a helper's existence, a file split. **Stage 2 never sees this
  section.**

The incidental section is the whole point of the split. An over-specified incidental reaches stage 2
as a false constraint, stage 2 designs around it, and stage 3 scores the agreement as evidence the
implementation is justified. That failure is invisible in the final report.

Give the agent the boundary test: *if a from-scratch implementation did this differently, would any
test fail, any caller break, or any documented promise be violated?* Yes → requirement or
constraint. No → incidental.

### Evidence discipline

Every claim must be cheap for stage 3 to check: cite a `path:line`, a test name, a commit SHA or a
docs path. Anything asserted but not confirmed must either be run (`uv run python -c ...`, `uv run
pytest <test>`) or marked `[UNVERIFIED]`. No guessed numbers.

Ask for the counts and, in the final message, the two or three places where the
requirement/incidental boundary was genuinely hard to call. Those are where you should look first
when triaging stage 3.

## Stage 2 — the isolated design

Split the spec: everything above the incidental section becomes `inputs/requirements.md`. Add
`docs/design-philosophy/` and `docs/architecture/code-style.md` so the design lands in house idiom.

**Then check your own isolation before launching**, because run 2 found two leaks that neither the
brief nor the agent could have prevented:

```bash
# 1. Do any of the target's private helpers appear in the stage-2 input outside a citation?
grep -hoE '^\s*def (_[a-zA-Z0-9_]+)' <target>/*.py | grep -oE '_[a-zA-Z0-9_]+' | sort -u |
  while read -r f; do grep -n "$f" inputs/requirements.md; done
```

Check every hit lands on an `*Evidence:*` line. Then grep the *copied docs* for the same names —
`docs/design-philosophy/inherent-stability.md` names `_upsample_nwp_to_half_hourly` verbatim, so
copying it in leaked an internal that the spec had carefully kept out.

**The bigger leak is `CLAUDE.md`.** It is auto-injected into every sub-agent's context as project
instructions, the agent cannot decline it, and its "Architecture" section describes some modules
down to their private function names — for `ml_core.features` it names `_engineer_features`,
`_nullify_leaky_lags`, `ParsedFeatures.from_strings` and the five `*Feature` class names. Run
stage 2 with a **working directory outside the repo** so it is not picked up. If you cannot, name
the offending paragraph in the brief and tell the agent to treat those names as contaminated — a
declared contamination is recoverable, an undeclared one is not.

### The isolation rule, stated as why

Do not write "do not look at the implementation" and stop. Write the reason, and name the impulse:

> An implementation of this specification already exists in a repo on this machine. You must not
> look at it, and you must not look for it. A later stage compares your independent design against
> that implementation; if you have seen it, the comparison is worthless. No `git log`, no repo grep,
> no reading a skill or a `CLAUDE.md`, no "just checking how the tests call it". **The impulse to
> check how it is currently done is exactly what this exercise blocks** — when you feel it, note the
> question in your output as an open question instead, and design past it.

Running `python` to check a *library's* behaviour is fine and should be encouraged; that is about
the tools, not about the existing solution. In run 1 this wording made the agent decline to open a
repo skill twice, unprompted.

Two more clauses that earn their place:

- **Warn that `*Evidence:*` test names are citations, not a suggested decomposition.** Without this,
  an agent reads the module's structure out of its test names.
- **Demand a declaration of every file read, by path**, and say why: *an honest declaration is far
  more useful than a clean-looking one; a contaminated design that is labelled contaminated is still
  usable, a contaminated design labelled clean poisons the result.* Run 2's agent declared both
  leaks above without being asked about either.

Ask for: the shape (concrete names and signatures), how each hard part is handled and why,
**decisions with the alternatives considered and why they were rejected**, what it would leave out
and what that costs, and open questions. The decisions section is what stage 3 leans on hardest.

## Stage 3 — comparison under evidence inversion

A third fresh agent, with the **full** spec (incidentals included), the design, and full repo
access. Tell it stage 2's isolation declaration is in the design document, and should be read and
weighed.

### The rule that makes the report usable

> **Divergence is not a finding. *Unjustified* divergence is a finding.**

The two artefacts differ in dozens of ways and almost every difference has a reason. Reporting those
as "the fresh design was simpler" produces a report that is confidently wrong. So for every
divergence where the real implementation is more complex, it may only be reported after five
searches come up empty:

1. **`git log -S` / `git log --follow` on the code** — did it arrive in a commit whose message gives
   the reason?
2. **The tests** — is there a test that fails if you simplify it? Make the change, run it, see.
3. **The docs** — `docs/architecture/`, `docs/design-philosophy/`, `docs/techniques/`,
   `docs/ml_experimentation/`, `.claude/skills/`. This project writes its reasons down.
4. **The callers** — does anything outside the target depend on it? Grep `packages/` and `src/`.
5. **The constraints** — does a data volume, memory ceiling, dtype or null semantic make the simpler
   form wrong at production scale? Is there a measured number?

A search that finds the justification sends the divergence to the **"considered and rejected"**
table, with what was found. Ask explicitly for the places the fresh design was *more* complex too —
that is direct evidence the real code is already lean, and it is the half of the answer Jack's
question actually needs.

### Ask for bugs, not just complexity

This is the clause both runs proved out, and it was missing from run 1's brief:

> The previous run found that its most valuable outputs were not unnecessary complexity: they were a
> latent crash, an internal inconsistency and a performance defect that the comparison tripped over
> while looking for something else. So if you trip over a bug, an inconsistency between two parts of
> the module, a stale comment or doc, or a defect — report it, in its own section, whether or not it
> has anything to do with complexity. Hold these to the same evidence standard.

Hand it any leads stage 1 flagged in its final message. Both runs' stage 1 named something it could
not resolve, and both turned out to be real.

### Evidence discipline

Every claim names a `path:line` and the specific caller, test or commit. **Where it asserts a
simplification, it must actually make the edit and run the tests, report what happened, and restore
the file** — then confirm `git status` is clean. Findings ranked by what they buy: lines deleted, a
class of bug made impossible, a concept the reader no longer has to hold.

## Triage, then report

**Verify the top findings yourself against the code before repeating any of them to Jack.** Stage 3
runs 40-odd tool calls and reports confidently; the same rule as every other review in this repo
applies — findings are often wrong, and passing them on uncritically is worse than not running the
exercise. Both runs' reports survived spot-checking, but the checking is cheap and the claim you
repeat becomes yours.

Report: the verdict on whether the structure is justified, the findings ranked, the bugs separately
from the complexity, and what the "considered and rejected" table cleared. Then ask Jack which
findings he wants acted on — do not start fixing them, and do not roll several unrelated findings
into one PR without saying so.
