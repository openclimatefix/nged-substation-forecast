# Pin the sign-convention doc prose to the contract (#640)

## Status — read this first if resuming cold

**Where this stands:** the plan below is finished and has been through both `plan-issue` adversarial
reviews (simplicity, then correctness/testability — see "Review findings and triage"). **No
implementation code has been written.** This PR was opened purely to make the plan durable and
reviewable across a machine shutdown — it is not the usual `implement-issue` PR, and it should stay
a draft (or otherwise clearly unmerged) until a human has read and approved the plan.

**What's needed next, once approved:** hand this off to the `implement-issue` skill, resuming at its
step 2 (worktree already exists — see below), which will: make the four file edits under "What
changes, file by file", run the "Verification commands", open (or reuse) the PR with the diff, run
the two diff-level adversarial reviews `implement-issue` itself calls for under this issue's
**Complex** sizing, triage, then stop for human review again before anything is merged.

**Branch:** `pin-sign-convention-docs-640`, pushed to `origin` with the plan as its only commits so
far (`7bd3ab48`, `733880d4`, `3a586f15`). A local worktree for it exists on this workstation at
`.claude/worktrees/pin-sign-convention-docs-640` (nested under the bridge session's own worktree at
the time this was written) — if that local worktree is gone in a future session, `git worktree add
.claude/worktrees/pin-sign-convention-docs-640 pin-sign-convention-docs-640` recreates it from the
already-pushed branch; nothing local-only is at risk. No `.env` symlink was created because none was
found at `/home/jack/dev/python/nged-substation-forecast/.env` when the worktree was set up — check
whether that's expected before running anything that needs it.

**Flagged for human attention** (also called out inline where they arise): the **Complex** sizing
call in "Verdict, size, departures"; the departure from all three of the issue's suggested
mechanisms in favour of a fourth (single-sourcing via `include-markdown`) — "Why the README, not the
Field description"; the decision to shorten both contract Field descriptions to a pointer rather
than keep their full prose; and the one accepted residual risk — the `substation_type` enum's
five-value count isn't mechanically pinned to the new doc fragment — in the last bullet of "Risks
and open questions".

**Problem:** the power sign convention — what positive/negative `power` means, keyed to
`substation_type` — is stated as free-text prose in four places: twice inside
`packages/contracts/src/contracts/power_schemas.py` (identical text on `PowerTimeSeries.power` and
`PowerForecast.power_fcst`), once as the canonical doc statement in
`docs/roadmap/forecast-building-blocks.md`, and once more, independently worded, in
`docs/roadmap/cost-savings-metrics.md`. Nothing mechanically ties these together, so an edit to any
one copy can silently drift from the others — and one already has: the contract says power flows
"**back** into the grid"; `forecast-building-blocks.md` currently says "**backwards** into the
grid".

**Solution:** make the wording exist in exactly one place — a new "Sign convention" section in
`packages/contracts/README.md` — and have `docs/roadmap/forecast-building-blocks.md` pull that
section in verbatim at build time with the `include-markdown` MkDocs plugin already installed in
this repo, rather than restating it. The two contract Field descriptions stop carrying their own
copy of the multi-sentence rule and instead point at that one section (readable locally, since
it's a file in the same package, and at the published URL). `cost-savings-metrics.md` stops
restating the rule too, replacing it with a link — matching the pattern `docs/index.md` and
`docs/roadmap/delivery-tables.md` already use. Drift becomes structurally impossible rather than
merely detected: there is nothing left to fall out of sync, so no new test is needed.

## Verdict, size, departures (step 2/3)

- **Verdict:** worth doing, as scoped. Verified against the code, not just the issue body — see
  "What changes" below for the exact current text of each copy.
- **Size: Complex.** Triggered by "more than one design would defensibly satisfy it" — the issue is
  explicitly a request to choose between several mechanisms, so the choice needs approval before
  code moves. Gets the plan and all four adversarial reviews (two here, two on the diff during
  implementation).
- **Departure from the issue body:** the issue lists three options (a parsing test, generated
  prose, or an accepted-drift decision) without preferring one. This plan is closest to the second
  ("generating the doc prose... so there is a single source text"), except the single source is a
  README.md fragment rather than the Field description itself — see "Why the README, not the Field
  description" below.
- **Revised after the first adversarial review (step 5/6):** the review (recorded in full below)
  found that this repo already has `mkdocs-include-markdown-plugin` installed and in active use
  (all seven `docs/api/*/index.md` pages `{% include %}` their package's `README.md`), and that its
  `start`/`end` fragment arguments make single-sourcing markdown text mechanical with no new test
  code. The plan below replaces the original four-`Final[str]`-constants-plus-pytest design with
  this include-based one. See "Review findings and triage" for what was accepted, adjusted, or
  rejected.

## What changes, file by file

### `packages/contracts/README.md`

Add a new `## Sign convention` section (after `## Key Data Contracts`, before `## Design
Principles`), wrapped in HTML comment markers so only the body — not the heading — gets pulled
into other pages:

```markdown
## Sign convention

<!-- sign-convention:start -->
Sign convention depends on `substation_type` in `TimeSeriesMetadata`, whose five values (`BSP`,
`EHV Customer`, `GSP`, `HV Customer`, `Primary`) partition into two behavioural cases:

- **Substations** (`BSP`, `GSP`, `Primary`): positive = power flowing **towards end-users**;
  negative = excess generation flowing **back into the grid**.
- **Customer meters** (`EHV Customer`, `HV Customer`): positive = the customer is **sending**
  power to NGED's grid; negative = the customer is **drawing** power from it. A customer meter can
  sit at a demand site or a generation site, so this case is not "generators only".

<!-- sign-convention:end -->
```

The blank line before the end marker is required: `pymarkdown` (`MD032`, lists must be surrounded
by blank lines) fails without it — checked by running the real linter against this exact block
during the second adversarial review, not assumed.

This is the exact wording currently in `forecast-building-blocks.md`, with the one already-drifted
word fixed ("backwards" → "back", matching the contract). `docs/api/contracts/index.md` already
`{% include %}`s the whole of this README, so this section also appears on the published API
reference page for free.

### `packages/contracts/src/contracts/power_schemas.py`

Today, `PowerTimeSeries.power` (lines 49–56) and `PowerForecast.power_fcst` (lines 392–398) each
embed this identical sentence group inside their Field `description=`:

> Sign convention depends on `substation_type` in `TimeSeriesMetadata`. At a substation (`BSP`,
> `GSP`, `Primary`), positive means power flowing towards end-users and negative means excess
> generation flowing back into the grid. At a customer meter (`EHV Customer`, `HV Customer`),
> positive means the customer is sending power to NGED's grid and negative means the customer is
> drawing power from it. Those five values are the whole enum, so every series falls into exactly
> one case.

Replace that sentence group, in both descriptions, with a one-line pointer:

```text
Sign convention depends on `substation_type` in `TimeSeriesMetadata` — see the Sign convention
section in this package's README.md, also published at
https://openclimatefix.github.io/nged-substation-forecast/roadmap/forecast-building-blocks/#sign-convention.
```

Everything else in both descriptions (the unit sentence, the rounding-precision sentence, the
`PLANNED` comment on `power_fcst`) is unchanged. No dtype, constraint, or validation behaviour
changes — this is a description-text-only edit.

### `docs/roadmap/forecast-building-blocks.md` (`## Sign convention`, lines 46–56)

Replace the hand-written bullets with an include of the README fragment:

```markdown
## Sign convention

{% include-markdown "../../packages/contracts/README.md" start="<!-- sign-convention:start -->" end="<!-- sign-convention:end -->" %}
```

The `#sign-convention` anchor is unchanged (the heading itself stays in this file), so every
existing link to it — `docs/index.md`, `docs/roadmap/delivery-tables.md`,
`docs/roadmap/cost-savings-metrics.md` — keeps resolving.

### `docs/roadmap/cost-savings-metrics.md` (`### Which direction is the constraint on?`, ~line 124)

Current text restates the rule in its own wording, with a link appended parenthetically:

```markdown
There is no single sign rule. This repo carries two conventions — at a substation, positive power
flows towards end-users; at a customer's meter, positive means the customer is *exporting* to
NGED's grid (see [sign convention](forecast-building-blocks.md#sign-convention)) — and the trial
area contains both,
```

Replace the restatement with a link-only reference, matching `docs/index.md`'s and
`delivery-tables.md`'s existing pattern:

```markdown
There is no single sign rule — see [sign convention](forecast-building-blocks.md#sign-convention)
— and the trial area contains both conventions, plus battery sites that both charge and discharge.
Constraint-side direction is therefore resolved **per `time_series_type`**, reusing the mapping the
[tail and exceedance
metrics](metrics-and-leaderboard.md#tail-exceedance-metrics-scoring-the-question-nged-actually-asks)
already need, with the ambiguous types confirmed by NGED.
```

(Rewrapped in full so the trimmed sentence doesn't leave the next sentence starting mid-paragraph
on a short dangling line — the original "plus battery sites..." continuation was two lines because
it followed the now-deleted restatement; folding it into one flowing sentence after the link avoids
that.)

### Why the README, not the Field description

The issue's "generate the doc prose from the contract's field description" option assumes the
Field description is the natural canonical source. It can't play that role cleanly here: it is a
plain-text `pt.Field(description=...)` string, so it cannot carry the doc's bulleted, bold-styled
markdown without either flattening the doc's formatting or teaching the include plugin to convert
Python string literals to markdown. `packages/contracts/README.md` is already markdown, is already
included wholesale into the API reference page, and — per its own "Design Principles" section — is
already "the authoritative account of what the data means" for this package, so moving the
full-prose canonical copy there (rather than leaving it embedded in a `pt.Field()` call) does not
relocate authority anywhere new.

## Review findings and triage (steps 5/6)

The first adversarial review (simplicity) found:

1. **Single-source via `include-markdown` instead of pinning two copies together.** *Accepted*,
   with one adjustment: the review suggested shortening the Field descriptions to "one sentence
   plus the published-docs URL"; this plan keeps a `substation_type` cross-reference plus a pointer
   to the local README (not only a URL), because a reader of `power_schemas.py` with no network
   access can still open `packages/contracts/README.md` in the same package. The review's core
   point — the multi-sentence rule doesn't need to live in the Field description at all — still
   holds and is taken.
2. **mkdocstrings already renders the contract via `show_source: true`, so pointing docs at
   `docs/api/contracts/index.md` instead might be simpler still.** *Checked and rejected*: `griffe-
   pydantic` is not installed (confirmed absent from `uv.lock`), so `pt.Field(description=...)`
   text only appears inside the page's collapsed "Source code" block, not as rendered prose. A
   reader would have to expand a code block to find the rule. Not simpler in practice.
3. **The in-contract duplication is arguably out of the issue's scope.** *Accepted as in-scope
   anyway*: the issue's own problem statement explicitly names "the contract (twice)" as part of
   what's wrong, so removing it is answering the issue as written, not scope creep — and under the
   include-based design it costs nothing extra (the two Field descriptions were always going to
   change to add the pointer).
4. **Four short `Final[str]` "atom" constants make a weaker regression test than one full-paragraph
   constant, and can't catch the same-wrong-wording-in-both-places case.** *Moot*: this plan drops
   the atom/test design entirely in favour of the include, so there is no test whose granularity
   this critique applies to.
5. **The new test file (`tests/test_sign_convention_docs.py`) could be one assertion in an existing
   file instead of a new file.** *Moot*, for the same reason as (4) — no test is added.
6. **The `cost-savings-metrics.md` change (drop restatement, link only) is proportionate.**
   *Accepted*, unchanged from the original plan.

The second adversarial review (correctness and testability, run against this revised plan)
verified every current-state and tooling claim above by reading the actual files and running real
builds, and found:

7. **The README fragment as originally drafted fails `pymarkdown scan`'s `MD032`** ("lists should
   be surrounded by blank lines") because the last bullet was immediately followed by
   `<!-- sign-convention:end -->` with no blank line. *Accepted and fixed*: a blank line now
   separates the last bullet from the end marker, in the snippet above.
8. **The trimmed `cost-savings-metrics.md` paragraph left a dangling short line** ("plus battery
   sites that both charge and discharge" as its own near-empty line, an artefact of deleting the
   restatement it used to follow). *Accepted and fixed*: the paragraph is now rewrapped in full, in
   the snippet above.
9. **`uv run mkdocs build --strict` and the plugin's failure modes were confirmed by actually
   running them**, including deliberately breaking a delimiter and a path in a scratch copy (not
   committed). *Accepted as confirming, not changing, the plan* — no edit needed beyond the wording
   fix in "Tests" acknowledging the two distinct failure modes (warning vs. hard error).
10. **The `substation_type` enum has five values, and the README fragment's "whose five values...
    partition into two behavioural cases" isn't tied to `pl.Enum([...])` at
    `power_schemas.py:225` by anything mechanical** — adding a sixth value would go stale silently.
    *Acknowledged, not fixed* — see "Risks and open questions": this is a different drift than the
    one the issue reports (which is about what positive/negative *means*, not how many
    `substation_type` values there are), and closing it would need a new test, reintroducing the
    exact test-and-constants machinery the first review's simplification removed. Left as an
    explicit residual risk rather than silently uncovered.
11. **The Field description's pointer text and the fact-check on double-inclusion, the docs
    inventory, and the existing-tests claims were all verified and found accurate** — no changes
    needed.

## Design-philosophy check

Pure documentation — no production asset, no Delta table, no serving path, no degradation rule, no
new Python code beyond shortening two description strings. Runs at doc-build time, never in
production, so the inherent-stability rules (WARN-not-ERROR checks, degrade-don't-raise) don't
apply. No principle in `design-principles.md` is traded away: this removes duplication rather than
adding an abstraction, and it reuses tooling (`mkdocs-include-markdown-plugin`) already adopted and
in active use for the same purpose (whole-README includes on every `docs/api/*/index.md` page).

## Tests

No new test. Drift between `forecast-building-blocks.md` and the contract's *positive/negative
meaning* prose — the specific drift this issue reports — becomes structurally impossible: the doc
page no longer contains its own copy of that prose to fall out of sync, only an include directive
re-resolved from `packages/contracts/README.md` on every build. What replaces a test is
`uv run mkdocs build --strict` (already run for any change touching links, per the existing
verification convention) plus reading the rendered HTML: the `include-markdown` plugin logs a
`mkdocs.plugins.include_markdown` warning — which `--strict` promotes to a build failure — if
either the `start` or `end` delimiter string is not found in `packages/contracts/README.md`, and
raises a hard `PluginError` (unconditionally, `--strict` or not) if the target file path is wrong.
Both were checked empirically during the second adversarial review: the review ran real builds
against deliberately broken delimiters and a deliberately broken path and confirmed each failure
mode, rather than trusting the plugin's source alone.

This does **not** make every possible drift in the section impossible — see "Risks and open
questions" for the one gap the second review found (the `substation_type` enum's five-value count),
which is deliberately left unclosed as out of this issue's scope.

No existing test needs to change. `packages/contracts/tests/test_power_time_series.py` and
`test_power_forecast.py` don't currently assert on field-description text (verified by grep before
writing this plan), so shortening the two descriptions doesn't touch them.

## Docs to update

- `packages/contracts/README.md` — new `## Sign convention` section, as above.
- `docs/roadmap/forecast-building-blocks.md` — the section body becomes an include, as above.
- `docs/roadmap/cost-savings-metrics.md` — drop the restatement, link only, as above.
- No roadmap status banner or "Implementation details" section applies: this issue isn't tied to a
  milestone page, it's a standalone documentation/tooling issue.

## Verification commands

```bash
uv run ruff check .
uv run ruff format . --check
uv run ty check
uv run pytest packages/contracts/tests/test_power_time_series.py packages/contracts/tests/test_power_forecast.py
uv run pytest   # full suite — this touches a shared contract file
uv run pymarkdown scan -r docs README.md CLAUDE.md packages/*/README.md
uv run mkdocs build --strict
```

Read the rendered HTML for `forecast-building-blocks.md`, `cost-savings-metrics.md`, and
`docs/api/contracts/index.md` after the build: confirm the included Sign convention section renders
with its bullets and bold styling intact (not collapsed by the include, and not double-included via
the API page's whole-README include plus its own explicit include, since the API page only includes
the README once, wholesale), and that the trimmed `cost-savings-metrics.md` sentence still reads
naturally.

## Risks and open questions

- **Is a plain silent no-op an actual risk if the include tag is written wrong (e.g. wrong relative
  path to the README)?** No — checked in the plugin source: a missing/unreadable target file raises
  an exception during the directive's own path resolution (before the start/end matching even
  runs), which fails the `mkdocs build` outright, strict or not. Only a *wrong delimiter string*
  degrades to a warning, and that warning is promoted to a failure by `--strict`.
- **Should the Field-description pointer sentence also appear on `TimeSeriesMetadata.substation_type`
  itself, since that's the field the rule keys off?** Not proposed here — `substation_type`'s
  current description ("Substation voltage level / role...") already stays factual and doesn't
  claim to explain `power`'s sign; adding a forward-reference there would be new scope the issue
  didn't ask for. Recommendation: leave it.
- **Should `cost-savings-metrics.md` keep zero restatement, or a one-clause hint (e.g. "(see sign
  convention)")?** The plan proposes a link with no restatement, matching the two pages that
  already do this. Recommendation: keep it consistent with the existing pattern rather than
  inventing a third style.
- **The `substation_type` enum's five-value count isn't mechanically tied to the README fragment**
  (finding 10 above): if `pl.Enum(["BSP", "EHV Customer", "GSP", "HV Customer", "Primary"])` at
  `power_schemas.py:225` gains a sixth value, "whose five values... partition into two behavioural
  cases" goes stale with nothing to catch it. Recommendation: accept this as residual risk rather
  than adding a test for it — it's a different drift than the one #640 reports, the enum's value
  list already appears as unlinked free text in several other places (e.g.
  `TimeSeriesMetadata.substation_type`'s own description), and closing every instance of that
  broader pattern is a larger, separate issue if it's wanted at all.
