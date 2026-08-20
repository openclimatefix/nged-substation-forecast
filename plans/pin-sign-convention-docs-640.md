# Pin the sign-convention doc prose to the contract (#640)

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
— and the trial area contains both conventions,
```

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

## Design-philosophy check

Pure documentation — no production asset, no Delta table, no serving path, no degradation rule, no
new Python code beyond shortening two description strings. Runs at doc-build time, never in
production, so the inherent-stability rules (WARN-not-ERROR checks, degrade-don't-raise) don't
apply. No principle in `design-principles.md` is traded away: this removes duplication rather than
adding an abstraction, and it reuses tooling (`mkdocs-include-markdown-plugin`) already adopted and
in active use for the same purpose (whole-README includes on every `docs/api/*/index.md` page).

## Tests

No new test. Drift between `forecast-building-blocks.md` and the contract's canonical wording
becomes structurally impossible: the doc page no longer contains its own copy of the prose to fall
out of sync, only an include directive that is re-resolved from `packages/contracts/README.md` on
every build. What replaces a test is `uv run mkdocs build --strict` (already run for any change
touching links, per the existing verification convention) plus reading the rendered HTML: the
`include-markdown` plugin logs a `mkdocs.plugins.include_markdown` warning — which `--strict`
promotes to a build failure — if either the `start` or `end` delimiter string is not found in
`packages/contracts/README.md`, so a typo'd marker or an accidentally-deleted fragment breaks the
build rather than silently including nothing or the whole file. This was checked directly against
the installed plugin's source
(`.venv/lib/python3.14/site-packages/mkdocs_include_markdown_plugin/{event,logger}.py`), not
assumed from its documentation.

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
