# Pin the sign-convention doc prose to the contract (#640)

**Problem:** the power sign convention — what positive/negative `power` means, keyed to
`substation_type` — is stated as free-text prose in five places: twice inside
`packages/contracts/src/contracts/power_schemas.py` (identical text on `PowerTimeSeries.power` and
`PowerForecast.power_fcst`), once as the canonical doc statement in
`docs/roadmap/forecast-building-blocks.md`, and once more, independently worded, in
`docs/roadmap/cost-savings-metrics.md`. Nothing mechanically ties these together, so an edit to any
one copy can silently drift from the others — and one already has: the contract says power flows
"**back** into the grid"; `forecast-building-blocks.md` currently says "**backwards** into the
grid".

**Solution:** extract the rule's four factual atoms (what positive/negative means at a substation,
what positive/negative means at a customer meter) into four short `Final[str]` constants in
`power_schemas.py`, build both contract field descriptions from them (removing the existing
in-contract duplication), add a regression test asserting each atom is a verbatim substring of
`forecast-building-blocks.md`, and stop `cost-savings-metrics.md` restating the rule at all —
replace its restatement with a link, matching the pattern `docs/index.md` and
`docs/roadmap/delivery-tables.md` already use.

## Verdict, size, departures (step 2/3)

- **Verdict:** worth doing, as scoped. Verified against the code, not just the issue body — see
  "What changes" below for the exact current text of each copy.
- **Size: Complex.** Triggered by "more than one design would defensibly satisfy it" — the issue is
  explicitly a request to choose between several mechanisms, so the choice needs approval before
  code moves. Gets the plan and all four adversarial reviews (two here, two on the diff during
  implementation).
- **Departure from the issue body:** the issue lists three options (a parsing test, generated
  prose, or an accepted-drift decision) without preferring one. This plan picks the first,
  narrowed to checking four short phrases rather than one long paragraph — see "Why four short
  atoms, not one paragraph" below.

## What changes, file by file

### `packages/contracts/src/contracts/power_schemas.py`

Today, `PowerTimeSeries.power` (lines 49–56) and `PowerForecast.power_fcst` (lines 392–398) each
embed this identical sentence group inside their Field `description=`:

> Sign convention depends on `substation_type` in `TimeSeriesMetadata`. At a substation (`BSP`,
> `GSP`, `Primary`), positive means power flowing towards end-users and negative means excess
> generation flowing back into the grid. At a customer meter (`EHV Customer`, `HV Customer`),
> positive means the customer is sending power to NGED's grid and negative means the customer is
> drawing power from it. Those five values are the whole enum, so every series falls into exactly
> one case.

Add four module-level constants near the top of the file (after the imports, before
`DropImplausibleRowsResult`):

```python
SIGN_CONVENTION_SUBSTATION_POSITIVE: Final[str] = "power flowing towards end-users"
SIGN_CONVENTION_SUBSTATION_NEGATIVE: Final[str] = "excess generation flowing back into the grid"
SIGN_CONVENTION_CUSTOMER_POSITIVE: Final[str] = "the customer is sending power to NGED's grid"
SIGN_CONVENTION_CUSTOMER_NEGATIVE: Final[str] = "the customer is drawing power from it"
```

Rewrite both field descriptions to build the same sentence group from these four constants (an
f-string interpolating them into the existing surrounding grammar), so the wording is unchanged but
now has one literal source instead of two. No dtype, constraint, or validation behaviour changes —
this is a description-text-only refactor.

### `docs/roadmap/forecast-building-blocks.md` (`## Sign convention`, lines 46–56)

Current text:

```markdown
- **Substations** (`BSP`, `GSP`, `Primary`): positive = power flowing **towards end-users**;
  negative = excess generation flowing **backwards into the grid**.
- **Customer meters** (`EHV Customer`, `HV Customer`): positive = the customer is **sending**
  power to NGED's grid; negative = the customer is **drawing** power from it. A customer meter can
  sit at a demand site or a generation site, so this case is not "generators only".
```

Three of the four atoms already appear verbatim as substrings ("power flowing towards end-users",
"the customer is sending power to NGED's grid", "the customer is drawing power from it"). Fix the
one that has already drifted: "flowing **backwards into the grid**" →
"flowing **back into the grid**", so `SIGN_CONVENTION_SUBSTATION_NEGATIVE` is also a verbatim
substring. No other wording changes — the bullet structure and bold styling stay as they are; only
the drifted word changes.

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

This removes the only other free-text restatement in the docs, so no mechanical check is needed for
this page: a stale link would be caught by the existing `mkdocs build --strict` link-anchor
validation (`validation.links.anchors: warn` in `mkdocs.yml`, promoted to a hard failure under
`--strict`), which this change's verification set already runs.

### New test: `tests/test_sign_convention_docs.py`

A cross-cutting test belongs in the root `tests/` directory (not `packages/contracts/tests/`)
because it reads a `docs/` file as well as importing `contracts` — see "Where tests and their
dependencies live" in `docs/architecture/testing.md`.

```python
from contracts.power_schemas import (
    SIGN_CONVENTION_CUSTOMER_NEGATIVE,
    SIGN_CONVENTION_CUSTOMER_POSITIVE,
    SIGN_CONVENTION_SUBSTATION_NEGATIVE,
    SIGN_CONVENTION_SUBSTATION_POSITIVE,
)
from contracts.settings import PROJECT_ROOT


def test_forecast_building_blocks_states_the_same_sign_convention_as_the_contract():
    doc_text = (
        PROJECT_ROOT / "docs" / "roadmap" / "forecast-building-blocks.md"
    ).read_text()
    for atom in (
        SIGN_CONVENTION_SUBSTATION_POSITIVE,
        SIGN_CONVENTION_SUBSTATION_NEGATIVE,
        SIGN_CONVENTION_CUSTOMER_POSITIVE,
        SIGN_CONVENTION_CUSTOMER_NEGATIVE,
    ):
        assert atom in doc_text
```

Uses `PROJECT_ROOT` (from `contracts.settings`, already regression-tested in
`packages/contracts/tests/test_project_root.py`) rather than a path relative to `__file__`, so the
test is robust to the non-editable-install layout that motivated `PROJECT_ROOT` in the first place.

### Why four short atoms, not one paragraph

The contract's field description is one flowing paragraph; the doc's is two markdown bullets with
bold styling. Forcing them to be byte-identical would mean either flattening the doc into plain
prose (losing the bullets that make it skimmable) or templating the doc paragraph out of the
contract string (new tooling — a Jinja include or codegen step — for a two-sentence rule, which is
disproportionate). Four short exact-wording phrases survive both surrounding styles: the contract
sentence can say "positive means the customer is sending power to NGED's grid" and the doc bullet
can say "positive = **the customer is sending power to NGED's grid**" and both still contain the
same atom verbatim. This is also, concretely, the granularity that would have caught the
"back"/"backwards" drift already present on `main`.

## Design-philosophy check

Pure documentation and test-time-only code — no production asset, no Delta table, no serving path,
no degradation rule. Runs at test time and doc-authoring time, never in production, so the
inherent-stability rules (WARN-not-ERROR checks, degrade-don't-raise) don't apply. No principle in
`design-principles.md` is traded away: this is a size-neutral internal consolidation (four short
constants replacing one duplicated paragraph), not a new abstraction serving a hypothetical future
caller.

## Tests

One new test, `test_forecast_building_blocks_states_the_same_sign_convention_as_the_contract`
(above). **Assertion that fails on `main` today:** before the doc's "backwards" → "back" fix, the
loop's `assert SIGN_CONVENTION_SUBSTATION_NEGATIVE in doc_text` fails, because
`SIGN_CONVENTION_SUBSTATION_NEGATIVE = "excess generation flowing back into the grid"` is not a
substring of the doc's current "...flowing **backwards into the grid**." This is a genuine
regression test: it pins wording, not structure, so an editor who changes the doc's bullet styling
freely can, but one who silently reworks what positive/negative *means* cannot.

No existing test needs to change. `packages/contracts/tests/test_power_time_series.py` and
`test_power_forecast.py` don't currently assert on field-description text (verified by grep before
writing this plan), so the description-text refactor in `power_schemas.py` doesn't touch them.

## Docs to update

- `docs/roadmap/forecast-building-blocks.md` — one-word fix ("backwards" → "back"), as above.
- `docs/roadmap/cost-savings-metrics.md` — drop the restatement, link only, as above.
- No roadmap status banner or "Implementation details" section applies: this issue isn't tied to a
  milestone page, it's a standalone documentation/tooling issue.

## Verification commands

```bash
uv run ruff check .
uv run ruff format . --check
uv run ty check
uv run pytest tests/test_sign_convention_docs.py packages/contracts/tests/test_power_time_series.py packages/contracts/tests/test_power_forecast.py
uv run pytest   # full suite — this touches a shared contract file
uv run pymarkdown scan -r docs README.md CLAUDE.md packages/*/README.md
uv run mkdocs build --strict   # this change touches a doc-to-doc link
```

Read the rendered `cost-savings-metrics.md` HTML after `mkdocs build --strict` to confirm the
trimmed sentence still reads naturally and the link resolves to the right anchor.

## Risks and open questions

- **Is substring-matching four short phrases too weak a guarantee?** It cannot catch every kind of
  drift — e.g., if both the contract and the doc were edited to say the same *wrong* thing, the
  test would still pass. But that's true of any prose-comparison mechanism that doesn't re-derive
  meaning from a formal model, and building one for a five-value enum's sign convention is not
  proportionate. Recommendation: accept this as the residual risk; it's a large improvement over
  today's "nothing mechanically ties them together" baseline.
- **Should `cost-savings-metrics.md` keep zero restatement, or a one-clause hint (e.g. "(see sign
  convention)")?** The plan proposes a link with no restatement, matching the two pages that
  already do this. Recommendation: keep it consistent with the existing pattern rather than
  inventing a third style.
- **Should the four atoms also cover `PowerForecast.power_fcst`'s doc restatement, if one existed
  elsewhere?** Checked: no other doc page restates the rule in its own wording today (`index.md`
  and `delivery-tables.md` already link-only). Nothing further to cover.
