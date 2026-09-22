# Correct NGED's half-hour-late power stamps (#790)

**The problem.** NGED stamped every half-hourly power reading half an hour late until 08:30 UTC on
26 March 2026, then corrected the feed without marking the change. `PowerTimeSeries.time` is
documented as period-ending — the value stamped `T` is the mean over `(T − 30 min, T]` — and for
roughly 95% of the rows in our `power_time_series` Delta table that is false: the value stamped `T`
is the mean over `(T − 60 min, T − 30 min]`. Every model trained on that table has learnt a
half-hour timing error, and the error is not uniform across the record. The measurement, and the
three independent signals that pin the changepoint, are in the [beam/diffuse
appendix](https://openclimatefix.github.io/nged-substation-forecast/results/beam-diffuse-split/#the-power-stamps-before-26-march-2026-are-half-an-hour-late).

**The planned solution.** Correct on read, through exactly one door, and leave the stored table
holding what NGED sent. Add a `PowerTimeSeries.scan_delta()` classmethod — mirroring the
`Nwp.scan_delta()` the repo already has — that scans the Delta table and rolls every stamp before
the changepoint back by 30 minutes. Route the six production and script read sites that care about
what `time` means through it, and leave the ingest, the dedupe and the freshness watermark reading
the raw table, because those three are about what NGED sent rather than about what it measured.

## Verdict, size and departures

**Verdict: worth implementing, and the fault is real.** The issue's account of the current code is
accurate: `read_nged_json.py:85` renames `endTime` to `time` and stores it unchanged, and nothing
downstream corrects it. The changepoint instant, the direction and the magnitude are already
settled and published on `main`; this issue is the engineering that acts on them, not a
re-measurement.

**Size: complex.** Answering each of the five triggers in turn:

| Trigger | Answer |
|---|---|
| Changes what gets stored | **Fires.** The plan's central choice is whether the stored table holds corrected stamps. Even under the recommended correct-on-read design, the meaning of a stored `time` is restated in the `PowerTimeSeries` contract. |
| Touches the production serving path | **Fires.** `live_forecasts` reads power through `_engineering_inputs.py::_scan_inputs`, which this change re-points at the new door. |
| Touches a degradation rule | Does not fire. No degradation ladder, asset check or warning path is added or edited. |
| More than one defensible design | **Fires.** Correct-on-ingest-plus-backfill and correct-on-read are both defensible, and the issue body asks for the choice explicitly. |
| Callers you could not name without searching | **Fires.** Eight modules touch the power table and the list came from a grep, not from memory. |

Four of five fire, so the issue gets a plan, both plan reviews, and both diff reviews.

**Departure 1 — the issue frames correct-on-read as "every consumer has to know the rule". It does
not have to.** The issue's "What to decide" section reads the choice as a trade between a one-off
backfill and a rule spread across every consumer. There is a third shape: one function owns the
rule and every consumer goes through that function. The repo already uses it for the NWP table —
`Nwp.scan_delta()` in `packages/contracts/src/contracts/weather_schemas.py:527` is the only door to
`nwp`, and it casts and types the frame on the way through. `PowerTimeSeries.scan_delta()` is the
same pattern, and it costs one new classmethod plus six one-line call-site edits.

**Departure 2 — no backfill of the stored table, and no change to the ingest.** Design principle 15
("Transform data in feature engineering, not in the ingest, unless it saves a lot of storage") is
squarely on point: a 30-minute shift destroys nothing, so feature engineering can produce either
form on demand and the transform has nothing to show for a place in the ingest. Principle 15's
stated cost of getting an ingest transform wrong — "re-downloading and re-writing the whole
archive" — is exactly what a stored correction would cost us if the changepoint moves, if the
correction turns out to be per-feed rather than fleet-wide, or if NGED republishes a corrected
history. All three are live possibilities, and the third is a question the issue itself says to ask
NGED before building anything.

**Departure 3 — the retrain and the NGED conversation are not in this issue's scope.** The issue
lists both under "What to decide". Retraining is a run of the existing CV assets once this merges,
not a code change; asking NGED whether the correction was fleet-wide is a human action. Both are
recorded under "Risks and open questions" so that neither is lost, and the second is a blocker on
the *retrain*, not on this code.

## What changes, file by file

### `packages/contracts/src/contracts/power_schemas.py`

- **Add `POWER_STAMPS_CORRECTED_AT: Final[datetime] = datetime(2026, 3, 26, 8, 30, tzinfo=UTC)`**, a
  module-level constant with a docstring stating what the fault is, which three measurements pin the
  instant, and a link to the published appendix. The same constant already exists twice on the
  `beam-diffuse-split-experiment` branch (`build_dataset.py:101` and `site_e_commissioning.py:51`);
  this is its permanent home, and that branch should import it rather than re-declare it once both
  have merged.
- **Add `PowerTimeSeries.correct_late_stamps(lf)`**, a classmethod taking and returning a
  `pt.LazyFrame[PowerTimeSeries]`, which applies
  `pl.when(pl.col("time") < POWER_STAMPS_CORRECTED_AT).then(pl.col("time").dt.offset_by("-30m")).otherwise(pl.col("time"))`.
  Separate from `scan_delta` so a caller holding an already-scanned frame — a test, a notebook, the
  beam/diffuse scripts — can apply the same rule without going back to disk.
- **Add `PowerTimeSeries.scan_delta(path=None, storage_options=None) -> pt.LazyFrame[Self]`**,
  mirroring `Nwp.scan_delta`: resolve `path` from `get_settings().power_time_series_data_path` and
  `storage_options` from `get_settings().storage_options` when either is `None`, scan, `set_model`,
  `cast`, then `correct_late_stamps`. The docstring says plainly that the frame it returns carries
  corrected stamps while the table on disk does not, and names `time_series_coverage` as the one
  read that deliberately stays raw.
- **Rewrite the `time` field description** so the contract stops asserting something false about the
  stored table. It should say that `time` is period-ending, that rows stamped before
  `POWER_STAMPS_CORRECTED_AT` are stored half an hour late exactly as NGED sent them, and that
  `scan_delta` is the door that corrects them. This is a documentation change to a contract, not a
  widening of a field or a relaxation of a range — see "Risks and open questions".

The sort order is worth a note in the implementation: the shift is monotonic and applies to a
contiguous prefix of each series, so it cannot reorder rows or create a duplicate `(time_series_id,
time)` key. A reading stamped exactly at the changepoint stays put, and the reading before it moves
to 08:00, where no row can already sit because the pre-correction feed published on the same
half-hour grid. Worth an explicit test rather than a comment alone.

### Read sites routed through the new door

| File | What it does with `time` | Change |
|---|---|---|
| `src/nged_substation_forecast/defs/_engineering_inputs.py:125` | Feeds `_engineer_features`; `time` becomes `valid_time` and is joined to NWP | Replace `pl.scan_delta(...)` + `pt.LazyFrame.from_existing(...).set_model(...)` with `PowerTimeSeries.scan_delta(...)`; keep the existing `.filter()` |
| `src/nged_substation_forecast/defs/cv_assets.py:987` | `cv_forecast_metrics` actuals, joined to forecasts on `(time_series_id, valid_time)` | Same substitution |
| `src/nged_substation_forecast/defs/cv_assets.py:199` | `effective_capacity`, a per-series P99 of `power` | Same substitution. The P99 is invariant to a time shift, so this is a no-op on the numbers; routing it through the door anyway keeps one rule for "how to read this table" |
| `packages/dashboard/view_forecasts.py:311` | Plots actuals beside forecasts | Same substitution |
| `packages/dashboard/map_and_timeseries.py:122` | Plots one series' history | Same substitution |
| `scripts/forecasting/export_baseline_forecasts.py:170` | Left-joins `observed_power` onto delivered forecast rows | Same substitution inside `_observed_power` |

Each dashboard edit is a marimo notebook, so the `marimo-notebooks` skill applies: the import must
live in `with app.setup:`, and `ruff check --fix` must never run over either file.

### Read sites that deliberately stay raw

- **`nged_data/storage.py::select_new_rows`** and **`_existing_power_time_series_keys`**. The
  anti-join that decides which parsed rows are new compares `(time_series_id, time)` against the
  table on disk. Both sides must be raw or the ingest would re-append every historic row it ever
  re-reads. Nothing here changes; the plan adds a sentence to `select_new_rows`' docstring saying
  why it reads raw while everything else reads through `scan_delta`.
- **`nged_data/storage.py::time_series_coverage`**, and therefore `defs/checks.py:400`
  (`power_data_is_fresh`) and `defs/cv_assets.py:140` (`eligible_time_series`). Coverage answers
  "when did NGED last send us anything", which is a fact about the feed rather than about the
  measurement. The correction cannot move `last_time`, because every recent row is post-changepoint;
  it moves `first_time` by 30 minutes, which no eligibility threshold measured in months can
  notice. Same docstring sentence.

### `packages/nged_data/src/nged_data/read_nged_json.py`

**No change.** Stated here because "correct on ingest" is the option this plan rejects, and a
reviewer should see that the rejection is deliberate rather than an omission.

## Design-philosophy check

**This code path runs in production, and the change cannot degrade it.** `live_forecasts` reads
power through `_engineering_inputs.py` over a lookback window measured in days, so every row it
reads is post-changepoint and the correction is an identity on all of them. The live service's
behaviour is therefore unchanged by this merge, and stays unchanged until a model is retrained.
That is the safety property that makes this change shippable ahead of the retrain rather than
coupled to it.

**Nothing raises and nothing degrades.** `correct_late_stamps` is a pure Polars expression on a lazy
frame: it has no I/O, no branch on data availability, and no failure mode short of the column being
absent, which the contract forbids. No asset check is added or edited, so the `WARN`/`blocking=False`
rule has nothing to bite on here.

**Principle 15 is the principle this plan follows, and principle 7 is the one it strains.**
Principle 15 — transform in feature engineering, not in the ingest — is why the stored table stays
raw. Principle 7 ("Strict contracts at every boundary") is strained in return: a
`pt.DataFrame[PowerTimeSeries]` will now carry either raw or corrected stamps with nothing in the
type to say which. What is bought is that the rule stays a one-line edit for as long as its three
open questions are open. What is given up is compile-time certainty, replaced by a single documented
door and, if the reviewer agrees, a grep guard (see "Risks and open questions").

**Hypotheses.** No hypothesis in `engineering-hypotheses.md` is delivered by this change. It is a
data-correctness fix, not a claim about the engineering. The eventual retrain bears on forecast
accuracy, which is measured on the leaderboard rather than by a hypothesis label.

## Tests

New tests in `packages/contracts/tests/test_power_schemas.py`, each named with the assertion that
fails on `main` today:

1. **A stamp before the changepoint moves back 30 minutes.** `correct_late_stamps` on a row stamped
   `2026-03-26 08:00Z` returns `2026-03-26 07:30Z`. Fails on `main`: the method does not exist.
2. **A stamp at the changepoint and a stamp after it are unchanged.** Rows stamped
   `2026-03-26 08:30Z` and `2026-03-26 09:00Z` come back identical. This is the boundary the whole
   change turns on, and a `<=` where the rule wants `<` passes test 1 and fails this one.
3. **The corrected frame still validates.** Applying `correct_late_stamps` to a multi-series frame
   spanning the changepoint and calling `PowerTimeSeries.validate()` raises nothing — proving the
   shift creates no duplicate key and leaves the `time` column sorted within each series. Fails on
   `main` for the same reason as test 1, and would fail on a buggy implementation that shifted rows
   on both sides of the boundary by different signs.
4. **`scan_delta` returns corrected stamps from a table written raw.** Write a small frame spanning
   the changepoint to a `tmp_path` Delta table with `write_power_time_series`, read it back with
   `PowerTimeSeries.scan_delta(path)`, and assert the pre-changepoint stamps came back 30 minutes
   earlier while the table on disk still holds the raw values. This is the test that pins the
   "stored raw, read corrected" contract, and it is the one a future correct-on-ingest change would
   have to delete deliberately.
5. **`scan_delta` resolves its defaults from settings.** With `power_time_series_data_path`
   monkeypatched, a no-argument `scan_delta()` reads that table. Mirrors the equivalent coverage
   `Nwp.scan_delta` has, if any; if `Nwp.scan_delta` has no such test, this one is optional and the
   simplicity reviewer should say so.

Changed tests:

- **`packages/nged_data/tests/test_storage.py`** — add an assertion to the existing `select_new_rows`
  coverage that a pre-changepoint row already on disk is *not* re-selected as new. Fails on a
  version of this change that mistakenly routed `select_new_rows` through `scan_delta`, which is the
  single most likely way to get this change wrong.
- **`tests/test_assets.py`** — the ingest asset tests assert stored `time` values. They must keep
  asserting the raw values, and gaining a comment saying so; no assertion changes.

## Docs to update

- **`packages/contracts/README.md`** — if it describes `PowerTimeSeries.time`, restate it to match
  the new field description. Check before editing.
- **`docs/roadmap/data-cleaning.md`** — add a section for the stamp correction alongside the
  commissioning ramp and the export cap, written in the present tense: what the fault is, where the
  correction lives now, and that the stored table holds NGED's own stamps. This page is the catalogue
  of trial-area data faults and is currently silent on the one fault we have actually fixed.
- **`docs/architecture/`** — whichever page describes how the power table is read. A grep for
  `power_time_series` across `docs/` returns 17 files; the implementer reads them and edits only
  those left inconsistent, rather than working from this list.
- **`docs/ml_experimentation/dagster-workflow.md`** — if it tells an experimenter how to load power
  for a notebook, point it at `PowerTimeSeries.scan_delta()`.
- **No roadmap status banner moves.** This issue completes no roadmap item: `docs/roadmap/data-cleaning.md`
  stays 🚧 Planned, because the commissioning ramp and the shape-change detection are still open.

## Verification commands

The green-before-push set from `implement-issue`, plus:

```bash
uv run ruff check . && uv run ruff format --check .
uv run ty check
uv run pytest
uv run pymarkdown scan -r docs README.md CLAUDE.md packages/*/README.md
uv run mkdocs build --strict          # docs/ pages and a new cross-link are edited
uv run python scripts/lint/check_marimo_notebooks.py   # two dashboard notebooks are edited
```

No network-gated tests are needed: nothing here touches NWP conversion conventions or an external
feed.

## Risks and open questions

**1. Is the correction fleet-wide, or only the six metered solar farms?** The measurement needs
solar geometry, so it cannot speak for a substation load series. Two of the three signals — the
rows-per-day drop from 48 to 26 and the disappearance of exact-zero rows — are properties of what
NGED publishes rather than of PV physics, and both are consistent with a fleet-wide change, but
neither establishes the *direction* or *magnitude* of the offset on a non-PV series.
*Recommendation:* ship the correction fleet-wide, because a single feed correcting at a single
instant is by far the likeliest reading and the constant is one line to scope down. Ask NGED in
parallel. If the answer comes back "per feed", the fix is to make
`POWER_STAMPS_CORRECTED_AT` a per-`time_series_id` lookup, which the one-door design makes a local
change. **This is the reviewer's call to confirm.**

**2. Should NGED be asked to republish a corrected history first?** The issue raises it and it is a
human action, not a code change. *Recommendation:* ask, but do not block on the answer. Correct-on-read
is the design that costs nothing if a corrected history later arrives — the constant moves to the
epoch, or the method is deleted, and no stored bytes have to be rewritten. That asymmetry is itself
an argument for this design over the backfill.

**3. Should a guard stop a future read site scanning the power table raw by accident?** Nothing
enforces the door, and a new `pl.scan_delta(settings.power_time_series_data_path)` added in six
months would be silently wrong. The repo already runs plain-regex pre-commit hooks of exactly this
shape (`no-sphinx-roles` in `.pre-commit-config.yaml`) and a `scripts/lint/` directory of small
Python linters. *Recommendation:* add a `pygrep` hook forbidding `power_time_series_data_path`
outside an allow-list of `contracts/settings.py`, `contracts/power_schemas.py`, `defs/assets.py`,
`nged_data/storage.py` and the tests. It is about six lines of YAML. **The reviewer should say
whether this earns its place or is process for its own sake.**

**4. Re-scoring changes every stored leaderboard number.** `cv_forecast_metrics` joins actuals to
forecasts on `valid_time`, so correcting the actuals changes the score of forecasts already on disk
— including forecasts from models trained on late stamps, which will look worse rather than better.
That is the honest number, but it means leaderboard rows computed before and after this merge are
not comparable. *Recommendation:* state it in the PR body, re-materialise `cv_forecast_metrics`
after merge, and treat the pre-merge rows as superseded rather than as a regression to investigate.

**5. Retraining before v0.2.1 goes to AWS (#646).** Every promoted model has learnt the late stamps.
Retraining is a run of the existing CV assets, not a code change, so it belongs after this merge
rather than in this PR. *Recommendation:* the PR body links #646 and says the retrain must land
between the two. Worth a follow-up issue rather than a note that only lives in a PR body — **the
reviewer's call.**

**6. Editing a Patito contract.** CLAUDE.md says to ask before changing one. What this plan changes
is the `time` field's `description` and the class's method set; no field is widened to `| None` and
no range is relaxed, so the rule's stated failure mode — hiding a defect to make a failing
`validate()` pass — is not in play. The description is currently false about 95% of the stored rows,
so the edit makes the contract more accurate rather than less. **Flagged for the reviewer's
explicit approval anyway.**
