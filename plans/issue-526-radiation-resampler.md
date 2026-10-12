# Interval-aware short-wave and long-wave radiation (#526)

The production half-hourly resampler treats backward-looking radiation means as instantaneous
samples. This displaces short-wave daylight energy and gives both radiation fields the wrong time
support. The study demonstrates the short-wave timing defect independently of forecast scores.

Replace point interpolation for the two radiation fields with interval-aware reconstruction.
Short-wave follows a clear-sky shape with interpolated midpoint indices, then conserves each native
window's energy. Long-wave fits a piecewise cubic Hermite interpolating polynomial (PCHIP) through
native-window midpoint means. Keep raw numerical weather prediction (NWP) storage intact, use the
same lazy feature path in training and inference, and defer precipitation, wind, and total cloud
cover (TCC).

## Verdict, scope and process

Worth implementing. Issue #526 is broader than this approved scope; narrow it to radiation and
transfer its remaining precipitation/temperature work to issue #1165, with wind tracked by
issue #525. The draft PR will carry `Closes #526` once that scope transfer is recorded. TCC interpolation
belongs in a note on #1157, not in this implementation. Do not alter model feature selections.

Complex: two sequential independent plan reviews (simplicity, correctness), followed after approval
by three independent implementation reviews (simplicity, correctness, mutation testing). Available
Codex reviewers substitute for unavailable Claude models; disclose that substitution. The local
implementation review requested by implement-issue is additional if available.

- Contracts and storage: no Patito field, raw Delta schema, rounding, or asset change. Regenerated
  feature caches will differ. Do not narrow `Nwp.continuous_var_names()` in this PR.
- Production serving: yes, `_engineer_features` uses the shared resampler during inference.
- Degradation: yes, missing radiation windows and unusable solar denominators require a policy.
- Design alternatives: yes, solar shaping, conservation and missingness admit different choices.
- Callers: yes, direct resampler tests, feature engineering, storage and dashboard classification
  dependents needed searching to identify.

Departures from the older issue/roadmap: solar shaping applies only to short-wave; long-wave needs
no solar geometry. Temperature interpolation cannot recover absent extrema. No circular-class or
wind-storage change is required to deliver this radiation-only fix. The study candidate is a design
reference, not the production implementation: it does not guarantee native-window conservation and
uses pvlib through `studies.solar`, which production must not import.

## Solution

### Walk through

#### Happy path

`TabularFeatureEngineer.engineer` maps H3 weather cells to `time_series_id` through
`_attach_containing_nwp_cell`, then calls `_engineer_features`. Before upsampling,
`_engineer_features` checks whether the native input contains short-wave and only then passes
`metadata_lf.select("time_series_id", "latitude", "longitude")` as
the new keyword-only `solar_coordinates` argument to `_upsample_nwp_to_half_hourly`. Pass metadata
separately because the resampler currently treats every non-weather input column as a group key.
Use site coordinates consistently with the study; the raw source remains the containing H3 cell.

In `_upsample_nwp_to_half_hourly`, separate the two radiation columns from the existing point
interpolation list locally. Keep its group keys, categorical handling, and non-radiation behavior.
Create a native-window relation before grid expansion: every radiation mean at `T` belongs to
`(T - width, T]`, with width 3 hours through lead 144 and 6 hours after that for the supported ECMWF
model. Lead zero has no interval. Derive width from the model's native schedule, never from the
previous non-null value or a sparse observed-row difference, which would reinterpret a missing
window as a wider accumulation. Reuse the native timestamp and group keys as window identity;
derive missingness from native radiation rather than adding a redundant mask column. Reuse
`ECMWF_ENS_LEAD_TIME_HOURS` from contracts for the schedule instead of duplicating its constants.

The new production `_radiation.py` helpers construct six target half-hours for each supplied 3-hour
window and 12 target half-hours for each supplied 6-hour window, before joining the existing output
grid. Target timestamps mark the ends of those half-hours. A private expression in that module
evaluates a vectorised, dependency-free solar-position approximation in Polars, including
longitude and equation of time, and Haurwitz irradiance. Use 15-minute midpoint quadrature: evaluate
at `T - 22.5 min` and `T - 7.5 min` for a target half-hour. Average those half-hour means over each
native window to get the matching denominator; this makes both integrations internally consistent.
Evaluate geometry on unique `(time_series_id, target_time)` rows and join the resulting means back
to member/run windows, so shared coordinates and timestamps do not repeat trigonometry 51 times.
Use the study's Haurwitz coefficient 0.059; its expression is the implementation reference rather
than the roadmap's illustrative 0.057. Reference independent geometry in tests before choosing a
documented accuracy tolerance near sunrise/sunset.

For finite short-wave windows with a clear-sky mean of at least 1 W/m², divide the native radiation
mean by the clear-sky mean and anchor the resulting index at the native window midpoint. Interpolate
ratios in actual elapsed time at target midpoints inside contiguous supported daylight blocks. Detect
support gaps from scheduled-window adjacency, not just adjacency among surviving daylight anchors.
Define daylight blocks from night gaps in the generated half-hour clear-sky support, not from
entirely dark native windows: northern UK summer nights can be shorter than a 6-hour window. Make
a native index available to each daylight block its window overlaps, retaining its native midpoint
anchor, and interpolate only among indices for that block and supported source segment. A window
overlapping two daylight blocks contributes its same index to both; conservation still applies
once across that native window's complete target set. Do not interpolate a cloud-index trend
from one day's sunset into the next day's sunrise. At sunrise/sunset
hold the closest usable daylight ratio; never interpolate toward an artificial zero night ratio.

**Each complete short-wave window must conserve its supplied energy.** Multiply by target
half-hour clear-sky means, then rescale the targets within each complete native
window so `sum(target_mean * 1800 s) == native_mean * width_seconds` within Float32 tolerance.
There is no clipping at clear-sky index 1: enhancement and quadrature error must not destroy energy.
The value at a native timestamp becomes that final half-hour's mean, not the original 3/6-hour mean.

For long-wave, fit shape-preserving PCHIP through native-window midpoint means and evaluate at
half-hour midpoints. Hold the nearest value outside the anchors within each contiguous source
sequence; never bridge null or absent windows. One anchor gives a constant and two give a line.
Prioritise smooth machine learning (ML) features over exact interval conservation, as requested in
human review.
Join radiation targets onto the same output grid and remove all internal helper columns.
Bulk and single-run feature paths then consume the identical corrected NWP relation before their
existing joins, lags, and rolling features. The output remains lazy and retains existing dtypes.

#### Main branches

- No radiation columns: retain today's path and do not compute solar geometry. Long-wave alone
  needs no coordinates. `solar_coordinates` defaults to None for existing non-short-wave callers;
  production supplies it when short-wave is present. Direct short-wave callers must supply
  coordinates explicitly. Minimal metadata fixtures without short-wave remain valid.
- Lead zero: retain its null radiation. A finite lead-3 mean supports targets from lead 0.5 through
  3, even though lead zero is null. Reconstruct this first window rather than copying the study's
  cache-boundary fallback, because original native input gives its support.
- Missing native row or null native radiation: leave that native window's targets null and split
  ratio interpolation there; never bridge a missing window or borrow another run/member's values.
  A finite later mean supports its own scheduled window. A sliced input supports its first supplied
  finite window only where its start can be established from the native schedule, not guessed from
  the next sample. Keep the existing output-grid extent unless extending it is explicitly approved.
- All-zero dark window: output zero, without requiring a daylight index. Dark endpoint with daylight
  earlier in the window: use the integrated solar shape normally.
- Small solar denominator or no usable daylight ratio in a supported segment: normalise the local
  positive clear-sky weights directly to the native total when possible. If weights sum to zero,
  retain the native mean as a constant over the window. This preserves source energy, including
  small positive fully-dark source values; do not invent daylight or silently discard them. Exact
  nighttime zero is guaranteed for zero-energy dark windows and for dark target half-hours in a
  successfully solar-shaped window, not for that explicitly documented fallback.
- Zero native mean: output zero regardless of neighboring ratios. Do not divide by a zero
  reconstructed sum. Conservation always runs on complete generated native windows before joining
  the existing grid; cropping therefore needs no separate renormalisation branch.

#### Error handling

Absent radiation stays null and reaches the existing model missing-input path; never raise because
a weather window is absent. Coordinate joins must be unique by time series; validated production
metadata provides finite coordinates. A missing coordinate in a short-wave helper call is invalid
programmer input, not a weather-absence degradation; do not silently change its reconstruction.
Reject duplicate keys and malformed programmer input at the existing boundaries. Do not claim
that `Nwp.validate` already rejects off-schedule leads: the completeness report tolerates them.
Join to supported scheduled widths and leave unsupported radiation windows null, retaining the
existing grid and non-radiation path. Introduce neither a new serving exception for off-schedule
weather nor an eager full-NWP validation pass. Do not introduce an eager collect to
emit per-row warnings. Document the fallback and expose counts in the regression/diagnostic report;
new operational degradation columns and general imputation monitoring are separate work.

## Changes by file

- `packages/ml_core/src/ml_core/features/_radiation.py` (new): private pure Polars solar-position
  and Haurwitz expressions, native interval support, short-wave midpoint-index interpolation and
  short-wave conservation, long-wave PCHIP and null handling. Plain lazy functions, group keys supplied
  by `_nwp.py`; use joins/window expressions rather than Python group loops. No pandas, pvlib,
  study imports, config object or new package. Extract solar helpers only when actual reuse needs it.
- `packages/ml_core/src/ml_core/features/_nwp.py`: local radiation dispatch; preserve all other
  interpolation behavior and output keys. Document the new interval and null semantics.
- `packages/ml_core/src/ml_core/features/tabular_feature_engineer.py`: pass the coordinate relation
  before upsampling; avoid introducing coordinate columns into the generic group-key calculation.
- `packages/ml_core/tests/test_radiation.py` (new): synthetic known-answer interval tests and solar
  reference checks. `test_features.py`: adapt radiation-specific expectations and cover dispatch.
- `packages/ml_core/tests/test_cross_mode_equivalence.py`: add radiation to meaningful cross-mode
  assertions on shared forecast targets, including hindcast rows used by rolling/lag features.
  Its existing fixture supplies half-hourly NWP; create a separate native 3-hour radiation fixture
  with site coordinates and half-hourly power timestamps. Retain existing known-answer fixtures
  rather than inserting overlapping 3-hour radiation windows at every half-hour.
- Check, without changing numeric membership, `packages/delta_store/src/delta_store/nwp.py`,
  `packages/contracts/tests/test_weather_schemas.py` and
  `packages/dashboard/tests/test_forecast_chart.py`. Their rounding/partition/plotting behavior
  must survive local dispatch unchanged.

The existing `plans/` directory contains inherited study plans. Leave those unrelated tracked files
alone; this task adds only this one implementation plan in its existing isolated desktop worktree.

## Tests and acceptance

Regression assertions that fail on current main for deliberately varying native values:

1. Independent half-hour backward clear-sky reference, aggregated to 3- and 6-hour native windows:
   corrected daylight centroid and nighttime output match the reference within a justified geometry
   tolerance; current point interpolation is delayed. Include winter and summer UK cases.
2. Changing-cloud known-answer inputs distinguish midpoint-anchored ratios from endpoint anchoring
   through final half-hour outputs, even after conservation. Check shape and native totals
   separately. Use at least three nonconstant daylight indices and independently calculated
   normalised target shares; explicitly show that endpoint anchoring gives different final output.
   A constant-index clear day alone cannot test anchoring.
3. Each finite complete short-wave window conserves `mean * duration`, including lead 144 to 150
   and the final lead-360 window. Long-wave matches an independent SciPy PCHIP reference through
   interval midpoints, including the 3/6-hour transition, extrema, and one/two-anchor sequences.
4. The first finite interval supplies its early half-hours, lead zero remains null, and a missing
   middle source window remains null across its whole support. Current linear interpolation loses
   the first window and fills interior source nulls. Include an actually absent native row, not
   only a row with a null value, to detect incorrect width inference.
5. Finite short-wave mean at a dark endpoint reconstructs its preceding daylight; target nighttime
   zero is checked separately from per-window conservation, using a daylight/dark transition
   where point interpolation leaks energy into night.

Additional preservation checks, explicitly not claimed to fail on main: shuffled rows; independent
series/runs/members, using contrasting member labels including 0 and 50; categorical and
non-radiation columns; stable schema and lazy execution across collection engines; solar/no-solar
dispatch; and cross-mode agreement. Exercise tiny-denominator, all-dark positive, all-dark zero and
isolated finite-window fallbacks without NaN/inf or energy loss. Check that cropping does not
concentrate a full window's energy into visible rows by comparing with an unsliced reconstruction
with identical native support. Include off-schedule radiation input that stays null without a
serving exception, and two daylight days with different indices to detect interpolation across a
night gap, including a northern-UK summer 6-hour case with no entirely dark native window.
Reference-check solar geometry at leap-year dates and western/eastern UK coordinates.
Existing tests may already cover some checks: extend rather than duplicate them. Constant-input
and generic finiteness/cropping assertions alone do not demonstrate that main's resampler is wrong.

Attempt a small reproducible real-native-data check for a short-lead case, the 144-hour transition,
and a run with missing radiation, including all 51 members when supplied. Compare bulk versus
selected single-run engineered values on identical
forecast targets; aggregate target means back to source windows before target-time filtering.
Record the command, runtime, timing/energy/fallback counts and results in the PR; no new permanent diagnostic
framework or profile publication is required. Never use engineered
caches to infer original missingness. If data is unavailable, record that validation as pending
and do not describe synthetic parity as real-data parity or deploy before it is checked.

## Design-philosophy check

Principles 3 (one execution path), 9 (provenance), and 11 (lazy computation) are preserved: raw NWP
is untouched, original null support survives, and both modes call the same lazy resampler. H1's
absence-tolerant service is respected by null/fallback behavior; the plan adds no asset check or
new failure-producing warning path. Malformed input and internal duplicate joins remain bugs.
No new model feature, model-selection branch, asset, or raw table is introduced. Conservation is a
constraint on reconstruction, not a claim to recover unknown within-window weather or improve
forecast scores. Long-wave is absent from the current 24-feature model recipe.

## Documentation and rollout

- `docs/architecture/nwp-variable-conventions.md`: describe actual backward half-hour output,
  changed native-timestamp values, conservation, fallback, and source-null behavior. Keep the old
  radiation section anchor if needed by existing inbound links, but rewrite its text in present
  tense. Preserve unresolved wind and temperature limitations.
- `docs/roadmap/xgboost-improvements.md`: remove the completed radiation implementation recipe and
  link to permanent architecture decisions; correct the two-radiation clear-sky claim. Keep other
  resampling work and the shared-helper/physics-feature dependency accurate.
- `docs/roadmap/index.md` and `docs/roadmap/forecast-building-blocks.md`: update only relevant
  completion/dependency statements; the broader milestone is not complete.
- `docs/design-philosophy/inherent-stability.md` (missingness in learned models) and
  `docs/architecture/ecmwf-ens-known-issues.md`: distinguish preserved radiation-null windows from
  precipitation's existing interior interpolation; their current descriptions bundle all three.
  Update both learned-missingness explanations, including the earlier passage around line 325.
- `docs/studies/forecasts/production-feature-faults.md`: leave recorded historical results intact;
  update the production handoff with the implementation link and conservation-policy departure.
  Coordinate with PR #1164's small nighttime-threshold wording edit.
- Issue #1157: record instantaneous TCC point-interpolation recommendation, source-metadata verification,
  percent units, preserved native values/bounds/nulls, and explicit exclusion from this PR.
- Before deployment, regenerate affected engineered caches and retrain under the new semantics.
  Never serve an existing model with new features and call that a controlled comparison. Repeat
  radiation-only contrasts on actual production outputs; forecast gains are measured separately,
  not a correctness acceptance criterion. Retain a baseline implementation commit and cache IDs.
- At merge, archive this plan's substance in the PR and remove this task's plan file. The closing
  link is for #526 after its residual scope is transferred; study and TCC issues remain separate.

## Verification commands

For implementation, load `code-style` first. Run the implement-issue green-before-push checks:

```bash
uv run ruff check .
uv run ruff format .
uv run --all-packages ty check
uv run pytest
uv run pymarkdown scan -r docs README.md CLAUDE.md packages/*/README.md
uv run mkdocs build --strict
```

Run targeted radiation/cross-mode tests first, inspect the rendered convention/roadmap pages, then
the full suite. Add `--run-studies` if production-boundary checks or study files change. Run the
real-native-data check and record its exact inputs, source revision and result. During planning,
validate Markdown, diff whitespace and the plan/PR scope; no production tests are implied to have run.

## Risks and reviewer decisions

- Per-window rescaling may add discontinuities at native boundaries. Recommendation: preserve
  source energy explicitly now, measure profiles; defer a smoother conservative reconstruction.
- Long-wave PCHIP treats native means as midpoint estimates and changes interval energy. Human
  review prioritises smooth electricity forecasting features; forecast-score comparison remains.
- Preserve positive all-dark source means or discard them? Recommendation: preserve energy with
  constant fallback, report the anomaly, and never pretend exact zero-night and strict conservation
  can both hold for contradictory source data.
- Solar approximation accuracy and daylight floor: recommend 1 W/m² for usable native denominators,
  15-minute midpoint integration and independent pvlib reference checks in tests only. Establish
  explicit geometry/timing tolerances before implementation; do not copy the study's pandas path.
- Missing-window behavior differs from current accidental filling. Recommendation: intentionally
  preserve radiation null windows; broader bounded filling and imputation flags remain separate.
- Caches/models: regeneration and retraining are mandatory before deployment, with radiation-only
  comparisons retaining unchanged forecast cases and evaluation configuration.

## Review record

GitHub drafts received fresh factual/scope reviews before posting, with truthful Codex attribution.
The PR body includes a guard against merging this draft before implementation and validation.

Simplicity review accepted: select coordinates only for short-wave; require valid coordinates rather
than silently substituting a different shape; reconstruct complete native windows before cropping;
reuse natural window keys and source nulls; keep solar expressions in one private radiation module;
test anchoring through final outputs; extend existing preservation tests; and keep retraining/cache
work as deployment gates rather than mandatory new report infrastructure. Verified minimal-metadata
callers in `test_features.py` and `test_cross_mode_equivalence.py`, mandatory coordinate fields in
`TimeSeriesMetadata`, and the existing min/max grid in `_nwp.py` before adopting those changes.

Alternative deferred: independent constant cloud attenuation per native window would remove
midpoint-index interpolation and its segment logic, but drops an explicit agreed requirement.
Retain interpolated indices; reconsider the simpler option only with the maintainer's approval.
No broader rearchitecture was proposed.

Correctness review accepted: add a native-grid radiation fixture rather than reusing half-hourly
NWP; separate true regressions from preservation/edge-policy assertions; use supported-width joins
and null radiation for off-schedule weather because the existing contract does not reject it; test
final shares with three varying daylight indices; verify source-window adjacency; test solar
geometry over leap years and UK longitude extremes; record runtime; and update both learned-
missingness documentation passages. Geometry is evaluated on unique series/time rows, and daylight
blocks explicitly prevent cross-night cloud-index interpolation. A follow-up check caught that
entirely dark native windows cannot identify all UK summer nights; use half-hour solar support
instead, and test a 6-hour northern-UK summer case without a wholly dark native window. Benchmark
the distinct/join cost of geometry reuse rather than assuming laziness alone makes it fast.
Verified the test timestamp builder,
`Nwp.validate` and `assess_nwp_run_completeness` before revising the plan.

No correctness findings were rejected. The native radiation upper bound was checked: engineered
`AllFeatures` imposes no matching bound, so solar-shaped half-hours above the native mean are not a
contract defect and must not be clipped merely to satisfy the native schema. No implementation code
has been written.
