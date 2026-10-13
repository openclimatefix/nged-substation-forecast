# Post hoc extension: two solar sets and three wind representations

**This extension answers the user's requests for five fixed solar/wind feature sets; it does
not continue the stopped G4 ladder.** The original ladder remains stopped at its first
addition. Label this extension, its contrasts, results and figures post hoc even though the
following specification precedes these new fits. Existing study results and provenance remain
immutable. Two independent Codex plan reviews accepted this design subject to the execution gates below.
Implementation proposals are being reviewed; no extension code has run. The original fitting
code and queue remain unchanged until their successful completion.

## Scope and fixed recipes

The questions are whether either requested solar set improves loss against the full-input
control, whether expanded solar improves on minimal solar, whether five-input minimal wind
improves on the full-input and original lean-wind controls, and whether either requested wind
representation improves on five-input minimal wind. Other listed comparisons are exploratory,
using the same registered fits and both settings.
Use the historical `production_cleaned` per-series squared-error recipe, with the same cleaning,
row subsampling, fitting targets, device, fold windows and seeds. Do not substitute the public
production training path or pooled quantile winner. Add exactly five fixed type-specific
recipe policies: `solar_minimal_requested`, `solar_expanded_requested`,
`wind_minimal_requested`, `wind_minimal_uv_requested` and `wind_minimal_circular_requested`.

- Minimal solar: `temperature_2m`, `downward_short_wave_radiation_flux_surface`.
- Expanded solar: `temperature_2m`, `dew_point_temperature_2m`, `wind_speed_10m`,
  `downward_long_wave_radiation_flux_surface`, `downward_short_wave_radiation_flux_surface`,
  `precipitation_surface`, `categorical_precipitation_type_surface`, `sun_position`,
  `clear_sky_irradiance`.
- Minimal wind, exactly five inputs: `temperature_2m`, `wind_direction_100m`,
  `wind_direction_10m`, `wind_speed_100m`, `wind_speed_10m`.
- Minimal wind plus components, exactly nine inputs: the preceding five plus `wind_u_10m`,
  `wind_v_10m`, `wind_u_100m` and `wind_v_100m`.
- Circular minimal wind, exactly seven inputs: `temperature_2m`, `wind_speed_10m`,
  `wind_speed_100m`, `wind_direction_10m_sin`, `wind_direction_10m_cos`,
  `wind_direction_100m_sin` and `wind_direction_100m_cos`. These replace the raw angles;
  this recipe contains neither raw direction columns nor u/v components.

Expand `sun_position` into three continuous components below, making 11 real expanded columns:
seven weather columns, three position columns and one clear-sky column. Minimal solar has two real
columns. Neither new PV set contains power lags, clock/calendar columns, a numeric site
identifier or any implicit extra predictor. Both remain separate models per solar series.
For each solar recipe, non-PV series retain the full-input historical control policy; for the
wind recipes, non-Wind series retain it. Demand remains unchanged. All five new sets exclude
windchill, lags, calendar fields and site identifiers. The five- and nine-input wind sets
preserve the two original direction columns exactly, including their units and circular wrap;
the seven-input set replaces them with the specified sine/cosine pairs. These bundled removals
cannot isolate windchill's contribution or a causal removal effect. Component and circular
contrasts test deterministic representations, rather than new weather information.
Do not change untargeted series' targets or use their outcomes to select the new inputs.

Keep column subsampling at one and the original maximum width at 24. Sort each recipe's real
columns, then cycle through its own sorted retained columns to add uniquely named deterministic
duplicates to width 24. Never borrow an omitted input. Print real and trained ordered columns
and counts; verify the existing registry's maximum stays 24 after adding these smaller sets.
Preserve the original main recipe, including its fixed rounds and absence of early stopping,
and use the existing `SENSITIVITY_HYPER_PARAMETERS` unchanged for the second setting. No tuning,
feature search, calibration or additional ladder is proposed.

Both extension-plan reviewers' padding diagnostic is accepted for all five sets: repeat each
new recipe with `--padding none` in both main folds and all three seeds. These are additional
main-setting representation checks, not replacements for the equal-width deciding comparisons.

## Geometry and irradiance definitions

**Position uses the midpoint of the half-hour ending at `valid_time`.** The existing frame
builder evaluates apparent zenith and azimuth at `valid_time - 15 minutes` using
`studies.solar.zenith` and `studies.solar.azimuth`. Their documented units are degrees, with
azimuth clockwise from north. Verify the actual frozen angles against those semantics and
their pinned source/dependency versions before reusing them. Convert to radians and add:

```
sun_east  = sin(zenith) * sin(azimuth)
sun_north = sin(zenith) * cos(azimuth)
sun_up    = cos(zenith)
```

These dimensionless unit-vector components represent position without an azimuth wrap or a
zenith-coordinate singularity. Do not clamp the vector's below-horizon components to zero.
Require finite geometry and a unit norm within a declared Float32-appropriate tolerance
(fixed absolute norm error at most 5e-7); retain higher precision during calculation and
record the final dtype. Check north crossings, near-zenith and below-horizon synthetic cases.

**Clear-sky irradiance is a half-hour mean of Haurwitz global horizontal irradiance.** For
each private coordinate and half-hour ending at `valid_time`, evaluate apparent zenith at the
30 one-minute midpoints: 29 minutes 30 seconds before the label through 30 seconds before it.
Pass those angles to `studies.baselines.haurwitz_w_m2` and average the 30 values, producing
`clear_sky_irradiance` in W/m². Both primitives already exist; no shared-package change is
needed. Do not use `hourly_clear_sky`'s different sampling or the old three-hour-smoothed
`clear_sky_ghi_window_mean`. The Haurwitz primitive returns zero below the horizon; a whole
half-hour mean is zero only when all its sample contributions are zero. A night midpoint alone
does not imply a wholly dark interval. Check sunrise/sunset and fully dark intervals explicitly.

This is a deterministic clear-sky model approximation, not measured irradiance or tilted-panel
power. It adds no cloud observations, private observed power or future target information.
Time and private coordinates make both derived quantities available before forecast issuance.
Compute once per series and valid time, then join to all matching members/initialisations with
strict many-to-one validation and unchanged row order. Never publish coordinates or the private
series mapping. No external data, download, paid service or new licence is needed.

**Wind components and circular directions derive only from the corrected processed polar
values already in the frames.** For each of 10 m and 100 m, convert the stored meteorological
bearing to radians in Float64, then compute `u = -speed * sin(bearing)` and
`v = -speed * cos(bearing)`, and separately compute the direction's sine and cosine. Cast final
derived columns to Float32, recording both calculation and storage precision. Preserve nulls
and existing nonfinite/missing-input handling rather than substituting zeros or filtering rows.
The four u/v columns reconstruct processed polar values; they are not native upstream components
or recovery of information lost during archive reconstruction. The four dimensionless circular
columns encode the same angles with a wrap-continuous pair. Check cardinal directions, the
north crossing, calm speeds, null propagation and circular-pair norm independently. The
nine-input recipe retains speed/raw direction plus u/v; the seven-input recipe keeps speed and
substitutes sine/cosine for raw direction, without u/v.

### Frozen implementation choices

Use Float64 for trigonometry and irradiance reductions and Float32 for all 12 stored additions.
Verify both frozen midpoint angles against the existing primitives within 0.0001 degrees,
using circular error for azimuth. Keep the existing pvlib solar-position defaults: NREL NumPy
SPA, altitude 0 m, pressure 101325 Pa and refraction temperature 12°C. These are geometry-model
assumptions, not observed site weather. Pin the existing pvlib version and primitive source hashes.
The public synthetic integration guard compares one-minute with 30-second midpoint averages,
with a fixed 0.25 W/m² absolute tolerance; it is not a guarantee for every atmospheric condition.

Audit precipitation by both weather lead (`valid_time - nwp_init_time`) and served forecast lead
(`valid_time - power_fcst_init_time`), separately. Freeze bins below 72 hours, 72 to below
264 hours and from 264 hours, plus explicit lead-zero support counts. A lack of lead-zero rows
in engineered frames says nothing about the raw archive's lead-zero nulls. Keep separate
before/from-13-November source-run eras and training/validation subsets. Precipitation rate has
units kg m⁻² s⁻¹: count nulls, nonfinite values, zeros, negatives and values above 0.01, the pinned
contract bound. Count type codes outside 0–255, preserving the existing numeric representation.
These are diagnostic counts, never grounds for feature-dependent row filtering or imputation.

## Immutable frame augmentation and preflight

Create a uniquely named study-only augmentation builder in `studies/quick_wins/`, with a fresh
private extension namespace. Read the completed pinned combined frames for each fold; never
rebuild their original weather columns through a newer pipeline. Copy the original metadata,
capacity and bounded-power artifacts byte for byte. Add exactly 12 columns to training and
validation frames: three solar position components, clear-sky irradiance, four wind components
and four circular direction columns. Preserve every original column's dtype,
null positions, row order, keys and finite values exactly, including signed floating-point
representations where present; additions necessarily change the augmented parquet byte hashes.
Retain original file hashes and prove original-column equality independently of parquet layout.

Expanded solar, nine-input wind and seven-input circular wind require augmented frames.
Minimal solar and five-input minimal wind read the original frozen combined frames directly.
Canonical training/validation keys, targets and all
old feature values must remain identical across original and augmented roots; the added frame
columns do not alter either minimal recipe's information set.

The new manifest must satisfy the existing fitter's frame contract and bind the parent manifest,
every parent file, new builder/dependency source hashes, feature definitions, units, dtypes,
coordinate-source fingerprint, fold window and complete new inventory. Verify parent hashes
before and after augmentation. Write only to absent destinations and finalise completion only
after every validation succeeds. Recheck member labels, unique sample keys, target support,
training/scoring bounds and lag masking, even though the new recipes do not consume lags.

Keep component archive version 924, power version 70, capacity version 1 and eligibility
snapshot 1; all observations remain before 1 July 2026 and within their original fold roles.
The frozen full-history effective-capacity scale enters scoring only and remains retrospectively
conditional. Do not add a missing-input row filter or impute missing inputs to zero. Preserve
existing missing/nonfinite handling and report its counts. Stop on unexpected nonfinite geometry,
invalid metadata, changed original columns, lost keys or changed targets rather than silently
dropping rows. Record the weather fields' existing units from their pinned contract; do not
reinterpret `precipitation_surface` as precipitation type or change its accumulation/rate semantics.

Audit precipitation amount/rate and categorical precipitation type separately by lead and era.
Type is wholly missing before the source-run archive boundary of 13 November 2024 at 00:00 UTC.
The amount/rate field has lead-zero and scattered-corruption limitations of its own. Report
nulls and suspicious values without removing matched rows or conflating the two fields' histories.
Preserve the original folds and descriptive availability-era limitations; neither an era summary
nor these bundled feature sets identify precipitation information content or an operational
weather-model upgrade effect.

## Fits, control reuse and source lineage

**The design has 90 new GPU invocations plus one CPU diagnostic:** 60 equal-padding CUDA fits
(five recipes, two folds, seeds 0/1/2 and both main/sensitivity settings) and 30 main-setting
unpadded CUDA fits (five recipes, two folds and three seeds). These counts include all accepted
padding checks. One GPU job runs at a time, with CPU preparation capped at eight
cores. No source or queue used by the original running work is edited. Wait for the original GPU
queue's successful exit and completion footer before applying the reviewed registry-only addition
to `fixed_pipeline_fit.py`. The feature builder is new; the existing frame builder, fitting
helpers and report estimator remain unchanged. Reuse the reporter's canonical training-key
pairing across the old and augmented frame roots, with strict cleaned-count and validation-key
checks, rather than requiring identical feature-parquet bytes.

Before reusing an old control fit, prove that new augmentation and registry membership leave
its selected feature preparation, ordered training/validation rows, cleaning decisions, targets,
capacities, hyperparameters and device semantics unchanged. Retain the old control's original
frame root and manifest in report specifications. Do not relabel it as trained on augmented
frames, rewrite its provenance or bind an old run to the new fitter hash. Preserve historical
source bytes and execution records separately from the extension's newly reviewed source hashes.
The registry's membership changes, so its new inventory is a new artifact, not a replacement
historical inventory.

If that preflight cannot establish valid control reuse, stop before interpreting new contrasts
and explicitly register refits of the full-input, original lean-PV and original lean-wind
controls in the new namespace. Keep conditional fallback groups separate: up to 36 core
full/lean-control fits across both settings; 12 main unpadded original-lean-PV/lean-wind fits;
and 24 guarded-full/guarded-lean-wind fits across both settings. Each group is needed only if
its reuse/preflight checks fail, and is outside the 90 new-recipe GPU count. Do not launch
fallbacks merely because the recipes were added.
Do not silently mix source versions or treat a failed preflight as evidence that controls are
unaffected.

An additional immutable audit must compare each new recipe's untargeted series forecasts
against the corresponding full-input controls, with exact key and forecast-value comparisons,
before publication. Identical feature policies alone are insufficient proof. Keep this separate
from the intended changed PV/Wind predictions and from comparisons between padding policies.
The study skill also requires a device check: propose one fixed new minimal-solar Fold A
seed-0 main CPU refit against its CUDA counterpart, reported solely as a numerical diagnostic.
The original page already includes a production CPU/CUDA comparison; this proposed extension
check is an additional recipe-specific diagnostic, explicitly outside the 90 CUDA-fit design; it
does not add another feature candidate or enter deciding contrasts.

**Guard diagnostics are mandatory but add no guarded versions of the five new recipes.** All
five new targeted feature sets omit windchill; preflight must prove applying only that guard
cannot change their targeted prepared matrices or predictions. Reuse those same target
predictions and compare them with existing guarded full-input controls for all five sets,
and with existing guarded lean-wind controls for all three new wind sets, in both folds and
settings. Bind each reference to its original frames and provenance and the matching padding
policy. The accepted equal-padding diagnostic uses the existing equal-padding guard references;
do not silently substitute a padded guarded-lean reference into an unpadded contrast. Non-targeted
types are excluded from these guard comparisons. These direct comparisons bound dependence
on the full/lean controls' windchill handling; they do not isolate a causal windchill-removal
effect from the new bundles. Guard-reference fallback fits above apply only if their reuse
preflight fails.

## Deciding contrasts, uncertainty and stopping rules

Freeze these seven deciding contrast families now, all labelled post hoc in published material:

1. Minimal solar minus the full-input control, on solar.
2. Expanded solar minus the full-input control, on solar.
3. Expanded solar minus minimal solar directly, on solar.
4. Minimal wind minus the full-input control, on wind.
5. Minimal wind minus original lean wind, on wind.
6. Nine-input wind plus u/v minus five-input minimal wind, on wind.
7. Seven-input circular wind minus five-input minimal wind, on wind.

Additionally report seven exploratory post hoc contrasts: minimal solar minus original lean
solar; expanded solar minus original lean solar; nine-input wind minus full control; nine-input
wind minus original lean wind; seven-input circular wind minus full control; seven-input
circular wind minus original lean wind; and seven-input circular wind minus nine-input wind.
All run in both folds and both settings using the same fits. Do not promote an exploratory
comparison to deciding status after seeing its result. For main unpadded representation checks,
repeat all listed families against matching unpadded controls and compute direct padded-minus-
unpadded paired contrasts for each of the five new recipes. This checks padding dependence
without inferring differences from two separate baseline contrasts. Report every recipe's absolute fair CRPS,
ensemble-mean MAE and its matched contrasts, with common target
support and seed spread. Fair CRPS decides these 51-member point-model comparisons; common
19-midpoint pinball and coverage may remain descriptive secondary diagnostics without a
calibration claim. Keep each run's full output for later inspection and report support/exclusions.

Use the existing equal-series central estimate, three fitting seeds and 2,000 paired
whole-calendar-month/seed bootstrap resamples. Also report the separate equal-month mean and
t-interval and counts of months with lower loss. Explain their different estimands and seed
treatment. Evaluate both main and sensitivity settings for every deciding comparison and state
sign or significance disagreements. Fold A uses the original forward window; Fold B remains
retrospective, and both were already examined in earlier feature experiments. These five fixed
sets were requested after those results, so their evidence is not an untouched confirmation.

Stop at these five sets and the specified seven deciding, seven exploratory, padding and guard
diagnostic comparisons. Do not continue G4,
tune settings, add predictors, select a favourable era or discard an unresolved setting.
Stop execution on provenance/key or
feature validation failures. No new automatic adoption threshold is invented: report magnitude,
both interval methods, both settings and both folds before making any scoped recommendation.
The old 0.15 G4 continuation rule does not become a retroactive selection rule for this new
experiment. An unresolved interval is not evidence of zero effect. Retain unadjusted 5% testing
with explicit multiplicity and shared-month limitations, rather than implying familywise control.

## Bounded outputs, reviews and local cost

Prepare one fixed publication selector family containing absolute scores, seven deciding paired
contrasts and seven separately labelled exploratory contrasts for each fold/setting, with the
specified main unpadded/padded-minus-unpadded and guarded-reference diagnostic tables,
generated only by committed reporting code from completed aggregate tables. Add exact emitted
tables to a labelled post hoc section of the existing feature-results appendix; do not
hand-transcribe or double-round them. Keep the two main narrative study pages. Print real/padded feature lists and null/era support in the emitted report.
Use one compact finding-led figure for the seven deciding paired differences, separated by
series type and fold, with the
second setting as a distinct marker rather than a duplicate chart. Show absolute scores in
labels or the appendix, define capacity points and direction, and distinguish bootstrap bounds
from equal-month t-intervals. No individual private output or generator mapping is published.
Add five descriptive gain-importance figures from saved extension point-model boosters, one
for each requested feature set. Show main and sensitivity settings separately. Combine duplicate
padding credit with its retained source, include zero-gain real inputs, normalise total gain
within each point booster and average equally over series before folds/seeds. Show the mean
and range across the six fold/seed refits per setting, using a logarithmic share display where
needed. These ranges describe variability across refits, not confidence intervals. Importance
does not establish that an input helps; held-out paired contrasts decide that question. No
importance-only fit or feature search is added.

This is complex work: stored study outputs yes, production-serving change no, incomplete-input
policies yes, defensible alternative geometry/averaging choices yes, existing callers requiring
compatibility checks yes. Obtain two fresh independent extension-plan reviews and triage them
before implementation. Every new or changed script, registry addition, execution spec and chart
script requires fresh review before its first run. Require an independent first scientific review
of the design/results and a second fresh scientific review of the revised extension page;
perform requested reruns. Record any model substitutions or shortened process honestly. Apply
first-reader and prose reviews to chart text before rendering, inspect renders, and complete
evidence/prose reviews of the integrated page. Existing parent reviews do not substitute for
extension reviews. No subagent is launched as part of this proposal task.

Planning allowance, not a measured runtime or delivery commitment: 90 serial CUDA invocations
would take roughly 3–8 GPU-hours assuming 2–5 minutes per invocation on the existing workstation, plus
roughly 1–3 local CPU-hours for geometry, frame copying, audits and reporting. The CPU diagnostic,
control-refit fallback and requested review reruns add time. Replace these assumptions with the
first approved extension job's measured timing before giving an ETA or scheduling the remainder. Reviews and
publication work are additional elapsed time. There is no proposed paid compute or data spend.
All results remain weekly-initialisation study diagnostics, with no official leaderboard,
deployment, calibration or joint-scenario claim.

## Decisions requiring root review of this proposal

Confirm the exact geometry/irradiance convention, immutable manifest contract, control-reuse
preflight and proposed recipe names before implementation. The requested five-set matrix has
90 new CUDA jobs and one CPU numerical check; conditional core, unpadded-lean and guard-control
fallbacks remain separate and require failed reuse preflights. Preserve the reviewed three-set
plan in `minimal-solar-extension-plan-proposal.v1-three-sets.md` for amendment review; both fresh
plan reviewers must review these added representations and diagnostics before implementation.
No result-dependent feature expansion or
commitment to production adoption is proposed.
