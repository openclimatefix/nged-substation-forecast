# Audit the production forecast's engineered features

The quick-wins study found suspicious feature values by plotting its inputs. This study checks
Part 1 of issue 1156 against both the cached engineered frames and the current production code.
The user asks for a report with plots and recommendations, with no production changes.

The study will audit the existing frames, run known-answer probes through existing production
functions, and publish a diagnostic report. It will distinguish invalid calculations, temporal
aggregation effects, encoding choices, and missing observations. Following the user’s added request, it will also fit study-only corrected-feature models and
measure forecast-score changes. Production code remains unchanged.

## Verdict and process

Worth doing: the visual anomalies are reproducible hypotheses, not established causes. Complex:

- Stored outputs: yes, a new study page, figures, and reproducible diagnostic tables.
- Production serving path: read and probe only; no changes.
- Production degradation rule: no changes.
- Multiple defensible designs: yes, full-frame diagnostics versus example-only checks.
- Callers needing search: yes, weather conversion, interpolation, static and calendar features.

Two sequential plan reviews, script review before execution, two scientific-validity reviews,
diff simplicity and correctness reviews, and first-stumble/prose/evidence reviews. The prescribed
Claude Opus/Sonnet models are unavailable; use fresh available reviewers and disclose this
substitution. No shared study package changes; no mutation pass. No spending or downloads.

## Planned diagnostic contrasts

1. Period-ending short-wave radiation during fully dark half-hours versus daylight, by lead band;
   separate native period means from inserted half-hours. Probe both 3-hour and 6-hour means through
   the current upsampler; do not treat a dark last half-hour as a dark whole native interval.
2. Raw wind-direction and weekday encodings versus circular encodings; probe directions either
   side of north through the current upsampler. Check existing weekday sine/cosine too.
3. Production windchill versus its reconstructed formula and versus temperature outside its stated
   domain (temperature <=10 C and wind >=4.8 km/h). Emulate 6-hour sampling from validation control
   member's native 3-hour temperature knots; compare daily ranges on complete matched days.
4. Precipitation-type missingness before versus after the observed availability boundary, grouped
   by NWP initialisation month; compare training and validation separately.
5. Missing 336-hour lag before its availability horizon versus at/after it; compare lag values
   with observed power at valid_time minus 336 hours on all unique control-member lookup keys and distinguish
   absent observations from a failed join. Never fill from observations at/after issuance.

These are the deciding planned contrasts. Extra checks after the first run are labelled post hoc.
The order follows the updated issue's priority assessment, which is a prior, not a measured score
ranking. Wider requested checks (bulk/single-run equivalence, lag equality boundary, control-member
selection, units, and time alignment) are diagnostic checks, not new learned comparisons. Include
synthetic boundary/equivalence controls where practical and explicitly disclose untested items.
Solar geometry is computed once per unique series/time and joined across members. Lag lookup
conclusions are restricted to the control-member rows actually checked, with denominators printed.
Following correctness review, the detailed audit uses member 0 in both splits; input manifests
also count every member. All prevalence statements are scoped to these control-member rows.
Each affected-population table holds eligible/affected rows, affected series and lead bands.
Wind discrepancies use shortest-arc interpolation between finite native knots, exclude native
rows and ambiguous 180-degree pairs, and are not errors against observed wind. Precipitation
availability is checked at run resolution before monthly summaries. The reverse fold is untested.

Radiation has an independent synthetic half-hour backward-mean daylight curve, aggregated into
3-hour and 6-hour means. Compare the production reconstruction on common supported interior
rows for energy, timing centroid and dark leakage; retain lead-zero null and all-dark controls.
Lag lookup categories distinguish missing keys, null readings, matching values and unexplained
cache/source mismatches; none alone proves a historical join failure. Boundary probes use a
nonzero NWP publication delay. Temperature emulation retains the actual six-hour grid phase,
uses complete native three-hour days before lead 144, and measures knot-range information loss,
not error against observed temperature. A monotone interpolant cannot exceed retained extrema.
Full real bulk/single-run feature equivalence and power/weather clock-offset correlations remain
explicitly untested; synthetic probes do not certify either.
Counts and rates describe the audited finite rows; they are not uncertainty estimates for other periods.

## Data and files

Read the quick-wins forward-fold cache: April 2024 to June 2025 control-member training and the
weekly sampled July 2025 to June 2026 validation runs with all members. Validate actual row dates,
duplicate keys, feature null/NaN shares and finite ranges. No final-test data. Read local metadata
only for solar geometry and local power only through studies.power.scan_power. Never write
generator identifiers, names, coordinates or unnormalised power to study outputs. Aggregate
population plots; synthetic diagnostic examples are clearly identified.

- studies/production_feature_faults/audit_production_features.py: audited tables, source probes,
  input manifest and generated report.md under a user-supplied output directory.
- studies/production_feature_faults/plot_production_features.py: figures from those tables, with
  figure text emitted separately before rendering.
- studies/production_feature_faults/README.md: inputs, scope, commands and provenance.
- docs/studies/forecasts/production-feature-faults.md: findings, plots and production recommendations.
- docs/studies/assets/production_feature_faults_*.svg: optimised figures.

No src/, production package, contract, configuration, dependency or CI edits. Use a relative link
from the new report to quick-wins; do not change mkdocs.yml solely to add navigation.

## Solution

### Walk through: happy path

audit_production_features.py reads --frames and --data-root, checks keys and date bounds, and
aggregates diagnostics before writing anything. Solar position comes from studies.solar. It calls
the existing production windchill expression and NWP upsampler on synthetic frames to isolate
mechanisms. plot_production_features.py reads only aggregate tables and saves standalone figures.
The page quotes only numbers printed by the audit script in report.md.

### Walk through: branches

Training and validation are reported separately because their ensemble populations differ. Lag
nulls are classified using lead <336 hours before interpreting missingness. Radiation is classified
using both ends and midpoint of the half-hour; native interval means can remain positive at a
sunset label, so a positive endpoint is not itself a fault. If the actual source convention is
unresolved, recommend a convention audit rather than silently zeroing native interval means.

### Walk through: errors

Missing paths, duplicate keys, non-finite geometry, or observed-power key duplication fail fast.
No data gaps are replaced with zeros. Current code and historical caches are distinguished using
source-file hashes and cache metadata; a current-code probe cannot prove historical provenance.

## Verification and limitations

Run ruff, formatting, ty on changed scripts, study boundary tests and the studies suite, markdown
lint, and mkdocs --strict plus inspect the rendered report. Render and view every chart. Source
probes serve as known-answer positive and negative controls. Scripts receive fresh review before
running. Reviews may require re-runs. Score experiments below estimate forecast effects; production implementation remains follow-up work.


## Forecast-score extension (planned before fitting)

The user additionally requested measured score improvements and GPU training. Diagnostic results
already exist; the forecast contrasts below are planned before any forecast result, rather than
being selected from those results. Only study scripts implement corrections.

Baseline: fresh per-series XGBoost refits of the current 24-feature production recipe, control
member training and all 51 validation members. Main controlled comparisons use column subsampling
1 as required by the study skill, with the production 500 trees, depth 6, learning rate .05,
child weight 1, row subsampling .8. A separate exact-production .8 column-subsampling comparison
runs baseline versus all corrections. Every arm retains all 24 columns; masking means replacing
a column with missing values, with the retained count explicit. No target cleaning changes.
Train and predict rows stay identical, including missing inputs. Use CUDA, one fit at a time,
8 CPU threads, seeds 0, 1, 2. Repeat every planned contrast with
studies.cross_validation.SENSITIVITY_HYPER_PARAMETERS and column subsampling 1.

Five deciding contrasts:
1. All corrections minus baseline.
2. All corrections minus all except short-wave radiation correction.
3. All corrections minus all except wind-direction correction.
4. All corrections minus all except guarded windchill.
5. All corrections with precipitation type masked minus all corrections.

Corrections: reconstruct short-wave radiation with the reviewed clear_sky_index_resample helper,
using native 3/6-hour means divided by Haurwitz clear-sky means over those same windows, anchored
at their midpoints, nearest daylight index outside daylight, and half-hour backward clear-sky
means at the targets. Geometry uses 15-minute midpoint quadrature, no observed power. Recover
native knots from the engineered cache, retaining its endpoint support; retain production values outside verified cached support, and count all-daylight-missing groups. This is a candidate resampling policy,
not recovery of the true sub-grid radiation. Wind uses shortest-arc interpolation of native
directions at both heights (speeds unchanged to isolate the wrap fault); ambiguous 180-degree
pairs and unsupported boundary rows retain production values and are counted. Windchill falls
back to air temperature outside temperature <=10 C and wind >=4.8 km/h. Missing values remain
missing. Long-wave is absent from the baseline feature list, so cannot change its score.

Precipitation masking is an availability sensitivity, not reconstruction of missing history.
Temperature compression and unavailable lag observations have no defensible recoverable truth;
report their measured limitations without inventing corrected scores. Exploratory negative
control: mask ordinal weekday while retaining its sine/cosine. Known-answer correction probes
are positive controls; the one-hour displacement suggestion is left for a follow-up, since the known-answer probe already isolates timing.

Score against guarded observed power, normalise each series before averaging, then give series
equal weight. Report fair CRPS and ensemble-mean MAE as percentage points of effective capacity,
overall, by type and issuance lead band. Pair arm losses at identical series/run/valid-time keys;
report denominator and member checks. Resample whole validation months and one common fitting
seed, 2,000 draws. Print absolute arm errors, paired differences, intervals and monthly signs.
These are study scores on sampled runs, never leaderboard submissions or leaderboard skill.
Treat a -0.15 CRPS-point improvement as the issue’s practical threshold, requiring 9/12 improving
months and interval below zero before claiming a robust gain. Confirm the combined correction
against baseline on the cached backwards fold, with future-trained chronology disclosed as a
stress test, not prospective evidence. Use manual_heuristic only as a secondary reference if
identical cached row keys can be established. Plot total-gain feature importance descriptively.


Pre-fit review amendments: call the .8 refit a production-hyperparameter cached-data refit;
check native-grid linear round trips, preserve unsupported boundaries and missing targets,
sort training by complete forecast keys, enforce identical prediction keys and member labels,
and bind caches to source, metadata, helper and capacity provenance. Count candidate interval
energy discrepancies. Months are assigned by valid time. Bootstrap draws recompute per-series
means from monthly sums/counts; also print the issue’s monthly t-interval, which gives each month
equal weight and has a different estimand from the main equal-series score. Print monthly signs
and all scoring denominators. The backwards stress test spans April 2024–June 2025 (15 months,
66 weekly runs), trained on July 2025–June 2026. Main primary interpretation remains forward.
