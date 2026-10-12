# Audit the production forecast's engineered features

The quick-wins study found suspicious feature values by plotting its inputs. This study checks
Part 1 of issue 1156 against both the cached engineered frames and the current production code.
The user asks for a report with plots and recommendations, with no production changes.

The study will audit the existing frames, run known-answer probes through existing production
functions, and publish a diagnostic report. It will distinguish invalid calculations, temporal
aggregation effects, encoding choices, and missing observations. It will not fit forecast models
or claim that a correction improves forecast accuracy.

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
There are no learned arms, folds, fitting seeds, forecast scores or significance tests. Counts and
rates describe the audited finite rows; they are not uncertainty estimates for other periods.

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
running. Reviews may require re-runs. A diagnostic audit can recommend where to change code but
cannot estimate a forecast-score benefit. Production changes and scoring corrected forecasts are
follow-up work, outside this request.
