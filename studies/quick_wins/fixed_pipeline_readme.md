# Forecast comparisons after the wind and radiation corrections

This study refits the production XGBoost configuration and the quick-wins winner on matched
inputs, then rebuilds the lean solar and wind feature sets from issue 1156. Production means a
configuration refit, as in the original quick-wins study. The deployed model artifact is outside
this comparison.

The published study records the planned comparisons and completed scientific reviews in
`docs/studies/forecasts/fixed-pipeline-lean-features.md`. The review record is
`fixed_pipeline_review_log.md`.

## Isolated code and frozen inputs

The base is commit `4e820d24c0129c8c95c567ff7355abff8d416144`. The reviewed prerequisite heads are
PR 1167 at `38025f4eab457b9605b8c124006609869dbfca0d` and PR 1166 at
`0f094c356da82e75c4014664ad7fb33fe1df8b7f`. Runtime worktrees apply only their relevant contracts
and feature-engineering package changes to that base. They do not change this study branch's
production packages. `PYTHONPATH` selects the runtime before Python starts; no script changes
the import path. The [runtime reconstruction instructions](fixed_pipeline_runtime/README.md) and
source-hash manifest reproduce these package sources from the pinned commits.

| Treatment | Feature implementation | Wind archive |
| --- | --- | --- |
| Old | Base commit | Polar speed and bearing, snapshot 1830 |
| Radiation | PR 1166 | Same polar snapshot |
| Wind | PR 1167 | Validated component rewrite of snapshot 1830 |
| Combined | Both PRs | Same validated component rewrite |

In the combined runtime, pass the radiation implementation its solar coordinates and derive wind
features after resampling the component columns. The independent integration review checked this
combination against each PR and synthetic bulk/single-run tests.

The component archive reconstructs already-rounded polar values. Its round-trip validation is
against those archived values; it is not a lossless download of upstream components. Require the
migration readiness record to say ready, with full inventory, partition, unchanged non-wind/key
and wind round-trip checks, before using it. Record its destination snapshot in every frame build.

The study pins cleaned power snapshot 70 and capacity snapshot 1. Metadata, eligibility, bounded
observations, code and dependencies are fingerprinted. Both folds use the same eligible population.
Every observed outcome precedes 1 July 2026. Native forecast context is retained for engineering;
the engineered rows are then clipped to the fold windows.

Capacity snapshot 1 is the full-history 99th percentile of absolute power, and can include later
observations. It is a common scoring denominator only: fitting, cleaning, lag construction and
pooled training scales do not use it. Normalised results are retrospective comparisons conditional
on this frozen snapshot.

## Execution

Set `DATA_PATH_INTERNAL`, `DATA_PATH_DELIVERY` and `REPO_DATA_DIR` to the local data root.
Set `POLARS_MAX_THREADS=8`, `OMP_NUM_THREADS=8` and `OPENBLAS_NUM_THREADS=8`. Run GPU fitting
jobs serially. Use a new output namespace, separate from the original `quick_wins` outputs.

1. Run `fixed_pipeline_frames.py` with an explicit treatment, fold, code revision, archive path
   and all Delta snapshots. Its manifest records the complete inventory and refuses incompatible
   reuse. A pilot build is not accepted for fitting.
2. Run `fixed_pipeline_recipe_inventory.py` before feature fitting to print exact cuts, additions,
   duplicate padding and numeric pooled-site encoding without reading observations.
3. Run `fixed_pipeline_fit.py` once per recipe, seed, device, setting and padding policy. It
   verifies the complete scheduled validation inventory, writes private models and forecasts,
   then writes completion hashes. An incomplete fit requires diagnosis and a new output directory.
4. Run `fixed_pipeline_input_audit.py` on completed, paired frame directories. Its changed-row
   counts, domain screens and timing correlations are input diagnostics, not forecast improvements.
5. Give `fixed_pipeline_report.py` a JSON specification containing labelled fit paths and planned
   contrasts. Its public aggregate report includes each pinball level and actual feature lists.
   Give each run an explicit `frames` path so the pairing guard verifies the pinned frame and
   compares canonical training keys independently of Arrow chunk layout.
6. Use `fixed_pipeline_tables.py` to select and format public aggregate report rows for the page.
   Its JSON specification names exact sources, filters and columns. It performs no estimation,
   refuses existing output directories and records source, specification, script and output hashes.
7. Give `fixed_pipeline_charts.py` an explicit report-backed figure specification. Export text
   first with `--text-only`; review that text before rendering with `--reviewed-text`. Inspect the
   PNG rendered from each optimised SVG before copying publication assets into the docs.
8. Run `fixed_pipeline_importance.py` on the six combined full-feature point-control fits to
   describe per-booster normalized training gain. Use `fixed_pipeline_importance_charts.py` with
   the same text-review gate for its figure. Its ranges describe refit variability, not uncertainty
   about a feature's benefit.
9. Run `fixed_pipeline_unaffected.py` after the corresponding combined-pipeline fits complete.
   Use both folds, main and sensitivity settings with equal padding, and the main setting with
   no padding. It verifies exact sample keys and forecast values for unaffected types, checks
   complete file inventories and hashes before and after comparison, and writes an aggregate
   proof to a new private file. This does not compare affected-type predictions across padding
   policies. The page gives the full commands.

Raw frame exports, models, row-level scores and manifests containing private identifiers stay
under the private study data directory. Public reports contain type aggregates and feature names.
Never copy private manifests into the documentation.

## Requested feature extension

The published study documents two post hoc solar sets and three wind sets with 2, 11, 5, 9
and 7 real inputs respectively. The extension retains the cleaned
per-series point recipe, the two folds, both hyperparameter settings and three fitting seeds.
Untargeted series keep the full-input policy. Main-setting unpadded checks supplement the
equal-width comparisons; they do not replace them or reopen the stopped solar ladder.

`requested_features_frames.py` adds solar position, half-hour clear-sky irradiance, reconstructed
wind components and circular bearings to separate immutable frame copies. Minimal solar and
five-input wind consume the original combined frames. Expanded solar, wind plus u/v and circular
wind consume the augmented copies. The added wind representations derive from processed polar
values and contain no new upstream weather information. Missing precipitation remains missing.

Before fitting, the `reuse_preflight` phase of `requested_features_audit.py` checks preparation,
feature order, training counts and the limited target-specific windchill guard argument in all
six fold/setting/padding scopes. `requested_fit_queue.py` requires their completed proofs and
the exact reviewed execution bindings. It serializes ninety CUDA fits and one CPU numerical
diagnostic, writes private receipts and stops on failure. The queue and its private job inventory
are not public reproduction records or evidence that unfinished fits completed.

After each scope's fifteen new CUDA fits finish, the audit's `prediction_audit` phase compares
forecasts for untargeted series with the matching full-input controls. The existing reporter
then evaluates the fixed post hoc contrasts, and the existing table selector exports the
eighteen typed result tables. `requested_features_charts.py` draws the paired-loss figure.
`requested_feature_importance.py` reads saved point boosters from the sixty equal-width fits;
`requested_feature_importance_charts.py` draws all five importance figures. Both chart scripts
require text export and review before rendering. Gain remains descriptive training credit.

The requested comparisons use diagnostic fair CRPS as their deciding loss because every
compared forecast has the same 51-value point-model construction. Mean absolute error,
19-midpoint pinball and coverage are secondary diagnostics. Publication uses the scoped table
headings and metric explanation; the generic private report preamble's cross-recipe pinball
statement applies to the original comparisons between different distribution constructions.

The extension is in progress. Actual completed artifacts, source hashes and separate review
records govern reporting and publication. Private path-bearing queue, audit and chart
specifications stay private; reproduction copies must bind the reader's own authorised paths
and freshly generated artifacts. Further Part 2 implementation and experiments are on hold.

## Scores and decisions

The common cross-recipe score is twice mean pinball loss at 19 midpoint probabilities. Delivered
samples use empirical inverse-CDF ranks `ceil(n*q)-1`. A quantile recipe's 459 values are not
exchangeable weather members; its diagnostic fair CRPS must not decide a cross-recipe comparison.
Coverage is percent of cases; normalised losses are percentages of capacity.

Primary estimates weight series equally. Intervals resample whole valid-time months and one paired
fitting seed. The central estimate averages three refits; the bootstrap interval includes variation
from an individual refit. The separately reported equal-month t-interval averages the refits within
each month before estimating month-to-month variation and uses equal month weights. Three
seeds and the retrospective second fold guard against a lucky refit; the second hyperparameter
setting checks robustness. Weekly sampled scores are study diagnostics. A leaderboard claim
requires a separate exact-reference-key submission through the official scorer.

The solar ladder stops at the first three-seed mean gain below 0.15 points. Its continuation rule
does not turn an uncertain gain into a decisive recommendation. Wind additions remain separate
one-column tests. Total cloud cover waits for the specified full post-introduction year.

Part 2 is a separate saved-prediction study under `studies/forecast_calibration/`, with its own
plan and output namespace. It fits no calibrators and cannot pass G5 or G9 by itself.
