# Forecast comparisons after the wind and radiation corrections

This study refits the production XGBoost configuration and the quick-wins winner on matched
inputs, then rebuilds the lean solar and wind feature sets from issue 1156. Production means a
configuration refit, as in the original quick-wins study. The deployed model artifact is outside
this comparison.

The preregistration is `plans/issue-1156-fixed-pipeline-and-lean-features.md`. The review record
is `fixed_pipeline_review_log.md`. Results remain provisional until the scientific reviews of
the results and page are complete.

## Isolated code and frozen inputs

The base is commit `4e820d24c0129c8c95c567ff7355abff8d416144`. The reviewed prerequisite heads are
PR 1167 at `38025f4eab457b9605b8c124006609869dbfca0d` and PR 1166 at
`0f094c356da82e75c4014664ad7fb33fe1df8b7f`. Runtime worktrees apply only their relevant contracts
and feature-engineering package changes to that base. They do not change this study branch's
production packages. `PYTHONPATH` selects the runtime before Python starts; no script changes
the import path.

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

## Execution

Set `DATA_PATH_INTERNAL`, `DATA_PATH_DELIVERY` and `REPO_DATA_DIR` to the local data root.
Set `POLARS_MAX_THREADS=8`, `OMP_NUM_THREADS=8` and `OPENBLAS_NUM_THREADS=8`. Run GPU fitting
jobs serially. Use a new output namespace, separate from the original `quick_wins` outputs.

1. Run `fixed_pipeline_frames.py` with an explicit treatment, fold, code revision, archive path
   and all Delta snapshots. Its manifest records the complete inventory and refuses incompatible
   reuse. A pilot build is not accepted for fitting.
2. Run `fixed_pipeline_fit.py` once per recipe, seed, device, setting and padding policy. It
   verifies the complete scheduled validation inventory, writes private models and forecasts,
   then writes completion hashes. An incomplete fit requires diagnosis and a new output directory.
3. Give `fixed_pipeline_report.py` a JSON specification containing labelled fit paths and planned
   contrasts. Its public aggregate report includes each pinball level and actual feature lists.
4. Run `fixed_pipeline_input_audit.py` on completed, paired frame directories. Its changed-row
   counts, domain screens and timing correlations are input diagnostics, not forecast improvements.
5. Give `fixed_pipeline_charts.py` an explicit report-backed figure specification. Export text
   first with `--text-only`; review that text before rendering with `--reviewed-text`. Inspect the
   PNG rendered from each optimised SVG before copying publication assets into the docs.
6. Use `fixed_pipeline_tables.py` to select and format public aggregate report rows for the page.
   Its JSON specification names exact sources, filters and columns. It performs no estimation,
   refuses existing output directories and records source, specification, script and output hashes.

Raw frame exports, models, row-level scores and manifests containing private identifiers stay
under the private study data directory. Public reports contain type aggregates and feature names.
Never copy private manifests into the documentation.

## Scores and decisions

The common cross-recipe score is twice mean pinball loss at 19 midpoint probabilities. Delivered
samples use empirical inverse-CDF ranks `ceil(n*q)-1`. A quantile recipe's 459 values are not
exchangeable weather members; its diagnostic fair CRPS must not decide a cross-recipe comparison.
Coverage is percent of cases; normalised losses are percentages of capacity.

Primary estimates weight series equally. Intervals resample whole valid-time months and one paired
fitting seed. The separately reported equal-month t-interval targets a different estimand. Three
seeds and the retrospective second fold guard against a lucky refit; the second hyperparameter
setting checks robustness. Weekly sampled scores are study diagnostics. A leaderboard claim
requires a separate exact-reference-key submission through the official scorer.

The solar ladder stops at the first three-seed mean gain below 0.15 points. Its continuation rule
does not turn an uncertain gain into a decisive recommendation. Wind additions remain separate
one-column tests. Total cloud cover waits for the specified full post-introduction year.

Part 2 is a separate saved-prediction study under `studies/forecast_calibration/`, with its own
plan and output namespace. It fits no calibrators and cannot pass G5 or G9 by itself.
