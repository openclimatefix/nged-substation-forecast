# Row sets and scoring

**Two scores are comparable only when they cover the same rows, the same series, and the same
quantile levels.** This page lists the named row sets, the score and its intervals, the tail and
curtailment reporting that every forecasting experiment carries, and the retraining defaults. The
procedure is in the [`ml-experiment` skill](https://github.com/openclimatefix/nged-substation-forecast/blob/main/.claude/skills/ml-experiment/SKILL.md),
and terms are in the [glossary](../glossary.md).

## Series counts

- **The trial area has 32 series with data:** 16 Primaries, 6 solar farms, 3 wind farms, 2 GSPs,
  2 BSPs, and 3 other generators or storage units ([NGED power
  telemetry](nged-power-telemetry.md#the-series)).
- **The fold-A demand set has 22 demand-side series (D1 to D22):** 16 Primaries, 4 BSPs or GSPs,
  and a battery and a biofuel generator ([glossary](../glossary.md#d-codes)).
- **The generator experiments score 9 series** (6 solar farms and 3 wind farms).
- **The differentiable-physics (DP) series uses five long-history MW Primaries as its decision set**,
  because only five Primaries hold long histories ([NGED power
  telemetry](nged-power-telemetry.md#gaps-ramp-up-and-regime-changes)).

## Row sets

- **A row set is a named set of forecast origins, series, and valid times**, and
  `experiment_store.read_predictions` raises when a prediction set covers a different row set.
  State the `row_set_id` in the plan, and read the benchmark from the same row set.
- **No experiment scores a valid time on or after 2026-07-01.** That period is kept for a finalist.

| `row_set_id` | Rows | Defined in |
|---|---|---|
| `fold_a_tuesdays_v1` | 53 Tuesday ENS runs and the 9 generator series; every case of the first day sits on a Tuesday or Wednesday, which does not suit demand | Experiments 1 and 2 |
| `fold_a_every5d_v1` | Every fifth day at 00:00 UTC from 2025-07-01 (73 runs); each lead band has 14% of its cases on each weekday | Experiment 3 |
| `fold_a_daily_gap_v1` | The 292 daily 00:00 UTC runs from 2025-07-01 to 2026-06-30 that `fold_a_every5d_v1` does not hold | Experiment 48 |
| `control_fold_b_daily_v1` | 274 runs, 2023-07-01 to 2024-03-31, 9 km grid | Experiment 81 |
| `era5_val_*` | ERA5-based validation years (for example `era5_val_3folds_2023_07_to_2026_06_v1`) | Experiments 6 and 11 |
| `dp_mw_v1` | 508,341 half-hours at the MW decision series, after the start cuts, the fixed lockdown window, and the switching mask | Experiment 103 |
| `dp_mva_dark_v1` | 352,841 dark half-hours at the MVA-only Primaries | Experiment 103 |
| `dp_christmas_daily_v1`, `age1_daily_2025_07_to_2026_06_v1` | Christmas-period origins; the old-year training check | Experiments 173 and 93 |

- **Read the defining experiment before reusing a row set.**
- **The DP series uses season-years from July to June.** Tuning is 2023-24, the replicate year is
  2024-25, and the scored year is 2025-26 (`FIRST_SEASON_MONTH` in `studies.dp_calendar`;
  experiment 109).
- **Count eligible long-history sites at each tuning year's first day.** Five Primaries qualify at
  2024-07-01 (experiment 103).
- **Choose a row set that covers every weekday**, and give each experiment a climatology row
  (experiment 3).

## The score

- **The score is the quantile score in points of capacity.** It is twice the probability-mass
  weighted mean of the pinball losses over the quantile levels, and a lower score is better. One
  point is 1% of the series' capacity ([glossary](../glossary.md#capacity-and-points-of-capacity)).
- **The quantile levels changed.** Experiments 33, 34, and 48 use nine levels (0.05 to 0.95).
  Experiment 87 and later use 11 levels (those plus 0.01 and 0.99). Scores with different levels
  cannot be compared ([glossary](../glossary.md#quantile-score-and-pinball-loss)).
- **Skill is 100 times (1 minus the best guess's score over `manual_heuristic`'s score)**, pooled
  over the series being compared ([glossary](../glossary.md#skill)).
- **The baselines are `manual_heuristic` and `climatology`** (`baseline_forecasters`).
  `manual_heuristic` takes the same weekday and time in the last 6 weeks and 49 to 55 weeks earlier,
  and uses no weather. `xgboost_baseline` is the stored production-style forecast; none of the
  experiments we read confirms that it matches the deployed one ([glossary](../glossary.md#xgboost_baseline)).
- **The sample resolves gains of roughly 0.3 points of capacity or more**, on 16 Primaries,
  9 generator series, and one validation year ([experiments README](../README.md)).

## Intervals

- **Intervals resample whole calendar months (month blocks), and their level follows the number of
  planned contrasts.** One contrast gives a 95% interval. N contrasts give a Bonferroni-corrected
  level, such as 99.44% for nine ([glossary](../glossary.md#interval-level-convention)).
- **Compute them with `studies.bootstrap.bootstrap_difference_at_level`** on the loss frame from
  `experiment_store.losses_for_bootstrap`, which refuses arms with different seeds.
- **A contrast counts as a gain when the interval over repeated seeds excludes zero.** Use the
  scored year and report the replicate year beside it where the experiment has one.
- **Half-hours within a day, and members within a run, are strongly correlated**, so a row
  bootstrap understates the interval (experiment 111).

## Tails and curtailment

- **Every forecasting experiment reports tail calibration beside the headline score.** The table
  gives coverage and pinball loss at 0.01, 0.05, 0.95, and 0.99, through `studies.tail_calibration`.
- **Write `q01` to `q99`, `actual`, `lead_hours`, and `time_series_id` in the predictions**, so the
  tail and curtailment metrics can score them.
- **The curtailment metric is `studies.node_curtailment`** (node plus its generators). The headline
  is the scaled limit: L is f times the node's maximum observed export before the scored year, with
  f = 0.9 by default and 0.75 and 0.5 as sweeps. The real published thermal limit appears only as a
  reference mark and in the event census, because NGED curtails so that one more fault still leaves
  headroom. Metric details: [cost-savings metrics](../../roadmap/cost-savings-metrics.md#metric-2-curtailment-cost).
- **Report the manual-heuristic baseline arm (`capacity_forecast`) as the maintainer's unconfirmed
  understanding of NGED's practice.** Experiment 84 is the pattern to copy.
- **The scored lead band for curtailment is 12 to 36 hours.**

## Initialisation and lags

- **Experiment 106 initialises at 09:00 UTC**, and treats a power reading as reaching the
  forecast 3.5 hours after its half-hour (an assumption; see [NGED power
  telemetry](nged-power-telemetry.md#timestamps)).
- **Power lag features shorter than the lead time are nullified** to avoid lookahead bias
  (`_nullify_leaky_lags` in `ml_core`).

## Retraining defaults in R&D

- **Screening a contrast of 0.05 points or more: refit monthly on a rolling 12-month window, 150
  rounds, eta 0.15 ("P-roll").** It saves 90% to 97% of compute (experiment 85).
- **Conclusions about cadence or adaptation, and any absolute or tail score: refit weekly on the
  whole history at 150 rounds.** Reserve weekly refits at production settings for best-guess runs
  and production.
- **Do not use quarterly refits** (0.38 points worse than weekly) **or warm starts** (failed in
  experiment 34).
- **Say in each write-up which schedule was used and whether the contrast could depend on it.**

## Audit triggers

- **A result that fails a control, contradicts its own prediction, or matches one of the other
  triggers gets an audit before merging.** The seven triggers are in the `ml-experiment` skill,
  section "Audit before merge".

Last checked against experiment 183.
