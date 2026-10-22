# Row sets and scoring

**Two scores are comparable only when they cover the same rows, the same series, and the same
quantile levels.** This page lists the named row sets, the score and its intervals, the tail and
curtailment reporting that every forecasting experiment carries, and the retraining defaults. The
procedure is in the [`ml-experiment` skill](https://github.com/openclimatefix/nged-substation-forecast/blob/main/.claude/skills/ml-experiment/SKILL.md),
and terms are in the [glossary](../glossary.md).

## Series sets

- **The fold-A demand set has 22 demand-side series (D1 to D22):** 16 Primaries, 4 BSPs or GSPs,
  and a battery and a biofuel generator ([glossary](../glossary.md#d-codes)) (exp 3; fold A;
  established).
- **The generator experiments score 9 series** (6 solar farms and 3 wind farms) (exp 1; fold A;
  established).
- **The differentiable-physics (DP) series uses five long-history MW Primaries as its decision
  set** ([NGED power telemetry](nged-power-telemetry.md#gaps-ramp-up-and-regime-changes)) (exp 103;
  5 Primaries; established).

## Row sets

- **A row set is a named set of forecast origins, series, and valid times**, and
  `experiment_store.read_predictions` raises when a prediction set covers a different row set.
  State the `row_set_id` in the plan, and read the benchmark from the same row set (code).
- **No experiment scores a valid time on or after 2026-07-01.** The `ml-experiment` skill
  owns the rule and the reason (code).

| `row_set_id` | Rows | Defined in |
|---|---|---|
| `fold_a_tuesdays_v1` | 53 Tuesday ENS runs and the 9 generator series; every case of the first day sits on a Tuesday or Wednesday, so the set does not suit demand | Experiments 1 and 2 |
| `fold_a_every5d_v1` | Every fifth day at 00:00 UTC from 2025-07-01 (73 runs); each lead band has 14% of its cases on each weekday | Experiment 3 |
| `fold_a_daily_gap_v1` | The 292 daily 00:00 UTC runs from 2025-07-01 to 2026-06-30 that `fold_a_every5d_v1` does not hold | Experiment 48 |
| `control_fold_b_daily_v1` | 274 runs, 2023-07-01 to 2024-03-31, 9 km grid | Experiment 81 |
| `era5_val_*` | ERA5-based validation years (for example `era5_val_3folds_2023_07_to_2026_06_v1`) | Experiments 6 and 11 |
| `dp_mw_v1` | 508,341 half-hours at the MW decision series, after the start cuts, the fixed lockdown window, and the switching mask | Experiment 103 |
| `dp_mva_dark_v1` | 352,841 dark half-hours at the MVA-only Primaries | Experiment 103 |
| `dp_christmas_daily_v1`, `age1_daily_2025_07_to_2026_06_v1` | Christmas-period origins; the old-year training check | Experiments 173 and 93 |

- **Read the defining experiment before reusing a row set** (code).
- **The DP series uses season-years from July to June.** Tuning is 2023-24, the replicate year is
  2024-25, and the scored year is 2025-26 (`FIRST_SEASON_MONTH` in `studies.dp_calendar`;
  exp 109; DP series; established).
- **Count eligible long-history sites at each tuning year's first day.** Five Primaries qualify at
  2024-07-01 (exp 103; 5 Primaries; established).
- **Choose a row set that covers every weekday**, and give each experiment a climatology row
  (exp 3; fold A; provisional).

## The score and its intervals

- **The score, skill, baselines, capacity, and interval level are defined in the glossary.** See
  [quantile score](../glossary.md#quantile-score-and-pinball-loss),
  [skill](../glossary.md#skill), [capacity](../glossary.md#capacity-and-points-of-capacity),
  [interval-level convention](../glossary.md#interval-level-convention), and
  [rolling refit](../glossary.md#refit-and-rolling-refit). The procedure is in the `ml-experiment`
  skill (code).
- **Experiments 33, 34, and 48 use 9 quantile levels (0.05 to 0.95).** Experiment 87 and later use
  11 levels (those plus 0.01 and 0.99). Scores with different levels cannot be compared (exp 33,
  48, 87; all experiments; established).
- **The sample resolves gains of roughly 0.3 points of capacity or more**, on 16 Primaries,
  9 generator series, and one validation year ([experiments README](../README.md)) (exp 3; 16
  Primaries, 9 generators, one year; provisional).
- **Half-hours within a day, and members within a run, are strongly correlated**, so a row
  bootstrap understates the interval (code: `studies.bootstrap` resamples whole months).

## Tails and curtailment

- **Every forecasting experiment reports tail calibration beside the headline score.** The tail table
  gives coverage and pinball loss at 0.01, 0.05, 0.95, and 0.99, through
  `studies.tail_calibration` (code).
- **Write `q01` to `q99`, `actual`, `lead_hours`, and `time_series_id` in the predictions**, so the
  tail and curtailment metrics can score them (code).
- **The curtailment metric is `studies.node_curtailment`** (node plus its generators). The headline
  limit is a fraction f of the node's maximum observed export before the scored year, with f = 0.9
  by default and 0.75 and 0.5 as sweeps. The real published thermal limit appears only as a
  reference mark and in the event census, because NGED curtails so that one more fault still leaves
  headroom. Metric details: [cost-savings metrics](../../roadmap/cost-savings-metrics.md#metric-2-curtailment-cost)
  (code).
- **Report the manual-heuristic baseline arm (`capacity_forecast`) as the maintainer's unconfirmed
  understanding of NGED's practice.** Experiment 84 is the pattern to copy (exp 84; 2 bulk supply
  points; provisional).
- **The scored lead band for curtailment is 12 to 36 hours**, because scoring needs a stored forecast
  in that window ([curtailment theme](../themes/curtailment-cost/index.md); exp 84; curtailment;
  provisional).

## Retraining defaults in R&D

- **Do not use quarterly refits, three refits a year, or warm-start boosting** as a proxy for monthly
  retraining (exp 34, 85; XGBoost refit cadence, Primary demand; established).
- **Screen a demand-feature contrast with monthly refits on a rolling 12-month window, and judge
  cadence, absolute scores, and tail scores with weekly refits.** The
  [rolling refit](../glossary.md#refit-and-rolling-refit) glossary entry gives the settings (exp 34,
  48, 81; Primary demand; established).
- **Say in each write-up which schedule was used and whether the contrast could depend on it**
  (exp 34, 85; refit schedules; established).

## How to score and gate an experiment

- **Run a known-answer gate before reading any contrast.** Inject a known step or shift into real
  residuals, or use a fake-ensemble gate, so that a null cannot be a wiring fault. Experiments
  128, 142, and 172 narrowly missed or failed their gates, so their contrasts are inconclusive,
  not refuted (exp 111, 128, 142, 172, 175; DP series; established).
- **State the row definition and the lead range before any score.** A "sunny" gap was 0.05 MW over
  six substations, but 63% of it came from D5, and it shrank to 0.015 MW under a different
  definition of sunny (exp 178; 6 substations; provisional).
- **Use month-block intervals, and report D5 separately where one substation dominates** (exp
  111, 178, 180; DP Primaries; provisional).
- **Judge calibration where the loss sits: cold nights and daylight bands, not the all-row
  average.** Cold-night q95 coverage was 0.846 under experiment 160's smooth spread table and 0.896
  with widening of the upper quantiles only, while ensemble spread added nothing over all rows (exp
  160, 164, 169; DP Primaries; provisional).
- **Score a new model component on top of the best-guess arm, not on a partial stack.** The drift
  term improved the score by 0.40 on the bare GAM-K and worsened arm D by 0.03 once the filter, the
  winter night offset, and recipe R were in (exp 167, 170; DP Primaries; provisional).

Last checked against experiment 183.
