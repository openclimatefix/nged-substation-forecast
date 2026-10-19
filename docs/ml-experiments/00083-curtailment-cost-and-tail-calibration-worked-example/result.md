---
id: "00083"
slug: curtailment-cost-and-tail-calibration-worked-example
theme: curtailment-cost
also_relevant_to: []
status: done
size: simple
hypothesis: >-
  A tail-calibration table, built in `packages/studies`, can rank the stored quantile forecasts of
  experiment 48 by how well they cover the tails of the forecast distribution. This is a worked
  example of new code, not a test of a forecasting idea.
prediction: >-
  None was registered before the run. Every number below is exploratory.
decision_rule: >-
  None. The example shows what the tail table reports and where it cannot be used.
benchmark: >-
  The best-guess stack (experiment 48's `frozen_cs`), `manual_heuristic`, and `climatology`.
row_set_id: fold_a_daily_gap_v1
metric: >-
  Coverage and pinball loss at the 0.05 and 0.95 quantiles.
code: ml-experiments/00083-curtailment-cost-and-tail-calibration-worked-example/
verdict: inconclusive
verdict_note: >-
  A worked example with no pre-registered contrast, so no verdict on the forecasts is drawn. The
  single-series curtailment-cost half of this experiment was withdrawn, because curtailment binds at
  a substation and not at a single meter; the tail-calibration result stands.
best_arm: >-
  Weekly retrained XGBoost has the lowest pinball loss at both the 0.05 and the 0.95 quantile.
headline: >-
  Every forecast is too narrow in both tails: coverage of the 0.05 quantile is 0.066 to 0.085 and of
  the 0.95 quantile is 0.891 to 0.927.
lessons: >-
  The stored sets stop at the 0.95 quantile, so the 0.99 tail is out of reach; later experiments
  should write the 0.01 and 0.99 quantiles.
production_status: not-for-production
adopted_in: none
advice:
  - >-
    Write the 0.01 and 0.99 quantile columns in every forecasting experiment's prediction set, so
    that the tail table can be read at those levels (this page, section D).
  - >-
    Score tails with `studies.tail_calibration`.
---

# Experiment 83: tail calibration, on experiment 48

**Every stored forecast of experiment 48 is too narrow in both tails, and weekly retraining has the
lowest pinball loss at both levels.** We built a tail-calibration table for the studies code and ran
it on the stored demand forecasts of experiment 48. The stack is the demand model fitted once, with
a correction from its own recent errors. This experiment first also built a single-series
curtailment-cost metric. We withdrew that metric, because curtailment binds at a substation, so a
metric that scores each meter alone against its own history does not measure it.

## Results (exploratory, 20 demand series pooled, every lead and run from 2025-07-02)

- **Coverage:** coverage of the 0.05 quantile is 0.066 to 0.085 and of the 0.95 quantile is 0.891 to
  0.927, so every forecast is too narrow in both tails. The target coverage is the level itself.
- **Pinball loss:** weekly retraining has the lowest pinball loss at both levels.
- **Levels not delivered:** the 0.01 and 0.99 rows read "not delivered", because the stored sets
  hold nine quantiles from 0.05 to 0.95.

## Caveats and next steps

- **Not scored:** experiment 64's stored sets carry no `actual` column, so they were not scored here.
- **Next:** write the 0.01 and 0.99 quantiles in new experiments.

Reproduce with `uv run python ml-experiments/00083-*/e00083_report.py`. The printed report is in
`data/ml-experiments/00083-curtailment-cost-and-tail-calibration-worked-example/report.md`.
