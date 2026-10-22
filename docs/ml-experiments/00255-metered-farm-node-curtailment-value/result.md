---
id: '00255'
slug: metered-farm-node-curtailment-value
title: >-
  Does a better metered solar farm forecast lower the curtailment cost at its bulk supply point?
theme: metered-solar-farms
also_relevant_to:
- curtailment-cost
status: planned
size: >-
  medium. Stored: private joined farm frames and tables under
  data/ml-experiments/00255-metered-farm-node-curtailment-value/ only. Serving path: not touched.
  Degradation rule: not touched. Designs: more than one defensible way to pair a 00 UTC farm
  forecast with a 09 UTC node forecast, so the pairing is fixed in this plan before any scoring.
  Callers: this folder's scripts, copied from experiments 84 and 243, and the unchanged
  studies.node_curtailment. Asset graph: not touched.
hypothesis: >-
  When six metered solar farms beneath two bulk supply points are curtailed by the
  node-plus-generators rule of studies.node_curtailment, a forecast of each farm's output (the
  plant, XGBoost, or the plant with a neural head) avoids more curtailment cost than the NGED
  baseline, which assumes every generator runs at full capacity, and the neural head's lower 0.99
  pinball loss does not show in the metric, because the rule uses only a farm's median.
prediction: >-
  Every forecast arm avoids more cost than the NGED baseline, statistically significant at the 5%
  level, at a limit of 0.5 times each node's earlier maximum export. H5 and XGBoost differ by under
  5% of XGBoost's avoidable cost, with an interval that includes zero. A difference of 10% or more
  would mean the median error at high-export half-hours differs from the all-row median error.
decision_rule: >-
  Avoidable cost (energy proxy at GBP 2,000 per MWh, Curtailment Limit 0) at f = 0.5, from
  month-block resampling of the months of the forecast origin (20,000 draws) at the 99% level
  (Bonferroni over five planned contrasts). C1 to C3: plant, XGBoost, and H5 minus the NGED
  baseline. C4: H5 minus XGBoost. C5: H5 minus the plant. Supported when C1 to C3 are all below zero.
  Porting the neural head in place of XGBoost for curtailment value would be worth its added
  complexity only if C4 is below zero by at least 10% of XGBoost's avoidable cost; the threshold is
  the cost of serving and retraining a neural network beside a tree model. Stop rule: if the scored
  rows hold fewer than 20 event half-hours at f = 0.75 over the two nodes, report the event census
  and f = 0.5 only, and say that the metric cannot rank farm forecasts on this window.
benchmark: >-
  The NGED baseline (capacity_forecast, the maintainer's unconfirmed understanding of NGED's
  practice), the perfect forecast, and experiment 224's two floors
row_set_id: >-
  Experiment 48's node forecast origins (09 UTC, 12 to 36 hour leads) that fall in experiment 243's
  scored year, 2025-09-01 to 2026-06-30
metric: >-
  Avoidable cost in GBP per MW of forecast generator capacity per year (and GBP in total) at
  limits of 0.9, 0.75, and 0.5 times each node's earlier maximum export
code: ml-experiments/00255-metered-farm-node-curtailment-value/
---

# Experiment 255: does a better metered solar farm forecast lower the curtailment cost at its bulk supply point?

**This page is the plan; the write-up replaces it when the experiment finishes.**

## Question

**We ask whether the better farm forecasts of experiments 224 and 243 lower the curtailment cost at
the two bulk supply points (132 kV to 33 kV substations) that the six metered farms sit beneath.**
[Experiment 84](../00084-substation-limit-curtailment-cost-worked-example/result.md) built the
curtailment metric (`studies.node_curtailment`) and scored it on the pilot with experiment 48's
generator forecasts. Experiments 224, 243, and 248 scored the farm forecasts by quantile score and
pinball loss and none scored a curtailment value.

**The rule reads only a farm's median.** The node's own 0.95 quantile sets the planned cut. A
farm's forecast enters through its median `q50` alone, which sets the farm's share of the cut and
the cap on its output. The 29% lower 0.99 pinball loss of H5 against XGBoost in [experiment
243](../00243-metered-farm-architecture-comparison/result.md) therefore cannot show in this metric.
H5's median error is 0.5% lower than XGBoost's, with an interval spanning zero.

## Design

- **Node forecast:** experiment 48's best-guess node arm (`weekly`), held fixed, for the two nodes
  (Boston and Skegness). The three other generators beneath the nodes keep experiment 48's
  best-guess generator forecast (`gen_monthly`).
- **Farm arms:** (1) the NGED baseline; (2) the experiment 224 manual-heuristic and climatology
  floors; (3) the plant with the two-number summary (experiment 224, `a3_two_number`); (4) XGBoost,
  own history (experiment 243); (5) H5, own history, the mean of the five seeds' quantiles, with each
  seed's score as a check; (6) H5, zero history; (7) the perfect forecast. All arms carry seed 0.
- **Units:** the stored farm forecasts are fractions of the farm's effective capacity. They are
  multiplied by the capacity of the generator table, and capped at it, as in experiment 84.
- **Night:** the farm forecasts hold daylight half-hours only (sun more than 5 degrees above the
  horizon). Every other half-hour of a farm gets a forecast of 0 MW in the forecast arms, because a
  missing forecast would count as not instructed. The NGED baseline keeps its capacity forecast at
  all times, as in experiment 84.
- **Pairing of origins (fixed before any scoring).** A node forecast is made at 09 UTC on day D for
  valid times 12 to 36 hours later. A stored farm forecast is the 00 UTC run, at 12 to 36 hours
  lead. For a valid time after 12 UTC on day D+1 the farm forecast comes from the 00 UTC run of day
  D+1, which exists only after the node forecast was made. The headline is therefore **causal**:
  node rows are scored when every daylight farm forecast of the row comes from a run at or before
  the node's origin. The **valid-time pairing**, which gives every node row the farm forecast stored
  for its valid time whatever its run, is a sensitivity analysis, labelled optimistic for the farm
  arms.
- **Missing farm rows:** a daylight half-hour of a farm that is not in experiment 243's common rows
  (a flagged outage, for example) has no forecast in any arm, the baseline included, so every arm
  instructs the same generators. The result's counts of uninstructed generator-half-hours are
  printed.
- **Rows and scoring:** origins of experiment 48 from 2025-09-01 to 2026-06-30, leads 12 to 36 hours,
  Curtailment Limit 0, scaled limits at f = 0.9, 0.75, and 0.5. Headline limit f = 0.9, with the
  prediction's contrasts at f = 0.5 because the pilot holds too few events at 0.9 to rank arms.
- **Intervals:** month-block resampling of the months of the forecast origin.
- **Also written:** the joined farm frames with `q01`, `q05` to `q95`, `q99`, `actual`, `lead_hours`,
  and `time_series_id` (farm label), scored with `studies.tail_calibration` at 0.01, 0.05, 0.95, and
  0.99 on the scored rows.
- **Anonymisation:** farms A to F only. The cost is split by node only as a sum over generators and
  never per farm. Every generator figure is a sum over at least 4 generators.

## Weather and forecast time

- **Weather:** the ECMWF ensemble 00 UTC run at the nearest 0.25 degree cell, available at forecast
  time. Lagged farm power one or two days earlier, available at forecast time.
- **Schedule:** the screening schedule of experiment 243, one fit on the months before 2025-09. No
  fit is run here; every farm forecast is read from storage.

## Earlier advice followed

- Experiment 84: score at a scaled limit, label every cost "calibration to history, not a saving",
  and show the published thermal limit only in the event census.
- Experiment 243: read H5 and XGBoost from their stored predictions on the rows that every arm holds.
