---
id: "00288"
slug: pooled-head-on-public-wind-farms
title: >-
  Does an 11-quantile correction head trained on dozens of public Great Britain wind farms lower the
  quantile score of the forecasts for the three metered wind farms, and does more history help the
  XGBoost model more than the physics model?
theme: generator-forecasting
also_relevant_to:
  - calibration-and-uncertainty
  - curtailment-cost
status: planned
date: 2026-10-23
size: >-
  complex. Stored: private farm tables, ERA5 and ENS extracts at the farms' cells, trained weights,
  and predictions under data/ml-experiments/00288-pooled-head-on-public-wind-farms/ and
  data/studies/downloads/market_wind00288/ only. Serving path: not touched (the experiment decides
  whether a later port is worth proposing). Degradation rule: not touched. Designs: several are
  defensible (which farms, how to drop curtailed half-hours, which base model the head corrects,
  how folds avoid sharing weather), so this plan fixes one before any fit. Callers: this folder's
  scripts, which copy experiment 259's, and `studies.*` helpers only. One plan review, one code
  review, one write-up review, and one prose review, each by a fresh Sonnet sub-agent; an audit
  runs if a trigger fires.
hypothesis: >-
  A neural head that adds 11 quantile offsets to the XGBoost wind forecast (R0) and was trained on
  dozens of public onshore wind farms (metered output from Elexon's Balancing Mechanism Reporting
  Service) lowers the quantile score of the three metered wind farms (W1 to W3) against R0 alone,
  and a head trained on the pool beats the same head trained on W1 to W3 only, because experiment
  259's head had too few farm-years to learn a correction that transfers. A second hypothesis:
  XGBoost keeps improving as the training history grows from 12 to 48 months while the physics
  core is flat after 24 months, and a pool of public farms replaces the missing history.
prediction: >-
  Written before any fit. C1, P2 minus P1 on W1 to W3, kept rows, lead days 1 and 2: at most -0.10
  points of capacity (pooling repairs the small-sample failure). C2, P3 minus P0: inside -0.30 to
  0.00 points, with P3's 0.99 pinball loss at least 5% lower than P0's. C3, P2 minus X2 (the
  non-neural pooled arm): at least 0.03 points lower, which is the threshold for keeping the
  network. Blind share, (BL2 minus P0) divided by (P2 minus P0): under 0.5. Held-out farms: a head
  scored on a held-out fifth of the public farms scores within 5% of the same farms scored by a head
  that trained on them. Learning curve: R0's quantile score on W1 to W3 falls by at least 0.15
  points from 24 to 48 months of history, the physics core's changes by less than 0.05, and a
  base XGBoost model trained on the public farms alone scores within 0.30 points of R0 trained on
  12 months of the farm's own history.
decision_rule: >-
  Three decisive contrasts, C1 to C3, each paired per case on the quantile score on W1 to W3's
  kept rows at lead days 1 and 2 (rows kept by experiment 210's mask), read at 98.33% (Bonferroni
  across three), resampling whole scored months and the three seeds, 50,000 resamples. The
  threshold for porting the pooled head is C2 wholly below zero, because the head adds a trained
  network and an Elexon feed to the training pipeline (not the serving path). The tail score T (mean
  of the 0.01 and 0.99 pinball losses) is reported beside the headline and is not a decisive test.
  C3 decides whether the network is kept over the non-neural stage; a C3 interval that does not lie
  wholly below -0.03 means the non-neural stage is preferred. Learning curves, the blind share, the
  held-out-farm check, all leads, and everything per farm are descriptive. The intervals describe
  these three farms.
benchmark: >-
  P0, the XGBoost recipe of experiment 230's A0r (R0) fitted on ERA5 before 2024-04-01 and run on
  51 ensemble members with the spread factor. Experiment 259's stored R0 rows serve W1 to W3.
  Reference rows: month-by-half-hour climatology and experiment 230's stored A0r.
row_set_id: >-
  experiment 33's wind cases (`fold_a_tuesdays_v1`: 53 Tuesday runs from 2025-07-01 to 2026-06-23,
  W1 to W3). Headline on the rows kept by experiment 210's mask; all rows descriptive.
metric: >-
  Quantile score (twice the mass-weighted mean pinball loss over nine levels), in points of
  capacity; pinball loss at 0.01, 0.05, 0.5, 0.95, and 0.99; the tail score T; coverage of the 5% to
  95% and 1% to 99% bands; by lead band
production_status: not-for-production
adopted_in: none
code: ml-experiments/00288-pooled-head-on-public-wind-farms/
---

## Plan: a correction head trained on public wind farms (planned, no result yet)

**This page is the plan and is committed before any model is fitted.** Experiment 288 follows
[experiment 259](../00259-wind-plant-plus-quantile-offset-head/result.md), whose head lowered the
wind quantile score by about 0.5 points of capacity on three metered farms but learned from only
15 months of one farm each. Elexon publishes half-hourly metered output for every Balancing
Mechanism Unit (BMU), and public BMU data may be named, so the experiment gathers a pool of public
onshore wind farms and asks whether a head trained on the pool transfers to W1 to W3.

## Data stage (already run)

- **Farm candidates:** 287 BMUs have register fuel type WIND. The 65 that match an operational
  onshore row of the Renewable Energy Planning Database (REPD) by name (similarity at least 0.85)
  and lie inside the ENS table are the candidates (`e00288_farms.py`).
- **Output:** Elexon dataset B1610, half-hourly, 2019-09-01 to 2026-06-30
  (`e00288_fetch_b1610.py`, validated by `e00288_validate_b1610.py`). Curtailment: Elexon bid-offer
  acceptances (BOALF) and final physical notifications (PN) for the same BMUs, fetched with
  `studies/market_downloads`. Contains BMRS data © Elexon Limited copyright and database right 2026.
- **Weather:** ERA5 wind at 10 m and 100 m at each farm's nearest 0.25 degree cell, hourly,
  2019-09-01 to 2024-03-31, read from Google's ARCO-ERA5 store (`e00288_fetch_era5.py`; no cost). ENS
  members at the same cells from the ENS Delta table.

## Design fixed before any fit

- **Base model (R0):** experiment 230's A0r recipe fitted per farm on ERA5 before 2024-04-01, run on
  the 51 members, mixed, and spread-adjusted as in experiment 259. The brief allowed R0 or R1; R0 is
  used because experiment 259 stored R0 for W1 to W3 and R0 fits quickly for dozens of farms.
- **Farm selection:** keep a BMU if, after the curtailment drop, it has at least 2 years of clean
  output before 2024-04-01 and at least 12 months of clean output after, a share of exact zeros
  under 0.3, and a mean capacity factor between 0.10 and 0.50. Capacity is the 99.5th percentile of
  the farm's clean output. Negative readings (station load) are set to zero.
- **Curtailment drop:** drop a half-hour that overlaps any BOALF acceptance for the BMU, and a
  half-hour whose PN exceeds the metered output by more than 5% of capacity. The share dropped is
  reported per farm.
- **Head (H):** experiment 259's H5c network with the same 11 base quantiles, 7 member summaries and
  lead; the one-hot farm columns are replaced by the farm's capacity (log MW), turbine rating, and
  hub height. Hub height is REPD's where present and otherwise predicted from turbine rating by a
  linear fit over the REPD rows that have both; W1 to W3 take the pool's median hub height because
  their real values are private.
- **Arms:** P0 (R0 alone), P1 (H trained on W1 to W3), P2 (H trained on the public farms, zero
  shot), P3 (P2 fine-tuned on W1 to W3's training runs, at most 3 epochs at a tenth of the learning
  rate), X2 (non-neural XGBoost stage, `colsample_bytree` 1, trained on the pool), BL2 (pooled head
  with lead, capacity, hub height and calendar columns only).
- **Folds inside the pool:** five spatial blocks (k-means of farm cell positions), because farms in
  one ERA5 cell see identical weather and would leak across a random split. Each fold's farms are
  scored by a head trained on the other four blocks and compared with the same farms scored by the
  all-farm head.
- **Learning curve and core-only arm (manager's note, 2026-10-23):** R0 fitted per farm on the last
  12, 24, 36, and 48 months before 2024-04-01, scored on W1 to W3 and on the public farms; the
  physics core (experiment 230's B1, with standard air density at the public farms) on the same
  windows for W1 to W3; and an XGBoost base model trained on the public farms alone, applied to W1
  to W3.
- **Seeds and rows:** three seeds on W1 to W3 (R0 seeds 0 to 2 and head seeds 0 to 2); five head
  seeds in the pool. Training runs start 2024-04-01 to 2025-04-15, early stopping uses runs from
  2025-04-16 to 2025-06-16, and no valid time on or after 2025-06-17 enters training.
  The public farms' scored rows are the same 53 Tuesday runs, at the ensemble's native 3-hourly and
  6-hourly valid times (descriptive).

## Stop rules

- Fewer than 15 farms with at least 2 years of clean output: stop and report.
- The base model scores worse than climatology at lead days 1 and 2 on more than a quarter of the
  farms: stop and find the cause before training a head.

## Rules

W1 to W3 stay anonymous and keep experiment 210's mask private; public BMU farms may be named. No
change under `src/` or `packages/`.
