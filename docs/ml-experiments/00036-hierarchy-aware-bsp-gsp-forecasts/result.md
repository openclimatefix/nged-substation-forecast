---
id: "00036"
slug: hierarchy-aware-bsp-gsp-forecasts
status: planned
size: complex
hypothesis: >-
  For the two bulk supply points (BSPs) Boston and Skegness and the two grid supply points (GSPs)
  Walpole and Bicker Fen, forecasts of the series beneath each point, and an N1 model trained on
  ERA5 and applied to ENS members, lower the quantile score below that of N1 trained on ENS.
prediction: >-
  At each BSP, adding the sum of the best-guess median forecasts of the Primary substations and
  metered generators beneath (B1) lowers the quantile score over all leads by 0.5 to 1.5 points of
  capacity, and a regression of the BSP on the same forecasts (B2) lands within 0.5 of B1. At each
  GSP, the ERA5-trained N1 applied to the 51 ENS members (B3) lowers the score by 0.5 to 1.5, with
  the gain largest beyond day 2, and regional wind and solar forecasts as columns (B4) lower it by
  0.2 to 0.8. The oracle that gives B1 the observed values of the series beneath (O) scores below B1
  at both BSPs.
decision_rule: >-
  Six confirmatory contrasts, each on the quantile score over all leads of one series, read at
  99.17% (Bonferroni across six): B1 minus B0 at Boston and at Skegness, B3 minus B0 at Walpole and
  at Bicker Fen, and B4 minus B0 at Walpole and at Bicker Fen. Months and seeds are resampled. A
  family of two contrasts (BSP B1, GSP B3, GSP B4) is supported when both intervals lie wholly below
  zero and both estimates are at most -0.5, because a gain below 0.5 does not justify porting a
  model that reads other models' forecasts; it is refuted when both lower bounds exceed -0.5; it is
  a small gain when both intervals lie wholly below zero but an estimate exceeds -0.5; it is mixed
  when only one series passes; otherwise it is inconclusive. The experiment is supported when all
  three families are supported, partly supported when one or two are, refuted when all three are
  refuted, and inconclusive otherwise. Stop rule, fixed before any fit: if O minus B0 is above -0.5
  at both BSPs, B1 and B2 are not fitted and the BSP family is refuted. Every other contrast is
  exploratory. The four series cannot speak for other BSPs or GSPs.
benchmark: >-
  B0, N1 refitted by this experiment's code on the same rows, checked against experiment 13's
  stored `n1`.
row_set_id: fold_a_every5d_v1
metric: >-
  Quantile score (twice the mass-weighted mean pinball loss over nine levels, the estimate of the
  CRPS), in points of capacity; MAE of the median
advice: []
production_status: not-for-production
adopted_in: none
code: ml-experiments/00036-hierarchy-aware-bsp-gsp-forecasts/
---

# Experiment 36: do forecasts of the series beneath a BSP or GSP lower its error?

**This page is the plan, committed before any code runs.** The results replace the sections below
when the experiment finishes.

Units are points of capacity (1 point is 1% of a series' effective capacity). The rows are the 73
runs of `fold_a_every5d_v1` (one run every 5 days, valid times 2025-07-01 to 2026-06-30) for four
series, each fitted with seeds 0, 1, and 2. No valid time on or after 2026-07-01 is scored.

## Why the question is worth asking

**A bulk supply point (BSP) flow is almost exactly a linear sum of the series beneath it, and N1
does not use that.** On the validation year, a least-squares fit of each BSP on the Primary
substations and metered generators beneath it gives R² 0.994 at Boston and 0.992 at Skegness, with
a residual standard deviation of 1.2 to 1.3 MW against 14 to 15 MW for the BSP itself. The two grid
supply points (GSPs) carry flows from outside the trial area (R² 0.16 at Walpole and 0.74 at Bicker
Fen against their BSP and generators), so a sum of children cannot reproduce them, but the metered
wind and solar generation beneath them is a large share of their variance. Experiment 33's interim
ceiling arms (not yet published) suggest that an N1 trained on ERA5 and fed ERA5 removes about a
fifth of the error at BSPs and GSPs, three times the gain of an ENS-trained N1 given the same ERA5
inputs. This experiment asks how much of that headroom a deployable forecast can capture.

## Series and hierarchy

NGED's hierarchy file (private, never committed) assigns each Primary substation and metered
generator to a BSP and a GSP. The series are:

| Type | Series | Beneath it |
|---|---|---|
| BSP | Boston | 9 Primary substations; solar farms B, D, and F; wind farm W1 |
| BSP | Skegness | 7 Primary substations; solar farms A and E; wind farms W2 and W3 |
| GSP | Walpole | Boston and the series beneath Boston |
| GSP | Bicker Fen | Skegness and the series beneath Skegness |

The 16 Primary substations of the earlier demand experiments are exactly the Primaries beneath these
two BSPs. Solar farm C does not appear in the hierarchy file and is left out. Four series beneath
Boston have no forecast (a biofuel generator, a battery, a landfill-gas solar site, and one other
generator), so their flows stay in the residual. Every generator beneath Skegness has a forecast. Generator
names and identifiers never appear on the published page.

## Arms

All arms are N1 per series, with experiment 13's recipe: a nine-level quantile XGBoost model
(500 rounds, `eta` 0.05, depth 6) trained on the ENS control member of every daily run, valid times
2024-04-01 to 2025-06-30, recency weights with a 2-year half-life anchored at 2025-06-30, the
Primary column set, `lead_hours`, and the three power lags from the forecast origin. The 51 ENS
members are pooled into nine quantiles as in experiment 13. No 14-day correction is applied to a
BSP or a GSP, because experiment 21 found it unproven there.

- **B0:** N1 refitted by this experiment's code. The fit is checked against experiment 13's stored
  `n1`, and the check is printed.
- **B1, bottom-up columns:** B0 plus two columns in MW at the valid time. The first is the sum over
  the Primary substations beneath of the pooled median of experiment 13's `n1` (no correction, no
  spread factor). The second is the sum over the metered generators beneath of the pooled median of
  experiment 20's best-guess recipe (ERA5-trained, with the power-curve margin for wind). The
  spread factor leaves the median unchanged, so it is not applied. At a GSP, B1 holds the
  generator column only, as the proposal for this experiment specifies. A BSP variant **B1g** has
  the generator column alone and is exploratory.
- **B2, bottom-up sum (BSPs only):** a least-squares regression of the observed BSP power on an
  intercept and the median forecast of each child, fitted on the training rows. The quantiles are
  the regression's median plus the empirical residual quantiles of the training rows, taken per lead
  band. Nothing else enters B2.
- **B3, perfect prognosis:** N1 trained on ERA5 at the valid times of the same training rows (same
  window, half-life, lags, and `lead_hours`), applied to each of the 51 ENS members and pooled. Its
  ceiling twin **B3c** feeds the same model ERA5 at the valid time instead of ENS (one input per
  case, so the forecast is the model's nine quantiles). All four series.
- **B4, regional proxies (GSPs only):** B0 plus two columns computed from the GSP's own ENS cell.
  The first is a generic wind power curve (0 below 3 m/s, a cubic ramp to 1 at 12 m/s, 1 to 25 m/s,
  0 above) applied to the ENS 100 m wind speed of each member. The second is experiment 28's
  PV_Live solar share (an ERA5-trained XGBoost model of PV_Live's share of its own capacity, applied
  to the ENS members and averaged), with the additive offset between the control-member forecast
  and PV_Live removed, fitted per GSP and lead band on training-period daylight rows. Experiment 28
  found the member mean 1.8 to 3.0 points of PV_Live capacity too high. Attribution: Sheffield
  Solar PV_Live, CC BY 4.0.
- **O, oracle (BSPs only):** B1 with the observed values of the children at the valid time, and the
  child's forecast median where the observation is missing. The page reports the share of filled
  children per BSP.

A month-by-half-hour climatology row (`ClimatologyForecaster`) is scored for reference, repeated
once per seed.

## How the child columns are built without leakage

**A child's forecast in the training rows must not come from a model that saw those rows.** An
XGBoost model with 500 rounds and depth 6 reproduces its own training rows closely, so an in-sample
column would look far better in training than at validation, and B1 would learn to trust it too much.
The experiment cross-fits the training columns in time instead:

- **Blocks.** The training window is split into five blocks by valid time: 2024-04 to 2024-06,
  2024-07 to 2024-09, 2024-10 to 2024-12, 2025-01 to 2025-03, and 2025-04 to 2025-06. The column
  for a training row in block *k* comes from a child model trained without block *k* and without the
  15 days either side of it.
- **Primary substations.** Each Primary's N1 is refitted five times per seed with a block left out,
  on the same recipe as above (ENS control member, 2-year half-life).
- **Generators.** Each generator's ERA5-trained model of experiment 20 is refitted five times per
  seed with the block left out, and applied to the control-member ENS rows of the block. The
  validation columns use the full models (training to 2025-06-17), pooled over the 51 members.
- **Validation columns** for Primaries are read from experiment 13's stored `n1`. The validation
  columns for generators need ENS rows at the generator cells for the 73 runs, which are not stored
  (the pilot stored Tuesday runs only), so this experiment extracts them from the ENS archive with
  the same upsampling that `demand_build_ens.py` uses, and checks the refitted models against
  experiment 20's stored predictions on the runs the two row sets share.
- **Same child sets everywhere.** B2's coefficients, B1's columns, and the oracle all use the same
  children.

**A training column is a control-member median and a validation column is a 51-member pooled
median.** Their errors differ, because pooling lowers the error of the median. This mismatch is
shared by B1, B2, and the oracle, and the page reports it as a limit.

## Scoring

Quantile score and MAE of the median, per series, over all leads and in the five lead bands (0 to
24 h, 1 to 2 days, 3 to 6 days, 7 to 10 days, 11 to 15 days). Contrasts use
`experiment_store.losses_for_bootstrap` and `studies.bootstrap.bootstrap_difference_at_level`,
resampling the 12 validation months and the three seeds, with `n_resamples` raised for the 99.17%
level. The page reports the absolute score of every arm, per series. Intervals describe these four
series and say nothing about other BSPs or GSPs. B3 minus B0 at the BSPs, B1 minus B0 at the GSPs,
B2 minus B1, O minus B0, and the lead-band contrasts are exploratory.

## Sizing

- **Stored data:** prediction sets and cross-fitted columns under `data/ml-experiments/`; no
  production table changes.
- **Production serving path:** not touched. Porting B1 would add a dependency of the BSP model on
  the Primary and generator models in the Dagster asset graph.
- **Degradation rule:** not touched. A missing child forecast would need a degradation rule at
  porting time, and that rule is outside this experiment.
- **Defensible designs:** more than one (the child columns, the cross-fitting scheme, the regression
  in B2, the filling rule of the oracle). Each is fixed above.
- **Callers:** the scripts copy experiment 13's N1 fit, experiment 20's generator fit, experiment
  28's PV_Live loader and solar model, and experiment 33's ENS extraction. The author could name
  each without searching.

Size: complex. Sonnet writes the plan, Opus reviews it, Sonnet implements, Opus reviews the code,
Sonnet runs the code and writes up, and Sonnet reviews the write-up.

## Earlier advice followed and departed from

- **Followed:** treat BSP and GSP findings as provisional with four series (experiments 6, 21, 24);
  report intervals over validation months (13); keep the 14-day correction off BSP and GSP (21);
  pool ERA5-trained quantiles over ENS members (2); add a climatology row (10); report the absolute
  score of every arm.
- **Departed from:** the experiment adds no calendar-placebo or shuffled-input control. The columns
  are forecasts of real series, and the oracle already bounds what a perfect column gives.

## Past weather

B3c, the ceiling twin, feeds ERA5 at the valid time, and ERA5 arrives about 5 days late, so no
serving system could run it. B3, B1, B2, B4, and every child forecast use ENS weather only. The lag
features use observed power from the forecast origin, as in experiment 13. B4's PV_Live model trains
on ERA5 weather and PV_Live's revised values, both of which exist at training time.

## Limits

- **Four series.** Two BSPs and two GSPs cannot support a release decision alone.
- **Unforecast flows.** Four series beneath Boston are not forecast, and the GSPs carry flows from
  outside the trial area.
- **Cross-fitted training columns are out-of-sample, not out-of-time.** A block's child model sees
  months both before and after it.
