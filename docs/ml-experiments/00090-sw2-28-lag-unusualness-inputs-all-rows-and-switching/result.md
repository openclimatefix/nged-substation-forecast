---
id: "00090"
slug: sw2-28-lag-unusualness-inputs-all-rows-and-switching
status: done
size: simple
hypothesis: >-
  For the 16 Primary substation demand series, telling the monthly-refitted (P-roll) XGBoost model
  how unusual each of its three lags is (the probability integral transform, PIT, of each lag
  against a probabilistic climatology of the series) lowers the quantile score on all half-hours
  outside switching, and widens the quantiles on the half-hours whose lag lies inside a logged
  switching event, when both arms train on normal-arrangement targets only.
prediction: >-
  On all rows outside switching the PIT arm changes the quantile score by -0.05 to +0.01 points of
  capacity (experiment 18's residual lags: -0.04, -0.12 to +0.07, with ERA5 at lag times). On the
  rows outside switching with a lag inside a logged event it changes the score by -0.3 to 0.0 and
  moves the coverage of the 5%-to-95% band from about 0.78 towards 0.90. The switching gain is small
  because masked training holds few rows with an unusual lag and a normal target.
decision_rule: >-
  Two planned contrasts, `clim_pit` minus `plain`, quantile score over all leads, 97.5% interval
  (Bonferroni across two), whole validation months resampled. C1, all rows outside switching
  (the general headline). C2, the rows outside switching whose lag is inside a logged (oracle)
  event (the switching headline). The PIT inputs are `supported` for the general question when
  C1's upper end is below -0.02 points (0.4% of the plain score), `small gain` when C1's upper end
  is below zero but not below -0.02, `refuted` when C1's lower end is above zero, and
  `inconclusive` otherwise. The same words, from C2, answer the switching question. The threshold
  is -0.02 because the inputs add a climatology table and three columns to the serving path, which
  is justified only by a gain whose interval excludes zero. Tail calibration (coverage and
  pinball loss at q01, q05, q95, q99) is descriptive.
benchmark: >-
  The `plain` arm refitted by this experiment's own code, 11 quantile levels, targets inside any
  switching event or closure removed from training.
row_set_id: fold_a_daily_gap_v1
metric: >-
  Quantile score (twice the mass-weighted mean pinball loss over 11 levels), in points of capacity
code: ml-experiments/00090-sw2-28-lag-unusualness-inputs-all-rows-and-switching/
verdict: refuted
best_arm: >-
  `plain`. On the 2,928,842 half-hours outside switching it scored 5.349 points of capacity, and
  `clim_pit` scored 5.703. On the 37,394 of them with a lag inside a logged event, `plain` scored
  10.506 and `clim_pit` 11.301.
headline: >-
  C1, `clim_pit` minus `plain` on all rows outside switching: +0.355 points (97.5% interval +0.209
  to +0.513). C2, the same on rows with a lag inside a logged event: +0.795 (-0.325 to +2.787).
lessons: >-
  Giving the XGBoost model the PIT of each lag against a climatology raised the error at every lead
  time and narrowed the 5%-to-95% band, so the extra inputs added noise and no usable signal of
  "this lag is unusual". The model did not learn to widen its quantiles on rows whose lag lies
  inside a logged event: their coverage fell from 0.770 to 0.740.
advice:
  - >-
    Do not add the PIT of each lag against a per-series climatology to the Primary demand
    inputs: it raised the score by 0.355 (0.209 to 0.513) on all rows outside switching (C1).
  - >-
    Make the model widen its quantiles on suspect lags with an explicit rule outside the booster
    (SW2-27), because lag-unusualness inputs did not teach it to (C2, figure 3).
  - >-
    If a lag-unusualness input is retried, compute the training PITs out of sample (leave the
    lag's own month out of the climatology), so that the training and forecast inputs have the
    same distribution (caveats).
production_status: not-for-production
adopted_in: none
---

# Experiment 90: does telling the model how unusual each lag is improve the forecast?

**Take-home message: telling the XGBoost demand model how unusual each of its lags is made the
forecast worse, not better.** In this experiment the extra inputs added no useful signal beyond what
the lag values and the calendar already give. The three lags are the newest reading, the reading at
the same time yesterday, and the reading at the same time last week. The unusualness of a lag is the
share of that series' ordinary readings, at the same time of day, kind of day, and month, that were
lower than the lag. On the half-hours whose lag lies inside a logged switching event, the extra
inputs did not make the XGBoost model widen its quantiles either. A switching event is a period when
the electricity network operator closes a normally open switch, so that one Primary substation (a
substation that steps down to the local distribution network) carries a neighbour's load. The next
approach to test is an explicit widening rule (SW2-27, a planned experiment). Do not retry the
unusualness inputs in this form. The verdict is **refuted**.

We asked two questions with the same model fits. First, does the unusualness of each lag improve the
forecast on all half-hours outside switching? Second, on the half-hours whose lag lies inside a
logged switching event, does the model learn to give wider quantiles? We refitted the XGBoost demand
model monthly for 16 Primary substations, with and without the three unusualness inputs, and
scored 292 daily forecast runs from July 2025 to June 2026. Both versions train only on half-hours
outside every switching event. The score is the quantile score in points of capacity (smaller is
better), over 11 quantile levels from q01 to q99.

![The two planned contrasts](figure_1.svg)

![The all-row contrast by lead time](figure_2.svg)

![Coverage of the 5%-to-95% band](figure_3.svg)

- **Arms** (seed 0, one refit a month on a rolling 12-month window, 150 rounds, eta 0.15; the cheap
  schedule of [experiment 49](../00049-vp1-screening-reference-under-p-roll-demand/result.md)).
  `plain`: the inputs of [experiment 86](../00086-sw2-12-detector-labels-for-nra-forecast/result.md)
  with raw lags. `clim_pit`: the same plus the three lag PITs (the probability integral transform:
  the share of reference values below the lag, ties counted as half). The reference is the series'
  readings outside switching in the refit's training window, in cells of local month, weekday or
  weekend, and half-hour of day, each pooled with the two half-hours either side.
- **Training rows.** Both arms drop half-hours inside an oracle event, a logged closure, or an
  event of the RO-NC detector (called XL on earlier pages). This is a lab shortcut: these are
  hindsight labels, and the weather inputs are ERA5 at lag times, which a live service would not
  have.
- **Rows scored.** 2,928,842 half-hours outside switching (the general headline). 37,394 of them
  have a lag inside a logged event (the switching headline, 1.3%, spread over 8 of the 12 months).
  The spread factor of 1.15 is applied to both arms, as in the production recipe.
- **Planned contrasts** (`clim_pit` minus `plain`, 97.5% interval, whole months re-drawn). C1, all
  rows outside switching: +0.355 (+0.209 to +0.513), `refuted`. C2, rows with a lag inside a logged
  event: +0.795 (-0.325 to +2.787), `inconclusive`. The `supported` bar was an upper end below -0.02.
- **Absolute scores** (`plain` / `clim_pit`). Outside switching 5.349 / 5.703; with a lag inside a
  logged event 10.506 / 11.301. By lead band on all rows outside switching, the difference runs from
  +0.30 (10 days and beyond, 95% interval +0.14 to +0.46) to +0.41 (2 to 6 days, +0.29 to +0.53).
- **Descriptive contrasts** (95%). All half-hours: +0.334 (+0.203 to +0.471). Rows with a lag
  inside a detector event: +0.097 (+0.002 to +0.243); inside a detector event only: +0.083 (-0.013
  to +0.234).
- **Tail calibration** (`plain` / `clim_pit`; coverage nominal 0.98, 0.90, 0.80; pinball loss in
  points).

| Rows | Coverage q01-q99 | Coverage q05-q95 | Coverage q10-q90 | Pinball q01 | q05 | q95 | q99 |
|---|---|---|---|---|---|---|---|
| Outside switching | 0.985 / 0.985 | 0.888 / 0.874 | 0.789 / 0.759 | 0.298 / 0.300 | 0.918 / 0.969 | 1.172 / 1.203 | 0.390 / 0.391 |
| Lag inside a logged event | 0.971 / 0.972 | 0.770 / 0.740 | 0.668 / 0.612 | 0.482 / 0.491 | 1.494 / 1.530 | 3.629 / 3.573 | 0.665 / 0.678 |

- **What it rules out.** The unusualness inputs did not make the XGBoost model widen its band: on
  rows with a lag inside a logged event the band covers 0.740 instead of 0.770 with the inputs. Only
  the upper tail on those rows improved (pinball at q95 3.573 against 3.629), and the 97.5% interval
  of C2 contains zero.
- **Caveats.** The training PITs are in sample (a lag usually lies inside the climatology window
  that defines its PIT), while forecast-time PITs for lags after the refit are out of sample. A
  cell's pooling also crosses midnight, so the first and last half-hours of a day borrow values from
  the neighbouring day type and month. The in-sample training PITs and the midnight pooling may each
  add to the harm. This experiment tested neither. The result uses one seed and the monthly
  schedule: this experiment did not test a weekly refit. Only the climatology reference was run. The
  lag-free-booster PIT and neighbour arms of SW2-28 need the booster of SW2-27, which does not exist
  yet, and experiment 88's injected cases were not rescored. The curtailment metric was not scored.
- **Open question for the next experiment.** Whether an explicit rule that widens the quantiles
  from the probability that a lag is inside an abnormal arrangement (SW2-26 feeding SW2-27) fixes
  the 0.77 coverage that the inputs tested here did not fix.
