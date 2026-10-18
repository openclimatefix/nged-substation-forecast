---
id: "00085"
slug: cheaper-retraining-schedules-rank-arms-like-weekly-demand
status: done
size: medium
hypothesis: >-
  For the 16 Primary substation demand series, a cheaper retraining schedule ranks two pairs of
  arms the way full weekly retraining ranks them: the cadence pair (monthly minus weekly
  retraining) and the starting-value pair (a zero starting value minus XGBoost's estimated
  starting value, the contrast of experiment 80). The cheaper schedules are (a) a monthly refit on
  a rolling 12-month window with 150 rounds and `eta` 0.15 (P-roll), (b) the same refit every
  quarter, (c) a monthly refit at the full setting (whole history, 500 rounds, `eta` 0.05), and
  (d) a weekly refit on the whole history with 150 rounds and `eta` 0.15. A schedule that passes
  can replace weekly retraining in research experiments.
prediction: >-
  Full weekly retraining gives a cadence gap (monthly minus weekly) of about +0.13 points of
  capacity (monthly 5.57 against weekly 5.44 in experiment 48) and a starting-value gap (zero
  start minus estimated start) of about -0.07 (experiment 78 and the pilot of experiment 80). The
  starting-value gap comes from the first rounds of a fit, not from the cadence, so P-roll
  monthly, P-roll quarterly, and the full monthly refit reproduce its sign and lie within 0.05 of
  it. P-roll quarterly scores 0.10 to 0.20 above P-roll monthly. The weekly refit with 150 rounds
  and `eta` 0.15 scores within 0.05 of full weekly retraining, and passes both pairs. The cadence
  gap under P-roll is within 0.05 of the full-setting gap.
decision_rule: >-
  Every contrast is computed on one run set: the daily-gap runs that experiment 80's stored
  zero-start weekly arm serves (28 weekly anchors: 2025-07-01, 07-08, 07-15, then every 14 days to
  2026-06-23). A run is in the set when its latest weekly refit, at or before its start, is one
  of those 28 anchors. Every weekly arm of this experiment is fitted at exactly those anchors, and
  the stored
  arms (`w`, `m`, `pm`) are restricted to the same runs. Two reference contrasts come from stored
  predictions: R1 is full monthly minus full weekly (experiment 48's `n1h` monthly and weekly
  regimes), and R2 is the zero start minus the estimated start at full weekly retraining
  (experiment 80's `n1h_m0` minus experiment 48's weekly). The schedule contrasts are K1(S), the
  schedule's monthly arm minus its weekly arm, and K2(S), the schedule's zero-start arm minus its
  estimated-start arm. K1 is scored for P-roll (`pm` minus `pw`) and for the whole-history
  150-round recipe (`fm` minus `fw`). K1 of quarterly (`pq` minus `pw`) is descriptive, because no
  full-quarterly reference exists on these rows. K2 is scored for P-roll monthly, quarterly, and
  weekly, the full monthly refit (`m0` minus `m`), and the whole-history 150-round weekly refit.
  The quantile score over all leads (spread factor 1.15, no correction) is read per contrast with a
  95% interval from 20,000 resamples of whole origin months, paired on series, run, and valid
  time. GPU fits are not bit-reproducible, so the noise floor N is the largest of 0.01 points, the
  absolute difference in mean score between `pm` and its repeat `pm_rep`, the same for `pm0` and
  `pm0_rep`, and the absolute difference between the two K2(P-roll monthly) values; the page also
  prints how far `pm_rep` is from the stored `pm`. The tolerance is T = max(0.05, N). The
  rank-agreement rule, fixed here: a schedule agrees on a contrast when (i) the reference
  interval excludes zero and the reference size exceeds N, and the schedule's estimate has the
  same sign, its own interval excludes zero, and it lies within T of the reference estimate; or
  (ii) otherwise, the schedule's estimate lies within T of zero. A schedule passes when it agrees
  on every contrast it is scored on, and the page states how many that is. The whole-history
  weekly refit's absolute score against full weekly is reported and does not affect the pass.
  Cost is the mean wall-clock fit seconds of one refit under the schedule's own recipe, measured
  in this experiment's own runs (full weekly uses `m0`'s mean, the same full setting), times the
  schedule's refits in a year (52 weekly, 12 monthly, 4 quarterly) times 16 series, in compute
  minutes. The recommendation: a schedule that passes and costs under a third of full weekly
  becomes the R&D default for the contrasts it passed on; a schedule that fails a contrast is not
  used for that kind of contrast. The threshold is zero in score because the experiment ranks
  schedules and ports nothing to production; the pass rule above is the whole decision.
benchmark: >-
  Experiment 48's stored full weekly, full monthly, and P-roll monthly (`roll12`) predictions and
  experiment 80's stored zero-start weekly predictions, plus `manual_heuristic` and `climatology`
  as stored.
row_set_id: fold_a_daily_gap_v1
metric: >-
  Quantile score (twice the mass-weighted mean pinball loss over nine levels), in points of capacity
  (1 point is 1% of the series' effective capacity); MAE of the median, coverage of the 0.01 to
  0.99 quantile band, and pinball loss at the 0.01, 0.05, 0.95, and 0.99 quantiles are reported
verdict: partly supported
best_arm: >-
  Weekly refits on the whole history with 150 rounds and `eta` 0.15: 5.408 points of capacity,
  against 5.405 for full weekly retraining, at about 0.31 of its fit time.
headline: >-
  Monthly minus weekly retraining is +0.125 (95% interval +0.036 to +0.229) at the full setting,
  +0.118 (+0.033 to +0.216) with a 12-month window, and +0.125 (+0.038 to +0.227) with 150 rounds.
  The zero-start gap is -0.017 (-0.062 to +0.037) at full weekly retraining, too small to rank
  schedules by.
lessons: >-
  The cadence gap is reproduced by a monthly refit on a 12-month window and by a weekly refit on
  the whole history with 150 rounds, so those two schedules can stand in for full weekly retraining
  on cadence contrasts. A quarterly refit loses 0.38 points to weekly, so it ranks arms of
  different cadence wrongly. Repeating a 12-month-window fit moves the score by 0.000 to 0.001, so
  GPU noise is far below the contrasts.
advice:
  - >-
    Use monthly P-roll (12-month window, 150 rounds, `eta` 0.15) for research contrasts between
    features or recipes, because it reproduced the monthly-minus-weekly gap to within 0.01 at about
    a tenth of the fit time (K1).
  - >-
    Use the weekly refit on the whole history with 150 rounds and `eta` 0.15 when weekly cadence
    matters but full settings are too slow: it scored within 0.003 of full weekly at about 0.31 of
    the fit time (absolute scores).
  - >-
    Do not use quarterly refits to compare arms that differ in cadence or in how fast they adapt:
    they score 0.38 above weekly (K1 quarterly).
  - >-
    Do not rely on a 0.017-point contrast to rank schedules: on 158 runs the zero-start gap is
    inside the noise, so a pass on it carries no information (R2).
production_status: not-for-production
adopted_in: none
code: ml-experiments/00085-cheaper-retraining-schedules-rank-arms-like-weekly-demand/
---

# Experiment 85: do cheaper retraining schedules rank arms the way weekly retraining does? (SPD1)

**A monthly refit on the latest 12 months of data (150 rounds), or a weekly refit with 150 larger
rounds on the whole history, ranks cadence contrasts the way full weekly retraining does, at 0.10
to 0.31 of its fit time. A quarterly refit does not, and the starting-value contrast is too small
on these rows to say anything about any schedule.** The write-up covers only cadence (monthly
against weekly) and starting value (zero against XGBoost's estimated start) for the Primary
substation N1 model. It says nothing about the `am`-against-`d0` contrast of experiment 64 or the
five-model blend of experiment 49, which the manager excluded because experiment 64 has no weekly
reference and the blend mixes schedules.

We refitted the 16 Primary substations' demand model under cheaper schedules and compared each
schedule's score gaps with the gaps that stored full weekly retraining gives, on 158 daily forecast
runs. The verdict is partly supported: cadence is supported, and starting value cannot be tested.

![Contrasts under each schedule, with the full-weekly reference](figure_1.svg)

![Absolute score of each schedule](figure_2.svg)

![Fit cost of each schedule](figure_3.svg)

## Results

- **Rows:** 16 Primary substations, 158 runs of `fold_a_daily_gap_v1` (the runs whose latest
  weekly refit is one of experiment 80's 28 anchors), seed 0, N1 model with the spread factor 1.15
  and no correction, ENS weather. Weekly schedules refit at those 28 anchors only.
- **Noise floor N:** 0.01 points. A repeat of the 12-month-window monthly fit scored 5.563 against
  5.563, and its zero-start repeat 5.547 against 5.546. Tolerance T is 0.05.
- **Cadence gap (monthly minus weekly):** reference 0.125 (0.036 to 0.229). The 12-month window
  gives 0.118 and the 150-round recipe 0.125, so both agree (rule i). Quarterly minus weekly,
  descriptive, is 0.381 (0.207 to 0.565).
- **Starting-value gap (zero minus default start):** reference -0.017 (-0.062 to +0.037), smaller
  than the -0.04 to -0.07 planned. Every schedule gives -0.003 to -0.020, so each passes rule ii,
  but any value within 0.05 of zero passes, so the pass is weak evidence.
- **Pass table:** monthly 12-month window 2 of 2, weekly 150 rounds 2 of 2, weekly 12-month window
  1 of 1, quarterly 1 of 1 (on starting value only), full monthly 1 of 1 (on starting value only).

| Arm | Score | 95% interval | Coverage of q05 to q95 | Pinball at q05 | Pinball at q95 |
|---|---|---|---|---|---|
| Weekly, full setting | 5.405 | 5.111 to 5.716 | 0.889 | 0.910 | 1.236 |
| Weekly, 150 rounds | 5.408 | 5.119 to 5.711 | 0.897 | 0.908 | 1.233 |
| Weekly, 12-month window | 5.445 | 5.157 to 5.749 | 0.889 | 0.912 | 1.245 |
| Monthly, full setting | 5.530 | 5.179 to 5.912 | 0.888 | 0.907 | 1.294 |
| Monthly, 150 rounds | 5.533 | 5.185 to 5.913 | 0.896 | 0.911 | 1.288 |
| Monthly, 12-month window | 5.563 | 5.219 to 5.939 | 0.889 | 0.920 | 1.288 |
| Quarterly, 12-month window | 5.826 | 5.446 to 6.252 | 0.883 | 0.946 | 1.372 |

## Cost

- **Fit time of one refit (one series, 2 threads):** 114 s at the full setting (500 rounds, whole
  history), 33 to 36 s for the 150-round whole-history recipe, and 13 to 51 s for the 12-month
  window. Other jobs shared the machine, so the same recipe varied by up to 4 times.
- **A year for 16 series:** full weekly about 1,590 compute minutes (the monthly full-setting fit
  time times 52, not measured separately), full monthly 366, weekly 150 rounds 460 to 500, monthly
  12-month window 42 to 164, quarterly 54.
- **Against the one-third rule:** monthly 12-month window (0.03 to 0.10 of full weekly) and weekly
  150 rounds (0.29 to 0.31) pass. Weekly on the 12-month window (0.31 to 0.35) is borderline.

## Caveats and what to try next

- **Starting value is untested:** the stored weekly zero-start arm covers half of the weeks, and
  the gap on those runs is within the noise. A later experiment should pick a contrast of 0.1 points
  or more, such as the `am` against `d0` gap, to test schedules.
- **One series class, one seed:** Primary N1 demand only. Tail scores stop at q05 and q95, because
  the stored quantile models have nine levels from 0.05 to 0.95.
- **Cost is noisy:** read the cost as an order of magnitude, not a measurement.
- **Not covered:** the `am`-against-`d0` and five-model-blend contrasts, as stated above.
- **Next:** use monthly 12-month-window refits for feature and recipe screening, the 150-round
  weekly refit where weekly cadence matters, and keep full weekly retraining for the best-guess
  experiments and production.

## Recommended R&D defaults

- **Feature and recipe contrasts of 0.05 points or more:** use monthly refits on a 12-month window
  with 150 rounds and `eta` 0.15. It reproduced the cadence gap to within 0.01 and repeated to
  within 0.001. It saves about 1,420 to 1,540 of the 1,590 compute minutes a weekly year takes (90
  to 97%). Only the cadence contrast was tested, so a feature contrast is assumed to carry over.
- **Contrasts about cadence, retraining rules, or how fast a model adapts:** use the weekly refit on
  the whole history with 150 rounds and `eta` 0.15. It scored within 0.003 of full weekly and saves
  about 1,090 minutes (69%). Experiment 34 found a weekly-analogue gain of 0.05 shrank to 0.01
  under monthly retraining, so a conclusion about recent-data features, the 14-day correction, the
  training window, or the half-life can change with the schedule. Check it on this weekly refit.
- **Absolute scores and tail scores:** quote them from weekly retraining only. Monthly refits on a
  12-month window score 0.16 points above weekly (5.563 against 5.405) and 0.05 above at q95
  pinball loss, so the same arm reads worse on every schedule except weekly.
- **Quarterly refits:** do not use them. They lose 0.26 points to monthly and 0.38 to weekly, and
  this experiment cannot show that they rank any contrast correctly.
- **Contrasts under 0.05 points:** this experiment cannot say which schedule keeps them, because
  the only one tested, the zero-start gap, was -0.017.
