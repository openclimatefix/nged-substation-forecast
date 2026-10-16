---
id: "00063"
slug: sw2-10-linear-joint-fit-v2-in-miniature
status: planned
size: medium
hypothesis: >-
  On the 8 Primary substations metered in MW, one linear fit per Primary that holds a weather
  baseline without radiation inputs, the embedded-solar capacity of experiment 00045 times the CAMS
  unit PV shape, and one share-of-the-donor's-net-power term per labelled transfer, (C1) lowers the
  error of the median forecast outside switching on held-out months, in hindsight, below that of the
  same baseline with radiation inputs by at least 0.35 points of capacity, (S1) recovers an
  injected solar capacity to within 0.15 of its size, and (S2) recovers an injected transfer share
  to within 0.05.
prediction: >-
  C1 is -0.4 points (-0.8 to 0.0) on daylight half-hours from April to September, so the verdict is
  likely `small gain` or `inconclusive`. The precedent is experiment 00046, where the true sunshine
  (the CAMS oracle) lowered the quantile score of the best guess by 0.97 points but the served ENS
  solar term lowered it by 0.06. S1 recovers 0.80 to 0.95 of an injected capacity (median error 0.05
  to 0.20). S2 has a median absolute share error of 0.02 to 0.06 on summer intervals, worse than on
  winter intervals because donor and receiver solar are then collinear.
decision_rule: >-
  C1 decides the verdict, with S1 and S2 as validity gates. C1 is the mean absolute error of the
  median (q50), in points of the series' capacity, of the joint fit (baseline without radiation plus
  fitted solar term) minus that of the frozen v1 baseline (the same leave-one-month-out baseline with
  its radiation inputs), over the held-out rows: daylight half-hours outside logged and labelled
  switching, 2023-09 to 2026-06. The interval is a 95% bootstrap over whole months, resampled within
  series then pooled over series (50,000 resamples). The threshold is 0.35 points: porting the joint
  fit would add a capacity table, a PV shape input and a fitting step to the serving path, the same
  cost experiment 00046 priced at 0.35. `supported` when C1 lies wholly below -0.35 and S1 and S2
  hold. `small gain` when C1 lies wholly below zero but not wholly below -0.35, and S1 and S2 hold.
  `refuted` when C1 lies wholly above -0.35 and S1 and S2 hold, or when C1 lies wholly above zero.
  `inconclusive` otherwise, including when S1 or S2 fails (the fit then cannot be read as a
  decomposition). S1 holds when the median over injections of |recovered added capacity / injected
  capacity - 1| is at most 0.15. S2 holds when the median absolute error of the recovered share is at
  most 0.05 on injected transfers of 0.1, 0.25 and 0.5 of the donor's net power lasting 6 hours or
  more, and the 95th percentile of the absolute recovered share on placebo intervals (true share
  zero) is at most 0.10. Descriptive contrasts, read at 95% and never changing the verdict: C2 (fixed
  scale 1 minus fitted scale), C3 (the recommended weekly capacity series, post hoc pair a=2, r=10,
  minus the season-year constant), C4 (joint fit without the transfer terms minus with them), and R1
  (on the real labelled events, the share of the residual's variance explained inside each event
  against the same statistic on 20 placebo intervals of equal length per event, shifted by whole weeks
  into unlabelled periods).
benchmark: >-
  The frozen v1 baseline: experiment 00041's leave-one-month-out XGBoost (median, 13 input columns
  including shortwave and longwave radiation, logged closures padded by 24 hours excluded from
  training). A second arm of the joint fit uses the same XGBoost without the two radiation columns.
row_set_id: >-
  none (hindsight rows: every daylight half-hour of the 8 Primaries outside switching from 2023-09-04
  to 2026-06-30, not the shared forecast rows, because the baseline uses weather at the valid time)
metric: >-
  Mean absolute error of the median, in points of the series' capacity (the 99th percentile of its
  half-hourly power magnitude); pinball loss at q10 and q90 as a secondary view; for S1 and S2 the
  recovered quantity against the injected one.
production_status: not-for-production
adopted_in: none
code: ml-experiments/00063-sw2-10-linear-joint-fit-v2-in-miniature/
---

# Experiment 63: does a linear joint fit of baseline, solar, and transfers beat the v1 baseline?

**Plan, written before any fit.** We ask whether one small linear fit per substation can explain a
substation's metered demand as three separate parts: ordinary weather-driven demand, the output of
solar panels behind the meter, and load lent by a neighbouring substation during a switching job.
Flexpectation v1 forecasts the metered total with one XGBoost model. Flexpectation v2 is meant to
forecast the three parts separately, so that NGED can read each one. This experiment is v2 in
miniature, with the switching times given, to learn whether the split costs accuracy and whether the
parts come out at the right size.

## Size

- **What gets stored:** private predictions and tables under `data/ml-experiments/`; nothing in the
  repository except the page and scripts.
- **Production serving path:** not touched; the experiment reads stored hindsight baselines.
- **Degradation rule:** not touched.
- **More than one defensible design:** yes. The capacity can be an input or a fitted value, and the
  baseline can include or exclude radiation. This plan fixes the capacity (the brief's primary), fits
  a scale on it, and excludes radiation from the joint fit's baseline.
- **Code whose callers could not be named without searching:** no; the scripts copy patterns from
  experiments 00041 and 00045, which no other code imports.

This is `medium`: a plan, one Sonnet plan reviewer, Sonnet implementation, one Sonnet code review,
and a Sonnet write-up review. Opus is not used because no verdict depends on a judgement that a
Sonnet reviewer could not check against the printed report.

## Design

- **Series.** The 8 Primaries metered in MW with a capacity in
  `capacity_weekly.parquet` of experiment 00045: Trusthorpe, Horncastle, Spilsby, Wrangle T1,
  Wrangle T2, Mount Bridge Boston, Kirton, and Donington. Mount Bridge Boston from 2023-09-04 to
  2025-08-25 is skipped, because its constant capacity is about 0 MW over 104 unobserved weeks.
  Rows use `status` clean or low-power and `unobserved` false.
- **Capacity.** The primary is the `season_year_constant` variant (recommended = true). The
  weekly recommended series (post hoc a=2, r=10) is a sensitivity (C3). The vintages file holds no
  season-year-constant rows, so this experiment is a hindsight decomposition and does not claim a
  real-time capacity.
- **Weather and the caveat.** Every weather input is ERA5 at the valid time, and the PV shape is the
  CAMS unit PV shape at the valid time (`u` in experiment 00045's half-hourly inputs). Neither exists
  at forecast time: ERA5 arrives about 5 days late, and CAMS is a past estimate. The experiment
  therefore bounds what a v2 decomposition could do with perfect sunshine. Experiment 00046 showed that
  served ENS sunshine keeps almost none of that gain.
- **Baseline.** Experiment 00041's leave-one-month-out XGBoost (13 columns, logged closures padded by
  24 hours excluded from training) is the frozen v1 baseline, read from its stored files. The joint
  fit's baseline is the same procedure refitted without `shortwave` and `longwave`, one file per
  series, on 4 cores. Every baseline prediction for a month comes from a model that never saw that
  month, so the joint fit's residuals are not shrunk by overfitting.
- **The joint fit.** For Primary p, with residual r_t = y_t - b_t, an ordinary least-squares fit of
  r_t = -k x C_w(t) x u_t + sum_j beta_j x x_j,t x 1[t in I_j] + e_t, where k is a scale on the
  capacity C_w(t) of the week, u_t the CAMS unit PV shape, I_j the labelled interval of transfer j,
  x_j,t the donor's net power in MW (receiver gains when the donor is switched onto it), and beta_j
  its share. Rows are daylight half-hours (for k) plus all half-hours inside labelled intervals, with
  rows in logged closures that carry no label dropped. The labelled intervals are experiment 00041's
  real visible events, conserved or one-sided, with `onset_utc` and `end_utc`.
- **Held-out scoring (C1).** Leave one calendar month out. Fit k on the other months, predict
  b_t - k x C x u_t for the held-out month, and score the median. Transfer terms are zero outside
  switching, so C1 is a test of the solar term only; the transfer terms are tested by S2 and R1.
  The v1 arm is b_t with radiation, and the control arm is b_t without radiation, so the page can say
  what the solar term buys over simply removing radiation.
- **Injection harness S1.** For each Primary, add the half-hourly AC output of a real solar farm from
  `farm_half_hourly.parquet` (experiment 00045's farms) scaled to 10%, 25% and 50% of the Primary's
  capacity, to the Primary's metered power in the scored period, and refit. The recovered added
  capacity is the change in k*C. The injected path is the truth.
- **Injection harness S2.** Pick 40 intervals per Primary at random starts outside labelled
  and logged switching, durations 6, 12, 24 and 72 hours, add beta times a donor's real net power
  (donor drawn from the other series with data, so the donor's own solar is included), beta in
  {0.1, 0.25, 0.5}, plus 40 placebo intervals with beta 0. Half the intervals fall in April to
  September and half in October to March. The recovered beta is compared with the injected one, with
  the interval given to the fit.
- **R1, real events.** For each real visible event that names one of the 8 Primaries as donor or
  receiver, fit beta in sample and report the variance explained, against 20 shifted placebo
  intervals. This is descriptive, because there is one parameter per event.
- **Compute.** The XGBoost refits (about 400 quick fits) run in 4 workers of 1 thread each in
  slot E; the linear fits are negligible. Each fit writes its own file atomically.

## Departures to report

The page will say plainly that C1 uses oracle sunshine, that the fit is in hindsight, and that the
only test of the transfer terms outside the injection harness is the descriptive R1.

## Advice followed

- Experiment 00045: the season-year constant is the primary capacity, because the weekly series
  did not beat it on injected changes.
- Experiment 00046: oracle sunshine is worth about 1 point, served sunshine about nothing; this
  plan labels its gain as a ceiling for that reason.
- Experiment 00041: labelled events are scored as given, with the don't-care bands for transfers
  under 25% of the receiving load named in the write-up.
