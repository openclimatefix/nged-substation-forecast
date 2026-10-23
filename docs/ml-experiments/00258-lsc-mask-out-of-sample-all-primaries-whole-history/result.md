---
id: "00258"
slug: lsc-mask-out-of-sample-all-primaries-whole-history
title: >-
  Does the carried LSC mask keep its gain and its no-harm out of sample on every Primary?
theme: lsc-detectors
also_relevant_to:
  - switching-labels-in-the-forecast
status: done
size: >-
  complex. Stored: private score tables and a reusable harness under
  data/ml-experiments/00258-lsc-mask-out-of-sample-all-primaries-whole-history/ only. Serving path:
  not touched. Degradation rule: not touched. Designs: more than one is defensible for how a mask
  that reaches back to the step stamp is made causal, for which fold's alarms score a Primary
  substation that sits in pairs of two folds, and for the scale of the quantile score; each is fixed
  below before any score is read. Callers: this folder's scripts only; they copy the loaders of
  experiments 130, 195, 208, 227, and 254 and read their stored tables.
hypothesis: >-
  Experiment 227 found that LR_b20 (the learned alarm score of experiment 195, at the threshold that
  leaves 20 unexplained odd-month candidates) with a mask recovers 83.5% of the logged-oracle gain
  in the 72 hours after a logged load shift and raises the ordinary-day score by -0.007 points
  (-0.039 to +0.014). That rests on 11 load shifts at 6 Primary substations, 27 months of cached
  ensemble forecasts, a threshold chosen on the scored rows, and months that were the re-ranker's
  training negatives. On all 15 Primary series with a residual from the frozen convex model of
  experiment 130 (arm B), over the whole residual history of about 4.5 years, and scored only on
  rows the learned score was not trained on, the LR_b20 mask keeps at least half of that gain and
  keeps its ordinary-day harm inside +/-0.05 points.
prediction: >-
  Written before any score is read, and only if both known-answer gates pass. (a) The LR_b20 mask
  recovers at least 50% (point estimate) of the logged-oracle mask's gain in the 72 hours after each
  of the 35 logged load shifts lasting over 1 day, with the lower 95% bound above 0%. (b) Its
  odd-month ordinary-day harm over all 15 series has a 95% interval inside +/-0.05 points. The
  prediction fails if either clause fails. If clause (b) fails only because of D8 or D9, the page
  says so first and gives the harm without them.
decision_rule: >-
  Pre-registered. Harness: for each of the 15 series, the frozen convex-model forecast of experiment
  130 (past-weather: ERA5 and CAMS, refit on the 1st of each month, `panel_1`) plus the two-regime
  Kalman filter of experiment 208 with the series' stored hyperparameters (experiment 130's
  `hyper.json`), in points of effective capacity (100 times megawatts over the series'
  `effective_capacity_mw`). Origins every 6 hours, data available 3.5 hours before the origin, leads
  0 to 7 days. Quantiles come from one fixed table per series, hour band, and lead band of K0
  residual quantiles fitted on odd months only and shared by every arm, so every contrast is a
  difference in the location of the forecast only. The score is twice the mean pinball loss over 11
  quantiles, in points. Gates, before any arm is read: (G1) the harness's ORACLE minus K0 gain on
  the 11 real load shifts at the six Primaries of experiment 227 is within 30% of 227's 5.339 points
  (10.129 minus 4.790) and of the same sign; (G2) on experiment 141's stored injected worlds at the
  six Primaries, PERFECT minus K0 in the AFTER phase is within 30% of experiment 208's 1.50 points.
  A failed gate makes the page `inconclusive`. Arms, all causal and on the same rows: K0 (no mask);
  ORACLE (the logged intervals, onset logged, end known at once); LR_b20 and LR_b80 (experiment
  195's re-ranker at those thresholds, fold of the Primary pair that holds the series out, alarm
  known 96 hours after the step, mask reaching back to the step stamp, end by experiment 148's rule
  with 6 hours of lookahead); RO_b80; HALF on LR_b80 (the average of K0 and the masked correction);
  and GAM_ONLY_ALWAYS (the convex forecast with no filter). A series that sits in pairs of two folds
  is run with each fold's alarms and weighted 1 over the number of folds, as in experiment 254. Row
  sets: AFTER (origin at or after the onset, target within 72 hours after the logged end), headline
  on the 35 load shifts over 1 day and the shorter ones separately; ORDINARY (rows outside every
  closure of experiment 37's log of 629 and the 144 half-hours after each), reported all, without
  D8's drops (experiment 159) and the 17 block days (experiment 175), and without D8 and D9.
  Headline harm on odd months only; even months are reported beside it as in-sample. Intervals: gain
  by resampling whole Primary pairs (20,000 draws), harm by resampling whole calendar months, and
  leave-one-Primary-out for both. The 50% bar and the +/-0.05 limit are experiment 227's, and are
  set by the cost of porting an alarm source and its use. Verdict: `supported` if both clauses hold,
  `partly supported` if the gain holds and the harm fails at D8 or D9 only, and `not supported`
  otherwise. Reported beside (not bars): the median forecast minus the mean forecast per series,
  tail coverage and pinball loss at q01, q05, q95, and q99, the net annual difference per
  series-year for LR_b20 and GAM_ONLY_ALWAYS against K0, and every load shift's gain against ORACLE.
benchmark: >-
  Experiment 227: LR_b20 recovers 83.5% (-0.1% to 123%) of the logged-oracle gain after 11 load
  shifts and has ordinary-day harm -0.007 points (-0.039 to +0.014) on six Primaries; the logged
  oracle gains 5.339 points over K0 there (10.129 against 4.790).
row_set_id: >-
  The harness's own rows (15 series, origins every 6 hours, leads 0 to 7 days, whole record); not
  comparable with the leaderboard or with ensemble-driven pages
metric: >-
  Mean quantile score (twice the mean pinball loss over 11 quantiles), in points of effective
  capacity; gain is the K0 score minus the arm's score.
verdict: inconclusive
best_arm: >-
  The LR_b20 mask: 47.1% (95% interval 18.4% to 85.0%) of the logged-oracle mask's gain in the 72
  hours after 34 labelled load shifts, against a bar of 50%, and an odd-month ordinary-day score
  difference from K0 of -0.009 points (-0.022 to +0.002), inside the limit of +/-0.05. The harness
  failed gate G2, so the verdict is inconclusive.
headline: >-
  The LR_b20 mask recovers 47% (18% to 85%) of the best-case gain after a load shift and changes
  ordinary days by only -0.009 points out of sample, but over every row of the 4.5 years it scores
  0.040 points worse than no mask, and 73% of its excess score sits in alarms over logged closures
  that are not labelled load shifts. The harness gains 2.39 points on experiment 141's injected load
  shifts where experiment 208 found 1.50, so the plan's rule gives `inconclusive`; had both gates
  passed, the rule would give `not supported`.
lessons: >-
  One load shift of 34 carries half of LR_b20's gain, and 22 of 34 gain almost nothing, so the
  pooled share of 47% rests on a few events. No filter at all (GAM_ONLY) recovers 88% of the
  oracle's gain after a load shift and scores 0.63 points worse on ordinary days. A mask that is
  judged only on ordinary days and the 72 hours after labelled load shifts misses its largest source
  of excess score in this harness, which is long alarms over logged closures.
take_home_message: >-
  On rows it was never trained on, the carried load shift mask recovers about half of the best-case
  gain after a load shift (47%, with a wide range of 18% to 85%) and changes ordinary days by less
  than 0.05 points. Over every row of 4.5 years it scores slightly worse than no mask, because a few
  long alarms over logged closures add more excess score than the mask removes elsewhere. The check
  against simulated load shifts failed, so the result is inconclusive, and this page gives no
  support for porting the mask.
advice:
  - >-
    Judge any LSC mask by its net score over every row of a year, not by ordinary days and the 72
    hours after labelled load shifts alone, because LR_b20 passes both of those checks and still
    scores 0.040 points worse over every row (net table, figure 4).
  - >-
    Test a limit on how long a mask may stay open, or a rule that releases the mask when readings
    stay away from the forecast for days, because 73% of LR_b20's excess score comes from alarms
    over logged closures that are not labelled load shifts (figure 5).
  - >-
    Report the gain of an LSC arm for each load shift beside the pooled share, because 1 of 34 load
    shifts carries 50% of LR_b20's gain and LR_b20's median per-shift share is 3.8% (figure 1).
  - >-
    Treat the no-filter forecast (GAM_ONLY) as the benchmark of any mask. The no-filter forecast
    recovers 88% of the oracle's gain after a load shift and scores 0.63 points worse on ordinary
    days (figure 2).
  - >-
    Match the quantile offsets of the harness to those of the experiment being reproduced before
    gating on injected load shifts, because multiplying this harness's quantile offsets by 1.5 cut
    its injected-world gain from 2.39 to 1.90 points against experiment 208's 1.50 (gate table).
production_status: not-for-production
adopted_in: none
date: 2026-10-22
code: ml-experiments/00258-lsc-mask-out-of-sample-all-primaries-whole-history/
---

**The result is `inconclusive`, because the harness failed one of its two known-answer gates.** The
harness's numbers still say that the carried LSC mask recovers about half of the oracle's gain and
scores worse than no mask over every row of 4.5 years. The load shift screen (LSC) detects load
shifts. A load shift is demand that the electricity network operator (NGED, National Grid
Electricity Distribution) moves between neighbouring Primary substations for hours or days. The mask
sets the readings inside a detected load shift aside, so the forecast filter does not learn the
temporary level. We scored experiment 227's carried setting, LR_b20, in a new forecast harness on 15
Primary series over about 4.5 years, taking each learned alarm from the fold that held its Primary
pair out and scoring ordinary days on odd months only. In the 72 hours after 34 labelled load shifts
of over 1 day, LR_b20 recovers 47.1% of the gain of the oracle mask (95% interval 18.4% to 85.0%),
which is under the bar of 50%. LR_b20's ordinary-day score difference from no mask is -0.009 points
(-0.022 to +0.002), inside the limit of +/-0.05. The harness reproduces experiment 227's gain on
real load shifts (gate G1). The harness gains 2.39 points on experiment 141's injected load shifts
where experiment 208 found 1.50 (gate G2, limit 30%). The plan makes a failed gate `inconclusive`.
Had both gates passed, the plan's rule would give `not supported`, because the share is under 50%.

![Figure 1: gain over K0 of the oracle mask and LR_b20 after each load shift](per_shift_gain.svg)

![Figure 2: share of the oracle's gain recovered, and ordinary-day score difference, for each arm](arms.svg)

![Figure 3: ordinary-day score difference of LR_b20 by Primary, odd and even months](harm_by_primary.svg)

![Figure 4: LR_b20 and no-filter score minus K0's over every row of each Primary and year](net_annual.svg)

![Figure 5: where the LR_b20 mask's excess score over K0 sits](net_loss_classes.svg)

## Terms

- **Points and score.** A point is 1% of a series' effective capacity. The score is twice the mean
  pinball loss over 11 quantiles, and lower is better. Every arm uses the same quantile offsets, so
  a score difference is a difference in the forecast's location only.
- **Arms.** K0 is the forecast with no mask: the frozen convex GAM forecast of experiment 130, which
  uses past weather (ERA5 and CAMS), plus the Kalman filter on recent power. A mask arm also sets
  the readings inside each known alarm span aside. ORACLE masks the 66 labelled load shifts as
  logged. LR_b20 and LR_b80 are experiment 195's learned alarm score at budgets of 20 and 80
  unexplained odd-month candidates, taken from the fold that holds the Primary pair out. RO_b80 is
  RO-NC at budget 80. HALF averages the K0 and LR_b80 corrections. GAM_ONLY is the GAM forecast with
  no filter.
- **Row sets.** A row is one forecast of one half-hour from one origin (every 6 hours, leads 0 to 7
  days). AFTER rows are the 72 hours of target time after a labelled load shift ends, from origins
  at or after its logged onset. ORDINARY rows lie outside every logged closure that names the series
  and the 72 hours after each closure. Odd months did not supply the learned score's negative
  training examples; even months did.

## Results

| Arm | Gain after load shifts (points) | Share of ORACLE's gain | Ordinary days, odd months | Ordinary days, even months | Every row |
|---|---|---|---|---|---|
| LR_b20 | +1.69 (0.50 to 3.14) | 47.1% (18.4% to 85.0%) | -0.009 (-0.022 to +0.002) | +0.003 (-0.011 to +0.024) | +0.040 |
| LR_b80 | +1.80 (0.62 to 3.23) | 50.2% (21.6% to 88.1%) | +0.039 (-0.007 to +0.108) | +0.018 (-0.006 to +0.050) | +0.071 |
| RO_b80 | +2.52 (0.75 to 4.47) | 70.2% (41.0% to 93.7%) | +0.082 (+0.003 to +0.194) | +0.026 (-0.022 to +0.074) | +0.099 |
| HALF | +1.19 (0.44 to 2.06) | 33.2% (14.8% to 57.9%) | 0.000 (-0.014 to +0.017) | -0.004 (-0.014 to +0.006) | +0.008 |
| GAM_ONLY | +3.16 (1.15 to 5.79) | 88.0% (66.5% to 96.8%) | +0.633 (+0.418 to +0.897) | +0.456 (+0.305 to +0.632) | +0.582 |
| ORACLE | +3.59 (1.60 to 6.23) | 100% | -0.016 (-0.032 to -0.004) | -0.017 (-0.028 to -0.007) | +0.000 |

Gain is the K0 score minus the arm's score on the AFTER rows of the 34 load shifts (146,354 rows, 18
Primary pairs). Ordinary-day columns are the arm's score minus K0's, so negative is better. "Every
row" is the arm's score minus K0's over every forecast with a reading, in all 5 calendar years, with
no interval.

| Gate | Result | Target | Passes |
|---|---|---|---|
| G1: K0 minus ORACLE on the 13 load-shift and series pairs that experiment 227 scored (32,148 rows) | 5.68 points | 5.339 within 30% | yes |
| G2: K0 minus PERFECT in the AFTER phase of 141's injected load shifts (717,698 rows) | 2.39 points | 1.50 within 30% | no |

- **Question and rule.** Does LR_b20's mask keep at least half of the oracle's gain after a load
  shift, with ordinary-day harm inside +/-0.05 points, on every Primary and on months it did not
  train on? The harm limit holds. The gain share misses the bar on the point estimate (47.1% against
  50%), and its lower bound is above 0%. The failed gate G2 makes the verdict `inconclusive`.

- **Gate G1 passes.** The harness's K0 scores 9.61 points on experiment 227's real load-shift rows
  against 10.13 in experiment 227, and the gain of the oracle is 5.68 points against 5.34. The gain
  rises with lead: 2.22 points under 24 hours, 4.49 from 24 to 72 hours, and 8.53 from 72 hours to 7
  days.
- **Gate G2 fails.** The harness's PERFECT minus K0 gain is 2.39 points, 59% above experiment 208's
  1.50. Each lead band is 42% to 69% above experiment 208's gain in the same band (1.98, 2.65, and
  2.61 points against 1.25, 1.57, and 1.84). Three post hoc diagnostics, all outside the
  pre-registered rule, change the gain. Multiplying the quantile offsets by 1.5 lowers the gain to
  1.90 points, which is within 30% of 1.50 and would pass the gate. Multiplying by 1.2 gives 2.17.
  Keeping only injected load shifts in experiment 208's scored year gives 2.15. The gate fails as
  pre-registered. Whether the gate passes depends on a post hoc setting of the quantile offsets.
- **The gain after load shifts rests on a few events.** The largest of the 34 load shifts carries
  49.9% of LR_b20's total gain and 20.0% of the oracle's. LR_b20 recovers at least half the oracle's
  gain after 6 of the 34 load shifts, and between -10% and +10% of the oracle's gain after 22. The
  median per-shift share (28 load shifts where the oracle gains) is 3.8% for LR_b20 and 98.8% for
  GAM_ONLY. Leaving out one Primary moves LR_b20's share between 38.8% (without D16) and 63.5%
  (without D18), so the 50% bar is crossed by dropping D18. The 30 load shifts of 1 day or less give
  the oracle 0.22 points and LR_b20 0.01.
- **Ordinary days.** LR_b20's difference stays inside +/-0.05 in every variant: -0.010 without D8's
  drops and the 17 block days, -0.011 without D8 and D9, -0.008 when closures of connected
  neighbours are also excluded, and -0.005 (-0.013 to +0.002) without the 7.5 alarms (of 44 that
  start on ordinary rows, each alarm weighted 1 over its series' number of folds) that are
  balanced-opposite candidates. The in-sample even months give +0.003. At 13 of 15 Primaries the
  absolute difference is under 0.05. At the other two, D18 (-0.195) and D22 (-0.052), the mask
  improves the score. D12 and D2 have no LR_b20 alarms. Leaving one Primary out moves the pooled
  difference between -0.006 and -0.012. Of the three alarm settings, LR_b20 has the smallest
  ordinary-day difference: LR_b80 gives +0.039 and RO_b80 +0.082, with an interval above zero.
- **Net over a year.** Over every forecast with a reading, LR_b20 scores 0.040 points worse than K0
  (by year: -0.008, +0.046, +0.125, -0.008, +0.040 from 2022 to 2026), the oracle 0.000, and
  GAM_ONLY 0.582 worse. D22 and D21 account for 65% and 26% of the summed excess over series, mostly
  in 2024 (+1.55 points at D22) and 2023 (+0.59 at D21). The excess sits in the alarm spans: 73% of
  LR_b20's excess score lies in spans that overlap a logged closure that is not a labelled load
  shift, 25% in spans that overlap a labelled load shift, and 1.5% in spans that overlap no logged
  closure (shares of the score summed over the alarm spans, each span counted once, which cover 82%
  of the net figure). The three alarms with the largest excess score last 27, 32, and 44 days, sit
  on D22, D21, and D22, and carry 59%, 31%, and 9% of the excess summed over spans. The 32-day alarm
  on D21 overlaps a labelled load shift.
- **Tails.** After load shifts, K0's readings fall below its 1% quantile 9.1% of the time and
  LR_b20's 4.0%, against 1.8% for ORACLE (nominal 1%). K0's readings fall below the 99% quantile
  93.4% of the time, LR_b20's 96.3%, and ORACLE's 98.6% (nominal 99%). On ordinary rows the tail
  shares of LR_b20, HALF, and ORACLE match K0's within 0.2 percentage points, and GAM_ONLY's
  readings fall below the 1% quantile 2.9% of the time and below the 5% quantile 9.9% of the time.
- **Row sets and data.** 15 Primary series (D15 has no readings), 2022-01 to 2026-06, origins every
  6 hours from 2022-02-01, 12,236,784 ordinary odd-month rows. D-numbers are the labels in the
  series table. The GAM forecast of D9 starts in 2023-12 and D18's in 2025-07, so those two series
  have fewer rows. The 35th labelled load shift (D15 and D18, 2025-03) has no AFTER rows, because
  D18's GAM forecast starts after it.
- **Limits.** The forecast uses past weather, so absolute scores are an upper bound on forecast
  skill and are not comparable with ensemble-driven pages. The effective capacity is the single
  stored value, not re-estimated from data before each origin. A row whose target lies after a refit
  of experiment 130's GAM sees a fit made up to 7 days after its origin. The intervals for gain
  resample 18 Primary pairs, so they are wide. The learned alarm's threshold is set on the pooled
  scores of all 15 series in a fold file, and those pooled scores include the scored series' own odd
  months. A series in two or three folds is scored with each fold's model, weighted equally, and
  each of those models trained on another pair that contains the series. The Kalman hyperparameters
  and the GAM refits come from experiment 130, which used the whole record. A mask arm's span ends
  at the end rule of experiment 148 with 6 hours of lookahead, and every learned alarm is known 96
  hours after its step (RO_b80's: 48 hours).
- **What to try next.** A limit on how long a mask stays open would test whether the excess score
  over a year disappears without losing the gain after load shifts, because 3 alarms of 27 to 44
  days carry 99% of the excess summed over spans. A like-for-like gate would first match the
  harness's forecast spread to experiment 208's.

## Checks

- **Gates.** G1 and G2 were fixed in the plan and run before any arm was read. G2 failed and was not
  changed.
- **Series 10 was read early.** The aggregated table of series 10 (D10) was printed while the
  scoring script was being tested, before the clarifications in the last section were committed.
- **Median against mean.** The median forecast minus the mean forecast differs by series, from -2.06
  points at D12 to +0.61 at D8, and is the same for every arm. The contrasts between arms do not
  depend on the median-minus-mean gap.
- **Direction.** The quantile-score gain and the mean and median absolute-error gains agree in sign
  for every arm except HALF. The exception is HALF, whose ordinary-day differences are within 0.003.
- **Audit triggers.** Three of the six audit triggers fired: the failed G2, a leave-one-out share
  that reaches 63.5% (without D18), and a harm interval under half as wide as experiment 227's. A
  Sonnet audit (Opus was not used, to save quota) recomputed the gain, the share, the harm interval,
  and the per-shift shares from the per-row scores without reading the report script. The audit
  agreed with every figure. The audit also found that leaving out the largest load shift (pair D5
  and D16) drops LR_b20's share to 29.5%, and that the harm interval is built from few non-zero
  months and cannot see an alarm over a logged closure, which the ordinary rows exclude.
- **Code review.** A Sonnet review found no leakage in the filter's cutoffs, the span masking, the
  fold use, the offsets, the row sets, or the gates. The review found that the per-alarm net-loss
  table counted overlapping spans of different folds more than once, so the split of the excess by
  span class was recomputed with each span scored by its own fold. The review also listed the limits
  above that were missing from the page.
- **Numbers review.** A Sonnet review checked every figure against the printed reports and found no
  wrong number. It asked for the post hoc G2 diagnostics to be reported in full, for the claims
  about the net annual loss to be softened, and for the bullet lists to be repaired; all three were
  applied.

## Choices fixed after the plan and before any gate or arm was read

These pin down points the plan left open. The harness code was run and its output for one series
(series 10, aggregated over all its rows) was printed to test the scorer before these choices were
committed; no gate and no contrast was read.

- **Origins and leads.** Origins every 6 hours from 31 days after the series starts (2022-01-01),
  each with 336 half-hourly targets (leads 0 to 7 days). Lead bands: under 24 hours, 24 to 72 hours,
  and 72 hours to 7 days. Hour bands are the four local-time bands of `studies.dp_calendar`.
- **Quantile offsets.** One table per series, hour band, and lead band: the 11 quantiles of the
  no-mask residual over odd UTC months of the target, shared by every arm.
- **Effective capacity.** The single value stored for each series in the stage 1 pilot's series
  table, not re-estimated from data before each origin. The plan said "from data before each
  origin"; the stored value is the one every earlier page used.
- **GAM forecast.** The frozen forecast of experiment 130 uses, for each target, the fit made at the
  latest refit on or before the target. A row whose origin lies before a refit and whose target lies
  after it therefore sees a fit made up to 7 days after the origin.
- **ORACLE.** Masks every one of the 66 labelled real load shifts that involves the series.
- **Gate G1 rows.** The 13 pairs of load shift and Primary series that experiment 227 scored (11
  load shifts in its scored and replicate months). Rows: forecast origin at or after the logged
  onset and target within 72 hours after the logged end, every origin in the harness, leads 0 to 7
  days. The day-ahead lead band is reported beside the pooled figure. The gate is the pooled K0
  minus ORACLE score on these rows, within 30% of 5.339 points.
- **Gate G2 rows.** Experiment 141's 44 stored worlds at the six Primaries, the stream being the
  real residual plus the injected megawatts (in points). Rows follow experiment 208: injected
  transfers of 25% of load or more lasting over 1 day, forecast cutoff at or after the injected
  onset and not within 96 half-hours of it (the AFTER phase is cutoff at least 96 half-hours after
  the onset and target at or after the end), target within 96 half-hours of the end, scored against
  the stream without the injected megawatts. PERFECT masks the true interval, known at the onset
  with the end known at once. The gate is PERFECT minus K0 within 30% of 1.50 points.
- **ORDINARY rows.** The target lies outside every closure of experiment 37's log of 629 that names
  the series as a side, and outside the 72 hours after each closure. A variant that also drops
  closures of connected neighbours is reported beside it.
- **Month for the odd and even split.** The UTC calendar month of the target.
