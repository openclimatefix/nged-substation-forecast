---
id: "00258"
slug: lsc-mask-out-of-sample-all-primaries-whole-history
title: >-
  Does the carried LSC mask keep its gain and its no-harm out of sample on every Primary?
theme: lsc-detectors
also_relevant_to:
  - switching-labels-in-the-forecast
status: planned
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
  Pre-registered. Harness: for each of the 15 series, the frozen convex-model forecast of
  experiment 130 (past-weather: ERA5 and CAMS, refit on the 1st of each month, `panel_1`) plus the
  two-regime Kalman filter of experiment 208 with the series' stored hyperparameters (experiment
  130's `hyper.json`), in points of effective capacity (100 times megawatts over the series'
  `effective_capacity_mw`). Origins every 6 hours, data available 3.5 hours before the origin, leads
  0 to 7 days. Quantiles come from one fixed table per series, hour band, and lead band of K0
  residual quantiles fitted on odd months only and shared by every arm, so every contrast is a
  difference in the location of the forecast only. The score is twice the mean pinball loss over 11
  quantiles, in points. Gates, before any arm is read: (G1) the harness's ORACLE minus K0 gain on the
  11 real load shifts at the six Primaries of experiment 227 is within 30% of 227's 5.339 points
  (10.129 minus 4.790) and of the same sign; (G2) on experiment 141's stored injected worlds at the
  six Primaries, PERFECT minus K0 in the AFTER phase is within 30% of experiment 208's 1.50 points.
  A failed gate makes the page `inconclusive`. Arms, all causal and on the same rows: K0 (no mask);
  ORACLE (the logged intervals, onset logged, end known at once); LR_b20 and LR_b80 (experiment 195's
  re-ranker at those thresholds, fold of the Primary pair that holds the series out, alarm known 96
  hours after the step, mask reaching back to the step stamp, end by experiment 148's rule with 6
  hours of lookahead); RO_b80; HALF on LR_b80 (the average of K0 and the masked correction); and
  GAM_ONLY_ALWAYS (the convex forecast with no filter). A series that sits in pairs of two folds is
  run with each fold's alarms and weighted 1 over the number of folds, as in experiment 254. Row
  sets: AFTER (origin at or after the onset, target within 72 hours after the logged end), headline
  on the 35 load shifts over 1 day and the shorter ones separately; ORDINARY (rows outside every
  closure of experiment 37's log of 629 and the 144 half-hours after each), reported all, without
  D8's drops (experiment 159) and the 17 block days (experiment 175), and without D8 and D9.
  Headline harm on odd months only; even months are reported beside it as in-sample. Intervals: gain
  by resampling whole Primary pairs (20,000 draws), harm by resampling whole calendar months, and
  leave-one-Primary-out for both. The 50% bar and the +/-0.05 limit are experiment 227's, and are set
  by the cost of porting an alarm source and its use. Verdict: `supported` if both clauses hold,
  `partly supported` if the gain holds and the harm fails at D8 or D9 only, and `not supported`
  otherwise. Reported beside (not bars): the median forecast minus the mean forecast per series,
  tail coverage and pinball loss at q01, q05, q95, and q99, the net annual difference per series-year
  for LR_b20 and GAM_ONLY_ALWAYS against K0, and every load shift's gain against ORACLE.
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
---

Planned; the write-up replaces this page when the scores are done.

## Choices fixed after the plan and before any gate or arm was read

These pin down points the plan left open. The harness code was run and its output for one series
(series 10, aggregated over all its rows) was printed to test the scorer before these choices were
committed; no gate and no contrast was read.

- **Origins and leads.** Origins every 6 hours from 31 days after the grid starts (2022-01-01), each
  with 336 half-hourly targets (leads 0 to 7 days). Lead bands: under 24 hours, 24 to 72 hours, and
  72 hours to 7 days. Hour bands are the four local-time bands of `studies.dp_calendar`.
- **Quantile offsets.** One table per series, hour band, and lead band: the 11 quantiles of the
  no-mask residual over odd UTC months of the target, shared by every arm.
- **Effective capacity.** The single value stored for each series in the stage 1 pilot's series
  table, not re-estimated from data before each origin. The plan said "from data before each
  origin"; the stored value is the one every earlier page used.
- **GAM forecast.** The frozen forecast of experiment 130 uses, for each target, the fit made at
  the latest refit on or before the target. A row whose origin lies before a refit and whose target
  lies after it therefore sees a fit made up to 7 days after the origin.
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
  the series as a side, and outside the 72 hours after it. A variant that also drops closures of
  connected neighbours is reported beside it.
- **Month for the odd and even split.** The UTC calendar month of the target.
