---
id: "00088"
slug: sw2-21-nra-forecast-recovers-normal-arrangement-during-and-after-injected-switches
status: planned
size: complex
hypothesis: >-
  When a Primary substation (a substation that steps down to the local distribution network) is
  inside a load transfer, or has just left one, an XGBoost demand
  model that was trained without the transfer's half-hours as targets, with the lags corrected by
  the transfer's signed MW and the MW as a column ("full NRA", NRA being the normal running
  arrangement), forecasts the power the substation would have drawn had no transfer happened
  better than the plain model does. This holds when the MW comes from the XL detector of
  experiment 82 (XL, a logistic score on the 24-hour residual step) run causally on data up to the
  forecast origin, but by less than when the MW
  comes from the injected truth.
prediction: >-
  On injected transfers whose forecast origin falls inside the transfer (family A), the full NRA
  booster trained on XL labels, given causal XL MW, closes 20% to 45% of the gap in MW-weighted
  mean absolute error between the plain booster and the ceiling (the plain booster on lags with no
  transfer). It closes about 0% in the first 24 hours after onset and little in the next 12 hours,
  because the
  24-hour step statistic needs 24 hours of data after the onset (plus the 3.5-hour telemetry delay),
  and 40% to 70% from the second day. In family B the causal detector keeps an ended transfer open
  for about a day after it ends, so arm 4 with causal XL MW is worse than the plain booster in the
  first day after the end. With the
  injected MW the oracle-trained booster closes 70% to 95%. Masking the training targets adds
  nothing to the lag correction: the XL-trained full NRA booster minus the lag-correction booster,
  both with causal XL MW, is between -2% and +2% of the plain booster's error (experiment 75
  found the mask changes the NRA-row score by under 0.03 points of capacity). On forecast origins
  with no transfer, false causal XL alarms raise the error by under 0.02 points of capacity.
decision_rule: >-
  Four planned contrasts on the family A cases (forecast origin inside the injected interval),
  shapes
  S2, S3, S4, and S6 pooled with equal weight. These are the shapes that the harness does not
  build by taking a share of the donor's profile, so the headline does not rest on the shape that
  matches the harness. The measure is the MW-weighted mean absolute error of the median forecast
  against the true unmodified series, in points of capacity, over leads up to 10 days. Each
  contrast is read at 98.75% (Bonferroni across four), from 20,000 resamples of whole calendar
  months of the transfer's onset. C1, arm 4 with causal XL MW minus arm 1 (what Flexpectation v2
  could run). C2, arm 3 with injected MW minus arm 1 (the oracle). C3, arm 4 minus arm 2, both
  with causal XL MW (whether masking the targets adds to the lag correction). C4, arm 2 with
  causal XL MW minus arm 1 (the lag correction alone). The verdict is `supported` when the upper
  end of C1 is below zero, C1's point value is at least 20% of the gap from arm 1 to the ceiling
  (arm 5), and C1 is negative in at least 5 of the 6 onset months. It is `small gain` when the
  upper end of C1 is below zero but one of the other two conditions fails, `inconclusive` when
  C1's interval contains zero, and `refuted` when its lower end is above zero. Six month blocks
  give few distinct resamples, so the per-month signs are part of the rule and the interval is
  read as a range. The threshold of 20% of the gap is set by the cost of porting: the full NRA
  forecast needs a causal detector, a second booster family, and a label table on the serving
  path of Flexpectation v2. The pooled family A mix includes the first 24 hours after onset,
  where the gain is about zero by design, so the 1-3 d and over-3 d bins are also reported with
  their own shares. C2 to C4 are inferential at the same level but decide nothing; the family B
  cases, shapes S1 and S5, the false-alarm check, and the real-event check are descriptive.
benchmark: >-
  Arm 1, the plain booster of experiment 86 on the contaminated lags, and arm 5, the same booster
  on the unmodified lags (the ceiling).
row_set_id: >-
  none (injected cases drawn from the 292 daily runs of fold_a_daily_gap_v1, the runs that
  experiment 86's boosters forecast; not comparable with the leaderboard)
metric: >-
  MW-weighted mean absolute error of the median forecast against the true unmodified series
  (points of capacity), with the quantile score, the pinball loss at the 0.05 and 0.95 quantiles,
  and 90% interval coverage beside it
code: ml-experiments/00088-sw2-21-nra-forecast-recovers-normal-arrangement-during-and-after-
injected-switches/
---

# Experiment 88: does the full NRA forecast recover the normal arrangement during a switch?

Plan only; the write-up replaces this page when the run finishes.

- **Question.** [Experiment 75](../00075-sw2-11-nra-forecast-with-labelled-switching-
  masks/result.md)
  could not tell whether a forecast trained without switching half-hours, with corrected lags,
  helps, because the metered target is abnormal during a switch and only 1.3% of scored
  half-hours had a lag inside an event. Injection makes the unmodified power known, so the
  forecast can be scored on what it is for.
- **Sizing (the five `plan-issue` triggers).** What gets stored: two new sets of boosters
  (arms 3 and 4), per-case forecast tables, and the injected-event tables, all private under
  `data/`; no table the production code reads. Production serving path: not touched. Degradation
  rule: not touched. More than one defensible design: yes (how a detector run causally reports an
  open event, and which shapes form the headline), so the plan fixes both. Code whose callers
  cannot be named without searching: no, the scripts copy the experiment 82 injection and
  detector code and the experiment 86 forecasting code. The experiment is `complex` because it
  has a new method (the causal detector and the injected-lag forecasting) and more than one
  defensible design. Per the brief, all plan and code reviews are Sonnet.
- **Earlier advice followed.** Experiment 75: test corrected lags with the MW columns (arm 2),
  judge a refit arm against a same-code repeat of the plain arm (arm 1 and arm 5 use experiment
  86's stored boosters, so no repeat is needed), and score on rows with a lag inside an event
  (every injected case has one by construction). Experiment 82: score a detector on shapes it
  does not assume (S6 is new), and report by donor step in noise units. Experiment 58 and 43:
  injected events are in quiet periods (no logged closure within 7 days, no alarm within 2 days).

## Design

- **Injection.** Experiment 82's harness: `draw_events` with the 600-event stratification (160 in
  the headline band, 55 in each of the other eight bands of size and duration), onsets restricted
  to the six even months of the validation year (2025-08, 2025-10, 2025-12, 2026-02, 2026-04,
  2026-06), each quiet on every series it touches. `inject_shaped` injects each event under five
  shapes: S1 donor profile, S2 constant MW, S3 receiver profile, S4 fitted mix (the profile
  shares fitted to the labelled real transfers), and S6, the donor's profile in time-reversed
  order across the interval (new; replaces experiment 82's S5, the profile shifted 12 hours,
  which is also reported). The headline pools S2, S3, S4, and S6. Each shape rescales its MW to
  S1's mean, as in experiment 82. The pilot sets the number of events: if forecasting all of them
  would take over 4 hours, the draw is halved to 300 before any scoring, and the number is
  recorded in the write-up.
- **Cases.** The 292 daily runs of `fold_a_daily_gap_v1` (origin 09:00 UTC; the runs that
  experiment 86's boosters forecast, so the 73 every-fifth-day runs are not used). Family A: the
  origins with onset at or before the origin minus 3.5 hours (so the newest usable lag is inside the
  interval), up to 7 days after the onset and before the end. Family B: the origins from 3.5 hours
  after the end up to 7 days after the
  end (the lag window; a lag at least 7 days and 3.5 hours before the origin cannot touch the
  interval). A case is an (event, series, origin, valid time), for the donor and each metered
  receiver, with lead up to 10 days. The donor, the metered receivers, and the unmodified power
  are read from the dense half-hourly grid of experiment 86; the injected power adds the
  injected MW to it, and the lags are rebuilt from the injected power. Cases whose true power is
  missing, or inside a logged closure or an oracle-labelled event, are dropped. Experiment 82's
  stamps start on 2022-01-01 and experiment 86's grid on 2019-09-01, an offset of 40,896
  half-hours. The injected MW is divided by each series' `effective_capacity_mw`, and a printed
  check asserts that the panel's observed power over that capacity equals the dense grid wherever
  both exist.
- **Truth.** The unmodified observed power over effective capacity. Weights: the case's event's
  mean injected donor MW, so a large transfer counts for more (the MW-weighted rule of
  experiments 37 and 82); unweighted means are secondary.
- **Bins.** Family A by time since onset at the origin: 0-24 h, 1-3 d, over 3 d. Also by lead
  band (0-24 h, 1-2 d, 2-6 d, 6-10 d), step in noise units (donor step over experiment 82's
  noise scale, bins 2-4, 4-6, 6+), shape, duration band, and metered receiver or not.
- **Boosters.** Arms 1, 2, and 5 re-use experiment 86's boosters
  (`work/boosters/<arm>/<site>_s0_<refit>.ubj`, monthly refits from 2025-07-01, N1 recipe, 150
  rounds, learning rate 0.15, seed 0): `plain` and `lagmw_xl`. Arms 3 and 4 are new boosters on
  the same schedule and recipe as experiment 86: `nra_oracle` and `nra_xl` drop the training
  rows whose valid time lies inside an oracle (arm 3) or XL (arm 4) event, replace each lag by the
  lag minus the signed MW, and add the signed MW as three columns. Labels are the hindsight
  labels of experiment 86 (oracle: experiment 41; XL: experiment 82's `xl_intervals.parquet`).
  Arm 2 uses `lagmw_xl` so that arms 2 and 4 differ only in the training-target mask.
- **Forecast-time MW.** Two sources for arms 2 to 4. (i) Injected: the event's median injected MW
  on each series (negative on the donor), constant over the interval, as the oracle labels
  are; used for a lag whose valid time lies in the interval. (ii) Causal XL: experiment 82's
  XL detector (refit-offset step statistic, logistic score `models.pkl`, threshold from
  `thresholds.json`), run on the injected residuals (fast path of experiment 43: the stored
  residual plus the injected MW) with every value at or after the cut stamp (the origin minus 3.5
  hours) set to missing, and with only the 45 days before the cut visible. The functions are those
  that built experiment 82's `xl_intervals.parquet`, with four causal rules added. A candidate must
  be a local peak with the whole 48-half-hour after-window present, so a rising step cannot
  fire early on a partial window. A return step or level failure ends an interval only when it lies
  at least 48 half-hours before the cut, and the first-return search stops at the cut. Any other
  interval is open and runs to the cut. Partners follow the rule that built `xl_intervals.parquet`
  (an opposite-sign neighbour step above 2.0 noise units whose level holds at least half the step
  for 24 hours). The signed MW of an interval is its series' 24-hour step at the onset over
  capacity. Known mild leaks, stated rather than removed: the noise scales, the logistic score and
  its threshold were fitted on odd months of the whole record, including months after an origin,
  and the injected power does not enter the monthly residual-model refits (the fast path). A
  property test perturbs every residual value at or after the cut over many (event, origin)
  pairs and asserts that the detector output is unchanged.
- **Contrasts and bootstrap.** As in the decision rule. Family B is reported by hours since the
  end of the transfer. The share of the arm 1-to-ceiling gap
  that arms 2, 3, and 4 close is reported overall and by time since onset, lead band, step
  size, shape, and duration, with 95% intervals by the same resampling. The quantile score and
  the pinball loss at the 0.05 and 0.95 quantiles are reported beside the MAE. The stored boosters
  have 9 quantile levels, so the 0.01 and 0.99 tails cannot be scored.
- **False-alarm check (descriptive).** For 500 (series, origin) pairs drawn at random outside every
  injected interval and outside logged closures, arm 4 and arm 2 with causal XL MW minus arm 5.
  This is how much the NRA boosters raise the error when the causal detector raises an alarm that
  is not a transfer.
- **Second check: real events.** The conserved two-sided events of experiment 41 in the
  validation year with both sides at most half missing (about 22). For each side the NRA
  estimate is the metered power minus the labelled signed MW of the other side of the event
  (the donor's estimate adds the receiver's step, and vice versa); inside the event the target is
  this estimate and outside it the metered power. Arm 1 and arm 4 (causal XL MW, with the oracle
  MW beside it) are scored on family A origins with the same MAE. This truth is itself an
  estimate.
- **Schedule.** The cheap monthly P-roll schedule of experiment 49: a refit on the 1st of each
  month, 365-day window, 150 rounds, eta 0.15, recency half-life 0.5 years, seed 0. The contrast
  is not about retraining cadence, so a weekly schedule is not used. The two new booster arms
  need 12 refits x 16 Primaries x 2 arms = 384 fits, on the slot's GPU share (12 GB) and 8 cores.
- **Compute.** The causal detector runs once per (batch of events, shape, origin), on arrays
  restricted to the 45 days before the cut. The pilot measures seconds per detector run and per
  booster prediction (about 480 valid times by 51 members per case and arm) and the plan's event
  count is fixed from them before scoring. Worker count times threads per worker stays within 8
  cores. Each worker handles one batch of events (disjoint series within 7 days) and writes its
  output to its own file. Scoring waits until experiment 86 has written all its boosters.

## Process

1. Sonnet writes this plan; one Sonnet reviewer reads it; findings are triaged below.
2. Sonnet implements. One Sonnet code reviewer checks for leakage (nothing at or after the
   origin minus 3.5 hours reaches the causal detector or the lags; the booster for a case is the
   latest refit at or before its origin; training masks use hindsight labels, which is allowed).
3. Sonnet runs and writes up. A Sonnet reviewer checks the write-up against the printed report,
   and a second Sonnet reviewer runs the prose review.
4. The PR merges under the skill's "Merging" rules.

## Plan review

A Sonnet reviewer found these, all accepted unless noted.

- Rising-step peaks fire early on a partial window: fixed by requiring a full after-window.
- "Open interval" was undefined on truncated data: fixed by the explicit end rule above.
- The first-return search could fix an end early: ends are accepted only 48 half-hours before the
  cut.
- Family B is hurt for about a day after the end: predicted, and reported by hours since the end.
- Fitted constants are a mild leak: stated. Partner rule: made explicit. Grid and capacity offsets:
  stated and asserted.
- Six month blocks: per-month signs are now part of the verdict. Bonferroni now covers four
  contrasts, adding the lag correction alone (C4).
- The 20% threshold is also reported for the 1-3 d bin. A share of the gap is reported only where
  the gap's interval excludes zero.
- Compute estimate and a pilot are added; the 73 every-fifth-day runs are dropped.
- Rejected: none.
