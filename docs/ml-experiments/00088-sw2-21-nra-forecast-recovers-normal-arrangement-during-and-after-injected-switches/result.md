---
id: "00088"
slug: sw2-21-nra-forecast-recovers-normal-arrangement-during-and-after-injected-switches
status: planned
size: complex
hypothesis: >-
  When a Primary substation is inside a load transfer, or has just left one, an XGBoost demand
  model that was trained without the transfer's half-hours as targets, with the lags corrected by
  the transfer's signed MW and the MW as a column ("full NRA", from the normal running
  arrangement), forecasts the power the substation would have drawn had no transfer happened
  better than the plain model does. This holds when the MW comes from the XL detector of
  experiment 82 run causally on data up to the forecast origin, but by less than when the MW
  comes from the injected truth.
prediction: >-
  On injected transfers whose forecast origin falls inside the transfer (family A), the full NRA
  booster trained on XL labels, given causal XL MW, closes 20% to 45% of the gap in MW-weighted
  mean absolute error between the plain booster and the ceiling (the plain booster on lags with no
  transfer). It closes about 0% in the first 24 hours after onset, because the 24-hour step
  statistic needs 24 hours of data after the onset, and 40% to 70% from the second day. With the
  injected MW the oracle-trained booster closes 70% to 95%. Masking the training targets adds
  nothing to the lag correction: the XL-trained full NRA booster minus the lag-correction booster,
  both with causal XL MW, is between -2% and +2% of the plain booster's error (experiment 75
  found the mask changes the NRA-row score by under 0.03 points of capacity). On forecast origins
  with no transfer, a false causal XL alarm costs under 0.02 points of capacity.
decision_rule: >-
  Three planned contrasts on the family A cases (forecast origin inside the injected interval),
  shapes S2, S3, S4, and S6 pooled with equal weight (the shapes that the harness does not build
  by taking a share of the donor's profile, so the headline does not rest on the shape that
  matches the harness), MW-weighted mean absolute error of the median forecast against the true
  unmodified series, in points of capacity, over leads up to 10 days. Each contrast is read at
  98.33% (Bonferroni across three), from 20,000 resamples of whole calendar months of the
  transfer's onset. C1, arm 4 with causal XL MW minus arm 1 (what Flexpectation v2 could run).
  C2, arm 3 with injected MW minus arm 1 (the oracle). C3, arm 4 minus arm 2, both with causal XL
  MW (whether masking the targets adds to the lag correction). The verdict is `supported` when the
  upper end of C1 is below zero and C1's point value is at least 20% of the gap from arm 1 to the
  ceiling (arm 5); `small gain` when the upper end of C1 is below zero but the point value is
  under that share; `inconclusive` when C1's interval contains zero; `refuted` when its lower end
  is above zero. The threshold of 20% of the gap is set by the cost of porting: the full NRA
  forecast needs a causal detector, a second booster family, and a label table on the serving
  path of Flexpectation v2, so the gain has to be material in the 24-hour-to-3-day stretch where a
  causal detector can act at all. C2 and C3 are descriptive, as are the family B cases ("just
  after": the interval ends before the origin but within the lag window), the shape S1 (share of
  the donor's profile) and S5 cases, the false-alarm check, and the real-event check.
benchmark: >-
  Arm 1, the plain booster of experiment 86 on the contaminated lags, and arm 5, the same booster
  on the unmodified lags (the ceiling).
row_set_id: >-
  none (injected cases drawn from the 292 daily runs of fold_a_daily_gap_v1 and the 73 of
  fold_a_every5d_v1, so every daily run of the validation year; not comparable with the
  leaderboard)
metric: >-
  MW-weighted mean absolute error of the median forecast against the true unmodified series
  (points of capacity), with the quantile score, the pinball loss at the 0.05 and 0.95 quantiles,
  and 90% interval coverage beside it
code: ml-experiments/00088-sw2-21-nra-forecast-recovers-normal-arrangement-during-and-after-injected-switches/
---

# Experiment 88: does the full NRA forecast recover the normal running arrangement during and after a switch?

Plan only; the write-up replaces this page when the run finishes.

- **Question.** [Experiment 75](../00075-sw2-11-nra-forecast-with-labelled-switching-masks/result.md)
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
- **Cases.** The 365 daily runs of the validation year (origin 09:00 UTC). Family A: the origins
  from the onset (when the newest usable lag is inside the interval) up to 7 days after the onset
  and before the end. Family B: the origins from 3.5 hours after the end up to 7 days after the
  end (the lag window; a lag at least 7 days and 3.5 hours before the origin cannot touch the
  interval). A case is an (event, series, origin, valid time), for the donor and each metered
  receiver, with lead up to 10 days. The donor, the metered receivers, and the unmodified power
  are read from the dense half-hourly grid of experiment 86; the injected power adds the
  injected MW to it, and the lags are rebuilt from the injected power. Cases whose true power is
  missing, or inside a logged closure or an oracle-labelled event, are dropped.
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
  residual plus the injected MW) with every residual value at or after the origin minus 3.5
  hours set to missing. The candidates, scores, and thresholds are those of experiment 82,
  so no parameter is refitted. An interval is open when its tracked level has not failed and no
  return step has been seen; an open interval is assumed to continue to the origin. Its signed
  MW is the 24-hour step at its onset, divided by capacity; neighbours with an opposite-sign
  step of more than 2.0 noise units get the same interval and their own step, as in experiment
  86's XL labels. A causal run reads nothing at or after the origin minus 3.5 hours, which a
  printed check on one injected event confirms (the output is unchanged when the future is
  changed).
- **Contrasts and bootstrap.** As in the decision rule. The share of the arm 1-to-ceiling gap
  that arms 2, 3, and 4 close is reported overall and by time since onset, lead band, step
  size, shape, and duration, with 95% intervals by the same resampling. The quantile score and
  the pinball loss at the 0.05 and 0.95 quantiles are reported beside the MAE. The stored boosters
  have 9 quantile levels, so the 0.01 and 0.99 tails cannot be scored.
- **False-alarm check (descriptive).** For 500 (series, origin) pairs drawn at random outside every
  injected interval and outside logged closures, arm 4 and arm 2 with causal XL MW minus arm 5.
  This is what the NRA boosters cost when the causal detector raises an alarm that is not a
  transfer, or when no transfer exists.
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
- **Compute.** Worker count times threads per worker stays within 8 cores. Each worker handles one
  batch of events (disjoint series within 7 days) and writes its output to its own file.

## Process

1. Sonnet writes this plan; one Sonnet reviewer reads it; findings are triaged below.
2. Sonnet implements. One Sonnet code reviewer checks for leakage (nothing at or after the
   origin minus 3.5 hours reaches the causal detector or the lags; the booster for a case is the
   latest refit at or before its origin; training masks use hindsight labels, which is allowed).
3. Sonnet runs and writes up. A Sonnet reviewer checks the write-up against the printed report,
   and a second Sonnet reviewer runs the prose review.
4. The PR merges under the skill's "Merging" rules.

## Plan review

Pending.
