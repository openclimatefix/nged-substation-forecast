---
id: "00096"
slug: sw2-23-synthetic-switches-injected-into-training
status: planned
size: complex
hypothesis: >-
  A forecast of the normal running arrangement (NRA, the network as it is when no switching transfer
  is on) recovers the power a Primary substation would have drawn had no transfer happened better
  when its XGBoost model was also trained on many synthetic transfers than when it was trained on
  the real logged transfers only. In a synthetic transfer, known MW is added to the recorded power
  of a donor and its receivers, the half-hours of the transfer are dropped as training targets, the
  lags are corrected by the transfer's MW, and the MW is given as a column. The gain holds when the
  MW at forecast time comes from the causal refit-offset neighbour-count detector (RO-NC) or from
  the largest detector step, and not only when the MW is the injected truth.
prediction: >-
  With the injected MW at forecast time (the oracle), synthetic training lowers the error on
  held-out injected transfers (family A, forecast origin inside the transfer) by 0.00 to 0.15 points
  of capacity against real-only training, because the real-only booster already closes 83% of the
  plain-to-ceiling gap and little is left. With the RO-NC MW or the largest-step MW the change is
  between -0.10 and +0.10 points: training on exact MW does not teach the booster to cope with a MW
  that is 1.5 times too large, so the gain does not carry over. The experiment fails if the oracle
  gain is zero (the booster learns nothing from synthetic transfers) or if the largest-step MW gain
  is at least half the oracle gain (the synthetic transfers teach more than the oracle suggests).
decision_rule: >-
  Four planned contrasts on family A cases, shapes S2, S3, S4, and S6 pooled with equal weight, as
  in experiment 88. The measure is the MW-weighted mean absolute error of the median forecast
  against the true unmodified series, in points of capacity, over leads up to 10 days. Each
  contrast is read at 98.75% (Bonferroni across four) from 20,000 resamples of whole calendar months
  of the transfer's onset. C1: arm 2 (synthetic-trained, injected MW) minus arm 1 (real-trained,
  injected MW). C2: arm 3 (synthetic-trained, RO-NC MW) minus arm 1 with RO-NC MW. C3: arm 4
  (synthetic-trained, largest-step MW) minus arm 1 with largest-step MW. C4: arm 4 minus the plain
  booster (the total gain over doing nothing). The verdict is `supported` when the upper end of C3
  is below zero and C3's point value is negative in at least 3 of the 4 scored onset months;
  `small gain` when only C1 or C2 has an upper end below zero; `inconclusive` when no interval
  excludes zero; `refuted` when the lower end of C3 is above zero. The threshold is zero: the
  injection lives in the training loop and adds nothing to the serving path, so a change whose
  interval excludes zero is worth porting once the full NRA forecast is. C1 to C3 each pair an arm
  with its real-trained twin on the same cases and the same MW source. Family B, the time since
  onset, the lead band, the quantile score, and the tail calibration at q01 to q99 are descriptive.
benchmark: >-
  Arm 1, the booster trained on the real logged transfers only (experiment 88's `nra_oracle`,
  refitted at 11 quantile levels), scored with the same forecast-time MW as each synthetic-trained
  arm. The plain booster and the ceiling (plain booster on unmodified lags) of experiment 88 are
  the references for the share of the gap closed.
row_set_id: >-
  none (experiment 88's injected cases: its 839 transfers, five shapes, and 292 daily runs of
  fold_a_daily_gap_v1; not comparable with the leaderboard)
metric: >-
  MW-weighted mean absolute error of the median forecast against the true unmodified series (points
  of capacity), with the quantile score, pinball loss and coverage at q01, q05, q95, and q99 beside
  it
verdict: null
code: ml-experiments/00096-*/
---

# Experiment 96: do synthetic switches in the training data make the NRA forecast recover the normal arrangement better?

*Plan, written before any code ran.*

## Question and design

- **Question.** [Experiment 88](../00088-sw2-21-nra-forecast-recovers-normal-arrangement-during-and-after-injected-switches/result.md)
  showed that a booster trained on the real logged transfers recovers the normal arrangement during
  an injected transfer only when the MW is known (83% of the gap closed), because the real record
  holds about 56 transfer intervals in the training window. Does adding about three times as many
  synthetic transfers to the training history help, and does the help survive the detector's MW?
- **Sizing (the five `plan-issue` triggers).** What is stored: two sets of boosters (arms 1 and 2),
  a table of the synthetic training events, and per-case scores, all private under `data/`.
  Production serving path: not touched. Degradation rule: not touched. More than one defensible
  design: yes (how many events, which shapes, which months, which MW the booster is trained with),
  so the plan fixes them below. Code whose callers could not be named without searching: no, the
  scripts copy experiment 88's harness. Size: complex (new method and several designs), with Sonnet
  reviewers only, as the brief says.
- **Training injection.** Experiment 88's harness (`draw_events`, `inject_shaped`) draws synthetic
  transfers in the odd calendar months from 2024-07 to 2026-05, quiet on every series they touch
  (no logged closure within 7 days, no alarm within 2 days). The shapes S2, S3, S4, and S6 are
  assigned in turn. The number of series-intervals injected is 3 times the number of real labelled
  transfer intervals (experiment 41's labels) that start in the 365-day windows of the refits,
  counted per series; the script prints both counts. Test transfers (experiment 88's 839) start in
  even months only, so no training transfer shares a calendar month with a test transfer.
- **What the booster sees in training.** The booster's rows are experiment 88's: the ENS control
  member, a 365-day window, monthly refits (P-roll, 150 rounds, eta 0.15, half-life 0.5 years, seed
  0), no calendar columns beyond the existing derived set. The power the lags are read from is the
  recorded power plus the injected MW. Rows whose valid time lies inside a real (experiment 41) or
  injected transfer are dropped as training targets (the ARA, abnormal running arrangement, is
  masked). Each lag is replaced by the lag minus the signed MW at the lag time, and the signed MW
  at the three lag times is added as three columns. The signed MW is each event's median injected
  MW, constant over the event, summed over events, plus the real oracle MW. No other lag correction
  is used. This is experiment 88's `nra_oracle` recipe with the injected events added.
- **Arms.** Arm 1: `nra_oracle`, real transfers only, refitted at 11 quantile levels (q01, q05 to
  q95, q99) with the injected MW, the RO-NC MW, and the largest-step MW at forecast time. Arm 2:
  synthetic-trained, injected MW (oracle). Arm 3: the same boosters, RO-NC MW from the causal
  detector as built. Arm 4: the same boosters, largest-step MW (post hoc in experiment 88; a
  planned arm here). Arms 2 to 4 share boosters, so the training cost is two booster sets.
- **Scoring.** Experiment 88's script copied: the same 839 test transfers, shapes, origins,
  causal detector, and truth. Plain and ceiling come from experiment 88's stored cells. A check
  that arm 1's q05 to q95 reproduce the stored 9-level `nra_oracle` scores to 0.01 points guards
  the copy. Families A and B, bins by time since onset and lead band, the quantile score, and tail
  calibration (coverage and pinball at q01, q05, q95, q99) are reported.
- **Labels and lab conditions.** Training labels use hindsight (oracle events and injected MW),
  the detector's residual model uses ERA5, and the MW at forecast time in arm 2 is the injected
  truth. Nothing here runs live. The forecast needs the weather the lags are corrected with only
  through the lags themselves, so no ERA5 enters the booster.
- **Schedule.** Monthly refits on a rolling 365-day window, because the contrasts are about
  training data and not retraining cadence. The contrast could depend on the schedule only through
  the number of injected events inside each window, which the write-up reports.
- **Compute.** 2 arms x 12 refits x 16 Primaries = 384 fits at about 25 s, at most 4 GPU hours;
  scoring about 3 CPU hours. If the pilot shows over 4 GPU hours, arm 1's refit at 11 levels is
  dropped (stored 9-level scores are used and the tails are reported for arm 2 only) and the manager
  is told.
- **Earlier advice followed.** Experiment 88: use the largest-step MW and not the sum (arm 4);
  test with the true MW first (arm 2); do not expect target masking to add (arm 1 is masked
  already). Experiment 89: lag masking is worse and the longest history no better, so the window is
  365 days. Departures: none.

## Status

Planned; no code has run.
