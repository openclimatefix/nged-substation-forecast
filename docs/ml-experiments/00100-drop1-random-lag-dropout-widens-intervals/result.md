---
id: "00100"
slug: drop1-random-lag-dropout-widens-intervals
status: planned
theme: calibration-and-uncertainty
also_relevant_to:
  - switching-labels-in-the-forecast
  - lagged-power-and-error-correction
size: complex
hypothesis: >-
  An XGBoost quantile model for Primary substation demand that is trained with labelled switching
  half-hours removed from its targets and with its lagged power inputs set to missing at random
  intervals, and that has those lags set to missing when they lie inside a labelled switching event,
  widens its 5% to 95% band on the half-hours whose lags are missing, so that the band's coverage
  moves from 0.784 towards the nominal 0.90, without widening the band elsewhere.
prediction: >-
  On the 37,394 real half-hours with valid truth and a lag inside a labelled event, the model trained
  with random dropout at a 10% row share (`drop10`) raises the coverage of the 5% to 95% band by 0.03
  to 0.10 over the plain model refitted at 11 quantile levels (`plain11`). On experiment 88's held-out
  injected transfers with the lags masked by the injected interval, it raises the coverage by 0.02
  to 0.08 and lowers the quantile score by 0.05 to 0.40 points of capacity. On real half-hours without
  a masked lag, the quantile score changes by under 0.02 points and the band widens by under 3%.
decision_rule: >-
  Four contrasts, `drop10` minus `plain11`, each read at 98.75% (Bonferroni across four), with whole
  months resampled (injected cases: the month of the transfer's onset), 20,000 resamples, paired on
  series, run, and valid time. P1: change in 5% to 95% coverage on the real masked rows. P2: the same
  on the injected family A rows. G1: change in quantile score on the injected family A rows, guard
  with an upper end of at most +0.02 points. G2: change in quantile score on the real unmasked rows,
  guard with an upper end of at most +0.02 points and a point widening ratio of at most 1.03.
  `supported` when the lower ends of P1 and P2 exceed zero, G1 and G2 pass, and `drop10`'s coverage
  on both kinds of masked rows is at most 0.95. `small gain` when P2's lower end exceeds zero, G1 and
  G2 pass, and P1's interval includes zero. `refuted` when P2's upper end is below zero or G1's lower
  end is above +0.02. `not-for-production` when G2 fails. `inconclusive` otherwise. The threshold is
  zero because random dropout lives only in the training loop and adds nothing to the serving path.
  Coverage is not a proper score, so every coverage gain is read beside the quantile score. All
  other reads are descriptive.
benchmark: >-
  `plain11`, experiment 86's `plain` recipe refitted at 11 quantile levels (q01 to q99), which must
  reproduce experiment 86's stored nine-level `plain` quantile score within 0.01 points. The
  `manual_heuristic` and `climatology` rows of experiment 48 are descriptive references.
row_set_id: fold_a_daily_gap_v1
metric: >-
  Coverage and mean width of the 5% to 95% and 1% to 99% bands, pinball loss at q01, q05, q95, and
  q99, and the quantile score in points of capacity, on rows with valid truth.
code: ml-experiments/00100-drop1-random-lag-dropout-widens-intervals/
sizing:
  stored: Boosters and predictions under data/ml-experiments/00100-*/; nothing in a Delta table.
  serving_path: >-
    No. Random dropout is a training-loop change; test-time masking needs an event label on the
    serving path, which is a later decision.
  degradation_rule: No.
  more_than_one_design: >-
    Yes (augmentation or replacement, per-lag or interval masks, rate), so the design is fixed in
    `opus_design.md`.
  callers_not_nameable: No; the scripts copy experiments 89 and 96 and import reviewed packages only.
---

# Experiment 100: does random lag dropout make the demand forecast widen its band when a lag is missing?

## Plan (written before any code ran)

- **Question.** [Experiment
  89](../00089-sw2-22-longest-history-nra-forecast-with-ara-lags-masked/result.md) set lags inside a
  union of detector and logged events to missing and made the forecast worse. [Experiment
  86](../00086-sw2-12-detector-labels-for-nra-forecast/result.md) showed that on the 37,394 rows with
  valid truth and a lag inside a logged event, the 5% to 95% band covers only 0.784 of actuals. We
  ask whether training with lags missing at random intervals teaches the model to widen its band when
  a lag is missing, instead of treating a missing lag as an extreme value.
- **Design.** An Opus design review is committed beside this file as `opus_design.md` (in
  `ml-experiments/00100-drop1-random-lag-dropout-widens-intervals/`). It departs from the
  maintainer's brief in two ways, both accepted: lags are set to missing in place (not by adding
  masked copies), so every dropout arm has the same rows and weights as `mask`; and the dropout masks
  whole pseudo-event intervals with durations drawn from the 82 labelled events (not each lag
  independently), so the missing-lag patterns match real events. The fourth arm is a 30% dose check
  rather than a group-dropout arm, because intervals already mask lags in groups.
- **Arms** (seed 0, 12-month rolling window, monthly refits, 150 rounds, eta 0.15, 11 quantile
  levels, spread factor 1.15). `plain11`: all rows, no masking. `mask`: targets inside a labelled
  event or logged closure dropped, lags inside a labelled event missing in training and test.
  `drop10`: `mask` plus pseudo-event dropout touching 10% of training rows. `drop30`: the same at
  30%, whose pseudo events include those of `drop10`.
- **Rows with valid truth.** Real rows whose target lies outside every labelled event and logged
  closure. Of these, the 37,394 with a lag inside a labelled event are the masked rows (headline);
  the rest are the unmasked rows. Real rows whose target lies inside an event or closure have no
  valid normal-arrangement truth and are never scored. Experiment 88's family A injected cases
  (truth: the unmodified series) cover the cases real data cannot validate, including a target
  inside the transfer. Point accuracy during a real abnormal arrangement is not a metric.
- **Interval behaviour is the primary read:** coverage and width of the 5% to 95% and 1% to 99%
  bands on masked and unmasked rows, the widening ratio against `plain11` on the same rows, and the
  quantile score as the proper score beside them.
- **Sizing (the five `plan-issue` triggers).** Stored: private boosters and predictions only.
  Serving path: not touched. Degradation rule: not touched. More than one defensible design: yes, so
  the design review fixed it. Callers not nameable without searching: no. Size: complex, with Sonnet
  reviewers only (the brief), and the Opus design review above.
- **Earlier advice followed.** Judge lag treatments on injected events of known size (experiments 88
  and 89); report tails at q01 to q99; do not mask on an over-marking detector label (experiment
  89). **Departure:** none.
- **Schedule.** Monthly P-roll refits; the contrast is about training-time exposure, and could depend
  on the schedule only through the number of refits in which pseudo events fall.
- **Lab conditions.** Hindsight event labels at training and test, oracle masking on injected cases.
  Causal RO-NC masks on injected cases are descriptive.
