---
id: "00339"
slug: wind-research-default-1-adapter-plus-mixture-head
title: >-
  Does a mixture-density head on the wind adapter's quantiles improve Wind farms research default 1
  once the idle half-hours are handled?
theme: generator-forecasting
also_relevant_to:
  - calibration-and-uncertainty
  - curtailment-cost
status: planned
kind: change
date: 2026-10-24
size: >-
  complex by the research-default rule. Stored: head outputs and tables under
  data/ml-experiments/00339-wind-research-default-1-adapter-plus-mixture-head/ only, plus the
  forecasts of the adopted arm if there is one. Serving path: not touched, but arm x2 would add a
  rule on the ensemble-mean wind speed and a floor value per farm, and every head arm would add a
  small neural network to serving; the page says so and does not do either. Degradation rule: not
  touched. Designs: several are defensible for handling the idle half-hours (rule, training set),
  so the plan fixes three arms and the order before any score is read. Callers: this folder's
  scripts, which import experiments 324, 326, 336, and 303 and call no reviewed module beyond
  `studies.*`. One Sonnet code review and one Sonnet prose review (maintainer's brief), no Opus.
hypothesis: >-
  A three-component Gaussian mixture head on the hindcast, fed the quantiles of the physics core
  plus adapter and the ensemble members' summaries of speed, direction, and temperature (experiment
  324's W4), lowers the quantile score of Wind farms research default 1 on the kept rows of the
  three metered wind farms W1 to W3 at 1 to 10 day leads. The head loses on the idle half-hours
  because it is trained on kept rows only, and the maintainer's view is that a forecast-time rule
  (predict the idle floor where the ensemble-mean wind speed is below cut-in) needs no idle
  detector. Three arms are tested: x1 the head as in experiment 324, x2 x1 plus the rule, x3 the
  head trained on all hindcast half-hours.
prediction: >-
  Written before any score. x1 keeps at least 0.2 of its 0.27 gain on the kept rows in the
  replicate period. x2 removes at least half of x1's loss on the all-half-hours rows (the +0.208
  from the idle rows) but recovers under a third of it, because the 210 mask covers idle readings
  at higher wind that the cut-in rule cannot reach. x3 loses under 0.1 on the kept rows against x1
  and gains on the all-half-hours rows. None of the arms fixes the upper tail (the share above q99
  stays above 1.8%). The head is no better than the default for a farm with no history
  (leave-one-farm-out).
decision_rule: >-
  The five adoption conditions of the research-default page, read in the order x2, x3, x1, and the
  first arm that passes all five is adopted (at most one). Condition 1, gain: the 99.67% month-block
  interval of (default plus x, minus default) on the kept-row quantile score, scored at 1 to 10
  days, lies wholly below zero (level 1 - 0.01/3 for three arms), and the point estimate is at
  least 0.03 points (a new head, minimum gain 0.03). Condition 2, no harm: the one-sided 95% upper
  bound of the harm to the pooled 0.01 and 0.99 pinball losses is at most 0.05 points each. The
  all-half-hours score is a reported group with tolerance 0.05 and no pass or fail; the page states
  whether each arm meets it. Condition 3, replication: heads trained on 2024-04 to 2024-12 and
  scored on the daily runs of 2025-01 to 2025-06 give the same sign, and a t test on the monthly
  paired differences does not contradict it. Condition 4: the reproduction gate, the shuffled-input
  placebo, the seed check, and the leakage check pass, and no audit trigger is unresolved.
  Condition 5: the page says what serving needs. A change that wins on condition 1 and fails
  another stays a candidate. The default plus the rule alone is descriptive and never adopted.
benchmark: >-
  The default: Wind farms research default 1, the ERA5-trained XGBoost on the 51 members plus
  experiment 259's offset table (experiment 303's P1, 11.352 points on the kept rows, as
  experiments 326 and 336 reproduced exactly). Also beside it: climatology, persistence on hours
  0 to 6, the default without the table, the physics core plus adapter alone and with a table, and
  the weather-only mixture network of experiment 313.
row_set_id: >-
  fold_a_tuesdays_v1 (53 Tuesday runs, 2025-07 to 2026-06; kept rows by experiment 210's mask are
  the headline, all half-hours beside). Replicate: the daily hindcast runs of 2025-01-01 to
  2025-06-16.
metric: >-
  Quantile score over 11 levels in points of capacity, scored at 1 to 10 days, with days 1 to 2 and
  days 3 to 10 separately; tolerance 0.02 for the headline, 0.05 for the all-half-hours rows and
  for the 0.01 and 0.99 pinball losses.
code: ml-experiments/00339-wind-research-default-1-adapter-plus-mixture-head/
---
**Plan.** Committed before any score is read.

- **Question.** Experiment 324's W4 scored 0.272 points better than the default on the kept rows
  but 2.1 points worse on the 10,642 idle half-hours, because its training set excluded them.
  Can a forecast-time rule (x2) or a wider training set (x3) keep the gain without the loss?
- **Family and default.** Wind farms research default 1 (the ERA5-trained XGBoost on the 51
  members plus experiment 259's offset table), stored frames from experiment 336 (reproduction
  gate: the refit equals experiment 303's P1 exactly, 11.352).
- **Arms.** x1: experiment 324's W4, retrained here with the same code. x2: x1, then at every
  half-hour where the ensemble-mean 100 m speed is below the farm's cut-in speed (experiment 336's
  definition and values, W1 2.69, W2 2.85, W3 3.38 m/s) all 11 quantiles are the farm's idle floor,
  the median observed power of experiment 210's idle-band rows in the hindcast training window.
  x3: the head trained on every hindcast half-hour with a finite observation. Descriptive: the
  default plus the same rule. Adoption order x2, x3, x1; at most one adopted.
- **Tier and gain.** Tier 2, a new head: minimum gain 0.03 points.
- **Standard architecture.** The best head arm is named "standard architecture: physics core +
  adapter + MDN head" in every table. The page reports its gap to the best single family
  (XGBoost, physics core plus adapter, or weather-only mixture network) in points and per cent,
  with an interval.
- **Controls.** A shuffled-input placebo (the head scores member summaries from another month of
  the same farm). Five head seeds on one base seed, with mean and spread. Early stopping on runs
  initialised before 2025-04-16 only (before 2024-12-01 in the replicate). A leakage check that
  the head reads ensemble weather and the physics core's output only, and the rule reads ensemble
  speed only. A leave-one-farm-out arm for the first head.
- **Replicate.** Heads trained on hindcast cases from 2024-04 to 2024-12 (early stopping on
  December), scored on 2025-01 to 2025-06, with the default's replicate frames from experiment 336.
- **Compute.** About 1 GPU hour; head fits only, the default and base quantiles are stored.
