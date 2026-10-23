---
id: '00301'
slug: metered-farm-outage-state-band
title: >-
  Does an availability state improve the solar-farm forecast band during and after outages?
theme: metered-solar-farms
also_relevant_to:
- calibration-and-uncertainty
status: finished
size: >-
  medium. Stored: private tables and predictions under
  data/ml-experiments/00301-metered-farm-outage-state-band/ only. Serving path: not touched.
  Degradation rule: not touched. Designs: more than one defensible state model (hand-set rule,
  hidden Markov model, classifier), so all are arms. Callers: this folder's scripts, copied from
  experiments 270 and 287. Asset graph: not touched.
hypothesis: >-
  A small availability state (online, partly down, offline) estimated from the last 6 to 48 hours of
  a solar farm's power against its plant forecast improves the forecast band's lower tail during and
  after an outage.
prediction: >-
  The state model lowers the 0.01 and 0.05 pinball loss on outage-affected half-hours by at least
  10% against the spread-binned plant with its additive band (experiment 270), and changes the
  all-rows quantile score by under 1%.
decision_rule: >-
  Stop rule, checked first. If outage half-hours number under 200 at 4 or more of the 6 farms,
  report that outages are too rare to score and stop. Otherwise score the lower-tail pinball loss on
  outage half-hours with a month-block interval; the threshold is a 10% fall, because an
  availability-state model adds a pipeline step. If the all-rows quantile score worsens by 1% or
  more, report that and stop.
benchmark: >-
  The spread-binned plant with its additive band (experiments 270 and 287)
row_set_id: >-
  Experiment 224's scored-year rows at 12 to 36 hour leads (daylight half-hours, farms A to F,
  2025-09 to 2026-08)
metric: >-
  Pinball loss at 0.01 and 0.05 on outage half-hours, and quantile score on all half-hours
code: ml-experiments/00301-metered-farm-outage-state-band/
verdict: inconclusive
best_arm: >-
  None run. The stop rule fired before any arm was fitted.
headline: >-
  Outage half-hours in the scored year number 0 to 34 at five of the six farms (A, B, C, D, F) and
  165 at farm E, so all six farms are under the 200 half-hours the stop rule requires.
lessons: >-
  Full outages of a metered solar farm are rare in the scored year, and no farm has enough outage
  half-hours to score an availability-state model there.
take_home_message: >-
  Solar-farm outages are too rare in the scored year to test an availability state: five of the six
  farms have at most 34 outage half-hours, and the sixth has 165. Too few half-hours remain to
  separate a 10% gain from chance, so the experiment stopped before fitting anything. If outages
  matter, the next test needs a longer period, such as the full history of farms A, B, and F.
advice:
- >-
  Do not score an availability-state model on the metered farms' scored year, because outage
  half-hours number 0 to 34 at five of the six farms and 165 at the sixth (Results).
- >-
  Test any outage idea on a farm's full history, because the months before the scored year hold
  more low half-hours at farms A, B, C, and F (Results).
production_status: not-for-production
adopted_in: none
---

**Outages of the six metered solar farms are too rare in the scored year to test an availability
state, so the experiment stopped at its stop rule without fitting any model.** National Grid
Electricity Distribution (NGED) meters six solar farms, labelled A to F here. We asked whether a
farm's state (running, partly down, or switched off), inferred from its last 6 to 48 hours of power,
improves the lower quantiles (0.01 and 0.05) of the day-ahead forecast band during and after an
outage. The stop rule says to stop if fewer than 200 outage half-hours fall in the scored year
(2025-09 to 2026-08) at 4 or more farms. An outage half-hour is a daylight half-hour (sun more than 5
degrees above the horizon) in which the farm produced under 5% of the [physical power model's
forecast](../00287-metered-farm-spread-binned-plant-heads/result.md), when that forecast exceeded
15% of the farm's capacity, on a day with at least 6 such half-hours. No farm has 200: five have at
most 34, and farm E has 165.

## Results

- **Outage half-hours in the scored year (daylight, 12 to 36 h leads).** A 15 (2 days), B 14 (2
  days), C 0, D 34 (5 days), E 165 (15 days), F 10 (1 day). All six are under 200, so the stop rule
  fires.
- **A looser count gives the same answer.** Counting every low daylight half-hour, without the
  6-per-day requirement, gives 26 (A), 22 (B), 5 (C), 227 (D), 215 (E), and 21 (F). Farms A, B, C,
  and F stay under 200. Farms D and E pass 200 only because the loose count includes many short dips
  on 96 days (D) and 47 days (E), which the 6-per-day requirement removes.
- **Months before the scored year hold more low half-hours at some farms.** The loose count before
  2025-09 is 118 (A), 46 (B), 47 (C), 515 (D), 50 (E), and 78 (F).
- **Not run.** The availability-state arms, their pinball loss, coverage, false alarms, and the
  control that shifts the power by one week.
- **The count preceded the plan commit.** `e00301_count.py` ran before the plan was committed, so the
  stop rule acted as a pre-check and not as a pre-registered test.

## Checks

- **Forecast checks do not apply.** No forecast was made, so there is no median-minus-mean forecast
  and no mean-forecast error to read beside a quantile-score contrast.
- **No audit trigger fired.**

## Caveats and next steps

- **Label.** The outage label uses hindsight output against the plant forecast. A looser label finds
  more outage half-hours at farms D and E, and a stricter label finds fewer.
- **Next step.** If outages matter for curtailment, test the availability-state model on the full
  history of farms A, B, and F, scoring outage half-hours only with blocked folds. That test needs
  its own plan.
