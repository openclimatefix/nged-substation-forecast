---
id: '00301'
slug: metered-farm-outage-state-band
title: >-
  Does an availability state improve the solar-farm forecast band during and after outages?
theme: metered-solar-farms
also_relevant_to:
- calibration-and-uncertainty
status: planned
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
  Stop rule, checked first. If outage half-hours number under 200 at 4 or more of the 6 farms, report
  that outages are too rare to score and stop. Otherwise score the lower-tail pinball loss on outage
  half-hours with a month-block interval; the threshold is a 10% fall, because a state model adds a
  pipeline step. If the all-rows score worsens by 1% or more, report that the state model costs too
  much and stop.
benchmark: >-
  The spread-binned plant with its additive band (experiments 270 and 287)
row_set_id: >-
  Experiment 224's scored-year rows at 12 to 36 hour leads (daylight half-hours, farms A to F,
  2025-09 to 2026-08)
metric: >-
  Pinball loss at 0.01 and 0.05 on outage half-hours, and quantile score on all half-hours
code: ml-experiments/00301-metered-farm-outage-state-band/
---

Planned. The stop rule's count (`e00301_count.py`) was run before this plan was committed, to decide
whether the full experiment is worth running.
