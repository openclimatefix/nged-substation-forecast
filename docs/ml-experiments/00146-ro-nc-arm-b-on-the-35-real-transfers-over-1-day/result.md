---
id: "00146"
slug: ro-nc-arm-b-on-the-35-real-transfers-over-1-day
theme: switching-detectors
also_relevant_to: []
status: planned
size: >-
  simple. Stored: private files under data/ml-experiments/00146-ro-nc-arm-b-on-the-35-real-transfers-over-1-day/
  only. Serving path: not touched. Degradation rule: not touched. Designs: one (rescoring a
  detector that is already fitted). Callers: this folder's scripts only, which read experiment
  138's stored arm B tables and the residual panel of experiment 130.
hypothesis: >-
  Descriptive, with no hypothesis to test. Experiment 130 reported only the count of the 35 real
  transfers over 1 day that RO-NC finds on the frozen convex residual (18 of 35). This experiment
  reports, for each of the 35, whether and when the detector alarmed, how much of the moved power
  it covers, and why it missed the transfers it missed.
prediction: >-
  Written before the table exists. MW-weighted recall on the 35 transfers is 0.45 to 0.65, below the
  injected 0.794, because the real transfers over 1 day include many with a donor step under 4 noise
  units and without a metered receiving substation. Most misses (more than half) have a step under
  4 noise units at the donor or a missing reading.
decision_rule: >-
  Descriptive only, so there is no verdict rule and no porting threshold. Each miss is assigned to the
  first of these reasons that holds: no residual at the onset (reading missing); first step under 2
  noise units; first step of 2 to 4 noise units; first step of 4 noise units or more and the alarm
  score below the threshold. The write-up reports counts with 95% Wilson intervals, the
  MW-weighted recall with a 95% interval from resampling transfers 20,000 times, and the timing
  error of the alarm against the labelled start.
benchmark: >-
  Arm B of experiment 130, re-scored in experiment 138: 18 of 35 found.
row_set_id: >-
  none (scores stored detector alarms against the labelled real transfers of experiment 41, with no
  forecast)
metric: >-
  Transfers found out of 35, MW-weighted recall (weights are the labelled median moved MW), timing
  error of the alarm in hours, and the reason for each miss.
code: ml-experiments/00146-ro-nc-arm-b-on-the-35-real-transfers-over-1-day/
production_status: not-for-production
adopted_in: none
---

# Experiment 146: which of the 35 real transfers does RO-NC find, and why does it miss the rest?

**This page is the plan. The result is not in yet.** The refit-offset neighbour-count detector
(RO-NC) on the frozen convex residual found 18 of the 35 labelled real transfers lasting over 1 day
in [experiment 130](../00130-ro-nc-detector-on-the-gam-k-residual/result.md). This experiment lists
each of the 35 with its moved megawatts (MW), whether and when the detector alarmed, and the reason
for each miss. No detector is fitted or changed, and the verdict is descriptive.
