---
id: "00146"
slug: ro-nc-arm-b-on-the-35-real-transfers-over-1-day
theme: lsc-detectors
also_relevant_to: []
status: done
size: >-
  simple. Stored: private files under data/ml-experiments/00146-ro-nc-arm-b-on-the-35-real-transfers-over-1-day/
  only. Serving path: not touched. Degradation rule: not touched. Designs: one (rescoring a
  detector that is already fitted). Callers: this folder's scripts only, which read experiment
  138's stored arm B tables and the residual panel of experiment 130.
hypothesis: >-
  Descriptive, with no hypothesis to test beyond the prediction below. Experiment 130 reported only
  the count of the 35 real transfers over 1 day that RO-NC finds on the frozen convex residual (18
  of 35). This experiment reports, for each of the 35, whether and when the detector alarmed, how
  much of the moved power it covers, and why it missed the transfers it missed.
prediction: >-
  Written before the table exists. MW-weighted recall on the 35 transfers is 0.45 to 0.65, below the
  injected 0.794, because the real transfers over 1 day include many with a donor step under 4 noise
  units and without a metered receiving substation. Most misses (more than half) have a step under 4
  noise units at the donor or a missing reading.
decision_rule: >-
  Descriptive only, so there is no verdict rule and no porting threshold. Each miss is assigned to
  the first of these reasons that holds: no residual at the onset (reading missing); an alarm within
  2 hours of the labelled start whose interval ends too far from the labelled end; first step under
  2 noise units; first step of 2 to 4 noise units; first step of 4 noise units or more and no alarm
  within 2 hours of the labelled start. The write-up reports counts with 95% Wilson intervals, the
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
verdict: partly supported
best_arm: >-
  Arm B finds 18 of the 35 real transfers over 1 day (0.51; 95% Wilson interval 0.36 to 0.67), which
  carry 28.1 of the 42.7 labelled moved MW: MW-weighted recall 0.657 (0.466 to 0.806).
headline: >-
  18 of 35 found, MW-weighted recall 0.657 (0.466 to 0.806). Found alarms start within 0.5 hours of
  the labelled start. Of the 17 misses, 8 have a first step of 4 noise units or more but no alarm.
lessons: >-
  RO-NC finds the large transfers well and starts its alarms on time, but 8 of the 26 transfers with
  a first step of 4 noise units or more got no alarm at the start, and 2 more have an alarm at the
  start with an end 6.5 and 28.0 hours off.
take_home_message: >-
  On the 35 real transfers lasting over 1 day, RO-NC finds 18, and those 18 carry about two thirds
  (0.657) of the moved power. When it finds a transfer, its alarm starts within half an hour of the
  labelled start. Only 7 of the 17 misses are small or unreadable starts. 8 have a first step of 4
  noise units or more but no alarm, and 2 have an alarm with the wrong end. The next experiment
  should find out why those 8 got no alarm, before anyone builds detectors for weaker onsets.
advice:
  - >-
    Find out why 8 of the 17 missed real transfers, each with a first step of 4 noise units or
    more, got no alarm at the start, before building detectors for weaker onsets (results, reasons
    for the misses).
  - >-
    Quote the real-transfer recall as 18 of 35 (0.51; 0.36 to 0.67) and 0.657 (0.466 to 0.806) by
    moved MW, next to the injected-transfer recall of 0.794 from experiment 130 (results, found).
  - >-
    Expect the end of an alarm to be within 0.5 hours of the labelled end for 13 of the 18 found
    transfers but up to 11.5 hours off, so judge an end rule on its tail (results, end error).
production_status: not-for-production
adopted_in: none
---

# Experiment 146: which of the 35 real transfers does RO-NC find, and why does it miss the rest?

**The detector finds 18 of the 35 real load transfers lasting over 1 day, and those 18 carry about
two thirds of the power moved.** A load transfer is a switching operation that moves part of one
Primary substation's demand onto neighbouring substations. The refit-offset neighbour-count detector
(RO-NC) looks for such moves as sudden steps in the forecast errors (the residual) of a demand
model, helped by an opposite step at neighbouring substations. We ran RO-NC unchanged, on the
residual of the convex demand model of [experiment
130](../00130-ro-nc-detector-on-the-gam-k-residual/result.md) with its coefficients held fixed. We
then recorded, for each of the 35 real transfers lasting over 1 day that experiment 41 labelled from
the switching log, whether and when RO-NC raised an alarm. RO-NC found 18 (0.51; 95% Wilson interval
0.36 to 0.67). Those 18 carry 28.1 of the 42.7 labelled moved megawatts (MW), a share of 0.657 (95%
interval 0.466 to 0.806). The injected transfers of experiment 130, which are synthetic transfers
added to real power history, gave 0.794. Every found alarm starts within 0.5 hours of the labelled
start. Of the 17 misses, 8 have a first step of 4 noise units or more but no alarm within 2 hours of
the labelled start. A noise unit is the usual size of the detector's 24-hour step in odd-numbered
months, when no transfer is injected. The page is descriptive. Its written prediction was half
right, so the verdict is `partly supported`.

![Moved MW by outcome](figure_1_moved_mw_by_outcome.svg)

![First step against moved MW](figure_2_first_step_and_outcome.svg)

## Results

- **Found:** 18 of 35 (0.51; 0.36 to 0.67). MW-weighted recall 0.657 (0.466 to 0.806), by resampling
  transfers 20,000 times. All 35 have a labelled moved MW, 42.7 MW in total. The injected-transfer
  recall of experiment 130 is 0.794.
- **Reasons for the 17 misses (count, moved MW):** first step of 4 noise units or more with no alarm
  at the start, 8 (7.2 MW); first step of 2 to 4 noise units, 5 (4.0 MW); alarm at the start with an
  end too far off, 2 (2.2 MW); first step under 2 noise units, 1 (0.5 MW); no residual at the start,
  1 (0.7 MW). The first step is the largest 24-hour step within 2 hours of the labelled start.
- **Timing:** the alarm of each found transfer is within 0.5 hours of the labelled start (median
  0.00, range -0.50 to +0.50 hours; a negative value means the alarm came first).
- **End error of found transfers:** median 0.50 hours, largest 11.50 hours, and at most 0.5 hours
  for 13 of the 18. The two alarms with a wrong end are 6.5 and 28.0 hours off.
- **Metering:** 15 of 31 transfers with a metered receiving substation are found, and 3 of 4 without
  one. The prediction that missing a metered receiver explains the misses is not borne out.
- **Prediction:** the recall by MW was predicted at 0.45 to 0.65 and is 0.657, just above. The
  prediction that more than half of the misses have a first step under 4 noise units or an
  unreadable start was wrong: 7 of the 17 misses have such a start.
- **The 8 misses with a first step of 4 noise units or more:** they are 8 of the 26 transfers with
  such a step, and their steps (4.1 to 6.2) sit just above the alarm threshold. 5 have no alarm of
  any size within 48 hours of the start, and 3 have an alarm 3.5, 28.0, and 30.0 hours away. The
  experiment did not test why. One possibility is that the alarm score ranked them below the
  threshold, which was set at 80 false alarms on odd-numbered months.

## Checks

- **Same detector as before:** the table reads the stored alarms of arm B in experiment 138, which
  reproduce experiment 130 row for row, and it counts 18 found as experiment 130 did.
- **Forecast checks:** no forecast is scored, so the checks on the median and mean forecast do not
  apply.
- **Audit triggers:** none of the seven conditions that the `ml-experiment` skill lists for an audit
  holds, because the page has no control, no nested arm, and no planned contrast.

## Caveats

- **Thirty-five transfers,** so the intervals are wide. The convex model was trained with the real
  transfers masked, as in experiment 130.
- **Each miss has exactly one reason,** assigned in this order: no residual at the start, alarm at
  the start with the wrong end, first step under 2 noise units, 2 to 4, then 4 or more. A transfer
  with several causes shows only the first cause that applies.

## What to try next

Find out why the 8 first steps of 4 noise units or more got no alarm, for example by listing the
alarm score of the best candidate near each, before building more detectors for weak onsets.

Code: `ml-experiments/00146-ro-nc-arm-b-on-the-35-real-transfers-over-1-day/`; every number above
except the injected-transfer recall of 0.794 and the reproduction of experiment 130 is printed by
`e00146_rescore.py`.
