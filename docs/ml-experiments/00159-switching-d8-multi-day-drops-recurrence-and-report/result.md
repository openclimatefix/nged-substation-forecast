---
id: "00159"
slug: switching-d8-multi-day-drops-recurrence-and-report
theme: switching-detectors
also_relevant_to:
  - switching-labels-in-the-forecast
status: done
size: >-
  simple. Stored: private tables under data/ml-experiments/00159-switching-d8-multi-day-drops-recurrence-and-report/
  only. Serving path: not touched. Degradation rule: not touched. Designs: one (a descriptive
  census of one series). Callers: this folder's scripts only, which read the stored residual panel
  of experiment 130 arm B and the logged closures of experiment 37.
hypothesis: >-
  Descriptive, with no hypothesis test beyond the prediction below. D8 is a Primary substation
  whose power drops by 3 to 5 MW for days at a time while the convex demand model's forecast does
  not follow. The experiment lists every such drop, asks whether the drops recur on a pattern, and
  asks whether a connected metered neighbour steps up by a similar amount at the same time (a load
  transfer) or does not (a customer load change or an outage).
prediction: >-
  Written before the table exists. D8 has 20 to 40 drops of 2 days or longer between 2022-01 and
  2026-06, nearly all after mid-2023. Fewer than a quarter of the drops have a connected metered
  neighbour that steps up by at least half the drop's depth within 2 hours of the drop's start, so
  most drops look like a customer load change or an outage and not like a transfer. The drops do
  not cluster on one weekday or hour of day. Of the 28 alarms that the refit-offset neighbour-count
  detector raised on D8 and that matched no logged closure, at least 20 fall within 6 hours of a
  drop's start or end.
decision_rule: >-
  Descriptive only: no verdict rule and no porting threshold. A drop is a run in which the 24-hour
  centred rolling median of the residual (metered power minus the model's expected power) is below
  -1.5 MW, with gaps of under 12 hours merged, lasting at least 2 days. Depth is the median
  expected-minus-metered MW during the drop. A neighbour (series 3, 16, and 17 in the
  experiment 130 adjacency) steps up when its 6-hour median power after the drop's start exceeds
  its 6-hour median before by at least half the drop's depth; the pre-registered counts use that
  rule and a sweep of 0.25 and 0.75. Recurrence is reported as weekday and hour-of-day histograms
  with chi-squared tests against uniform, the spacing between starts, and shares near bank
  holidays, by season, and by cold and mild week. Counts carry 95% Wilson intervals.
benchmark: >-
  Experiment 130 arm B alarms on D8 (28 unexplained alarms, 27 with a jump of over 3 MW in half an
  hour), from the false-alarm analysis.
row_set_id: >-
  none (describes stored residuals and alarms, no forecast is scored)
metric: >-
  Drops found, their start, end, depth, duration, weekday and hour, neighbour step share, alarms
  within 6 hours of the drop's start or end, and logged closures within 24 hours.
code: ml-experiments/00159-switching-d8-multi-day-drops-recurrence-and-report/
verdict: partly supported
best_arm: >-
  Not applicable (descriptive). 29 drops of 2 days or longer, median depth 2.7 MW (range 1.2 to
  5.4) and median duration 3.6 days (range 2.0 to 19.2), 28 of them since July 2023.
headline: >-
  29 drops. A connected neighbour steps up by half the depth at 8 of 29 (0.28; 0.15 to 0.46)
  against 0.35 at random times. 23 of 29 have a logged closure within 24 hours, against 0.40 for
  random windows. 2 of the 28 unexplained RO-NC alarms fall within 6 hours of a drop's start or
  end.
lessons: >-
  The drops do not look like transfers to D3, D14, or D5, they follow no weekday, hour, bank
  holiday, or season, and the unexplained RO-NC alarms on D8 are a different event from the drops.
take_home_message: >-
  D8's multi-day drops are 29 episodes since 2022, nearly all since mid-2023, in which D8 runs about
  2.7 MW below the demand model for 2 to 19 days. No connected metered neighbour steps up with them
  more often than at random times, so they do not look like load moved onto D3, D14, or D5, and
  they follow no weekday, hour, bank holiday, or season. The RO-NC alarms on D8 are a different
  event, because only 10 of the 52 alarms fall within 6 hours of a drop's start or end. The next
  step is to ask NGED what moved, starting with the 6 drops that have no logged closure.
advice:
  - >-
    Treat D8's multi-day drops as their own event type, separate from the 28 unexplained RO-NC
    alarms, because only 2 of those 28 fall within 6 hours of a drop's start or end (alarms and
    drops).
  - >-
    Do not read a drop on D8 as a transfer to D3, D14, or D5 unless a neighbour's own step shows
    it: the best neighbour step-up is at or below the chance rate at all three thresholds (neighbour
    check).
  - >-
    Ask NGED what D8's switching and load changes are before choosing a mask for the drops, because
    23 of 29 drops have a logged closure within 24 hours against 0.40 at random times, and 14 of 29
    still do once the long-running job 21D5583 is set aside (closures).
production_status: not-for-production
adopted_in: none
---

# Experiment 159: what are D8's multi-day drops, do they recur, and do they look like transfers?

**D8's multi-day drops are 29 episodes in which its power runs about 2.7 MW below the demand model
for 2 to 19 days. No connected neighbour steps up with them more than it does at random times.** D8
is a Primary substation. The false-alarm analysis of the refit-offset neighbour-count detector
(RO-NC) found that D8's power falls for days at a time while the forecast does
not follow. We listed every such drop in the convex demand model's residual of [experiment
130](../00130-ro-nc-detector-on-the-gam-k-residual/result.md), described when the drops happen, and
checked whether one of the three connected metered neighbours (D3, D14, and D5) steps up by a
similar amount at the same time, which would make a drop a load transfer. A neighbour does not step
up more often than at random times (8 of 29 drops against 0.35 of random times), so the drops look
like a customer load change, an outage, or a move to an unmetered substation. The drops follow no
weekday, hour of day, bank holiday, or season. 23 of the 29 have a logged closure of D8 within 24
hours, against 0.40 for random windows of the same lengths, so switching is probably involved. The
written prediction was half right, so the verdict is `partly supported`. This page is descriptive,
and all series are labelled D8 and its neighbours.

![D8 power with every drop shaded](figure_1_history.svg)

![Three shallower drops](figure_2_typical_drops_a.svg)

![Three deeper drops](figure_3_typical_drops_b.svg)

![Recurrence of the drops](figure_4_recurrence.svg)

![Depth and duration over time](figure_5_depth.svg)

![Neighbour step-up at a drop against chance](figure_6_neighbours.svg)

![Unexplained alarms against the drops](figure_7_alarms.svg)

## Results

- **Drops:** 29 runs of 2 days or longer in which the 24-hour median of the residual (metered power
  minus the convex model's expected power) is below -1.5 MW, with gaps under 12 hours merged. 148 of
  the 1,642 days (0.090) lie inside a drop. 28 of the 29 start on or after 2023-07-01, by year 1, 3,
  7, 15, and 3 from 2022 to 2026. The sensitivity to the threshold is large: -1.0 MW gives 48 drops,
  -2.0 MW gives 24, and -2.5 MW gives 15.
- **Depth and duration:** median depth 2.71 MW (range 1.19 to 5.36), and median duration 3.6 days
  (range 2.0 to 19.2). Inside drops the median metered power is 3.43 MW against an expected 6.20 MW,
  and 0.435 of half-hours are below 3 MW, against 0.158 outside drops after the first. The depth is
  not constant: its Spearman correlation with the start time is -0.40 (p = 0.032), because the four
  drops of October 2023 to January 2024 are the deepest (4.9 to 5.4 MW).
- **What the power does inside a drop:** the metered power alternates between the expected level and
  0 to 3 MW within each day, as in drop 20, which lasts 19 days. A drop is a run in which those low
  periods dominate, not a steady 3 MW shift.
- **Recurrence:** no weekday, hour, or interval pattern. Starts by weekday Monday to Sunday are 3,
  5, 7, 3, 1, 5, 5 (chi-squared p = 0.48), and by 6-hour block of the day 11, 3, 5, 10 (p = 0.10).
  The spacing between starts has a median of 16.4 days (quartiles 10.3 and 42.3). 1 of 29 starts
  falls within 1 day of a bank holiday (0.03; 0.01 to 0.17), against 0.069 of all days. By calendar
  month Jan to Dec the starts are 5, 0, 0, 0, 3, 6, 2, 4, 2, 3, 1, 3 (p = 0.12), and the mean
  temperature during a drop is 0.55 C below the month's mean (95% interval -1.61 to +0.49).
- **Neighbour check:** the best step-up among D3, D14, and D5 in the 6 hours after a drop's start is
  at least 0.25, 0.50, and 0.75 times the depth for 11 (0.38; 0.23 to 0.56), 8 (0.28; 0.15 to 0.46),
  and 5 (0.17; 0.08 to 0.35) of the 29 drops. At 5,000 random times away from any drop the rates are
  0.51, 0.35, and 0.21. A neighbour steps down by at least half the depth at the end of 13 of 29
  drops (0.45; 0.28 to 0.62). The rate at which a neighbour steps down at the end is the only sign
  of a shared cause, and no placebo for it was run.
- **Logged closures:** 23 of 29 drops (0.79; 0.62 to 0.90) have a logged closure of D8 within 24
  hours, against 0.399 for 2,000 random windows of the same lengths away from any drop. 14 of 29
  (0.48; 0.31 to 0.66) still have one when the long-running job 21D5583 (a closure between D8 and
  D14 that overlaps 12 of the drops of 2025) is set aside. 6 drops have no logged closure.
- **What the alarms did:** RO-NC raised 52 alarms on D8, of which 28 match no logged closure. Of
  those 28, 27 (0.96; 0.82 to 0.99) have a half-hour jump of at least 0.3 times the capacity (3.11
  MW) within 24 hours, and 16 step down and 12 step up. 5 of the 28 (0.18; 0.08 to 0.36) fall inside
  a drop, 0 within 6 hours of a drop's start, and 2 within 6 hours of its end. Of all 52 alarms, 10
  fall within 6 hours of a drop's start or end, and 4 of the 29 drops have an alarm within 6 hours
  of their start. The alarms respond to D8's within-day jumps, of which there are 2,642 of 78,736
  half-hours in all history, and not to the drops.
- **Prediction:** the count (20 to 40 drops, nearly all after mid-2023) and the lack of a weekday or
  hour pattern were right. The neighbour rate (below 0.25 for half the depth) was missed narrowly,
  at 0.28, and the comparison with chance is what matters. The alarm prediction (at least 20 of 28
  within 6 hours) was wrong, at 2 of 28.

## Checks

- **Same inputs as before:** the 28 unexplained alarms, the 27 of 28 with a jump, and the 78 logged
  closures touching D8 reproduce the false-alarm analysis.
- **Forecast checks:** no forecast is scored, so the checks on the median and mean forecast, the
  tail coverage, and the curtailment cost do not apply.
- **Audit triggers:** one of the seven fires. The result contradicts the written prediction for the
  alarms (2 of 28 against at least 20). The page tests an explanation: the alarms sit in 2023 and
  2024 while 22 of 29 drops start in 2025 or later (figure 7), and the alarms match D8's within-day
  jumps (27 of 28). The other six triggers do not apply, because the page has no control, no nested
  arm, no planned contrast, and no sister experiment.

## Caveats

- **The definition of a drop is ours.** The count changes from 15 to 48 as the threshold moves from
  -2.5 to -1.0 MW, and a drop's start and end can be off by up to 12 hours, because of the 24-hour
  smoothing.
- **The residual is the lab residual.** It comes from the frozen convex model of experiment 130,
  refitted on the 1st and 16th of each month, with ERA5 weather and the real transfers masked from
  training. A drop that the model partly follows is under-counted.
- **The neighbour rate is a crude test.** It compares 6-hour medians, so it cannot see a transfer
  that is spread over a day, and the placebo includes the daily rise of the neighbours' demand,
  which is why it is high. D5 carries about 3 MW, so it can absorb only part of a 5 MW drop.
- **Closures match more than transfers.** The logged closures touching D8 number 78 in 4.5 years,
  and the random-window rate of 0.399 shows that a closure within 24 hours is common. The match does
  not show that a closure caused a drop.
- **Not done:** NGED has not confirmed any drop, and no cause was tested beyond the neighbours, the
  closures, and the calendar.

## What to try next

Ask NGED what D8 switches and which customers it serves, listing the 6 drops with no logged closure
first. Whether a drop should be masked from training, or shown to NGED's operators as an event,
follows from the answer.

## Every drop

| Drop | Start | End | Depth (MW) | Duration (days) | Start weekday and hour | Best neighbour step (MW) | Logged closures within 24 hours |
|---|---|---|---|---|---|---|---|
| 1 | 2022-08-09 | 2022-08-23 | 2.0 | 14.2 | Tue 16:00 | +2.2 (D5) | 1 |
| 2 | 2023-10-21 | 2023-10-25 | 4.9 | 4.3 | Sat 11:30 | +1.4 (D3) | 0 |
| 3 | 2023-10-26 | 2023-10-28 | 5.0 | 2.2 | Thu 18:30 | -0.1 (D5) | 0 |
| 4 | 2023-11-29 | 2023-12-02 | 5.0 | 3.9 | Wed 03:30 | +0.5 (D3) | 1 |
| 5 | 2024-01-17 | 2024-01-19 | 5.4 | 2.2 | Wed 17:00 | +1.6 (D14) | 0 |
| 6 | 2024-05-27 | 2024-05-30 | 3.6 | 2.2 | Mon 21:30 | +0.3 (D5) | 0 |
| 7 | 2024-06-12 | 2024-06-15 | 3.8 | 3.8 | Wed 03:00 | -0.2 (D5) | 1 |
| 8 | 2024-06-16 | 2024-06-24 | 1.8 | 7.8 | Sun 18:30 | +1.5 (D5) | 0 |
| 9 | 2024-08-28 | 2024-08-30 | 3.5 | 2.0 | Wed 01:00 | -0.9 (D5) | 2 |
| 10 | 2024-09-30 | 2024-10-02 | 3.2 | 2.5 | Mon 10:30 | +4.0 (D3) | 1 |
| 11 | 2024-12-10 | 2024-12-13 | 2.9 | 2.9 | Tue 22:30 | -1.4 (D5) | 4 |
| 12 | 2025-01-19 | 2025-01-23 | 3.0 | 3.3 | Sun 21:00 | -1.3 (D5) | 1 |
| 13 | 2025-05-07 | 2025-05-10 | 2.3 | 3.3 | Wed 05:00 | +1.3 (D3) | 2 |
| 14 | 2025-05-18 | 2025-05-23 | 2.7 | 4.3 | Sun 21:30 | -0.4 (D5) | 1 |
| 15 | 2025-06-05 | 2025-06-07 | 1.6 | 2.1 | Thu 12:30 | +0.2 (D3) | 1 |
| 16 | 2025-06-13 | 2025-06-15 | 1.4 | 2.0 | Fri 19:30 | +2.1 (D5) | 1 |
| 17 | 2025-06-17 | 2025-06-21 | 3.0 | 3.6 | Tue 12:00 | -0.2 (D3) | 1 |
| 18 | 2025-06-28 | 2025-07-01 | 1.2 | 2.9 | Sat 22:30 | +0.1 (D5) | 1 |
| 19 | 2025-07-05 | 2025-07-13 | 2.4 | 8.3 | Sat 16:30 | +1.4 (D5) | 1 |
| 20 | 2025-07-16 | 2025-08-04 | 2.4 | 19.2 | Wed 04:00 | +1.7 (D14) | 2 |
| 21 | 2025-08-06 | 2025-08-15 | 1.9 | 9.4 | Wed 05:00 | +0.7 (D3) | 3 |
| 22 | 2025-08-16 | 2025-08-30 | 2.3 | 14.2 | Sat 04:00 | +0.0 (D3) | 1 |
| 23 | 2025-09-23 | 2025-09-25 | 2.1 | 2.6 | Tue 01:00 | -0.6 (D5) | 1 |
| 24 | 2025-10-11 | 2025-10-16 | 2.4 | 4.2 | Sat 20:30 | -0.3 (D5) | 0 |
| 25 | 2025-12-15 | 2025-12-22 | 3.2 | 7.0 | Mon 10:00 | +2.8 (D3) | 1 |
| 26 | 2025-12-28 | 2025-12-31 | 2.9 | 3.2 | Sun 02:30 | -0.4 (D5) | 1 |
| 27 | 2026-01-11 | 2026-01-13 | 2.1 | 2.1 | Sun 22:00 | -1.4 (D5) | 1 |
| 28 | 2026-01-15 | 2026-01-18 | 3.1 | 3.6 | Thu 01:00 | -0.4 (D5) | 5 |
| 29 | 2026-01-27 | 2026-01-31 | 2.7 | 4.8 | Tue 00:30 | -0.7 (D5) | 1 |

Code: `ml-experiments/00159-switching-d8-multi-day-drops-recurrence-and-report/`; every number above
is printed by `e00159_drops.py` into `printed_report.txt` in that folder, and `e00159_charts.py`
draws the figures.
