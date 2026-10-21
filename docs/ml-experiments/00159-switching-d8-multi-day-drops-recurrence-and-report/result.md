---
id: "00159"
slug: switching-d8-multi-day-drops-recurrence-and-report
theme: switching-detectors
also_relevant_to:
  - switching-labels-in-the-forecast
status: planned
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
---

# Experiment 159: what are D8's multi-day drops, do they recur, and do they look like transfers?

Planned; the write-up replaces this section when the experiment finishes.
