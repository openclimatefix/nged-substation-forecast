---
id: "00182"
slug: switching-d8-disaggregated-farms-and-drops
theme: switching-detectors
also_relevant_to:
  - solar-disaggregation
status: planned
size: >-
  simple. Stored: private tables under data/ml-experiments/00182-switching-d8-disaggregated-farms-and-drops/
  only. Serving path: not touched. Degradation rule: not touched. Designs: one (a descriptive
  comparison). Callers: this folder's scripts only, which read the stored drops of experiment 159 and
  the raw power table.
hypothesis: >-
  D8 (Marsh Lane) is a disaggregated Primary: the metadata says the output of two solar farms was
  added back to its reading, and that both farms' analogues are not working. When an analogue fails,
  D8's reading loses that farm's output, so D8's multi-day drops (experiment 159) coincide with a
  stuck, zero, or missing reading at the farms, and the drop depth matches the farms' lost output.
prediction: >-
  At least 15 of the 29 drops coincide with a stuck, zero, or missing reading at one of the two
  farms, against at most 5 of 29 for random windows of the same lengths and time of day. Stop rule:
  under 8 of 29 means the farms do not explain the drops.
decision_rule: >-
  Descriptive. Count the drops with a farm anomaly (a window with under half its half-hours present,
  all zero, or a run of 6 or more identical positive readings) and compare with 1,000 random windows
  per drop, drawn from 2023-07-01 onwards at the same time of day. Also test whether the drop depth
  by hour of day follows clear-sky irradiance.
benchmark: Experiment 159 drops table (29 drops).
row_set_id: none (describes stored residuals and raw power, no forecast is scored)
metric: Drops with a farm anomaly out of 29, against random windows; correlation of depth by hour with clear-sky irradiance.
code: ml-experiments/00182-switching-d8-disaggregated-farms-and-drops/
---

# Experiment 182: do D8's multi-day drops coincide with failed analogues of its two disaggregated farms?

Planned. Written before the comparison was run.
