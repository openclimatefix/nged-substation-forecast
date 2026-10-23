---
id: '00299'
slug: summer-gap-timing-night-level-and-error-correction
title: >-
  When does the best-guess demand model fall behind BG2 in spring, what causes D10's night offset,
  and does the error-correction layer make the spring excess worse?
theme: production-reference-and-best-guess
also_relevant_to:
  - differentiable-physics-forecast-engine
  - differentiable-physics-embedded-generation
status: planned
size: >-
  medium, three zero-fit parts in one experiment. Stored: private files under
  data/ml-experiments/00299-summer-gap-timing-night-level-and-error-correction/ only. Serving path:
  not touched. Degradation rule: not touched. Designs: the timing and night-level parts have one
  defensible design (rescoring stored forecasts); the error-correction part has two (stored pairs
  versus a refit), and this plan pre-registers the stored pairs and records a refit as not done.
  Callers: this folder's scripts, which read experiment 266's master frame and the stored forecasts
  of experiments 111, 155, 200, 201, and 289 and call reviewed `studies.*` modules unchanged. No
  fits, no GPU, no downloads, 2 CPU cores, thread caps 1.
hypothesis: >-
  The April-to-June excess of arm D (the convex demand model with a Kalman filter on recent errors)
  over BG2 (the production-style XGBoost forecast) at the substations D5 and D9 is a step that
  starts at the March clock change and sits at short leads, which would point at a local-time
  mismatch or at BG2's use of recent power. D10's night-time level offset of about -2 points
  (21:00 to 07:30 UTC) is a step on one date, not a smooth drift with temperature. The Kalman
  filter and the correction head are net harmful at D5 and D9 in April to June.
prediction: >-
  Written before any stored forecast is cut by week, hour, lead, or date. Each line can fail.
  (Part 1, timing) At least half of D5's and D9's April-to-June daylight excess of arm D over BG2
  appears within 4 weeks of the last Sunday of March (2026-03-29); and at least half of it sits at
  leads under 24 hours. If neither holds, the excess is gradual and physical. (Part 3, D10 night
  level) The nightly mean residual of arm D at D10 changes in a single step inside a 2-week window,
  not smoothly with temperature; if smooth, the heating hinge or the night demand term is mis-set
  for spring, and the page names which. (Part 2, error correction) Removing the Kalman filter or
  the correction head lowers the April-to-June excess at D5 and D9 by at least 30% and changes July
  to September by under 0.02 points.
decision_rule: >-
  Intervals resample whole ISO weeks of forecast initialisation time (about 13 weeks in April to
  June), the same blocks for every contrast. Part 1: the timing prediction holds only if both
  halves hold at D5 and at D9 separately; one half holding is reported as partial. A step after the
  clock change is also read against the October 2025 change as a mirror test. Part 3: a single step
  is supported when a one-change-point fit on the nightly mean residual lowers the squared error
  by at least 50% of the total variance explained by a fit of the same residual on night
  temperature band and month, and the change point's 95% interval (block bootstrap over weeks) is
  under 2 weeks wide; a candidate date is reported as "inferred, unconfirmed". Part 2: supported
  when the lower bound of the 95% interval of the reduction is at or above 30% at both D5 and D9
  and the July-to-September change has an interval inside -0.02 to +0.02; refuted when the upper
  bound is below 30%; otherwise undecided. A filter that helps in every season is reported as a
  finding. The threshold of zero applies to porting any removal, because dropping a layer adds no
  production complexity; the 30% bar is for how much of the spring excess it must explain to be
  the cause.
benchmark: >-
  BG2 (experiment 87 p_s6_bg2) on the 561,207 rows of experiments 228, 251, and 266 (six
  substations, 2025-07-02 to 2026-06-30, 09:00 UTC runs, leads 0.5 to 168 hours, outside
  hand-filtered load shifts). Part 2's without-filter and without-head arms are scored on the rows
  they share with arm D after an exact key-equality gate, with the row count stated.
row_set_id: fold_a_daily_gap_v1
metric: >-
  Quantile score (twice the mean pinball loss over 11 quantiles, in points of effective capacity),
  the signed residual of the median in points, and the share of the April-to-June excess that each
  cut accounts for.
code: ml-experiments/00299-summer-gap-timing-night-level-and-error-correction/
---

Planned. The write-up replaces this body when the scripts have run.

## Plan

- **Part 1, the gap by week, hour, lead, and clock change.** Rescore experiment 266's stored arm D
  and BG2 (and experiment 289's realised-sunshine variants) at D5, D9, D10, D11, D14, and D16 by
  ISO week from January to October (the scored year covers July 2025 to June 2026), by lead band
  (0 to 6, 6 to 24, 24 to 48, 48 to 168 hours), by local hour with the clock-change weeks marked,
  and by clear-sky band.
- **Part 3, D10's night level.** The stored arm D residual (actual minus mean and minus median) at
  D10 by night, month, weekday, and temperature band, with a one-change-point fit on the nightly
  mean residual. The same cut for 2024 and 2025 where experiment 201's past-weather rebuild of arm
  D stores forecasts for those years.
- **Part 2, the error-correction layer.** Stored pairs only: experiment 111's GAM-K and GAM0
  (with and without the filter), arm D against arm 1 of experiment 266 (the head and the summary),
  and experiments 200 and 201's head rows, scored by season on daylight half-hours. A refit of arm
  D without the filter is not part of this experiment, and the page says so if the stored pairs
  cannot answer the question.
- **Compute:** pandas-free Polars and NumPy on stored files; 2 cores, `OMP_NUM_THREADS=1`,
  `systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue`.
