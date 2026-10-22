---
id: "00210"
slug: wind-idle-noise-mva-artefact
title: >-
  Does the noise in NGED's metered wind series occur only at low wind speed, so that ERA5 wind speed
  alone can find the rows to drop?
theme: generator-forecasting
also_relevant_to: []
status: planned
size: >-
  simple. Stored: a per-row mask and summary tables under
  data/ml-experiments/00210-wind-idle-noise-mva-artefact/ only. Serving path: not touched.
  Degradation rule: not touched. Designs: one defensible design (a descriptive analysis of three
  series, no model fit). Callers: this folder's scripts only, which read the stored power through
  `studies.power.scan_power` and the stored ERA5 handover file.
hypothesis: >-
  H0: the noise in the three metered wind series (W1 and W3 in MVA, W2 in MW) occurs only at low
  wind speed, because the meters have a deadband, the reactive power drifts, and reactive power
  control runs only while the turbines turn. If H0 holds, a rule on ERA5 100 m wind speed alone,
  with no observed power, finds the rows to drop from training and evaluation.
prediction: >-
  Written before any power is plotted against wind. Above 1.5 times the cut-in speed, the spread
  (5% to 95% band width) of power within each 0.5 m/s wind bin is no larger than the spread of the
  power-curve residual at other speeds above cut-in, and under 1% of those rows sit in the idle
  band. The prediction fails for a farm if either part fails for it. A dip (median power below the
  idle-band median) is expected in the 2 to 5 m/s bins for W1 and W3 (MVA) and not for W2 (MW).
decision_rule: >-
  Definitions fixed before any plot. Wind: ERA5 100 m speed, linearly interpolated between the two
  hourly values either side of the half-hour's midpoint (the timestamp minus 15 minutes). Idle
  floor: the distribution of power at wind below 2 m/s; the idle band is its 5% to 95% range. Cut-in
  speed: the lowest 0.5 m/s bin whose median power exceeds the idle band's 95% value, taken as 3.0
  m/s if that estimate falls outside 2.5 to 4.5 m/s; "1.5 times cut-in" uses it. Spread above
  1.5 times cut-in: the width of the 5% to 95% band of power per bin, compared with the same width
  of the residual from the bin medians at speeds from cut-in to 1.5 times cut-in. Outage rows:
  at least 24 consecutive half-hours (12 hours) with wind above 1.5 times cut-in and power at or
  below the idle band's 95% value. Detector: drop rows with wind below a threshold v; v is the
  largest threshold whose false-drop rate on the training years is at most 5% (a false drop is a
  dropped row whose power exceeds the idle band's 95% value plus 2% of capacity); the training years
  are the odd calendar years, the test years the even ones, then swapped, and both directions are
  reported. Miss rate: the share of idle-band rows outside outage rows that the rule keeps. Verdict:
  `supported` if the prediction holds for all three farms, `partly supported` if for one or two,
  `not supported` if for none, `inconclusive` if a series fails data validation. No threshold on
  gain is set, because the mask costs nothing to apply in production.
benchmark: >-
  None. This is a descriptive precursor; the mask it writes is the input of the main wind-farm
  experiment.
row_set_id: >-
  all half-hours of W1 to W3 from 2019-09-17 to 2026-09-20 (the end of the ERA5 file) with a power
  reading; not comparable with the leaderboard
metric: >-
  Share of half-hours dropped, false-drop rate, and miss rate, with the median and 5% to 95% band
  of power over the retained rows.
code: ml-experiments/00210-wind-idle-noise-mva-artefact/
---

# Experiment 210: wind idle noise and the MVA artefact

Planned. The write-up replaces this page when the analysis has run.
