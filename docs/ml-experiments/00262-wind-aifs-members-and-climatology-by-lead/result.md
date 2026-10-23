---
id: "00262"
slug: wind-aifs-members-and-climatology-by-lead
title: >-
  Does mixing the ECMWF ensemble's members with the AIFS ensemble's members and with climatology,
  weighted by lead, lower the wind forecast's quantile score and tail score, once an offset table
  corrects the common low bias?
theme: generator-forecasting
also_relevant_to:
  - weather-products-and-blending
  - calibration-and-uncertainty
status: planned
date: 2026-10-23
size: >-
  complex. Stored: private member tables and predictions under
  data/ml-experiments/00262-wind-aifs-members-and-climatology-by-lead/ only. Serving path: not
  touched (the experiment decides whether a port is worth proposing). Degradation rule: not touched.
  Designs: several are defensible (how weights and offsets are fitted, how the two ensembles are
  pooled), so the plan fixes one before any fit. Callers: this folder's scripts, which copy
  experiments 259 and 230. One code review and one write-up review, by fresh Sonnet sub-agents,
  plus the prose review.
hypothesis: >-
  For the three metered wind farms (W1 to W3), a lead-dependent mixture of the ECMWF ensemble's
  member forecasts, the AIFS ensemble's member forecasts (both run through experiment 230's
  ERA5-trained XGBoost wind model A0r), and climatology lowers the quantile score and the 0.99
  pinball loss against the ECMWF members alone, after both are corrected by a table of offsets per
  lead day, quantile, and farm; and a small neural gate that reads the two ensembles' spread and
  disagreement chooses better weights than weights by lead alone.
prediction: >-
  Written before any fit. C1 (climatology weights by lead, M1 minus M0, both with the offset table,
  kept rows, lead days 7 and later): -0.15 to -0.40 points of capacity; at lead days 1 and 2 inside
  -0.02 to +0.02. C2 (AIFS pooled in, M3 minus M0, lead days 3 to 6): -0.10 to -0.30; at lead days 1
  and 2 inside -0.05 to +0.05; the 0.99 pinball loss at days 3 to 6 at least 3% lower. C3 (neural
  gate, M5 minus M4, all leads): inside -0.03 to +0.03. The linear pool M1p beats quantile averaging
  M1q on the tail score at lead days 7 and later. AIFS alone (M2) scores worse than the ECMWF
  members at lead day 0 to 2.
decision_rule: >-
  Three decisive contrasts, C1 to C3, each on the quantile score on the rows kept by experiment
  210's mask, in the lead band named, with the offset table applied to both arms, read at 98.33%
  (Bonferroni across three), resampling whole scored months and three seeds, 50,000 resamples.
  Thresholds are set by porting cost. C1 needs an interval wholly below -0.03, because the
  climatology weight adds one blending step to a model production already has. C2 needs an interval
  wholly below -0.15, because AIFS adds a weather feed and its pipeline (the threshold experiments 52
  and 54 used). C3 needs an interval wholly below -0.03, because the gate replaces a table of
  weights, and M5 must also beat M5b. Everything else (the versions without the offset table, other
  bands, arms M1q, M1p, M2, M5b, the run-timing check, the AIFS version split) is descriptive.
  Intervals describe these three farms.
benchmark: >-
  M0: experiment 230's A0r (ERA5-trained XGBoost quantile model with the 0.01 and 0.99 levels) on
  the 51 ECMWF members, with the spread factor and the offset table, both refitted on earlier scored
  months only. Experiment 259's stored R0 (spread factor and table fitted on its 15-month hindcast)
  and experiment 230's stored A0r are reference rows.
row_set_id: >-
  experiment 33's wind cases (`fold_a_tuesdays_v1`) restricted to the Tuesday runs from 2025-07-01
  that have an AIFS ensemble run; July to September 2025 only train the weights, spread factors,
  and offsets, and October 2025 to June 2026 are scored. Headline on the rows kept by experiment
  210's mask; all rows descriptive.
metric: >-
  Quantile score (twice the mass-weighted mean pinball loss over nine levels), in points of capacity;
  pinball loss at 0.01, 0.05, 0.5, 0.95, and 0.99; the tail score T; coverage of the 5% to 95% and
  1% to 99% bands; share of rows below the median
production_status: not-for-production
adopted_in: none
code: ml-experiments/00262-wind-aifs-members-and-climatology-by-lead/
---

**Planned, not yet run.** The design below is fixed before any fit.

- **Why.** Experiment 259's offset table removes most of the ECMWF members' low bias, and
  [experiment 261](../00261-wind-low-bias-decomposition/result.md) traced that bias to the ensemble's
  10 m and 100 m wind speeds being slower than ERA5's. A second ensemble (AIFS) has a different
  error, and climatology beats every trained arm beyond day 7 in experiment 230.
- **Source of the AIFS ensemble.** The ECMWF AIFS ensemble from Dynamical.org (CC BY 4.0), 51
  members, 6-hourly steps, 100 m and 10 m wind components, 2025-07-02 to 2026-09-25. East and north
  components are interpolated linearly to half-hours before speed and direction are computed, and
  each farm reads the area-weighted mean of the cells under its H3 resolution-5 cell, as the ECMWF
  ensemble does.
- **Run timing.** Both ensembles' 00 UTC runs are read as available at 09:00 UTC, the convention of
  the matched-lead study. ECMWF's published dissemination times were not found, so the assumption
  is untested. A descriptive check reads the AIFS run of 18 UTC the day before instead.
- **Arms** (A0r on every member, three seeds, CPU). M0: ECMWF members. M1q and M1p: M0 with
  climatology, by quantile averaging and by linear pool, one weight per lead day. M2: AIFS members
  alone. M3: linear pool of the ECMWF and AIFS member mixtures, one AIFS weight per lead day. M4: M3
  with climatology. M5: a neural gate (one hidden layer of at most 16 units, softmax over three
  weights) reading lead, each ensemble's standard deviation of 100 m wind speed, and the difference
  of their mean 100 m wind speeds. M5b: the same gate reading only the lead.
- **Fitting.** Weights, gate, spread factor, and offset table are refitted monthly on cases whose
  valid time is at least 3.5 hours before the month's first run. Weights and the gate minimise the
  continuous ranked probability score of the pooled mixture (a smooth stand-in for the quantile
  score that needs only per-case sums, not a sort), and quantile averaging uses the quantile score
  directly. The offset table follows experiment 259's recipe: for each lead day, quantile, and farm,
  the quantile of the observed-minus-forecast residual.
- **Offset table.** Every arm is scored with its own table, fitted after its weights and spread
  factor. The reference also has experiment 259's own table, fitted on its 15-month hindcast. The
  AIFS ensemble has no hindcast before 2025-07, so no arm that reads it can use that table.

**Stop rules.** If AIFS alone (M2) scores worse than climatology at lead days 1 to 6, stop the AIFS
arms and check the 100 m wind mean against the ECMWF ensemble and ERA5 by month. If more than 10% of
the scored runs lack an AIFS run, score on the runs that have one, and stop the AIFS arms above 25%.
