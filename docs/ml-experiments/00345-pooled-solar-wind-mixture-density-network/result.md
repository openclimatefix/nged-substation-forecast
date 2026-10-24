---
id: '00345'
slug: pooled-solar-wind-mixture-density-network
title: >-
  Does one pooled mixture-density network for all leads, trained on solar and wind farms together,
  match the best single-family forecast for each fuel, and help a farm with no history?
theme: metered-solar-farms
also_relevant_to:
- generator-forecasting
- production-reference-and-best-guess
status: planned
kind: change
date: 2026-10-24
size: >-
  medium. Stored: private predictions and tables under
  data/ml-experiments/00345-pooled-solar-wind-mixture-density-network/ only. Serving path: not
  touched. Degradation rule: not touched. Designs: the arms are fixed by the maintainer's brief, so
  no design choice is open. Callers: this folder's scripts, which import experiment 310's feature
  and head modules, experiment 285's wind modules, and no new reviewed module. Asset graph: not
  touched. One Sonnet code review and one Sonnet prose review, no Opus.
hypothesis: >-
  One mixture-density network (MDN) trained once for all leads on the six metered solar farms and
  the wind farms together, with a fuel-type indicator and weather inputs common to both fuels,
  scores as well on each fuel as that fuel's best single-family model, and scores better for a farm
  with no history.
prediction: >-
  Written before any score is read. Leads are 1 to 10 days for solar and 12 to 252 hours (days 1 to
  10 by the same arithmetic) for wind. P1: a solar-only pooled MDN (pure, lead features, plant
  column dropped, trained across the six farms) is within 1% of the single-family MDN on the plant
  at every day band, own history and zero history. P2: the solar-plus-wind pooled MDN is within 1%
  of the solar-only pooled MDN on solar and within 1% of the wind-only pooled MDN (the pure MDN
  of experiment 313, retrained here on all wind farms) on wind, at every day band. P3: with zero
  history (the farm held out), the solar-plus-wind network beats the solar-only network by at least
  2% on solar and the wind-only network by at least 2% on wind. P4: the pooled network's 0.99
  pinball loss is no worse than Metered solar farms research default 1's at days 3 to 7. Stop rule:
  if pooling worsens the solar quantile score by more than 2% at 3 or more of the 6 farms at days 1
  and 2, stop the pooled arms and report the solar-only pooled result alone. Each prediction can
  fail.
decision_rule: >-
  Quantile score in points of capacity, equal weight per farm, month-block resampling of the 12
  scored months at 98.75% (Bonferroni across the four predictions). "Within 1%" holds when the
  98.75% interval of the percentage change lies inside plus or minus 1% (or its upper end is below
  1% for a one-sided claim), read per day band. Plan items for Metered solar farms research default
  1: the family and version are Metered solar farms research default 1; the arms are "default" (the
  blend, read from experiment 310's stored predictions, and its single-family MDN refitted in
  experiment 318 with five seeds) and "default plus x" (each pooled network); tier 2, complexity
  tier "a new data source" (wind rows) or "a new fitted step" (pooling across farms), so a minimum
  gain of 0.02 points; controls are the shuffled fuel indicator, shuffled wind rows, `climatology`,
  and `manual_heuristic`. At most one pooled arm is adopted. The pooled network is a single-family
  candidate, so the best outcome against the blend is non-inferiority within 1%, which the page
  states plainly. The standard-architecture arm reports its gap to the best family.
benchmark: >-
  Solar: Metered solar farms research default 1's blend and its MDN on the plant (experiments 310
  and 318), experiment 313's pure MDN, experiment 318's pure MDN with lead features, `climatology`,
  and `manual_heuristic`. Wind: experiment 313's pure MDN and pure XGBoost, experiment 285's neural
  adapter on the physics core (the standard-architecture wind arm), experiment 276's physics core.
row_set_id: >-
  Solar: experiment 310's rows (farms A to F, 2025-09 to 2026-08, day bands 1 to 10). Wind:
  `fold_a_tuesdays_v1`, experiment 285's scored cases (W1 to W3, 53 Tuesday runs), kept rows at 12
  to 252 hours.
metric: >-
  Quantile score (twice the mean pinball loss over 11 levels, in points of capacity), pinball loss
  and coverage at 0.01, 0.05, 0.95, and 0.99
code: ml-experiments/00345-pooled-solar-wind-mixture-density-network/
---

**Planned, not yet run.** The title question is the whole experiment; the front matter holds the
pre-registered predictions. Scored at leads of 1 to 10 days for solar and at 12 to 252 hours for
wind (the stored wind cases reach 351 hours; the extra range is reported as a secondary row).

- **Arms.** (1) References: Metered solar farms research default 1, its MDN on the plant, and for
  wind experiment 313's pure MDN and experiment 285's neural adapter. (2) Solar-only pooled MDN.
  (3) Wind-only pooled MDN. (4) Solar-plus-wind pooled MDN. (5) Arm 4 plus the solar plant forecast
  as a column on solar rows. (6) Training cadence for arm 4: once before the scored year, quarterly
  (the screening schedule of experiments 310 and 318), and monthly. (7) Controls: shuffled fuel
  indicator, shuffled wind targets.
- **Weather at each lag.** Solar rows use the ECMWF ensemble mean at each lead (exists at forecast
  time). Wind training rows use ERA5, which arrives about 5 days late, so wind numbers on ERA5
  weather are a ceiling; the headline wind rows apply the network to each of the 51 ensemble
  members and mix the quantiles, as in experiment 313.
