---
id: '00183'
slug: metered-farm-flat-top-cause
title: >-
  Is the flat top of metered solar farms C, D, E, and F a fixed export cap or a panel-plane effect?
theme: solar-disaggregation
also_relevant_to:
- differentiable-physics-embedded-generation
status: finished
size: >-
  simple. Stored: a private table under data/ml-experiments/00183-metered-farm-flat-top-cause/
  only. Serving path: not touched. Degradation rule: not touched. Designs: one defensible design,
  because no model is fitted. Callers: this folder's script only, which calls reviewed `studies.*`
  modules without changing them.
hypothesis: >-
  The flat top at farms C, D, E, and F is a fixed export cap, so its level does not change with
  season or with the sun's azimuth.
prediction: >-
  At least 3 of the 4 farms have a flat level that varies by under 3% across the months April to
  September.
decision_rule: >-
  For each farm, take the 90th percentile of power over capacity in each month from April to
  September, over half-hours whose clear-sky irradiance on a horizontal plane exceeds 650 W/m². The
  level is seasonally constant when the range of the monthly levels over their mean is under 3%.
  Supported when at least 3 of the 4 farms are seasonally constant. When 3 or more farms vary by more
  than 3%, geometry is the cause, and the per-farm plant fit (follow-up 2) must fit tilt and
  azimuth, not only a clip level. Replication: the same test within each half of the years (even
  and odd calendar years). No score is computed, so no threshold for porting applies.
benchmark: none (data exploration, no fitting)
row_set_id: none
metric: seasonal range of the flat level over its mean, per farm and per half of the years
verdict: not supported
best_arm: none (data exploration, no fitting)
headline: >-
  The flat level falls from April to August by 6% to 9% (90th percentile) at all four farms, in both
  halves of the years, so the pre-registered prediction of at least 3 constant farms fails and the
  stop rule fires.
take_home_message: >-
  The flat top of farms C, D, E, and F is lower in summer than in spring (by 6% to 9% of capacity at
  the 90th percentile, in even and odd years alike), so it does not behave like a fixed export cap.
  The per-farm plant fit should therefore fit panel tilt and azimuth and a heat or soft-saturation
  loss, not only one clip level. A cloud-mix effect on the percentile cannot be excluded: the 99th
  percentile falls by less (1% to 6%).
production_status: not-for-production
adopted_in: none
advice:
  - >-
    Fit tilt, azimuth, and a summer loss term, not only a clip level, in the per-farm plant fit for
    farms C to F, because the flat level falls from April to August at all four (Figure 1).
  - >-
    Estimate a flat level from a percentile that cloud cannot lower, such as the 99th, and quote
    the 90th beside it, because the two give seasonal ranges of 1% to 6% and 6% to 9% (Checks).
---

**The flat top of the four metered solar farms that have one (C, D, E, and F) is lower in summer than
in spring, so it does not behave like a fixed export cap.** We asked whether the output of each farm,
as a share of its capacity, flattens at one level all year (an export cap, a limit on grid export) or
at a level that changes with the season (a panel-angle or heat effect). We took the half-hours with
clear-sky irradiance on a horizontal plane above 650 W/m², from the cleaned series of
[experiment 179](../00179-metered-solar-farm-exploration/result.md), and measured the 90th
percentile of power over capacity in each month from April to August. No model was fitted.

**The prediction fails and the stop rule fires.** The range of the monthly levels over their mean is
5.9% to 8.9% at the four farms, against a limit of 3%, so zero of four farms are seasonally
constant. The per-farm plant fit (follow-up 2) must therefore fit tilt and azimuth, not only a clip
level. September has under 30 qualifying half-hours at every farm and is left out.

![Figure 1: monthly flat level per farm](figure_1.svg)

## Numbers

- **Pre-registered level (90th percentile), range over mean, April to August,** farms C to F:
  7.2%, 8.9%, 5.9%, and 8.6%. Each farm's level is highest in April (1.04 to 1.05 times capacity)
  and lowest in July (0.96 to 0.98).
- **Replication by years:** the same test on even years only gives 5.0%, 8.7%, 6.4%, and 8.1%, and
  on odd years only 9.8%, 13.2%, 5.4%, and 11.9%. Every farm varies by more than 3% in both halves,
  so the count of 0 of 4 is replicated.
- **99th percentile (post hoc),** range over mean: 1.4%, 5.8%, 2.7%, and 3.4%. Farms C and E would
  pass the 3% rule at this percentile, F narrowly fails, and D fails. All four 99th-percentile
  levels still fall from April to August (for example farm F, 1.06 to 1.03).
- **Morning against afternoon** (90th percentile, same half-hours): the ratio is 0.99, 1.02, 0.99,
  and 1.00 for farms C to F, so the level does not depend on whether the sun is east or west of
  south. This is what a south-facing array or a fixed cap would show, and it does not separate them.

## Checks

- **Cloud-mix confound:** the clear-sky filter is on the model, not on observed irradiance, so
  cloudy half-hours enter the 90th percentile and lower it in cloudier months. The 99th percentile
  is less exposed, and it varies less, so part of the 90th-percentile range may be cloud. The
  seasonal fall at the 99th percentile remains at all four farms.
- **Not separated:** heat derating, panel-plane geometry, and an export cap that changes with the
  season are not distinguished, because no temperature, no panel tilt, and no export-cap record was
  used.
- **Capacity:** power is divided by the effective capacity from `studies.pv_dataset`, which may
  itself be estimated from the output, so the absolute level (near 1.0) is not meaningful, only its
  change across months.
- **Selection:** the high-irradiance half-hours in April and August are near midday only, so month
  and sun elevation are confounded.
- **Median minus mean forecast, quantile and mean-error contrasts:** not applicable, no forecast.
- **Audit triggers:** the trigger for a result that contradicts its own prediction fires in form.
  The explanation (cloud mix at the 90th percentile) was tested with the 99th percentile above. No
  audit was run.

## What the next experiment should answer

- Does a per-farm physical plant fit (tilt, azimuth, DC capacity, AC capacity, and a temperature
  loss) reproduce the fall from April to August better than a clip level alone? The prediction that
  could fail is that it does not.
- Does each farm's flat top coincide with export-cap hours (`studies.export_cap`)?
