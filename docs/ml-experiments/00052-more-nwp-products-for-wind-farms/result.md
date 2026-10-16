---
id: "00052"
slug: more-nwp-products-for-wind-farms
status: planned
size: complex
hypothesis: >-
  For the 3 metered wind farms, averaging the quantiles of the experiment 33 wind best guess (the
  ERA5-trained mixture over the 51 ENS members, with the power-curve margin and the spread factor)
  with the quantiles of the same ERA5-trained model fed ICON-EU's and IFS 0.25 degree's forecast
  wind lowers the quantile score at 1 to 2 days ahead, with weights chosen only on earlier months.
prediction: >-
  Wind quantile score at 1 to 2 days: -0.2 to -0.4 points of capacity against the best guess. Over
  all leads: -0.05 to -0.15. The interval on 3 farms is wide, and experiment 35's wind blend (-0.37
  at 0 to 24 h, -0.81 to +0.02) suggests `inconclusive` is the likeliest verdict. Solar (secondary,
  descriptive): 0 to -0.05.
decision_rule: >-
  One decisive contrast, C1: the blend minus the best guess on the quantile score for cases whose
  valid time falls 1 or 2 calendar days after the date of the ENS run (lead day 1 or 2, counted from
  the run's 00 UTC), read at 95% with whole months and the three seeds resampled, 50,000 resamples.
  The interval is conditional on the 3 farms, because farms are not resampled. The threshold is
  0.15 points, because the blend adds two weather feeds and two pipeline steps to the serving path,
  and a smaller gain does not repay them. C1 is `supported` when its interval lies wholly below
  -0.15, `small gain` when it lies wholly below zero but not wholly below -0.15, `refuted` when it
  lies wholly above zero, and `inconclusive` otherwise. C2, the same difference over all leads, is
  reported beside C1 at 95% and does not change the verdict. All other reads (per product, per farm,
  per lead day, solar) are exploratory, labelled so, and read at 95%.
benchmark: >-
  The experiment 33 wind best guess (`bg_full`) on the same cases, read from its stored table.
row_set_id: fold_a_tuesdays_v1
metric: >-
  Quantile score (twice the mass-weighted mean pinball loss over nine levels, the estimate of the
  CRPS), in points of capacity; coverage of the 5% to 95% band
production_status: not-for-production
adopted_in: none
code: ml-experiments/00052-more-nwp-products-for-wind-farms/
---

# Experiment 52: do more weather forecasts lower the wind farms' error at 1 to 2 days ahead?

**Status: planned. No code has run.** We ask whether giving the wind-farm forecast two further
weather forecasts, ICON-EU (the German weather service's European model) and IFS 0.25 degree (the
European Centre's high-resolution forecast, whose ensemble ENS the best guess already uses), lowers
its error one to two days ahead. The best guess feeds ENS's 51 members to a model trained on ERA5
(an estimate of past weather). We feed the same trained model each extra forecast's wind, and
average the resulting forecast ranges with the best guess's, using weights chosen on earlier months
only. The data are on disk, so the experiment downloads nothing.

## Sizing

- **What gets stored:** private prediction tables under `data/ml-experiments/`; nothing in
  production. Not a trigger.
- **Serving path:** untouched by the experiment. A `supported` verdict would add two weather feeds
  to the serving path, which the threshold prices.
- **Degradation rule:** untouched. A product missing at serving time would drop out of the average,
  as it does here beyond its horizon.
- **Defensible designs:** more than one (a weighted average of quantiles, extra pooled mixture
  atoms, product columns in a calibrator). The plan picks the first, because experiment 35 built and
  tested it, and experiment 23 warned against a calibrator trained on controls alone.
- **Callers:** the experiment copies experiment 33's and experiment 35's code, which it may not
  import.
- **Size: complex** (two triggers), so the plan gets one Opus review and the code a second Sonnet
  review.

## Coverage of the validation year (checked 2026-10-16)

| Product | Wind variables at the 3 farms | Validation year, July 2025 to June 2026 | Used |
|---|---|---|---|
| ICON-EU (Open-Meteo Previous Runs) | 10 m and 100 m speed and direction | All 8,760 hours at `previous_day1`; 98.9% at `previous_day2`; complete at days 3 and 4; no day 5 | Primary |
| IFS 0.25 degree (same archive) | Same | All 8,760 hours at days 1 to 3 (archive holds to day 7) | Primary |
| GFS (same archive) | Same | All 8,760 hours at lead days 1 to 3 | Exploratory |
| AIFS Single (Dynamical.org) | `wind_u_10m`, `wind_v_10m`, `wind_u_100m`, `wind_v_100m` | From the 2025-02-24 06 UTC run; version change in 2026-05 inside the year (PVDIS-8) | Not used: needs its own row set; left to a later experiment |
| GEFS and GFS native (Dynamical.org) | 10 m wind only | Whole year | Not used: no 100 m wind |
| The nwp-archive (ICON-EU-EPS, ICON-D2-EPS, MOGREPS) | n/a | Starts 2026-10-06 (DWD) or later | Not used: no overlap with the year |
| UKV | n/a | CEDA UKV differs statistically from the live feed | Not used |

## Design

- **Cases:** experiment 33's 53 Tuesday ENS runs of `fold_a_tuesdays_v1` for the 3 wind farms, the
  half-hourly cases its stored `bg_full` table holds, 3 seeds. Weights and spread factors for runs in
  month m use only cases with a run in an earlier month and a valid time before the first day of m
  (experiment 35's rolling origin). July 2025 has no history and uses equal weights and no extra
  spread factor.
- **Product forecast power:** for each seed, refit experiment 33's ERA5-trained wind model (`margin`
  variant: nine quantiles with the power-curve base margin, trained on valid times before
  2025-06-17). Feed it the product's 10 m and 100 m speed (converted from km/h to m/s) and the sine
  and cosine of its direction, at the farm's grid point. The model gives nine quantiles per case.
- **Lead day and availability:** the lead day d of a case is the number of calendar days from the
  date of its ENS run (`nwp_init_time`, 00 UTC) to its valid time, not `lead_hours` // 24, because
  experiment 33's `lead_hours` counts from `power_fcst_init_time` (09:00 UTC). A case reads
  `previous_day(d+1)`. That selects a run initialised no later than 21:00 UTC (ICON-EU) or 18:00 UTC
  (IFS, GFS) on the day before the ENS run, so it was published before the service reads ENS at
  09:00 UTC. ICON-EU reaches d = 3 (`previous_day4`); IFS and GFS reach d = 6. A case beyond a
  product's horizon, or with a null value, lacks that product.
- **Half-hour valid times:** the archive is hourly. A case at :30 averages the speed and the u and v
  components of the hour before and the hour after, then takes the direction from the averaged
  components, as the ERA5 build does; both ends use the same `previous_day(d+1)` column, where d is
  fixed by the case, so that the two ends never come from different runs.
- **Blend:** the Vincent average (mean of each quantile level, then sorted) of `bg_full` and the
  available product forecasts. The weights are on a 0.25 grid, chosen per lead-day band (day 0, days
  1 to 2, days 3 to 6, days 7 and beyond) on earlier months to minimise the quantile score, and are
  renormalised over the members a case has; the history for a band uses cases that have all the
  primary arm's products. A rolling-origin spread factor (experiment 33's `fit_spread_factor`,
  grid widened to 3.0) is then fitted on the blend per band on earlier months, because a
  perfect-weather model fed one forecast gives a band that is too narrow. A factor at the grid edge
  is reported. The arms are `ens+icon+ifs` (primary, fixed in advance), `ens+icon`, `ens+ifs`, and
  `ens+gfs` (exploratory), each with one weight vector for all bands as a second option.
- **Stop rule:** a product whose own spread-corrected forecast scores more than 1.0 points worse
  than `bg_full` at lead days 1 to 2 on the first 6 months is dropped from the exploratory arms and
  reported. A dropped product never leaves the primary arm, which is fixed in advance.
- **Solar (descriptive, only if the wind pipeline finishes with time to spare):** the 6 solar farms
  get the same blend, with the product's shortwave radiation replacing ERA5's in the single input
  column. The matched-lead study found no solar gain at this lead, so solar decides nothing.
- **Reports:** absolute scores of every arm per farm (`W1` to `W3`) and per lead band; coverage;
  a climatology row from `ClimatologyForecaster`; the chosen weights.

## Controls and leakage checks

- **Seed pairing:** the blend and `bg_full` use the same seeds; `losses_for_bootstrap` pairs them.
- **Calendar columns:** the ERA5-trained models use no calendar columns, as in experiment 33.
- **Past-weather caveat:** the ERA5-trained model is trained on ERA5 but fed forecast weather at
  serving time. No ERA5 weather is read at lag times, so nothing here depends on ERA5's delay.
- **Run availability:** a unit check asserts that for every case the selected run's initialisation
  time plus the product's publication delay (4 hours for ICON-EU, 8 hours for IFS and GFS, upper
  bounds taken from the providers' documentation) is no later than 09:00 UTC on the ENS run's date.
- **Product levels:** the report prints each product's mean 100 m speed at the farms against ERA5's.
  ICON-EU's 100 m wind is Open-Meteo's rescaling of its 120 m level.
- **A placebo arm:** the blend of `bg_full` with a product's forecast shuffled among cases of the
  same farm, month, and hour of day shows how much of any gain comes from averaging alone.

## Advice followed

- Experiment 35: blend with one fixed weight vector if a blend is ported; test wind on more than 3
  farms. This experiment runs both weight options and cannot add farms, so the interval is wide by
  construction.
- Experiment 33: grade wind on the 0.95 quantile's exceedance as well as the quantile score; the
  report prints both. ERA5 weather would lower the wind score by 5.46, the headroom that motivates
  this experiment.

## Compute

Slot E: 6 cores, GPU at most 12 GB. Refits are 3 farms times 3 seeds of small quantile models on
the GPU; the product tables are built on the CPU. Estimated 1 to 2 GPU hours, no download.
