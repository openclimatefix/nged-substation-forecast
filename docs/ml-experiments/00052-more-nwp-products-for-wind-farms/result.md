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
  Two decisive contrasts for wind, the blend minus the best guess on the quantile score: C1 on the
  1 to 2 day lead band (24 to 72 hours after the ENS run's 00 UTC), the primary contrast, and C2 over
  all leads. Both are read at 97.5% (Bonferroni across two) with whole months and the three seeds
  resampled, 50,000 resamples. The threshold is 0.15 points at 1 to 2 days, because the blend adds
  two weather feeds and two pipeline steps to the serving path (a gain under that does not repay
  them), and 0.05 over all leads. C1 is `supported` when its interval lies wholly below -0.15,
  `small gain` when it lies wholly below zero but not wholly below -0.15, `refuted` when it lies
  wholly above zero, and `inconclusive` otherwise. C2 does not change the verdict; it is reported
  beside C1 under the same four labels. All other reads (per product, per farm, per band, solar)
  are exploratory, labelled so, and read at 95%.
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
| ICON-EU (Open-Meteo Previous Runs) | 10 m and 100 m speed and direction | All 8,760 hours at lead days 1 and 2; day 3 has 98.9% of hours; no day 4 | Primary |
| IFS 0.25 degree (same archive) | Same | All 8,760 hours at lead days 1 to 3 (archive holds to day 7) | Primary |
| GFS (same archive) | Same | All 8,760 hours at lead days 1 to 3 | Exploratory |
| AIFS Single (Dynamical.org) | `wind_u_10m`, `wind_v_10m`, `wind_u_100m`, `wind_v_100m` | From the 2025-02-24 06 UTC run; version change on 2026-05 inside the year (PVDIS-8) | Exploratory, rows of one version only |
| GEFS and GFS native (Dynamical.org) | 10 m wind only | Whole year | Not used: no 100 m wind |
| The nwp-archive (ICON-EU-EPS, ICON-D2-EPS, MOGREPS) | n/a | Starts 2026-10-06 (DWD) or later | Not used: no overlap with the year |
| UKV | n/a | CEDA UKV differs statistically from the live feed | Not used |

## Design

- **Cases:** experiment 33's 53 Tuesday ENS runs of `fold_a_tuesdays_v1` for the 3 wind farms, the
  cases its stored `bg_full` table holds, 3 seeds. Weights and spread factors for runs in month m
  use only cases with a valid time before the first day of m (experiment 35's rolling origin). July
  2025 has no history and uses equal weights.
- **Product forecast power:** for each seed, refit experiment 33's ERA5-trained wind model (`margin`
  variant: nine quantiles with the power-curve base margin, trained on valid times before
  2025-06-17). Feed it the product's 10 m and 100 m speed (converted from km/h to m/s) and the sine
  and cosine of its direction, at the farm's grid point. The model gives nine quantiles per case.
- **Lead and availability:** a case whose valid time falls d days after the run's date (the ENS
  `lead_hours` // 24) reads `previous_day(d+1)`, which comes from a run initialised on or before the
  day before the ENS run, so the product was published before the service reads ENS at 09:00 UTC.
  This is the conservative lead of the matched-lead study. Cases beyond a product's horizon (ICON-EU
  after day 3) have no product quantiles and keep the best guess unchanged. A case whose
  `previous_day` value is missing is treated the same way.
- **Blend:** the Vincent average (mean of each quantile level, then sorted) of `bg_full` and the
  product forecasts, with weights on a 0.25 grid chosen per lead band (experiment 35's bands) on
  earlier months to minimise the quantile score, and a one-weight variant. The arms are `ens+icon`,
  `ens+ifs`, and `ens+icon+ifs`, the last being the primary arm.
- **Stop rule:** a product whose own forecast scores more than 1.0 points worse than `bg_full` at
  1 to 2 days on the first 6 months is dropped from the exploratory arms and reported (the rule of
  the ranking brief). A dropped product never leaves the primary arm, which is fixed in advance.
- **Solar (descriptive):** the 6 solar farms get the same blend where the product's shortwave
  radiation replaces ERA5's in the best-guess mixture's single input column. Solar decides nothing.
- **Reports:** absolute scores of every arm per farm (`W1` to `W3`) and per lead band; coverage;
  a climatology row from `ClimatologyForecaster`; the chosen weights.

## Controls and leakage checks

- **Seed pairing:** the blend and `bg_full` use the same seeds; `losses_for_bootstrap` pairs them.
- **Calendar columns:** the ERA5-trained models use no calendar columns, as in experiment 33.
- **Past-weather caveat:** the ERA5-trained model is trained on ERA5 but fed forecast weather at
  serving time. No ERA5 weather is read at lag times, so nothing here depends on ERA5's delay.
- **Run availability:** a unit check asserts that for every case the product run time is earlier than
  the ENS run date plus 9 hours, using the archive's documented run selection rule.
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
