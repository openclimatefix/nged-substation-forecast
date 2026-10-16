---
id: "00054"
slug: product-trained-xgboost-wind-farms
status: planned
size: complex
hypothesis: >-
  For the three metered wind farms, an XGBoost quantile model trained on ENS and given one other
  weather product's wind as extra input columns scores lower on the quantile score at lead days 1
  and 2 than the same model trained on ENS alone, with every arm trained on the same window, rows,
  and seeds. The deployable product is ICON-EU, read as the Previous Runs forecast issued at least
  d+1 days before the valid time. UKV, from the CEDA archive, gives a labelled ceiling read.
prediction: >-
  ICON-EU, lead days 1 and 2: the ENS-plus-ICON-EU arm scores -0.2 to -0.6 points of capacity
  against the ENS-only arm. UKV-CEDA, lead day 1: -0.2 to -0.5 (the UKV-CEDA blends study found
  -0.2 to -0.3 at lead days 1 and 2 for an ENS mean plus UKV winds). The product-alone arms score
  worse than the ENS-only arm. Every arm scores worse than the ERA5-trained best guess, as the
  ERA5-trained wind model beat the ENS-trained one by 1.70 points in experiment 1. The interval on
  three farms is likely wide, so `inconclusive` is possible.
decision_rule: >-
  Two decisive contrasts, each the ENS-plus-product arm minus the ENS-only arm refit on the same
  rows, on the quantile score for cases whose valid time falls on the lead days named: C1 for
  ICON-EU at lead days 1 and 2, and C2 for UKV-CEDA at lead day 1. Each is read at 97.5%
  (Bonferroni across two) with whole months and the three seeds resampled, 50,000 resamples, and
  the interval is conditional on the three farms. The threshold is 0.15 points, because an added
  weather feed and an added model input column are a pipeline step each, and a smaller gain does
  not justify them. Per product, the outcome is `supported` when the interval lies wholly below
  -0.15, `small gain` when it lies wholly below zero but not wholly below -0.15, `refuted` when it
  lies wholly above zero, and `inconclusive` otherwise. C2 is a ceiling, because the CEDA archive
  is not the live UKV product, so only C1 can make the verdict `supported`; a `supported` C2 alone
  gives `small gain`. The verdict is the better of the two outcomes under that rule, `refuted` when
  both are `refuted`, and `inconclusive` otherwise. The same contrasts over all the product's leads
  (C3 and C4) are reported and do not change the verdict. Every other read (the product-alone arms,
  IFS HRES 9 km, GFS, per farm, per lead day, against the ERA5-trained best guess) is exploratory,
  labelled so, and read at 95%.
benchmark: >-
  The ENS-trained XGBoost arm with the same training protocol (arm A), refit on each product's own
  rows and scored on the same cases. The ERA5-trained best guess of experiment 33 is a reference
  row, not the benchmark.
row_set_id: fold_a_tuesdays_v1
metric: >-
  Quantile score (twice the mass-weighted mean pinball loss over nine levels, the estimate of the
  CRPS), in points of capacity; coverage of the 5% to 95% band; share of cases above the 0.95
  quantile
production_status: not-for-production
adopted_in: none
code: ml-experiments/00054-product-trained-xgboost-wind-farms/
---

# Experiment 54: do other weather products, learned from their own history, help wind forecasts?

**Status: planned. No code has run.** [Experiment 52](../00052-more-nwp-products-for-wind-farms/result.md)
fed ICON-EU and IFS 0.25 degree into a wind-power model trained on ERA5, an estimate of past
weather, and could not tell whether they help. That design favours ENS, because the ERA5-trained
model never learned either product's own errors. This experiment trains every model on a forecast
product's own history instead, with one training protocol for every arm. The products are ICON-EU
and UKV, at the maintainer's request, relayed by the manager; IFS HRES 9 km and GFS are exploratory
extras.

## The three arms

**Each arm is an XGBoost model with nine quantile levels, trained per farm and per seed with the
same window, the same recency weights, and the same lagged-power inputs.** The training protocol is
that of experiment 33's ENS-trained model (N1): the ENS control member of every daily 00 UTC run
with valid times from 2024-04-01 to 2025-06-30, a recency weight with a half-life of 2 years, the
wind columns, `lead_hours`, and three lags of power from the forecast origin (09:00 UTC).

- **Arm A, ENS only:** the six ENS wind columns (speed and direction at 10 m and 100 m, direction as
  sine and cosine).
- **Arm B, ENS plus one product:** the six ENS columns and the product's wind columns, one product at
  a time. At prediction the model runs once per ENS member with the product's columns fixed, and the
  51 member quantile sets are pooled, as arm A does.
- **Arm C, one product alone:** the product's wind columns. At prediction the model gives one set of
  nine quantiles per case.
- **Reference row:** experiment 33's ERA5-trained best guess (`bg_full`) and its ENS-trained N1,
  scored on the same cases. They are not the control.

**Each product has its own rows, so arm A is refit once per product.** A product covers a different
range of leads, so arms A, B, and C of one product train on the same rows, and score the same cases,
which are the cases that product covers. Arm A for ICON-EU and arm A for UKV therefore differ.

## Each product, as read, and what the data on disk supports

| Product | Definition | Wind columns | Leads covered | Training window on disk |
|---|---|---|---|---|
| ICON-EU (Open-Meteo Previous Runs) | For a case at lead day d (calendar days from the ENS run's date to the valid time), the forecast issued at least d+1 days before the valid time (`previous_day(d+1)`), the same definition in training and scoring | 10 m and 100 m speed and direction | Lead days 0 to 3, because `previous_day5` is empty | The whole ENS window: 10,944 of 10,944 hourly values at each of the three farms for `previous_day1` to `previous_day4` |
| UKV (CEDA archive; a ceiling read) | The 00 UTC run of the ENS run's date, hourly leads 0 to 48 hours | 10 m speed and direction, and 925 hPa speed (about 750 m; no 100 m wind is archived) | Lead days 0 and 1 | To be measured at the start of the run, from the CEDA ledger, for the ENS window |
| IFS HRES 9 km (Open-Meteo Single Runs; exploratory) | The 00 UTC run of the ENS run's date, hourly leads to 240 | 10 m and 100 m speed and direction | Lead days 0 to 9 | 2024-03-14 onwards; 7 run days missing, 2 of them Tuesdays in the validation year |
| GFS (Dynamical.org native; exploratory) | The 00 UTC run of the ENS run's date, hourly to lead 120 and 3-hourly after | 10 m and 100 m u and v components | Lead days 0 to 9 (the IFS cases) | 2021-05 onwards; NaN values in some runs, which are dropped |

**ICON-EU is deployable by construction, with runs older than ENS's.** The Previous Runs archive
cannot give ICON-EU's 00 UTC run of the ENS run's date. The product is therefore the forecast
issued at least d+1 days before the valid time, which starts no later than the valid hour of day
on the day before the ENS run, so a service reading at 09:00 UTC has it. The product's lead
is `24(d+1) + (h mod n)` hours for a valid hour h and a run cycle of n hours, so the product runs
0 to 24 hours older than ENS's, about 12 hours on average. This lead structure is part of the
product's definition and holds in training and scoring alike. The ENS window gives about 456 days
of runs, with lead days 0 to 3 of each.

**Open-Meteo's UKV history cannot serve lead days 1 and 2, so UKV is read from the CEDA archive as a
labelled ceiling.** At the three wind farms, Open-Meteo's UKV has only `previous_day1` (a forecast
issued at least 1 day before): `previous_day2` to `previous_day7` are empty. `previous_day1` starts
on 2024-08-06 and holds 7,769 of the 10,944 hourly values of the ENS window at one farm (71%), so
it could support lead day 0 only, with a training window that is 71% full. The CEDA archive holds
UKV runs at 00, 06, 12, and 18 UTC with hourly leads to 48 hours. It is not the live product: its
licence is non-commercial, and the [UKV-CEDA blends study](../../studies/forecasts/ukv-ceda-blends.md)
found it statistically different from the live feed (and UKV was upgraded on 2026-01-21 inside the
validation year). A UKV-CEDA gain therefore says what a matched UKV might add, not what a
deployable one would. The experiment measures how many CEDA runs lie in the ENS window before any
UKV fit, and does not run the UKV arms when fewer than 200 00 UTC runs have a complete wind set.

**The latency each aligned product has live is stated, not measured.** The service reads ENS at
09:00 UTC. The experiment assumes that the 00 UTC runs of UKV, IFS HRES, and GFS are available
within 8 hours of their start, from the providers' published schedules, and does not measure when
any of them publishes.

## Caveats

- **ENS-trained models are weaker for wind.** Experiment 1 found the ERA5-trained wind model beat
  the ENS-trained model by 1.70 points, so arm A and its relatives sit above the best-guess
  reference row.
- **The products' histories are short.** Training uses at most 456 daily runs (about 15 months) per
  product; the cases of the validation year are 53 Tuesday runs for three farms, fewer for IFS HRES
  (51, because 2 Tuesday runs are missing) and for UKV (to be measured).
- **Model upgrades inside the windows.** IFS cycle 50r1 went live on 2026-05-12 and UKV was
  upgraded on 2026-01-21, both inside the validation year, so part of the validation year runs on a
  version the models never saw in training.
- **The cases are the same as experiment 52's**, so the contrasts are not confirmations on fresh
  data.

## Sizing

- **What gets stored:** private prediction tables under `data/ml-experiments/`. Not a trigger.
- **Serving path:** untouched. A `supported` verdict would add a weather feed and a model input.
- **Degradation rule:** untouched. A missing product run would fall back to arm A.
- **Defensible designs:** more than one (a product as extra columns, as a second model in a blend,
  or as a calibrator input). The plan follows the maintainer's request for arms A, B, and C.
- **Callers:** the experiment copies experiment 33's N1 recipe and experiment 52's scoring code,
  which it may not import.
- **Size: complex** (two triggers). One Sonnet plan review, one Sonnet code review, one Sonnet prose
  review, and one Opus result review only if the result needs it.

## Design details

- **Cases:** the half-hourly cases of experiment 33's stored `bg_full` table (three farms, three
  seeds, the Tuesday runs), restricted to each product's leads. A half-hour valid time averages the
  east-west and north-south wind components of the hours either side (and interpolates linearly
  between the steps of GFS, which is 3-hourly beyond lead 120). A case where the product is absent
  is dropped from all arms of that product.
- **Training rows:** every run of 2024-04-01 to 2025-06-30 for which the product exists, per farm,
  using the ENS control member's half-hourly valid times within the product's leads. The
  `lead_hours`, lag, and recency-weight definitions are experiment 33's. Arms A, B, and C of one
  product train on the same rows.
- **Fits:** the nine-level quantile loss, `eta` 0.05, depth 6, 500 rounds, on the GPU, as in
  experiment 33; seeds 0, 1, and 2.
- **Spread factor and blend:** none. Every arm is scored as the model outputs it.
- **Alignment assertions:** for the aligned products, every run's `init_time` is 00:00 on the ENS
  run's date, so the product's `lead_hours // 24` equals the case's lead day. For ICON-EU, the
  selected run starts at least `24(d+1)` hours before the valid hour by the archive's definition,
  which the matched-lead study checked on GFS for 2 days and this experiment does not recheck.
- **Scores reported:** the absolute quantile score of every arm per lead-day band and per farm; the
  coverage of the 5% to 95% band; the share of cases above the 0.95 quantile; each product's mean
  100 m wind speed (10 m for UKV) on the validation year against ERA5's; the number of training rows
  per arm and the number of runs per product.
- **Placebo:** arm B with the product's columns shuffled among rows of the same farm, month, lead
  day, and time of day, in training and in scoring. It shows how much of any gain comes from the
  extra columns' mere presence.

## Advice followed and departures

- **Followed:** experiment 52's advice to rerun with the product aligned to the ENS run where the
  data allow, and not to fit a spread factor on blends made under other weights.
- **Departed from:** experiment 52's advice to correct each product's wind speed. The trees learn
  each product's bias from its own history, so no separate correction is fitted.
- **Departures from earlier plans:** no ERA5-trained model, blend, or spread step is used; the
  primary contrast is each product's arm B minus arm A.

## Compute

Slot E: 6 cores, GPU at most 12 GB. For each of the four products, each seed fits arm A, arm B, and
arm C for three farms (9 fits), so about 108 fits in all, each on at most about 220,000 rows. No
download.
