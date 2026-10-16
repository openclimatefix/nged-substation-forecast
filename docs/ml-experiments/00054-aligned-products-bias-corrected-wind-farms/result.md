---
id: "00054"
slug: aligned-products-bias-corrected-wind-farms
status: planned
size: complex
hypothesis: >-
  For the three metered wind farms, averaging the quantiles of the experiment 33 wind best guess
  with the quantiles of the same ERA5-trained wind-power model fed the 00 UTC runs of IFS HRES 9 km
  and GFS from the ENS run's own date, after a per-farm, per-lead-day correction of each product's
  wind speed fitted on the training window, lowers the quantile score at lead days 1 and 2.
prediction: >-
  Lead days 1 and 2: -0.10 to -0.30 points of capacity against the best guess, because experiment 52
  fed runs about half a day older and uncorrected winds and found -0.055 (-0.200 to +0.072). The
  interval on three farms is likely to include zero again, so `inconclusive` is the likeliest
  verdict. All leads: -0.10 to +0.05. The correction should move each product's own score towards
  the best guess's, and the unreliable days 3 to 6 reading of experiment 52 should disappear because
  the primary arm has no spread step.
decision_rule: >-
  One decisive contrast, C1: the primary blend minus the best guess on the quantile score for cases
  whose valid time falls 1 or 2 calendar days after the date of the ENS run, read at 95% with whole
  months and the three seeds resampled, 50,000 resamples. The interval is conditional on the three
  farms. The threshold is 0.15 points, as in experiment 52, because the blend adds two weather feeds
  and two pipeline steps to the serving path, and a smaller gain does not justify them. C1 is
  `supported` when its interval lies wholly below -0.15, `small gain` when it lies wholly below zero
  but not wholly below -0.15, `refuted` when it lies wholly above zero, and `inconclusive`
  otherwise. `supported` also needs the primary blend's 5% to 95% coverage at lead days 1 and 2 to
  be at least 0.87 (nominal 0.90), because averaging a wide ensemble mixture with narrow product
  quantiles can under-cover; a blend below that is `small gain` at best. C2 (the same difference over all leads) is reported beside C1 and does not change the
  verdict. Every other read (per product, per farm, per lead day, uncorrected, with the spread step)
  is exploratory, labelled so, and read at 95%.
benchmark: >-
  The experiment 33 wind best guess (`bg_full`) on the same cases.
row_set_id: fold_a_tuesdays_v1
metric: >-
  Quantile score (twice the mass-weighted mean pinball loss over nine levels, the estimate of the
  CRPS), in points of capacity; coverage of the 5% to 95% band; share of cases above the 0.95
  quantile
production_status: not-for-production
adopted_in: none
code: ml-experiments/00054-aligned-products-bias-corrected-wind-farms/
---

# Experiment 54: do aligned, bias-corrected forecasts lower wind-farm error at 1 to 2 days?

**Status: planned. No code has run.** [Experiment 52](../00052-more-nwp-products-for-wind-farms/result.md)
could not tell whether ICON-EU and IFS 0.25 degree help the wind-farm forecast, and it fed runs
about half a day older than the ENS run, with no correction for each product's wind-speed bias.
This experiment asks the same question with runs that start at the same time as the ENS run (00 UTC
on the ENS run's date) and with a simple correction of each product's wind speed. It follows the
manager's request for a rerun, and experiment 52's advice (rerun at the 00 UTC runs, correct bias,
fit no spread step on blends made under other weights).

## Can the alignment be done honestly with data on disk?

**Partly: IFS HRES 9 km and GFS can be aligned, and ICON-EU cannot.** The three products on disk
are read as follows.

| Product | On disk | Run read | Wind variables | Used |
|---|---|---|---|---|
| IFS HRES 9 km (Open-Meteo Single Runs) | 2024-03-14 onwards, every day except 7 | 00 UTC run of the ENS run's date, hourly leads 0 to 240 | 10 m and 100 m speed and direction | Primary |
| GFS (Dynamical.org native store) | 2021-05 onwards | 00 UTC run of the ENS run's date, hourly to lead 120 and 3-hourly after | 10 m and 100 m u and v | Primary |
| ICON-EU | Previous Runs archive only | Not selectable: `previous_dayN` picks the freshest run at least N times 24 hours before the hour, so the run's age varies by the hour of day | n/a | Not used |
| AIFS Single | Dynamical.org, version changes in the validation year (PVDIS-8) | 00 UTC run, 6-hourly | u and v at 10 m and 100 m | Not used: needs its own rows |

**The latency each product has live is stated, not measured.** The ENS run is read at 09:00 UTC. The
experiment assumes that the 00 UTC runs of IFS HRES and GFS are available within 8 hours of their
start time, from the providers' published dissemination schedules. The experiment does not measure
when either product publishes, and the assumption would need to hold in the live service. ICON-EU's
nwp-archive starts in October 2026, so no ICON-EU run is on disk at its own initialisation time for
the validation year; an ICON-EU arm is not run.

## Sizing

- **What gets stored:** private prediction tables under `data/ml-experiments/`. Not a trigger.
- **Serving path:** untouched. A `supported` verdict would add two weather feeds.
- **Degradation rule:** untouched. A missing product run keeps the best guess, as here.
- **Defensible designs:** more than one (a correction by speed ratio, by quantile mapping, or by a
  trained calibrator; a weight per lead band or one weight). The plan picks the simplest correction
  that experiment 52's weight and band machinery can carry.
- **Callers:** the experiment copies experiment 52's code, which it may not import.
- **Size: complex** (two triggers). One Sonnet plan review, one Sonnet code review, one Sonnet prose
  review, and one Opus result review only if the result needs it.

## Design

- **Cases and benchmark:** the 53 Tuesday ENS runs of `fold_a_tuesdays_v1` for the three wind farms,
  the half-hourly cases of experiment 33's stored `bg_full` table, three seeds. A lead day counts
  calendar days from the date of the ENS run to the valid time.
- **Product forecast:** each product's 00 UTC run of the ENS run's date, read at the farm's grid
  point at lead `valid_time - 00:00` of that date. A valid time at :30 averages the east-west and
  north-south wind components of the hours either side (IFS HRES) or interpolates linearly between
  steps (GFS, which is 3-hourly beyond lead 120). Two of the 53 Tuesday runs (2025-08-05 and
  2026-06-23) have no IFS HRES run on disk, so the IFS arm keeps the best guess for them. Cases
  beyond a product's lead also keep the best guess.
- **Bias correction (primary):** for each farm, product, lead day, and height (10 m and 100 m), a
  multiplicative factor on speed equal to the mean ERA5 speed over the mean product speed on the
  whole hours of the training window (valid times before 2025-06-17, where the product exists),
  taken per `lead_hours // 24` of the product run. The fit keeps finite values only (the GFS store
  holds NaN, not null, in some 100 m wind rows), requires at least 500 hours per cell, and falls
  back to the factor pooled over lead days where a cell is thinner. Each factor must lie between
  0.7 and 1.5 or the script stops. The factor is fixed for the whole validation year, so no month
  uses ERA5 weather that arrives after forecast time. Directions are unchanged. An uncorrected arm
  and a quantile-mapped speed correction are exploratory.
- **Wind-power model:** experiment 33's ERA5-trained wind-power model, refitted per seed as in
  experiment 52.
- **Blend:** the average of each quantile level of the best guess and the available products, with
  weights on a grid of quarters chosen per lead-day band (day 0, days 1 and 2, days 3 to 6, days 7
  and later) on earlier months only, as in experiment 52, with ties going to the weights nearest
  equal. July 2025 has no earlier month, so it uses the best guess alone (weights 1, 0, and 0);
  experiment 52's equal weights in July scored +0.188 at lead days 1 and 2 against -0.070 in the
  other months. C1 is also reported without July as a named sensitivity. The primary
  arm applies **no spread step**, because experiment 52's spread step was fitted on blends made
  under other weights and worsened the blend. A blend with the spread step is exploratory.
- **Arms:** primary (best guess + IFS HRES + GFS, corrected, no spread step); the same with one weight
  for all leads; each product alone in the blend; uncorrected; with the spread step; and a placebo
  that shuffles the product forecasts among cases of the same farm, month, lead day, and time of
  day. The placebo must equal the best guess from August onwards, because the chosen weights
  reject it; any other result is a bug. Each product's own forecast, corrected and spread-scaled,
  is also scored.
- **Run alignment:** an assertion on the data checks that every selected run's `init_time` date
  equals the case's ENS run date at 00:00, so the product's `lead_hours // 24` equals the case's lead
  day. The assumed latency of 8 hours cannot be tested on data at rest.
- **Cells:** GFS reads the nearest 0.25 degree cell of its nine-cell crop to each farm
  (`studies.grid_sampling.nearest_cells`); IFS HRES reads Open-Meteo's nearest land cell. The two
  products may therefore read cells of different character.
- **Reports:** absolute scores per arm, lead-day band, and farm; coverage; the share of cases above
  the 0.95 quantile; the chosen weights; each product's mean and upper quantiles of 100 m speed
  before and after the correction against ERA5 on the validation year; IFS bias before and after
  2026-05-12 (IFS cycle 50r1) and by the 2024-11-12 cycle (49r1); July against the other months.

## Limits of the plan

**The cases are the same 53 Tuesdays and three farms as experiment 52, so C1 is not a confirmation
on fresh data.** The choice of no spread step in the primary arm follows experiment 52's exploratory
read on those cases, on the design argument that its spread factor was fitted on blends made under
other weights. The product set also changed after experiment 52 (ICON-EU dropped, IFS HRES 9 km
in). IFS cycle 50r1 went live on 2026-05-12, so about 7 weeks of the validation year run on a cycle
the correction never saw.

## Advice followed and departures

- **Followed:** experiment 52's advice to rerun at the 00 UTC runs and to correct each product's wind
  speed, and not to fit a spread factor on blends made under other weights.
- **Departures:** ICON-EU is dropped, because no aligned ICON-EU runs are on disk. IFS HRES 9 km
  replaces IFS 0.25 degree, because the Single Runs archive holds the 00 UTC run and the Previous
  Runs archive does not.

## Compute

Slot E: 6 cores, GPU at most 12 GB. Refits are three farms times three seeds of small quantile
models on the GPU, as in experiment 52 (about 8 minutes per seed). No download.
