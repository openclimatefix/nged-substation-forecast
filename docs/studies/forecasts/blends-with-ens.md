# Which single weather forecast should be added to ECMWF's ensemble mean first?

<!-- The Summary, the headline figures, Key findings, Results, Discussion, and the Limitations that
depend on the results are written after the fits in `nwp_forecast_comparison_product_blends`. -->

## Introduction

**The question is which one weather forecast, given to an XGBoost model beside the mean of ECMWF's
ensemble forecast (ENS), lowers the power forecast's error most.** The [matched-lead
page](matched-lead.md) scores each weather product alone against the ENS mean. It tests one blend,
of ENS with ICON-EU and IFS 0.25°, at day 1 and day 2. It cannot say which single product to add
first, because the ENS-plus-one-product blends exist only for AIFS Single, and AIFS Single's blends
are not charted against the other products.

**Each blend is one XGBoost model given the ENS mean's weather columns plus one product's weather
columns.** A blend never averages two forecasts. The products are AIFS Single (ECMWF's machine-learned
forecast), ICON-EU, UKV (the Met Office's UK forecast), and WeatherNext 3 (WN3, Google DeepMind's
machine-learned forecast).

| Product | Days the blend is fitted at | Rows | What limits the comparison |
|---|---|---|---|
| AIFS Single | 1, 2, 7, 14 | `single`: 16 months | None beyond the shared limits below. |
| ICON-EU | 1, 2 | `single` | Open-Meteo's archive holds ICON-EU's freshest run at least N days before the valid hour. |
| UKV | 1 | `single` | Open-Meteo's archive fills only `previous_day1` for live UKV. |
| WN3 | 1, 2, 7, 14 | `wn3`: 7 months | The months overlap WN3's training data, and its publication time is not established. |

## Data and methods

**Every blend is compared with an XGBoost model given the ENS mean's columns alone, and with a
control that has the same column count.** The control shuffles the product's columns among hours that
share a generator, a year-month, and an hour of day, so the control carries the product's values
without its timing. A blend is recommended at a lead only if it beats both the ENS mean alone and its
control.

**The rows, folds, settings, seeds, and device are those of the AIFS blends.** The `single` rows
hold the hours whose day-`N` AIFS Single run lies inside its version era, from March 2025 and without
the months 2025-08, 2026-01, and 2026-05. Folds are contiguous blocks of whole months within each
era. Every XGBoost model is fitted on the GPU at two hyperparameter settings with three seeds, and
each interval resamples whole calendar months and one seed. The AIFS Single blends, their controls,
and the ENS mean are read from the saved fits in `nwp_forecast_comparison_aifs_blends`. The fitting
script refits only the (arm, setting) pairs that folder lacks. It raises unless its build stamp (the
inputs' SHA-256, the settings, the seeds, the GPU, and the XGBoost version) equals that folder's, and
unless every new arm holds the saved ENS mean's `(site, time, seed, fold)` keys.

**ICON-EU is read at two leads, and UKV at one.** The optimistic ICON-EU blend reads the product's
day-`N` value beside the ENS mean's day-`N` value. The conservative blend reads ICON-EU's day-`N+1`
value, from the freshest run at least 48 hours before the valid hour (at day 1), as the published
blend P4b does. That run is older than the AIFS Single run, which is the 00 UTC run N days before the
valid day, so the conservative blend disadvantages ICON-EU against AIFS Single. UKV live has only a
day-1 value and no clear run cycle, so the UKV blend is an optimistic upper bound and is never ranked.

**Five contrasts were planned before any fit.** Each is at both hyperparameter settings.

- **C1:** the ICON-EU blend at the conservative lead minus the ENS mean alone, at days 1 and 2.
  The optimistic lead is exploratory.
- **C2:** the AIFS Single blend minus the ENS mean alone, at days 1, 2, and 7.
- **C3:** the UKV blend minus the ENS mean alone, at day 1.
- **C4:** each blend minus its own control.
- **C5:** the AIFS Single blend minus each ICON-EU blend, at days 1 and 2.

Every other number, including the day-14 rows and the WN3 rows, is exploratory.

**The ranking rule was fixed before any fit.** A blend lowers the error at a lead only if
`fit_aifs.lead_verdict` gives that verdict at both settings: the blend minus the ENS mean alone and
the blend minus its control both have an upper 95% bound below zero. AIFS Single ranks above ICON-EU
only if C5 against the optimistic ICON-EU blend has an upper bound below zero at both settings.
ICON-EU ranks above AIFS Single only if C5 against the conservative ICON-EU blend has a lower bound
above zero at both settings. Otherwise the two are not separable. WN3 and UKV are never ranked.

**About 1 verdict in 20 would pass by chance.** The fitting script's report prints how many
intervals the contrasts list, and every ranking is uncorrected for multiplicity.

## Data and code availability

**The AIFS Single, ICON-EU, UKV, and WN3 inputs and every fitted loss are in the private data store.**
Every output carries only the anonymised `site` label of the generator (A to F for solar, W1 to W3
for wind). The code is `studies/nwp_forecast_comparison/fit_product_blends.py` and
`studies/nwp_forecast_comparison/dot_interval_vs_ens.py --blends`, with `fit_aifs.py` and
`packages/studies/`.

## Reproducing the figures

```bash
D=/home/jack/dev/nged-substation-forecast/data/studies
P=$D/nwp_forecast_comparison
OUT=$D/nwp_forecast_comparison_product_blends
uv run python studies/nwp_forecast_comparison/fit_product_blends.py --lookahead-cleared \
  --workers 2 --published-dir $P --output-dir $OUT
uv run python studies/nwp_forecast_comparison/dot_interval_vs_ens.py --blends \
  --output-dir $D/nwp_forecast_comparison_blends_vs_ens_dots
```
