# Weather products and blending: research plan

**A lower number in the "Priority" column runs sooner.**

## Priority

| Id | Question | Priority | Waits for | State |
|---|---|---|---|---|
| [P11](#p11-ensemble-statistics-as-inputs) | Do ensemble statistics (the member mean, spread, and percentiles of temperature and irradiance) beat control-member rows? | 5 | Nothing | Deferred; an ERA5-trained stage 1 cannot take ensemble spread as an input, and experiment 00018 found the staged chain level with N1 for demand |
| [P13](#p13-perfect-prognosis-and-ceiling-checks) | Does a stage 1 trained on ENS early leads beat the ERA5-trained stage 1 at long leads, and how much skill remains with ERA5, CAMS, or short-lead fields as inputs? | 5 | Nothing | Deferred; 00001 found no gain beyond day 6, and 9 generator series cannot resolve smaller gains |
| [P15](#p15-a-second-weather-product-already-on-disk) | For solar and wind, does a shared stage 2 calibrator with a learned blend let a second weather product already on disk (CEDA UKV or Open-Meteo) enter without retraining? | 5 | A decision on the shape of the production model | Deferred; run for solar and wind only, ENS against ENS plus one product, because demand depends less on weather than generation does |

## Planned experiments

<!-- planned-table -->

## Proposals in detail

### P11: ensemble statistics as inputs

**The arms are the member mean, spread, and 10th and 90th percentiles of temperature and irradiance,
against rows from the control member only.** The planned contrast is the statistics arm against the
control-member arm.

### P13: perfect-prognosis and ceiling checks

**Experiment 00001 found no gain from ERA5 training beyond day 6.** The arms are the ERA5-trained
stage 1, a stage 1 trained on ENS early leads, and ceiling inputs (ERA5, CAMS, both, and each
product's short-lead fields).

**Resolution limit.** 9 generator series cannot resolve gains smaller than about 0.3 points of
capacity, and few solar farms share a weather cell.

### P15: a second weather product already on disk

**Blending products is the maintainer's strongest hunch for improving the forecast, and it can be
tested before any ingest.** CEDA UKV and products downloaded from Open-Meteo are already on disk.
The CEDA UKV archive differs statistically from the live feed, so a result may not carry over. The
arms are ENS alone; ENS plus the second product with the shared calibrator and the learned blend;
the per-product-columns control; a fixed equal-weight mixture; and a learning curve at 1, 2, 3, 6,
and 12 months of the second product's history, rotated across seasons. The planned contrast is ENS
plus the second product against ENS alone, with a safe-entry check at 1 and 2 months (the interval
lies inside plus or minus 0.15 points).

**Resolution limit.** With 9 generator series, a safe-entry interval of plus or minus 0.15 points is
not reachable at 1 or 2 months of history. The experiment compares ENS alone with ENS plus one
product, and does not run the learning curve.
