# Research plan of the ML experiments

**This page lists the ml-experiments that are planned or proposed, sorted by priority.** The
[research log](research-log.md) holds the finished experiments, their verdicts, and the lessons so
far. The design under test is on the [staged probabilistic
forecast](../roadmap/staged-probabilistic-forecast.md) page. In this plan, CRPS is the continuous
ranked probability score, MAE is the mean absolute error, ENS is the ECMWF ensemble, ERA5 and CAMS
are estimates of past weather, stage 1a is stage 1 as the experiments train it, and ECC-Q is
ensemble copula coupling by sorting quantiles. The maintainer adds proposals to this page, and the
ML experiment manager orders them, merges proposals that test the same hypothesis, and discards the
ones an earlier result has answered.

Each planned experiment has a `result.md` with status `planned`, committed before any code runs. An
idea that has no folder yet is marked "proposed" and carries a tentative id, P1, P2, and so on,
which the sections below use as headings. A proposal receives its numeric id when its plan is
committed, and the tentative id then disappears. A lower number in the "Priority" column runs
sooner.

| Id | Question | Priority | Waits for | State |
|---|---|---|---|---|
| [P2](#p2-embedded-solar-capacity-from-lagged-residuals) | Can summaries of the stage 1a residuals over recent days separate a change in a substation's embedded solar capacity since the stage 1a XGBoost model was trained from a fixed error in the irradiance input? | 2 | Nothing | Experiment 00007 (static CAMS correction) was inconclusive and experiment 00009 (CAMS-trained stage 1a) refuted a gain of 0.2 points; the slope statistics are still proposed and need a plan |

## Deferred until the first results are in

**These proposals wait for a decision on the shape of the production
model.** Each keeps its section below, and a result may promote it.

| Id | Question | Priority | Waits for | State |
|---|---|---|---|---|
| [P11](#p11-ensemble-statistics-as-inputs) | Do ensemble statistics (the member mean, spread, and percentiles of temperature and irradiance) beat control-member rows? | 5 | Nothing | Deferred; an ERA5-trained stage 1 cannot take ensemble spread as an input, and experiment 00018 found the staged chain level with N1 for demand |
| [P13](#p13-perfect-prognosis-and-ceiling-checks) | Does a stage 1 trained on ENS early leads beat the ERA5-trained stage 1 at long leads, and how much skill remains with ERA5, CAMS, or short-lead fields as inputs? | 5 | Nothing | Deferred; 00001 found no gain beyond day 6, and 9 generator series cannot resolve smaller gains |
| [P14](#p14-coupling) | Which coupling step, ECC-Q or the Schaake shuffle, gives better coherence scores at short and long leads? | 5 | The scoring harness and experiment 00016 | Deferred; no consumer needs coherent members yet |
| [P15](#p15-a-second-weather-product-already-on-disk) | For solar and wind, does a shared stage 2 calibrator with a learned blend let a second weather product already on disk (CEDA UKV or Open-Meteo) enter without retraining? | 5 | A decision on the shape of the production model | Deferred; run for solar and wind only, ENS against ENS plus one product, because demand depends less on weather than generation does |
| [P16](#p16-a-longer-ens-history) | Does the ENS backfill (2021-03 to 2024-03) give an extra validation fold, or improve the single-stage or stage 2 models? | 5 | The maintainer, who runs the fetch on a cloud machine | Deferred; tracked in [issue #959 (Extend ECMWF ENS training history)](https://github.com/openclimatefix/nged-substation-forecast/issues/959), and the 51-member fetch cannot run autonomously because it needs a cloud machine in `us-west-2` |

## Proposals in detail

**Resolution limits.** The sections below flag the experiments whose planned contrasts the available
sample cannot resolve. The sample is 16 Primary substations, 9 generator series, and one validation
year with 12 months, and it resolves gains of roughly 0.3 points of capacity or more.

### P2: embedded-solar capacity from lagged residuals

**The embedded-solar proposal rests on one follow-up check of experiment 00006's residuals, which
motivated two different questions. Experiment 00007 then ran a first test and was inconclusive.**
The check was run by the 00006 session on CAMS irradiance at each demand series' own location, and
its figures and script are in the write-up of [PR #1202 (Experiment 6: which input features work
best for the ERA5-trained demand
model)](https://github.com/openclimatefix/nged-substation-forecast/pull/1202). It was not
pre-registered. At Primary substation 12, the median-error Primary substation, the residual of the
chosen-feature stage 1a XGBoost model averaged -5.3 points of capacity from 08:00 to 16:59 in the
June, July, and August of the confirmation year (July 2025 to June 2026), against -1.0 at night. In
July 2025, a linear fit on 431 daylight hours explained 25.5% of the hourly residual's variance with
irradiance from the Copernicus Atmosphere Monitoring Service (CAMS), 3.8% with ERA5 shortwave, and
43.3% with both. Across the 14 Primary series that have CAMS values, CAMS alone explained more than
ERA5 alone in all 14: a median of 23.7% against 2.9%, and 39.8% with both. In one series the error
is a level shift that irradiance does not explain, and in another the residual correlates positively
with CAMS irradiance.

**The check cannot yet say whether the cause is embedded solar capacity or an error in ERA5's
irradiance.** At Primary substation 12 the joint fit gives coefficients of opposite sign and similar
size for ERA5 and CAMS, and CAMS minus ERA5 alone explains 33.9%, so much of the residual tracks the
difference between the two irradiance products. Stage 1a has already learned an irradiance response
from ERA5, so a fixed error in ERA5's cloud field would produce the same pattern without any change
in solar capacity. Daytime lighting load and behaviour on sunny days could also contribute. At
Primary substation 12, CAMS explains 25% of the variance on weekdays and 10% on weekends, so the
weekday mix matters too. The check has no time dimension, because it covers one month, one seed, and
one fold, and with hourly autocorrelation the effective sample is far below 431 hours. Because stage
1a already uses ERA5 shortwave, the comparison of 25.5% with 3.8% is not like for like, and the
partial R² of CAMS given ERA5 would be a fairer figure.

**Experiment 00007 was the cheap first test of an irradiance error against a capacity change.** It
added CAMS irradiance to ERA5 shortwave and fitted a static correction on one year of residuals. The
correction removed 46.6% of the next year's daylight squared residual against the 50% planned, and
one substation sets the pooled figure, so the verdict is inconclusive. A further test could compute
the three candidate statistics on residuals from walk-forward (out-of-fold) stage 1a fits, because
residuals from a model fitted on the same period are shrunk and would look different in operation.

**The three candidate statistics each compare what stage 1a predicted with what the substation
metered.** Each uses the residual, the metered power over capacity minus stage 1a's prediction at
the observed weather.

- **The solar slope:** regress the residual on irradiance over the trailing 14 to 28 days, in
  daylight hours only. A more negative slope means more embedded solar than stage 1a learned. The
  slope mixes capacity change, irradiance error, and other demand that varies with irradiance, so
  the regression could use the clear-sky index with day-of-week and temperature controls. Noise in
  the estimation irradiance biases the slope towards zero.
- **The sunny-minus-cloudy daytime residual:** the mean residual on sunny days minus the mean on
  cloudy days at matched hours, with a named clear-sky-index threshold. A day-minus-night residual
  is not a substitute. Commercial load is daytime-heavy, electric-vehicle charging and storage are
  night-heavy, and heat pumps are daytime and seasonal, so their trends would appear as false solar
  change.
- **The change from the same calendar window a year earlier:** this year's 28-day slope minus the
  slope in the same 28 days of the previous year. A ratio of the 28-day to the 365-day slope is
  unstable, because the slope is seasonal and the denominator can be near zero or change sign.

**The correction could be given to stage 2 as a linear offset, not as a tree feature.**
Gradient-boosted trees cannot extrapolate beyond their training range, and embedded solar capacity
grows steadily, so a product of the slope and the forecast shortwave would still be held flat by a
tree. A linear term, in stage 1b or in stage 2, would extrapolate correctly, and the tree feature
would be the comparison arm. The slope is estimated against observed irradiance and applied to
forecast irradiance, which differ in bias and scale, so the conversion has to be stated. The
past-weather source used at lag times has to be available at forecast time, because ERA5 arrives
about 5 days late, and training has to use that same source. CAMS is observed after the fact and has
no forecast, so the CAMS satellite-derived irradiance can feed the statistics only at lag times. The
CAMS irradiance at each demand series' own location comes from [PR #1201 (Download and validate CAMS
irradiance at the demand series'
locations)](https://github.com/openclimatefix/nged-substation-forecast/pull/1201), which covers the
22 series of experiment 00003 (16 Primary demand series, 4 bulk-supply and grid-supply point flows,
and 2 non-demand series, a battery and a biofuel generator). The experiment could compute each
statistic from ERA5 alone and from CAMS and ERA5 together.

**A decision rule could be fixed before any fit.** One candidate: the trend in the monthly change
from a year earlier is negative at 70% or more of the substations in the top quartile of registered
embedded solar capacity per unit of substation capacity, with a block-bootstrap 95% interval that
excludes zero, and it is absent in the night-time residuals. A second candidate: the linear-offset
arm lowers the confirmation-year daylight quantile score at Primary substations by at least 0.2
points of capacity, with a month-block bootstrap interval that excludes zero. Otherwise the
hypothesis is refuted. The thresholds are proposals for the planning step to confirm.

**Resolution limit.** The top quartile of 16 Primary substations by embedded solar capacity is 4
substations, so a rule of "70% of them" means 3 of 4, which no interval can support. The slope
statistics need more substations or a continuous measure of capacity.

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

### P14: coupling

**The coupling step turns calibrated quantiles back into 51 coherent members.** The arms are ECC-Q
with the interpolation, tail, and tie rules of the [staged forecast roadmap
page](../roadmap/staged-probabilistic-forecast.md), and the Schaake shuffle, scored by the variogram
score and the CRPS of summed solar farms in a weather cell, by lead band.

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

### P16: a longer ENS history

**The ENS backfill from Source Cooperative (2021-03 to 2024-03) could give an earlier validation
fold, and the full fetch cannot run autonomously.** [Issue #959 (Extend ECMWF ENS training
history)](https://github.com/openclimatefix/nged-substation-forecast/issues/959) describes the
fetch. The archive is in `us-west-2`. Fetching the control member alone is about 530 GB and 4 to 5
hours, and fetching all 51 members is about 30 TB, which needs a cloud machine in that region and
the maintainer's approval of the cost. The issue also ties autonomous use of the fetch to the
scorer-protection work in issue 958. Every experiment since 00003 scores the same year on the same
16 series, so choices tuned there overfit it, and an earlier fold is the remedy. For demand, a
longer training history has hurt so far, so the backfill is valuable first as validation data.
