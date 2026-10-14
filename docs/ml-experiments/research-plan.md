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
| [P6](#p6-primary-substation-demand-against-the-stored-baselines) | Where does Primary substation demand stand: how do `manual_heuristic`, `climatology`, the production recipe, an upgraded single-stage model (N1: a quantile model trained on ENS with the 00006 columns, recency weights, a 2023-09 start, and lags measured from the forecast origin), and stage 1 without lags score on one row set? | 1 | Nothing | Proposed; the three baselines are already stored |
| [P2](#p2-embedded-solar-capacity-from-lagged-residuals) | Can summaries of the stage 1a residuals over recent days separate a change in a substation's embedded solar capacity since the stage 1a XGBoost model was trained from a fixed error in the irradiance input? | 2 | Nothing | Experiment 00007 (static CAMS correction) was inconclusive and experiment 00009 (CAMS-trained stage 1a) refuted a gain of 0.2 points; the slope statistics are still proposed and need a plan |
| [P7](#p7-the-staged-chain-against-the-upgraded-single-stage-model) | Does the staged chain, with lag anomalies built from the sources serving has, beat the upgraded single-stage model N1 on Primary substations, by lead band? | 2 | P6 | Proposed; the go or no-go for the staged design on demand. It repeats the 00005 lag contrast without the ceiling, on purpose |
| [P8](#p8-training-window-and-drift) | Which training window and retraining rule handle the drift of demand best (sliding windows, recency half-lives, a monthly retrain)? | 2 | P6 | Proposed |
| [P17](#p17-input-features-for-solar-and-wind) | Which input features and which training start are best for the solar and wind stage 1 model (the weather block, the clear-sky ratio target for solar, and a power-curve `base_margin` for wind)? | 2 | Nothing | Proposed; a different data set from P6, so a separate session can run it |
| [P20](#p20-a-cams-trained-stage-1-with-a-stage-0-for-demand) | For Primary, BSP, and GSP demand series, does a stage 1 model trained on CAMS irradiance and fed ENS irradiance for all 51 members beat the ERA5-trained stage 1, and does a stage 0 model that maps ENS irradiance to CAMS irradiance help further? | 2 | Nothing; CAMS values must exist at the BSP and GSP series | Proposed; the follow-up to experiment 00009, which tested Primary substations only, without stage 0 |
| [P3](#p3-historical-electricity-prices-as-lagged-inputs) | Does the intra-day price at time t, with the weather at time t, help a stage 1 demand model explain power at time t on all 20 series, so that a later experiment can estimate the expected past power and measure how unusual the lagged power was? | 3 | Nothing; the Elexon price data is on disk | Proposed; follows experiment 00010, which was inconclusive |
| [P4](#p4-return-to-work-and-school-term-columns-for-the-holiday-periods) | Do columns for the return to work and for school terms lower the stage 1a demand error in the weeks after Christmas, at Easter, and in the school holidays, against the 14-day-window Christmas column of 00006? | 3 | School-term dates | Planned as experiment 00011, in [PR #1214 (Plan experiment 11: school-term and return-to-work columns for the stage 1a demand model)](https://github.com/openclimatefix/nged-substation-forecast/pull/1214), with no fits yet |
| [P18](#p18-lagged-power-and-summaries-of-recent-power-for-solar-and-wind) | Does lagged power in the stage 2 calibrator, or a summary of recent power as tested for demand in 00008, help solar and wind farms, as lagged power helps Primary substations in 00005? | 3 | P7 and P17 | Proposed |
| [P12](#p12-calibration-of-the-pooled-mixture-for-solar-and-wind) | Does a spread factor or the stage 2 calibrator bring the 5% to 95% band of the pooled mixture to 90% coverage for wind and solar, wind first? | 4 | P17 | Proposed; the demand part is in P7 |

## Deferred until the first results are in

**These proposals wait for the answer to P6 and P7, or for a decision on the shape of the production
model.** Each keeps its section below, and a result may promote it.

| Id | Question | Priority | Waits for | State |
|---|---|---|---|---|
| [P9](#p9-the-demand-weather-ceiling) | How much better would the demand forecast be with ERA5 in place of ENS weather at serving time? | 5 | Knowing what the production model will look like | Deferred; its size depends on the shape of the model |
| [P11](#p11-ensemble-statistics-as-inputs) | Do ensemble statistics (the member mean, spread, and percentiles of temperature and irradiance) beat control-member rows? | 5 | P7 | Deferred; an ERA5-trained stage 1 cannot take ensemble spread as an input, so revisit it only if N1 wins P7 |
| [P13](#p13-perfect-prognosis-and-ceiling-checks) | Does a stage 1 trained on ENS early leads beat the ERA5-trained stage 1 at long leads, and how much skill remains with ERA5, CAMS, or short-lead fields as inputs? | 5 | P6 | Deferred; 00001 found no gain beyond day 6, and 9 generator series cannot resolve smaller gains |
| [P14](#p14-coupling) | Which coupling step, ECC-Q or the Schaake shuffle, gives better coherence scores at short and long leads? | 5 | The scoring harness and P12 | Deferred; no consumer needs coherent members yet |
| [P15](#p15-a-second-weather-product-already-on-disk) | For solar and wind, does a shared stage 2 calibrator with a learned blend let a second weather product already on disk (CEDA UKV or Open-Meteo) enter without retraining? | 5 | P7 and P17 | Deferred; run for solar and wind only, ENS against ENS plus one product, because demand depends less on weather than generation does |
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

### P3: historical electricity prices as lagged inputs

**Experiment 00010 ran a first version and was inconclusive; a follow-up on all series is proposed.**
Experiment 00010 added the price from exactly 7 days earlier to the stage 1a demand model and judged
it at the two series known to host batteries, of which only one had training history. The follow-up
the maintainer asked for on 2026-10-14 does not forecast and uses no lags. It trains a stage 1 demand
model that maps the intra-day price at time t, with the weather at time t, to power at time t, on all
20 series, with the pooled effect as the headline. A later experiment could feed that model the
observed prices to estimate the power that should have been metered in the past, and so measure how
unusual the actual lagged power was. The follow-up keeps the shuffled-price control and adds the
climatology row that 00010 lacked.

### P4: return-to-work and school-term columns for the holiday periods

**Experiment 00006 found that the windowed days to Christmas lowers the Primary error in the weeks
around Christmas, and left the error after 5 January unchanged.** Over the three Christmases scored,
the Primary error with the windowed column was 6.4 to 9.3 points of capacity from 25 December to 1
January, against a base of 8.8 to 14.7. From 5 to 8 January it was 9.3 to 10.3, against a base of
9.8 to 10.5, so the column gains almost nothing there. Each of those days is a mean over three
Christmases. A day count from 25 December cannot encode a return to work or to school that is
staggered across a week or more. Easter and the summer school holidays have no column that encodes
them, and the existing holiday flags help Primary substations at Easter and hurt bulk supply points
and grid supply points.

**The plan has to fix its windows and contrasts before any fit.** A candidate design compares the
windowed Christmas column of 00006 with the same column plus the days since the first working day
after New Year, plus a school-term column by nation and by local authority, scored on the Christmas,
Easter, and half-term windows over all three folds, with a day-count control and a half-year control
as in the Christmas control of 00006. School-term dates have to be collected and checked by hand,
which is the data step the 00006 screen left out. The experiment stays out of 00006 because a column
designed after seeing early-January errors cannot be pre-registered there.

**Resolution limit.** All 20 series are in Lincolnshire, so the school-term column takes the same
value for every series and carries no variation across series. The Easter and half-term effects each
rest on three instances. The experiment can resolve the Christmas effect only, and it cannot
separate a school effect from a calendar effect.

The [quick-wins study](../studies/forecasts/quick-wins.md) found that holiday and ordinal-day
columns did not help demand on 15 months of ENS-era data (+0.06 points of CRPS), while experiment
00006 found that a windowed Christmas column helped on a longer history.

### P6: Primary substation demand against the stored baselines

**The leaderboard already holds the baselines.** The `forecast_metrics` Delta table has scores for
`manual_heuristic`, `climatology`, and the production recipe on the 16 Disaggregated Demand series
(the Primary substations) for the fold `mid_2025_to_mid_2026`. The mean normalised MAE over all
horizons is 0.086 for `manual_heuristic`, 0.108 for the production recipe, and 0.113 for
`climatology`, so the production recipe is about 2 points of capacity worse than `manual_heuristic`
in every horizon slice. The stored forecasts in `power_forecasts` allow re-scoring on the row set of
the experiments, `fold_a_every5d_v1`. The arms are the three baselines re-scored; N1, a quantile
model trained on ENS with the 00006 columns, recency weights, a 2023-09 start, and lags measured
from the forecast origin (the latest observed value at this half-hour before the origin, and the
same weekday in the latest week, allowing for NGED's real telemetry delay); and stage 1 pooled over
members without lags. The planned contrast is the best arm against `manual_heuristic`, in MAE and
CRPS by lead band. P6 tells the project roughly how well it forecasts substation demand, and it sets
the bar that every later experiment has to beat.

P6 also diagnoses why the production recipe loses to `manual_heuristic` by about 2 points of
capacity in every horizon slice, because that gap could be the largest mean absolute error gain
available.

**The quick-wins study already answered several questions that N1 should inherit and not repeat.**
The [quick-wins study](../studies/forecasts/quick-wins.md) found these results. Quantile loss at 9
levels lowered CRPS by 2.39 points against production, and the absolute-error objective lowered it
by 0.42 points, so N1 uses quantile loss and takes its median as the point forecast. Early stopping
lowered CRPS by 0.38 points. One model per type with a site identifier was the best quick-wins arm,
and experiment 00006 later found one model per Primary substation slightly better (7.11 against 7.34
points of capacity), by a small unpaired margin whose intervals overlap. Cutting the training window
from 15 to 12 months raised CRPS by only 0.07 points. The study left a climatology column (built so
that a row's own value never enters its own mean), training on CRPS, and features from the ensemble
itself as ideas for future work.

### P7: the staged chain against the upgraded single-stage model

**P7 is the go or no-go for the staged design on demand.** It compares the chain (stage 1 plus the
stage 2 calibrator) with N1 from P6, by lead band. Every lagged arm of experiment 00005 is a
ceiling, because the lag anomaly (observed power at a lag time minus stage 1's expectation at that
time) used ERA5 weather at past times. P7 builds the lag anomalies from the sources serving has:
ERA5T and CAMS at their serving latency, and the analysis proxy (the early leads of the latest ENS
run), and it allows for NGED's real telemetry delay at the forecast origin, which 00005 assumed
away. The arms are lag anomalies from ERA5, from the serving sources, from the analysis proxy, and
none (raw lagged power only). P7 also adds a spread-factor arm, because the 5% to 95% band of the
Primary substations covers only 83% to 85%. The planned contrasts are the chain with serving-latency
lag anomalies against the lag-free calibrated stage 1 (the 1.80-point contrast of 00005, measured
again), and against N1. A chain that does not beat N1 on Primary substations leaves the staged
design for demand, and stage 2 stays only for the multi-product work.

### P8: training window and drift

**Experiments 00003 and 00006 both point at drift as the largest lever.** The arms are sliding
windows of 6, 12, and 18 months, recency half-lives, and a monthly retrain against a frozen model,
scored rolling-origin on the winner of P6 or P7.

**Resolution limit.** One confirmation year cannot rank windows of 6, 12, and 18 months, because the
ranking would rest on one year of drift. A cheaper design scores stage 1a on ERA5 with rolling
origins over 2021 to 2026, as experiment 00006 did, which needs no ENS data and gives several years
of drift.

**The recency result differs between the quick-wins study and experiment 00006, so P8 repeats the
question on purpose.** The [quick-wins study](../studies/forecasts/quick-wins.md) found that recency
weights with a half-life of 120 days did not help on 15 months of ENS-era data (+0.05 points of
CRPS), and that cutting the window from 15 to 12 months changed CRPS by only 0.07. Experiment 00006
found that recency weights with a half-life of 2 years over 7 years of ERA5 helped. The two results
use different data, so P8 repeats the question on purpose.

### P9: the demand weather ceiling

**The experiment sizes how much skill demand gains from better weather.** The experiment gives N1
ERA5 in place of ENS weather at serving time, by lead. A small gap means weather-source work (P15,
P16) has low priority for demand.

### P11: ensemble statistics as inputs

**The arms are the member mean, spread, and 10th and 90th percentiles of temperature and irradiance,
against rows from the control member only.** The planned contrast is the statistics arm against the
control-member arm.

### P12: calibration of the pooled mixture for solar and wind

**Calibration is open for every technology.** Experiment 00006 found the 5% to 95% band of the
Primary substations covering about 83% to 85%, and the closed study of pipeline fixes (PR #1175)
found pooled wind quantiles under-covering and solar 80% bands covering above their label. The arms
are no correction, a scalar spread factor per technology, a location-and-scale calibrator, a free
quantile calibrator, and, for solar and wind, the PV ratio target and the wind `base_margin`. The
planned contrast is coverage and CRPS of the best arm against no correction, demand first.

The demand part of the calibration question moved into P7.

**Resolution limit.** 9 generator series in two weather regions cannot resolve coverage
differences of a few percentage points, so the result should be read per technology and as
descriptive.

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

### P17: input features for solar and wind

**An arm for the 6 solar farms trains stage 1 on CAMS irradiance in place of ERA5 shortwave.** The
maintainer asked for it on 2026-10-14, after experiment 00009 found no gain from a CAMS-trained stage
1a for demand (-0.035 points). The arm applies the CAMS-trained model to ENS irradiance, and a
variant adds a small stage 0 model that calibrates ENS irradiance to CAMS first. The two contrasts
are kept separate: CAMS-trained against ERA5-trained, then stage 0 against plain CAMS-trained.

**No experiment has screened inputs for solar or wind.** Experiments 00001 and 00002 used one fixed
list of 9 columns for the 6 solar farms and 6 columns for the 3 wind farms, with no calendar column
and no power lag. The screen follows 00006: select on 2024-07 to 2025-06 only and confirm once on
2025-07 to 2026-06, with the windchill-style columns in the production recipe tested for use, the
clear-sky ratio target for solar (power divided by clear-sky power, weighted by clear-sky power),
and a `base_margin` from a power curve of wind speed for wind. 9 generator series in two weather
regions cannot resolve gains under about 0.3 points of capacity, so the experiment reports every
score per technology and flags the claim as resting on 6 solar farms and 3 wind farms.

**The experiment also tests the training start for generators.** Generators change less over the
years than demand does, so a longer training window may help them where it hurt demand. Experiment
00001 found that a stage 1 trained on only the ENS-era months gave the same errors as one trained on
7 years, which is evidence of no difference, not of a benefit, so the test is an open one. The arms
are a start in 2019-09, a start in 2023-09, and the ENS-era months only, scored with the same
contrast on 9 generator series.

**Resolution limit.** 6 solar farms in one weather region and 3 wind farms cannot resolve gains
under about 0.3 points of capacity, so the claim rests on those farms.

### P20: a CAMS-trained stage 1 with a stage 0 for demand

**Experiment 00009 found no gain from training stage 1a on CAMS irradiance for Primary demand, and
this follow-up adds a stage 0 model and the other substation types.** The maintainer asked for it on
2026-10-14. Stage 1 trains on CAMS irradiance and is fed ENS irradiance for each of the 51 members
at inference. Stage 0 is a small model that maps ENS irradiance to CAMS irradiance, trained on the
period when both exist, and its output feeds stage 1. The arms are ERA5-trained stage 1,
CAMS-trained stage 1 without stage 0, and CAMS-trained stage 1 with stage 0, on Primary, BSP, and
GSP series, scored by type. Two contrasts, kept separate: CAMS-trained against ERA5-trained, then
stage 0 against plain CAMS-trained. The plan has to check first that CAMS values exist at the BSP
and GSP series ([PR #1201](https://github.com/openclimatefix/nged-substation-forecast/pull/1201)
covers the 22 series of 00003), and has to reuse the code of 00009.

### P18: lagged power and summaries of recent power for solar and wind

**Generators depend on weather more than demand does, so lags may help less.** The experiment
repeats the 00005 comparison for solar and wind farms: no lags, lagged power in the stage 2
calibrator, and lag anomalies built from the sources serving has (P7). It also repeats experiment
00008, which tested summaries of recent power (the minimum, mean, and maximum of the latest 24 hours
and of the latest day on the same weekday) for demand: raw lags against summaries against both.
Lagged power can also carry curtailment and capacity that the weather does not explain. The planned
contrast is the lagged calibrator against the lag-free calibrated stage 1, per technology.

**Resolution limit.** With 9 generator series, a lag effect smaller than about 0.3 points of
capacity will not be resolved, so the result should be read per technology.
