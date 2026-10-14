# Research plan of the ML experiments

**This page lists the ml-experiments that are planned or proposed, in the order they should run.**
The [research log](research-log.md) holds the finished experiments, their verdicts, and the lessons
so far. The maintainer adds proposals to this page, and the ML experiment manager orders them,
merges proposals that test the same hypothesis, and discards the ones an earlier result has
answered.

Each planned experiment has a `result.md` with status `planned`, committed before any code runs. An
idea that has no folder yet is marked "proposed" and carries a tentative id, P1, P2, and so on,
which the sections below use as headings. A proposal receives its numeric id when its plan is
committed, and the tentative id then disappears. A lower number in the "Priority" column runs
sooner.

| Id | Question | Priority | Waits for | State |
|---|---|---|---|---|
| P1 | Do statistical summaries of lagged power (the minimum, mean, and maximum of the most recent 24 hours, and of the most recent day on the same weekday) help, in the placement that 00005 finds best? | 1 | Nothing | Running as experiment 00008, in [PR #1209 (Plan experiment 00008: summaries of recent power in the stage 2 calibrator)](https://github.com/openclimatefix/nged-substation-forecast/pull/1209) |
| [P2](#p2-embedded-solar-capacity-from-lagged-residuals) | Can summaries of the stage 1a residuals over recent days separate a change in a substation's embedded solar capacity since the stage 1a XGBoost model was trained from a fixed error in the irradiance input? | 2 | Nothing for the second stage; the slope statistics wait for the second stage | Its cheap first test, experiment 00007, was inconclusive; the slope statistics wait for a CAMS-trained stage 1a |
| [P4](#p4-return-to-work-and-school-term-columns-for-the-holiday-periods) | Do columns for the return to work and for school terms lower the stage 1a demand error in the weeks after Christmas, at Easter, and in the school holidays, against the 14-day-window Christmas column of 00006? | 3 | School-term dates | Needs school-term dates, and follows 00006 |
| [P3](#p3-historical-electricity-prices-as-lagged-inputs) | Does the historical electricity price, as an input at lag times, reduce the error of the stage 1a XGBoost demand model at substations with price-responsive generation and storage? | 4 | Nothing | Planned as experiment 00010; the price pull is done |

## Proposals in detail

### P2: embedded-solar capacity from lagged residuals

**The embedded-solar proposal rests on one follow-up check of experiment 00006's residuals, which
motivates two different questions and tests neither.** The check was run by the 00006 session on
CAMS irradiance at each demand series' own location, and its figures and script are in the write-up
of [PR #1202 (Experiment 6: which input features work best for the ERA5-trained demand
model)](https://github.com/openclimatefix/nged-substation-forecast/pull/1202). It was not
pre-registered. At Primary substation 12 (P12), the median-error Primary substation, the residual of
the chosen-feature stage 1a XGBoost model averaged -5.3 points of capacity from 08:00 to 16:59 in
the June, July, and August of the confirmation year (July 2025 to June 2026), against -1.0 at night.
In July 2025, a linear fit on 431 daylight hours explained 25.5% of the hourly residual's variance
with irradiance from the Copernicus Atmosphere Monitoring Service (CAMS), 3.8% with ERA5 shortwave,
and 43.3% with both. Across the 14 Primary series that have CAMS values, CAMS alone explained more
than ERA5 alone in all 14: a median of 23.7% against 2.9%, and 39.8% with both. In one series the
error is a level shift that irradiance does not explain, and in another the residual correlates
positively with CAMS irradiance.

**The check cannot yet say whether the cause is embedded solar capacity or an error in ERA5's
irradiance.** At P12 the joint fit gives coefficients of opposite sign and similar size for ERA5 and
CAMS, and CAMS minus ERA5 alone explains 33.9%, so much of the residual tracks the difference
between the two irradiance products. Stage 1a has already learned an irradiance response from ERA5,
so a fixed error in ERA5's cloud field would produce the same pattern without any change in solar
capacity. Daytime lighting load and behaviour on sunny days could also contribute. At P12, CAMS
explains 25% of the variance on weekdays and 10% on weekends, so the weekday mix matters too. The
check has no time dimension, because it covers one month, one seed, and one fold, and with hourly
autocorrelation the effective sample is far below 431 hours. Because stage 1a already uses ERA5
shortwave, the comparison of 25.5% with 3.8% is not like for like, and the partial R² of CAMS given
ERA5 would be a fairer figure.

**A cheap first test separates an irradiance error from a capacity change.** Add CAMS, or CAMS minus
ERA5, as a lag-time input and see whether a static correction removes most of the residual. If it
does, the evidence points to better irradiance input and not to a drift statistic. The experiment
could then compute the three candidate statistics on residuals from walk-forward (out-of-fold) stage
1a fits, because residuals from a model fitted on the same period are shrunk and would look
different in operation.

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

#### Second stage: CAMS irradiance in place of ERA5 for stage 1a training

**A second stage of P2 could train stage 1a on CAMS irradiance, where the first test uses CAMS only
to read the residuals.** The stage 1a XGBoost model would train with CAMS shortwave in place of
ERA5's and be applied to ENS weather members, with the other weather columns staying ERA5. CAMS has
no forecast, so it can be an input in training only. This departs from the requirement above that
training use the source available at forecast time, and it creates a mismatch: the stage 1a XGBoost
model trains on satellite-derived irradiance and is applied to forecast irradiance. The comparison
shows what the departure costs or gains. The arms would be the ERA5-trained stage 1a XGBoost model
and the same model with CAMS in place of ERA5 shortwave, on the 20 series of experiment 00005 (16
Primary, 2 bulk-supply point, and 2 grid-supply point series), using experiment 00005's folds and
quantile score.

**The outcome of experiment 00007, the first test of P2, is the proposed gate for the second
stage.** If the outcome finds that a static CAMS correction removes most of the residual, which
points to an irradiance error, the second stage would ask whether training on CAMS removes the
residual at the source. If the residual remains, or the cause is a change in solar capacity, the
second stage would not be worth running. A candidate decision rule: the CAMS-trained arm lowers the
quantile score at Primary substations by at least 0.2 points of capacity, with a month-block
bootstrap interval that excludes zero. The threshold is a proposal for the planning step to confirm.

### P3: historical electricity prices as lagged inputs

**The price proposal asks a different question from price forecasting.** Future electricity prices
are hard to predict, so a price forecast is not the input. The proposal is to use prices that have
already been published, at lag times, as inputs that carry the response of price-dependent
generation and storage to the recent past, for example a battery that charges when prices are low.
The plan has to name the price. GB day-ahead auction prices for a day are published the day before,
so for leads inside that window the known price, not a lagged one, is the relevant input. The system
price, the intraday price, and a retail time-of-use tariff are other candidates.

**The recommended price is Elexon's Market Index Data, with the N2EX day-ahead price as the
fallback.** The intraday price that batteries trade against is mostly behind a paywall: the EPEX
continuous intraday and 30-minute auction prices are sold through the [EEX
webshop](https://webshop.eex-group.com/eex-public-market-data) from €665 a month, which is over the
£30 spending limit and needs the maintainer's approval. The nearest free series is the [Market Index
Data](https://www.elexon.co.uk/bsc/glossary/market-index-data/) that EPEX SPOT and Nord Pool supply
to Elexon: a half-hourly volume and price for the short-term wholesale markets, served without a key
by the [Elexon Insights API](https://www.elexon.co.uk/bsc/data/kinnect-insight-solution/) at
`/balancing/pricing/market-index`. It is known soon after each half-hour, so lagging it matches
what a forecast could use. The [N2EX day-ahead price](https://www.neso.energy/data-portal/day-ahead-power-exchange-prices-nordpool)
is free from NESO, hourly, and known at about 10:00 the day before ([GB price
page](../background/gb-price-data-and-ancillary-markets.md)).

**Three facts about the recommended series are unchecked, and the plan has to settle them before
any run.** Elexon's pages do not say which products the Market Index covers (day-ahead, within-day,
or both) or when its history starts, so the plan should pull 2019 to 2026 and confirm the coverage,
the product mix, and the periods in which one provider reported zero volume. The system price is
not a candidate input, because the [Elexon pricing page](https://www.elexon.co.uk/settlement/imbalance-pricing/)
describes it as set by balancing actions and a battery earns little from imbalance
([how GB batteries schedule
themselves](../background/gb-battery-scheduling.md)). The claim that batteries respond to the
intraday price rests on [Modo Energy's
description](https://modoenergy.com/research/en/gb-research-roundup-january-2025-battery-energy-storage-great-britain-revenues-markets-wholesale-capacity-market-balancing-mechanism)
of its own dispatch model, which re-optimises every 2 hours, so it is a modelling assumption rather
than a measurement. Retail time-of-use tariffs were not researched because grid-scale batteries do
not trade against them.

**The main risk is confounding, so a gain would not by itself show a battery response.** Price
follows national demand and wind and correlates with the clock, so any gain may come from those
proxies. The comparison arm is the chosen stage 1a feature set with clock and weather columns and no
price. The substations with price-responsive generation and storage have to be listed in advance
from data independent of the outcome, such as the lists behind [embedded battery
forecast](../studies/embedded-battery-forecast.md) and [unmetered battery
capacity](../studies/unmetered-battery-capacity.md). A candidate rule: the gain is concentrated at
the listed substations and is less than half as large at the others, because an equal gain at both
groups would refute the battery-response mechanism even if accuracy improves.

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
