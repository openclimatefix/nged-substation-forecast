# Solar disaggregation: research plan

**This page lists the experiments in this theme that are planned or proposed.** The
[Experiments](experiments.md) page holds the finished ones. A lower number in the "Priority" column
runs sooner.

## Priority

| Id | Question | Priority | Waits for | State |
|---|---|---|---|---|
| [P2](#p2-embedded-solar-capacity-from-lagged-residuals) | Can summaries of the stage 1a residuals over recent days separate a change in a substation's embedded solar capacity since the stage 1a XGBoost model was trained from a fixed error in the irradiance input? | 2 | Nothing | Experiment 00007 (static CAMS correction) was inconclusive and experiment 00009 (CAMS-trained stage 1a) refuted a gain of 0.2 points; the slope statistics are still proposed and need a plan |
| [PVDIS-T](#pvdis-t-the-capacity-table-a-study) | What is the unmetered solar capacity behind each Primary metered in MW, with error bars and outside checks? | 2 | The maintainer's review of this plan | Proposed; a study under `docs/studies/`, season-year table first from experiment 00040's outputs, weekly after PVDIS-4 |

## Planned experiments

<!-- planned-table -->

## Proposals in detail

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

### PVDIS: separating embedded solar from Primary demand

**The PVDIS line estimates how much unmetered solar capacity sits behind each Primary substation,
how that capacity changes over time, and whether an explicit solar term improves or explains the net
demand forecast.** The line serves three purposes, and the second matters about as much as the
first:

- **Skill:** a lower error in the net demand forecast at each substation.
- **Explanation:** a forecast that NGED can read as demand minus solar. A solar term that leaves the
  error level but splits the forecast into parts an engineer recognises is still worth keeping, so
  experiment 64 (PVDIS-5) tested non-inferiority as well as gain.
- **Capacity:** a time series of unmetered solar capacity per Primary. NGED holds almost no
  information on unmetered solar today, so an estimate with honest error bars is useful even when
  the estimate is far from perfect.

**The capacity estimate runs offline, weekly or monthly, and the production forecast only reads the
latest capacity table.** Nothing in this line adds a fitting step to the serving path. [Experiment
73](../../00073-pvdis-6-orientation-mix-and-dc-ac-ratio/result.md) (PVDIS-6) found that the mix of
panel orientations behind each Primary, the quantity the `UniversalSolarFleetNode` sketched in
[differentiable physics](../../../techniques/differentiable-physics.md) would fit, is not stable
across season-years, so the explicit solar forecast keeps one south-facing shape.

**Experiments 00039 and 00040 are parts 1 and 2 of the series, and the parts below follow the
exploratory protocol in the `ml-experiment` skill.** Opus plans each part after reading the result
of the part before, so the designs below are proposals that each part's plan may change. Experiment
00039 fitted one capacity per Primary per season-year, from a calendar and temperature demand term
minus one CAMS-driven solar shape. On real solar-farm output injected into the real demand, the
half-hourly fit recovered a median 0.773 (0.740 to 0.818) of the injected capacity, and a fit on
daily sums recovered 0.854 (0.815 to 0.866). Experiment 00040 compared six fits, which differ in how
much cloud-driven demand each allows, and took the six-arm capacity C as its estimate. C recovered a
median 0.992 (0.951 to 1.032) of the injected capacity. Experiment 00040 also attached a bound on
the error that an undetected demand load would cause, of 0.18 to 0.23 at five Primaries. A demand
load that rises with sunshine cannot be told apart from solar.

**The Opus review of experiment 00040 recommended stopping the series, publishing a season-year
capacity table, and not using capacity as a forecast feature; the parts below continue because the
goals changed.** That review judged the series against the questions it had then: whether a further
experiment could narrow the capacity bound, and whether capacity improves the forecast score. Four
experiments (00007, 00009, 00022, and 00031) found small or no gains from solar inputs. The
maintainer has since added two goals: a forecast that explains itself, and a capacity series that
changes week by week for NGED. Experiment 64 (PVDIS-5) rested on the explanation goal and on what
differs from those four experiments, not on a claim that the review missed the forecast question.
The same review found that the daily fit is about 2.6 times more exposed than C to an undetected
day-long load, so every part reports C as the capacity and keeps the daily fit as a cross-check.

**Rules shared by every part:**

- **Substations.** Only the 8 Primaries metered in MW, each carrying its status from the earlier
  review into every result: Kirton is reported only as a range (1.2 to 4.4 MW), because its daylight
  cloud load was flagged in 6 of 7 season-years; Wrangle T1 is excluded from any claim that a
  Primary is free of a confound; Trusthorpe is marked low-power. The 8 Primaries metered only in MVA
  fold solar into the magnitude of the flow, and experiment 00039 fitted a negative capacity at
  Stickney in every year. [Experiment 00079](../../00079-pvdis-7-mw-at-mva-only-primaries/result.md)
  (PVDIS-7) treats them separately.
- **Service areas.** `TimeSeriesMetadata.area_wkt` holds one polygon per Primary. Wrangle T1 and
  Wrangle T2 are two transformers at one substation, feeding different circuits, and share one
  polygon. Each transformer keeps its own capacity fit, and any comparison against an outside count
  of solar uses the sum of the two against the shared polygon. Before any outside comparison, a
  check reports overlaps and gaps between neighbouring polygons, whether each Primary's location
  falls inside its own polygon, and how the union of the polygons compares with the two BSPs' area.
- **Irradiance.** CAMS for past weather, because the [past-weather solar
  study](../../../studies/past-weather/solar.md) found the CAMS solar estimate the most accurate of
  the products it compared. ENS for forecasts. Where a capacity fitted on CAMS is applied to ENS,
  the ENS irradiance is first calibrated against CAMS, by lead, on data from the fold's training
  window only.
- **Capacity used in a forecast is a real-time vintage.** A forecast experiment reads the capacity
  that a fit truncated at the fold's training end would give, never a path smoothed over the whole
  record. A path fitted once over all data passes information from later weeks back to earlier weeks
  through the penalty and the constraint.
- **Switching is masked.** The logged closures, padded by 24 hours, are removed from every capacity
  fit until SW2-0b's labels exist. A transfer moves a share of the donor's net load, including the
  donor's solar, so an unmasked transfer looks like a step in capacity. The logs miss some real
  switching, so unlogged transfers remain a source of false steps.
- **Known answers first.** Every capacity method is scored on experiment 00039's injection harness
  before its result on the real series is trusted.
- **Loads that track sunshine are a named confound.** Seaside visitor load at the coastal Primaries
  (Skegness, Ingoldmells, Chapel St Leonards, and Trusthorpe), irrigation pumping, cooling of crop
  stores, and air conditioning all rise on sunny days and bias the fitted capacity low. Daytime
  domestic load falls on sunny days, as people go outside, and biases the fitted capacity high. Each
  write-up reports the coastal Primaries separately and compares school-holiday weeks with term-time
  weeks.

#### PVDIS-T: the capacity table, a study

**Publish a season-year capacity table now, built from experiment 00040's outputs with no refit, and
replace it with PVDIS-4's weekly series once that part has run.** NGED's need is to improve on
almost no information, so the season-year table is useful before the weekly series exists. The study
reports C as the capacity, keeps sampling uncertainty separate from the named systematic scenarios,
and carries each Primary's status from the earlier review. The joint fit of [experiment
00063](../../00063-sw2-10-linear-joint-fit-v2-in-miniature/result.md) read the same capacities.
