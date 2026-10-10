# Plan: which ERA5 variables help explain how much sunlight reaches a solar farm?

Status: signed off by the maintainer after two agentic reviews and a prose review. The `Spike` issue
under the studies epic carries this plan as its body. The ERA5 download has started, and the
analysis scripts are written and run on rungs G0 to G2.

## Question

**If an XGBoost model (a gradient-boosted tree library) is given every ERA5 variable that could
plausibly matter, does it predict solar photovoltaic (PV) output better than an XGBoost model given
the minimal ERA5 set?** ERA5 is the fifth-generation reanalysis of the European Centre for
Medium-Range Weather Forecasts (ECMWF). The same question is asked of a second target, global
horizontal irradiance (GHI) from the Copernicus Atmosphere Monitoring Service (CAMS) at the same six
solar farms. CAMS is a satellite retrieval, so CAMS GHI shows how much sunlight reached the ground
without a panel, inverter, or curtailment in the way. The scientific question behind both targets is
which ERA5 variables carry information about how much sunlight gets through the atmosphere that
ERA5's own downward short-wave radiation (`ssrd`) does not.

## Why ERA5 at all, when the aim is an IFS-driven forecast

**The study's ultimate aim is to choose which variables from ECMWF's Integrated Forecasting System
(IFS) to feed a solar power forecast.** Of the archives checked below, ERA5 is the only free one
that carries all 34 candidate variables for 2019-09 to 2026-09. The data-download coordinator (a
separate agent session) checked which IFS products the project can get:

- **Of the 34 candidate variables in the ladder table below, 16 are in the free ECMWF feed** (open
  data, and the Dynamical.org archive built from it): `ssrd`, `strd`, `t2m`, `d2m`, `u10`, `v10`,
  `sp`, `tp`, `tcc`, `tcwv`, `skt`, `sd`, `sf`, `asn`, `mucape` (standing in for `cape`), and the
  gust field `10fg` (standing in for `i10fg`).
- **Six more are on Open-Meteo's 9 km IFS:** `lcc`, `mcc`, `hcc`, `blh`, `cin`, and `fdir`. Whether
  the cloud layers and `fdir` are native or derived there is unverified.
- **The remaining 12 are only in ECMWF's full Meteorological Archival and Retrieval System (MARS):**
  `ssrdc`, `cdir`, `tclw`, `tciw`, `tcslw`, `cbh`, `tcrw`, `tcsw`, `tco3`, `uvb`, `fal`, and
  `deg0l`. The study screens whether these 12 are worth asking ECMWF for, and they are not
  candidate features today.
- **The IFS archives are short:** Open-Meteo's `ecmwf_ifs` starts 2024-03-14, the Dynamical.org
  ensemble archive (ENS) starts 2024-04-01, and the Source Cooperative backfill covers about 2021-03
  to 2024-03 (14 surface fields, no `fdir`).

**ERA5 is a screen, not the training source, and a result transfers to the IFS only in part.** ERA5
was produced with a frozen 2016 version of the IFS weather model (cycle 41r2), and its hourly
fields come from short forecasts. ERA5's `ssrd` comes from the same radiation scheme and the same
clouds as its cloud variables, so the cloud variables are largely redundant with `ssrd`, and a gain
can come only from radiation-scheme error. A day-3 IFS forecast at 9 km has a different error
structure, and its cloud layers may carry local information that ERA5's 31 km fields lack. A gain in
ERA5 therefore justifies a matched-lead IFS test and a limited MARS pilot fetch, and does not
justify adopting a variable. A null in ERA5 weakens the case for fetching the MARS-only variables
and does not exclude a gain in the IFS. The page reports each result against the variable's IFS
availability (free feed, Open-Meteo, or MARS-only), so the reader sees which winners are usable.
Confirming any winning rung on matched-lead IFS forecasts is a follow-up study, outside this plan.

**Aerosol makes ERA5 and the IFS differ in clear-sky irradiance.** Per the Opus aerosol review, the
operational IFS uses a fixed monthly aerosol climatology, not prognostic aerosol. Since cycle 43r3,
that climatology has been the CAMS interim reanalysis (2003 to 2013, 3° grid). Cycle 50r1
(operational 12 May 2026) revised the climatology, and ECMWF's release note for cycle 50r1 does not
say which climatology replaced it. ERA5 uses the older [Tegen et al.
(1997)](https://doi.org/10.1029/97JD01864) climatology, with a sulphate trend from the Coupled Model
Intercomparison Project Phase 5 (CMIP5). Neither weather model sees an individual dust or smoke
event. CAMS satellite irradiance (Heliosat-4: McClear for clear sky, McCloud for cloud extinction)
does use CAMS aerosol analyses and forecasts every 3 hours. The ERA5-versus-IFS climatology mismatch
is a second reason to train the production forecast on IFS forecasts, whatever this study finds.

## Arms: a ladder of variable groups, each adding one physical idea

The ladder comes from the Opus variable review, whose brief is at
`.claude/worktrees/era5-solar-variables-brief.md`. Each rung contains every rung below it.

| Rung | Adds | ERA5 variables (derived features in brackets) |
|---|---|---|
| G0 minimal | the "normal" set | `ssrd`, `t2m`, plus top-of-atmosphere flux and solar geometry from `studies.pv_dataset.add_solar_geometry` (a midpoint-zenith estimate of the top-of-atmosphere flux, close to but not identical with ERA5's hour-integrated `tisr`), and the clearness index `ssrd / extraterrestrial_horizontal_w_m2` |
| G1 cloud amount | total cloud | `tcc` |
| G2 cloud layers | low, medium, high cloud | `lcc`, `mcc`, `hcc` |
| G3 clear-sky normalisation | how bright the sky would be without cloud | `ssrdc` (clear-sky index `ssrd / ssrdc`) |
| G4 cloud optical thickness | water and ice in the cloud | `tclw`, `tciw`, `tcslw`, `cbh` |
| G5 beam and diffuse | direct beam against scattered light | `fdir`, `cdir` |
| G6 panel temperature | convective cooling and thermal radiation | `u10`, `v10`, `strd` |
| G7 humidity and haze | moisture in the air column | `d2m`, `tcwv`, `blh` |
| G8 snow and albedo | snow on the panel, ground reflection | `sd`, `sf`, `asn`, `fal` |
| G9 everything plausible | the remaining ERA5 variables | `tp`, `tcrw`, `tcsw`, `cape`, `cin`, `skt`, `tco3`, `uvb`, `i10fg`, `sp`, `deg0l` |
| G10 aerosol (not ERA5) | event-level aerosol, which neither ERA5 nor the IFS carries | CAMS EAC4 (the CAMS global reanalysis) total aerosol optical depth at 550 nm and dust optical depth at 550 nm |

**Every rung is a nested superset, so the number of feature columns differs between rungs.**
XGBoost runs with `colsample_bytree=1`, so column subsampling gives a wider rung no advantage. The
study skill records that, on synthetic data, about 0.4% of the mean absolute error still favours a
wider arm at `colsample_bytree=1`. With G9 at about 40 columns against about 6 for G0 or G2, that is
roughly 0.02 percentage points on the PV target, well below the smallest effect of interest (0.1
points). The negative control (below) measures what the extra columns do on their own.

**Two fields on the maintainer's original list, cloud ceiling (`ceil`) and convective cloud-top
height (`hcct`), are not ERA5 variables.** The Climate Data Store (CDS) download form (checked
2026-10-08) lists neither. `cbh` is the nearest available field to a ceiling, and no ERA5 field
gives a convective cloud-top height, so G4 and G9 come as close to those two fields as ERA5 allows.

**In the Opus aerosol review's judgement, aerosol is the ERA5 gap most likely to matter for this
question.** ERA5 has no aerosol optical depth, and its radiation scheme uses a climatology. CAMS
satellite irradiance uses CAMS aerosols. G10 tests the aerosol gap directly by adding CAMS EAC4
total and dust aerosol optical depth at 550 nm, which are not ERA5 fields. The expected sign is a
gain on the CAMS target (CAMS sees aerosol, ERA5 cannot) and a small gain on the PV target (aerosol
optical depth over Great Britain is usually low, about 0.1 to 0.2, from the reviewer's memory). Any
ERA5-versus-CAMS gap that survives G10 is not explained by aerosol.

**G10 is exploratory, and is only useful in production if the forecast also receives CAMS forecast
aerosol, a second live feed.** EAC4 ends in 2025, so G10 is compared with a G9 arm refitted on
exactly the G10 rows, with folds cut on that shorter span. EAC4 is 3-hourly at 0.75°. Let H be the
label of an hour ending at H. The aerosol join interpolates linearly in time to the labels H−1 and
H, takes the mean of the two, and reads the EAC4 cell nearest each farm. The aerosol download is
about 4 MB in total.

## Aerosol in unusual conditions: cloud-free skies and high dust

**A yearly mean error can hide the days that matter most to a trader, so G10 gets its own
conditional analysis.** One badly forecast day can cost an energy trader more than the year's gains.
Dust is the clearest case: on a cloud-free day with a dust plume overhead, irradiance is lower than
a clear-sky forecast expects, and the error is large and one-signed. Days like that are a few
percent of hours, so they barely move a yearly mean. The analysis below is pre-specified, which
means its conditions, measures, and reading rule are fixed here before any G10 result exists. The
analysis is not one of the ten planned contrasts, so the page labels every number in it
exploratory, with 95% intervals and no correction for multiple comparisons.

**Four conditions, each defined from variables a forecast would also carry.** All four use the G10
rows (daylight hours that EAC4 covers) and both targets, with PV primary.

- **Clear:** ERA5 `tcc` below 0.2, the primary regime threshold used elsewhere in the study.
- **Clear and dusty:** clear, and `duaod550` at or above its 95th percentile over the G10 rows
  (pooled over the six farms, which share two to four EAC4 cells). Dust over Great Britain is
  mostly Saharan, but the condition measures only high modelled dust optical depth, and the
  Saharan origin is unverified.
- **Dusty, any sky:** `duaod550` at or above its 95th percentile, whatever the cloud.
- **Clear and clean:** clear, and `aod550` below its median over the G10 rows. In clean, cloud-free
  air the extra aerosol columns have the least to say, so the page reports this condition as
  context. A gain here would mean aerosol helps in general, not only in dust.

**The contrast is G10 minus G9 on the aerosol rows, in each condition.** The sign is treatment
minus reference, so a negative difference is a gain. Four measures describe each condition.

- **Mean absolute error**, the study's metric, with a month-resampled 95% interval at each
  hyperparameter setting.
- **The 95th percentile of the farm-day mean absolute error**, and **the worst farm-day mean
  absolute error.** Each arm's bad days are its own, so the difference compares the two models'
  tails and not the same days. The page says so.
- **The mean signed error** (prediction minus measurement), which shows whether dust makes the
  XGBoost model over-forecast.
- **CRPS, interval coverage, and interval width** from the quantile fits (next section), at the
  primary setting only, with an interval for CRPS and a difference of means for the other two.

**The page counts the events before it reads any number.** For each condition the page prints the
hours, the farm-days, the calendar months with at least one hour (months are the resampling unit),
and the distinct days on which any farm met the condition, without dates on a farm's axis. The 95th
percentile of `duaod550` is by construction 5% of hours, but dust arrives in episodes, so the
number of independent events can be a handful. A condition spanning fewer than 6 calendar months
(`MIN_MONTHS_FOR_INTERVAL`) gets its means but no interval. The reading rule below needs at least 20
distinct days and 12 calendar months (`MIN_AEROSOL_DAYS`, `MIN_AEROSOL_MONTHS`), and below those
counts the page reports that G10 cannot be assessed in clear and dusty hours.

**Reading rule, fixed now.** A day-1 to day-3 CAMS aerosol feature is worth a production trial only
if, on the PV target at both hyperparameter settings, the whole 95% interval of G10 minus G9 in
clear and dusty hours lies below minus the smallest effect (0.1 percentage points of capacity), and
the condition meets the counts above. The report prints the outcome (`trial worth running`, `not
shown`, or `cannot be assessed`, from `aerosol_trial_recommendation`). This single interval, on one
measure, one target, and one condition, is the only deciding number in the aerosol analysis. Every
other interval in it is context.

**The page counts its intervals.** The condition analysis holds 16 intervals on mean absolute
error (4 conditions, 2 targets, 2 settings) and 8 on CRPS (4 conditions, 2 targets, primary
setting). The probabilistic contrasts below add 30. The page states the total, 54 at most, so a
reader can discount.

**Three limits the page states.** First, EAC4 is a reanalysis, so it knows the dust plume's actual
position, and a CAMS forecast issued the day before places the plume less well. The gain here is
an upper bound on what the forecast would give. Second, EAC4 assimilates satellite aerosol optical
depth mainly where the sky is clear, so the clear-sky subset is where the gap between reanalysis
and forecast is likely widest. Third, the CAMS-irradiance target contains CAMS aerosol by
construction (see Targets), so a gain on the CAMS target in dusty conditions is expected and says
little about PV.

## Probabilistic scores (exploratory, pre-specified)

**The point fits rank the arms, and a quantile fit on a few arms asks a second question: does an
input help the XGBoost model say how uncertain it is?** For an electricity-network forecast, a
narrow interval that covers the outcome is worth more than a small error on an average day, and a
variable can sharpen the interval without moving the median. Mean absolute error from the point
model stays the only planned ranking, and nothing in the ten planned contrasts, the Bonferroni
level, or the MARS decision rule changes. The quantile fit is a second XGBoost model
(`reg:quantileerror`, the nine levels 0.1 to 0.9 that `studies.cross_validation.QUANTILE_LEVELS`
already holds) fitted beside the point model on the same rows, folds, seeds, and columns. The point
model is fitted first with the same parameters, so each arm's point predictions are the same with
or without the quantile fit.

**Quantile fits cover the arms the questions need, at the primary setting only.** They are G0, G2,
G9, G9 without the 12 MARS-only variables (`g9_without_mars_only`), and the negative control, on
both targets, plus `g9_aerosol_rows` and G10 in the aerosol view. A multi-quantile fit builds one
tree per level, so each of these fits costs about nine times a point fit. Fitting every arm would
add 12 to 25 hours, and these arms add an estimated 4 to 8 hours (estimated from the superseded
G0 to G2 timings; the report records the real time). The second hyperparameter setting is not run
for the quantile fits, so every probabilistic result is labelled "primary setting only" and none
of them carries a verdict.

**The scores come from the quantiles after repairing them in three ways, identically for every
arm.** Each row's quantiles are sorted (XGBoost's multi-quantile head can cross), held at or below
the export cap in force, and floored at zero, because neither output nor the clearness index is
negative. Every score is divided by the row's own capacity (PV) or reported in index units (CAMS)
before averaging.

- **Continuous ranked probability score (CRPS)**, approximated from the nine levels as the existing
  `crps` does, on the repaired quantiles. The approximation leaves out the tails beyond the 0.1 and
  0.9 levels, so the score is a truncated one.
- **Coverage of the 0.1 to 0.9 interval** (nominal 80%) and its **mean width**.
- **A constant-width reference:** the width of one interval that holds 80% of the arm's own
  out-of-fold point errors (the 0.1 to 0.9 range of the signed error), which needs no second model.
- **A reliability table:** the share of outcomes at or below each of the nine quantile levels.
- **The same scores by ERA5 cloud regime** (the primary `tcc` split of the regime panel).

**Five contrasts, each on both targets, with 95% intervals from the same month-resampled paired
bootstrap, labelled exploratory.** They are G2 minus G0, G9 minus G2, G9 minus the negative control,
P4 (G9 minus G9 without the MARS-only variables), and G10 minus G9 on the aerosol rows. Each is
scored on CRPS, and also on interval width and on coverage as differences of means, which is 30
intervals in all. The page counts them with the aerosol intervals, as above.

**A claim that an input "helps the model estimate its own uncertainty" has to clear three bars.**
CRPS and interval width both fall when the median improves, so neither shows better uncertainty
estimation alone.

- **Narrower intervals at the same coverage:** the 95% interval of the coverage difference lies
  within plus or minus 2 percentage points, and the width difference lies below zero.
- **Narrower than the negative control:** the control's permuted columns keep each month-and-hour
  mean, which carries seasonal spread, so the page reports G9 against the control as well as G9
  against G0.
- **Narrower than the constant-width reference:** the quantile model's mean width at the same
  coverage is below the constant interval from the arm's own errors, so the width gain does not
  come from a better median alone.

A coverage difference of about one point may be resolvable. The tail quantiles beyond 0.1 and 0.9
and dust-episode spread are not, because the independent weather episodes number in the dozens and
the six farms share their weather. The page says so.

**Gain importance still comes from the point booster.** The quantile model is not used for the
importance figure.

## Targets

1. **PV target:** hourly mean output of each of the six NGED solar farms, as a percentage of the
   farm's capacity (its 99th percentile of metered output), for the hour ending at the label. One
   XGBoost model per farm, as in the past-weather solar page.
2. **CAMS target:** CAMS GHI (W m⁻², hour ending at the label) at each farm's location, read from
   the existing `reanalysis/CAMS` downloads. The target is the clearness index (GHI divided by
   top-of-atmosphere flux), and the page also reports error in W m⁻². The CAMS target has no
   curtailment and no panel, so the contrast between the two targets separates "the atmosphere" from
   "the panel".

**Gains from aerosol, `tcwv`, and `tco3` on the CAMS target are partly by construction.** McClear
computes CAMS GHI from CAMS aerosol, water vapour, and ozone, which come from the same IFS-based
assimilation family as ERA5's fields. Conclusions about those variables rest on the PV target. The
same holds for `ssrdc` (G3): McClear's clear-sky inputs differ from ERA5's aerosol climatology, so a
G3 gain on the CAMS target can be a mismatch in clear-sky climatology by construction. Every
conclusion about the MARS-only variables rests on the PV target alone. Which
CAMS aerosol product McClear uses (EAC4 or the operational analysis) is unverified.

## Rows, folds, metric, and fitting rules

- **Rows:** the daylight hours (top-of-atmosphere horizontal flux above 50 W m⁻², decision 6) on
  which the PV target and the CAMS target both exist, within the download span. Rows are set by the
  targets, the clock, and the span only, never by an ERA5 value. The build raises if any ERA5 value
  other than `cbh` and `cin` is missing on those rows.
- **Missing cloud base:** ERA5 sets `cbh`, and probably `cin`, missing where there is no cloud, so
  `cbh` and `cin` keep NaN, which XGBoost treats as missing. Each of `cbh` and `cin` goes through
  its own `hourly_from_snapshots` call (see Hour convention), whose hourly value is NaN unless both
  snapshots exist. The first fetched chunk reports each variable's share of NaN, split by `tcc`
  below and above 0.05.
- **ERA5 release:** the study window ends at a fixed date inside the months known to be final ERA5,
  at rows before 2026-08-01 (or before 2026-07-01 if the July check below fails). ERA5T, the
  preliminary release (`expver` 0005), can later be replaced by ECMWF, and the window drops it
  instead of flagging it. Google's ARCO-ERA5 copy carries no per-hour `expver`, so the ARCO
  downloads label every hour with a constant, and the build still raises if `expver` differs between
  variables for one hour. ARCO's own attributes put the end of final ERA5 at 2026-06-30, while the
  held CDS copy labels July 2026 as final. The download coordinator compares July 2026 with the held
  CDS files, and the window keeps July only if the two agree to float rounding.
- **PV cleaning:** reuse `studies.power`, `studies.export_cap`, and `studies.commissioning`, as the
  past-weather solar page does. Hours holding a zero half-hour are dropped, from the power table,
  for both targets.
- **Snow censoring:** dropping hours that hold a zero half-hour also drops fully snow-covered
  panels, which read exactly zero. G8's PV result and the snow days in figure 11 are therefore
  biased towards no effect. The page states this beside G8, and an exploratory G8 arm keeps zero
  hours where `sd > 0`.
- **Folds:** contiguous blocks of whole months (`studies.cross_validation.assign_folds`). No UKV era
  split is needed, because every input is ERA5.
- **Clearness index:** the clearness index is unstable at low sun, so the daylight threshold also
  bounds the clearness index. The CAMS target's clearness index uses the hour-integrated
  top-of-atmosphere value that the CAMS files carry, and W m⁻² is reported beside it.
- **Geometry in every rung:** the minimal set carries the sun's elevation and a midpoint-zenith
  estimate of the top-of-atmosphere flux, while `ssrd`, `ssrdc`, and `cdir` are hour integrals. To
  stop `ssrdc` and `cdir` winning by supplying hour-integrated geometry that can be computed for
  free, every rung also carries the hour-integrated top-of-atmosphere horizontal flux that the CAMS
  files hold (`cams_toa_w_m2`). The value depends on the sun's position alone, so a production
  forecast could compute it, and every arm sees the same column.
- **Metric:** mean absolute error is the main metric: as a percentage of capacity (PV), and as a
  clearness-index error and in W m⁻² (CAMS). Pearson correlation between out-of-fold prediction and
  measured value, pooled over each fold's rows, is reported beside mean absolute error for every
  rung (exploratory, and dominated by the daily cycle, so every rung reads close to 1), with
  intervals from the month-resampling bootstrap described under Intervals.
  Each farm's error is normalised by its own capacity before any mean or difference.
- **Intervals:** `studies.bootstrap.bootstrap_difference`, 2,000 resamples for exploratory intervals
  and 10,000 for the planned contrasts (a 99.5% tail holds about 25 of 10,000 resamples, against 6
  of 2,000) of whole calendar months,
  paired across rungs. Each arm is fitted with three random seeds, and each resample draws one of
  the three seeds. The page explains once what the test covers and what it does not.
- **Hour convention:** `ssrd`, `ssrdc`, `fdir`, `cdir`, `strd`, `tp`, `sf`, and `uvb` are
  accumulations: means over the hour ending at the label, the same as the PV and CAMS stamps. Every
  other variable is instantaneous, and is averaged over the labels H−1 and H so the hour matches,
  using `studies.hourly_means.hourly_from_snapshots(slot_offsets_minutes=(-60, 0))`. A test checks
  that each ladder variable has exactly one class.
- **Seam checks:** the fetch checks the hour-of-day profile of the mean absolute hour-to-hour change
  for steps at three families of seams. Accumulations change forecast run at 07 and 19 UTC. Cloud
  and cloud-water fields (from the 06 and 18 UTC forecasts) change at 06/07 and 18/19 UTC. Analysed
  fields (`t2m`, `d2m`, `u10`, `v10`, `sp`, `skt`, `tcwv`, `sd`) change at the 4D-Var
  (four-dimensional variational assimilation) window boundaries of 09 and 21 UTC.
- **Pairing guard:** before each contrast the code raises unless both arms hold the identical set of
  (site, time, seed) rows, because `paired_differences` inner-joins silently. Every arm runs on the
  same device, and the per-arm error table records which device.
- **Second hyperparameter setting:** every planned contrast is also run at a second hyperparameter
  setting (`SENSITIVITY_HYPER_PARAMETERS`), and `combine_setting_verdicts` combines the two
  verdicts.
- **GPU:** arms are fitted on a GPU (`device="cuda"`) if `nvidia-smi` shows a GPU, and the page
  states the device. No CPU refit is run for a noise floor.

## Planned contrasts (written before any result exists)

Each of the five contrasts below is run on both targets, so there are ten planned contrasts:

1. **P0:** G9 (every ERA5 variable) minus G0. P0 is the study question, and P0 equals P2 plus P3.
2. **P1:** G1 (adds `tcc`) minus G0. Does total cloud help at all beyond `ssrd` and `t2m`?
3. **P2:** G2 minus G0. Do the three cloud layers help beyond the minimal set?
4. **P3:** G9 minus G2. Does any variable beyond the three cloud layers help? P3 is the maintainer's
   headline question.
5. **P4:** G9 minus G9 without the 12 MARS-only variables (`ssrdc`, `cdir`, `tclw`, `tciw`, `tcslw`,
   `cbh`, `tcrw`, `tcsw`, `tco3`, `uvb`, `fal`, `deg0l`). P4 answers whether fetching them from MARS
   is
   worth it, against a reference that holds only variables the production forecast can already
   get. P4 is judged on the PV target, because the CAMS target is partly circular for these
   variables (see Targets).

**The decision rule for the MARS fetch is fixed before any result.** The adjusted interval of P4 on
the PV target, at both hyperparameter settings, decides:

- If the whole interval of the error difference lies below minus the smallest effect of interest, a
  limited MARS pilot (matched-lead IFS forecasts of the variables that carry the gain) is
  recommended.
- If the interval lies wholly above minus the smallest effect of interest, a gain as large as the
  smallest effect is ruled out, and the page recommends against the fetch.
- Otherwise the result is unresolved, and the page says what a larger sample would need.

The availability lists above (free feed, Open-Meteo, MARS-only) are re-verified against a recent
run before the page assigns a variable to a class.

**Planned verdicts use a Bonferroni-adjusted level.** With ten planned contrasts, the family-wise
level of 5% becomes 0.5% per contrast (a 99.5% interval, `bootstrap_difference_at_level`). Every
interval is also shown at 95%, labelled exploratory. The page states each planned result as "rules
out a gain larger than X" using the lower bound of the error difference (treatment minus
reference, so a negative difference is a gain), with a smallest effect of interest fixed before any
result (0.1 percentage points of capacity for PV, 0.01 for the clearness index).

**Every other number is exploratory:** each rung against G0, each rung against the one below it, the
drop-one-group runs, the per-farm numbers, the regime and season splits, the aerosol analysis in
unusual conditions (with its pre-specified reading rule), and the probabilistic scores. The page
labels those
numbers exploratory and does not correct those numbers for multiple comparisons.

## Controls

- **Negative control:** the G3 to G9 columns, permuted by `studies.blending.climatology_permutation`
  over site, calendar month, and hour of day, so the column count and each column's distribution
  match G9. The permutation removes the hour-to-hour information and keeps each month's mean at each
  hour, which is month-level weather information, so the control is not strictly information-free.
  All G3 to G9 columns are passed as one column group, so their joint distribution survives. A
  contrast of this arm against G2 shows how large a difference the pipeline produces from nothing.
- **Positive control:** G2 plus CAMS GHI itself as an input on the PV target. CAMS GHI is a column
  that must help. If the pipeline cannot see the CAMS GHI gain, a G9 result of no gain is not
  evidence of no effect. The gain is many times the smallest effect of
  interest, so a pass shows that the instrument detects a large effect only. A null planned result
  is read through the width of its interval, as "a gain larger than X is ruled out", and not through
  this control. The positive control is not run on the CAMS target, where CAMS GHI is the
  target itself.
- **Baseline for the CAMS target (the `ssrd`-only arm):** an XGBoost model on the CAMS target is
  first fitted
  with only `ssrd` and solar geometry, so the page shows how much of CAMS GHI the standard ERA5
  field already explains before any extra variable is tried.

## Drop-one-group runs

**The ladder is order-dependent, so a second instrument removes one group at a time from the full
set.** A group that adds nothing after G2 may add a lot if it came first. The drop-one-group run
starts from G9 and removes one rung's variables at a time, so the page can say for each group both
what the group adds on top of the minimal set (ladder) and what is lost without the group
(drop-one). A group is called useful only if both instruments give an error difference of the same
sign whose
95% interval excludes zero. Both instruments are exploratory, and a group of substitutes (`tcc`
against the cloud layers, `ssrdc` against `cdir`) can show a drop-one loss near zero even when the
set as a whole matters, so the planned contrasts decide, not the drop-one runs.

## XGBoost feature importance

**The fitted XGBoost models' own importance scores are a third, descriptive view of which variables
the models lean on.** They sit beside the ladder and the drop-one-group runs, and they decide
nothing. Gain importance splits credit between correlated columns arbitrarily (the cloud covers,
`tclw`, and `tciw` are all correlated), so a variable can score high yet add nothing when removed.
The planned contrasts and the drop-one-group runs stay the evidence for whether a group helps.
Importance shows what the models used, and where it disagrees with the two instruments the page says
so.

- **What is computed:** the total gain of every column in the fitted G0, G2, and G9 models, plus the
  G9 refit with shuffled copies,, on both targets, for each fold and each seed. Gain is read from
  `Booster.get_score(importance_type="total_gain")`. `studies.cross_validation.fit_one_fold` does
  not return its booster, so a new script, `era5_ladder_importance.py`, refits only those four arms
  with the same folds, seeds, and settings, and saves the gains. The shared fit loop stays
  unchanged. The refit costs about as much as the ladder's G0, G2, and G9 arms, and needs a slot
  from the study coordinator.
- **How it is summarised:** each model's gains are scaled to sum to 1, then averaged over seeds and
  folds. The page shows each variable's share, each rung's share (the sum over the rung's columns),
  and the range across folds and seeds, called the variability across refits because the fold models
  share most of their training months. The positional keys (`f0`, `f1`, ...) are mapped back to
  column names. Gain is measured on the training data, so it shows what the fit used, not what
  generalises.
- **Noise reference:** the importance refit of G9 also carries shuffled copies of the G3 to G9
  columns, shuffled over all rows of a farm, so each copy keeps its column's distribution. The
  largest shuffled copy's share is the line a real column must stand clear of. The refit is never
  scored in a planned contrast, because the extra columns change the trees.
- **Grouped permutation importance stays cut** (Review 1). Gain needs no refit and no extra rows.

## Page structure

**The page is figure-led, a hybrid of an academic paper and a data scientist's notebook, and the
figures tell the whole story.** The maintainer asked for this form on 2026-10-10, and the study
skill now describes it. Readers work at three depths, and the page serves each.

- **In about a minute, without scrolling:** a title, a bottom line of a few sentences, and the
  headline figure (figure 1). The take-home message is general, about solar power forecasting: which
  weather-model variables help a solar farm forecast, and whether fetching the MARS-only variables
  is worth it. It also says what it means for the Flexpectation project.
- **In a few minutes:** the Summary's question-and-answer bullets, each at most two sentences and
  each pointing at the figure that answers it, then the figures and their bolded leads in order.
- **In full:** the AI disclaimer, Key findings, Introduction, Data and methods, Results, and
  Limitations, with the minimum prose the figures need.

**After the headline figure, the figures tell the story in a fixed order.** First, simple plots of
what the data look like, with their issues (figures 2 to 5: three days of every variable, the cloud
covers against a cloud index, where ERA5's `ssrd` misses CAMS, and cloud water against optical
thickness). Then the proof that the technique works (figure 6: the XGBoost models track measured
output; figure 7: the leaderboard with the negative and positive controls). Then the more complex
results (the regime, season, drop-one, hour-of-day, importance, probabilistic, and aerosol figures,
8 to 14). Every figure stands alone: the title, subtitle, axes, and legend say what is plotted,
which direction is better, what each colour means, and the scope, and a figure that explains its
colours in text colours each phrase in the colour of the line it names. The Summary's questions
are:

1. Do ERA5 variables beyond `ssrd`, `t2m`, and sun position help predict solar farm output, and by
   how much?
2. Which groups of variables carry the information (cloud amount, cloud layers, clear-sky
   irradiance, cloud water, direct beam, wind and thermal radiation, humidity and haze, snow and
   albedo, the rest, CAMS aerosol)?
3. Is fetching the 12 MARS-only IFS variables worth it (planned contrast P4 and its decision rule)?
4. Does the answer on ERA5 carry over to IFS forecasts, and what would a matched-lead follow-up
   need?
5. Does CAMS aerosol help under cloud-free skies and high dust, where a yearly mean error can
   hide a costly bad day (the pre-specified conditional analysis)?
6. Do any inputs help an XGBoost model estimate its own uncertainty, measured as narrower intervals
   at the same coverage than the negative control (the exploratory probabilistic scores)?

## Figures, in page order

The page is figure-led, and the figures appear in the order below. Each has a bolded one-sentence
lead and a few sentences of support. Every chart is anonymised: farms are A to F, outputs are
normalised by capacity, and no coordinate appears. A dated per-farm series can identify a farm, so
a series of output or of a farm's weather carries no calendar date on its axis (days are counted
1 to n, with the month and year given in the text), has `aria=False` on its marks, and shows the
weather series without the farm's label. Figures 2, 4, 6, and 11 follow this rule.

1. **Headline (top of page).** The planned contrasts P0 to P4 on both targets, at the adjusted
   level, with the smallest effect of interest marked.
2. **What the variables look like.** Three days at a farm, chosen by a stated rule (the clearest,
   the most variable, the dullest): stacked small multiples of `ssrd`, `ssrdc`, CAMS GHI, PV output,
   the cloud covers, `tclw` and `tciw`, `cbh`, and `fdir`. The time axis is shared.
3. **Cloud covers against a cloud index.** CAMS clearness index (or PV output divided by its
   clear-sky expectation) plotted against `tcc`, and against `lcc`, `mcc`, and `hcc`.
4. **Where ERA5's `ssrd` misses CAMS.** ERA5 minus CAMS GHI as a time series for the same days, then
   binned against each candidate variable (`tclw`, `tciw`, `cbh`, `tcwv`, `d2m`, `blh`, `sd`).
   Figure 4 shows the scientific question directly, before any XGBoost model is fitted.
5. **Cloud water against optical thickness.** `tclw + tciw` against CAMS clearness index, coloured
   by low-cloud cover.
6. **The XGBoost models work.** Out-of-fold PV against measured for figure 2's three stated-rule
   days at every farm, G0 against G9, and each rung's error per farm.
7. **The leaderboard.** Every rung's own mean absolute error with a 95% interval on the PV target
   and the CAMS target, best first, with G0 and the two controls included. A second panel shows
   Pearson correlation for every rung. The controls sit here so a reader sees that the pipeline can
   fail and does not, before the results that rest on it.
8. **Weather regimes.** The difference in error between G9 and G0, between G1 and G0, and between G2
   and G0, split by regime: clear sky, broken cloud, and overcast. The primary panel splits by the
   ERA5 cloud cover `tcc` (clear below 0.2, overcast from 0.8), which
    is a forecast-time variable. A second panel splits by the CAMS clear-sky index (overcast below
    0.4, clear from 0.8), which is conditioned on the observed sky and can show regression to the
    mean. Both thresholds are fixed before any result. Exploratory.
9. **Seasons.** The same differences by season (winter, spring, summer, autumn), and for the
   clear-sky, broken-cloud, and overcast regimes within each season. Exploratory.
10. **Drop-one-group.** Error added when each group is removed from G9.
11. **Hour of day and snow.** Error by hour of day for G0 against G9, and the worst 20 days for G0
    with what G9 changed on them, anonymised by farm label.
12. **What the XGBoost models lean on.** Three panels, per target. First, the top 20 columns of the
    G9 model by share of gain, with the largest shuffled copy's share marked as a reference line.
    Second, gain share summed by rung, with the shuffled copies as one more group. Third, how
    the share of `ssrd` and of the cloud covers moves from G0 to G2 to G9, as a bar for each model.
    Importance is descriptive, and the caption says that gain splits credit between correlated
    columns.
13. **Probabilistic scores.** For G0, G2, G9, G9 without the MARS-only variables, and the negative
    control: contrasts in CRPS, width, and coverage (13), a reliability chart of the share of
    outcomes below each quantile level (13b), and interval coverage against mean width with the
    constant-width reference (13c). Exploratory, pre-specified, primary setting only.
14. **Aerosol in unusual conditions.** G10 minus G9 on the aerosol rows in the four conditions of
    the aerosol section, for mean absolute error and CRPS, with the event counts beside each bar
    and the outcome of the reading rule stated. Exploratory, pre-specified.

## Data and code

- **Reuse the existing ERA5 download for G0 and for `fdir`.**
  `data/studies/downloads/reanalysis/ERA5/beam_diffuse/` already holds `ssrd`, `fdir`, and `t2m`
  over a box of 20 ERA5 grid cells for 2019-09 to 2026-09 from the CDS, plus a CDS-versus-Open-Meteo
  check. CAMS GHI and site-level PV tables exist already (`reanalysis/CAMS/`, the `studies.power`
  loaders, `studies.pv_dataset`).
- **The new download is the 27 ladder variables not already held,** over the same 20-cell box and
  hours, from [Google's ARCO-ERA5](https://github.com/google-research/arco-era5) analysis-ready
  store (`gs://gcp-public-data-arco-era5/ar/full_37-1h-0p25deg-chunk-1.zarr-v3`, anonymous read).
  The Climate Data Store (CDS) was the first choice, but its queue ran at about 2.8 hours per
  chunk of 60 in tier 1b, which put the last ERA5 tier at about 25 October. The tiers 1 and 2 hold
  31 variables, and 4 of them (`tcc`, `lcc`, `mcc`, `hcc`, tier 1a) were already downloaded from the
  CDS, which leaves 27. The ARCO store has all 34 ladder variables.
    - **Agreement with the CDS:** on the 20-cell box, `ssrd`, `fdir`, and `t2m` correlate with the
      held CDS copy at 0.99999998 or better in 2024-03 and 2025-09, with mean absolute differences
      of 8 J m⁻² on `ssrd`, 7 J m⁻² on `fdir`, and 0.0004 K on `t2m`. For 2025-06, `tcc`, `lcc`,
      `mcc`, and `hcc` agree to a mean of about 2e-6 (float rounding), and `ssrdc`, `cdir`, `strd`,
      `cape`, `fal`, and `asn` agree to within 0.002% of their means or better (`cape` to 0.12 at
      most). Shifting by one hour makes the
      mean difference 1,000 to 50,000 times larger, so the hour convention is the CDS one: the
      accumulations cover the hour ending at the label, and the snapshots are at the label. The
      CDS values already held stay in place, and nothing in the study depends on the two copies
      being byte-for-byte identical.
    - **Cost of the read:** each hourly chunk is global (721 by 1,440 values, about 2 MB
      compressed), and the box is cut after reading. The download coordinator measured 14.8 hours
      of data per second at 16 threads, which is about 0.9 hours per variable for 60,624 hours, so
      about 31 hours and about 3.3 TB read for 27 variables (estimated from a sample of seven
      variables). The coordinator runs the download in the order below, while the CDS chain keeps
      running until tier 1b has been validated from ARCO.
    - **Tier 1b (needed for G3 to G7):** `ssrdc`, `cdir`, `tclw`, `tciw`, `tcslw`, `cbh`, `u10`,
      `v10`, `strd`, `d2m`, `tcwv`, `blh`.
    - **Tier 2 (needed for G8 and G9):** `sd`, `sf`, `asn`, `fal`, `tp`, `tcrw`, `tcsw`, `cape`,
      `cin`, `skt`, `tco3`, `uvb`, `i10fg`, `sp`, `deg0l`. `tisr` is not downloaded: the
      top-of-atmosphere flux comes from `add_solar_geometry`.
    - **Tier 3 (not ERA5, from the Atmosphere Data Store, ADS):** CAMS EAC4 total and dust aerosol
      optical depth at 550 nm, 3-hourly, 2019-09 to the end of EAC4 (the ADS catalogue gave 31
      December 2025 on 2026-10-08; the fetch sets the end month from the listing). A few MB and a
      handful of requests. The ADS account is separate from the CDS account, and the CAMS irradiance
      download already used the ADS account.
- **The fetch is the `data-download` skill's job:** resumable, one pilot month first, with the
  `data-validation` checklist on the first chunk and again after the last.
- **Code:** `studies/era5_solar_variables/` for the scripts (`era5_ladder_build_dataset.py`,
  `era5_ladder_fit.py`, `era5_ladder_report.py`, `era5_ladder_charts.py`, `era5_ladder_arms.py`, and
  `era5_ladder_importance.py`), with outputs under `data/studies/per_study/era5_solar_variables/`.
  The fetch scripts live with the download coordinator's `weather_downloads` folder. The
  clearness-index and aerosol-join
  functions, which a second study might use, go into `packages/studies/src/studies/` with tests. The
  page goes under Studies > Past weather, as `docs/studies/past-weather/era5-solar-variables.md`.
- **Order of work:** issue (type `Spike`), then the fetch scripts (reviewed by the download
  coordinator's session), then
the build, fit, importance, report, and chart scripts, each with at least one fresh Opus review
before it runs (done for the first four, the importance script still to write and review), a run on
rungs G0 to G2, the full run once the data arrives (point fits for every arm, quantile fits for the
seven arms in `QUANTILE_ARMS`, the aerosol view, and the importance refit, each after the
coordinator grants a GPU slot), the first Opus science review, charts and draft
page, the second Opus science review, the diff review, a mutation pass because `packages/studies`
changes (`era5_ladder`, `correlation`, `cross_validation`'s `probabilistic_scores`), the prose
review and persona reviews, and merge.

## Tests for the new `packages/studies` functions

- **Clearness index:** zero top-of-atmosphere flux gives null, not infinity, and rows below the
  threshold are excluded.
- **Accumulation to power:** 3600 J m⁻² over the hour is exactly 1 W m⁻².
- **Variable classification table:** every ladder variable has exactly one class.
- **Ladder:** each rung is a strict superset of the rung below, and G9 holds every ERA5 variable in
  the ladder table.
- **Aerosol join:** a linear ramp in 3-hourly values gives the analytic hour-ending mean, and a
  3-hour shift of the input fails the test.
- **Pairing guard:** a contrast between arms with different row sets raises.
- **Regime and correlation functions:** `sky_regime` returns each regime at and between its
  thresholds, and `pooled_correlation_interval` returns the known correlation of a constructed pair.
  Tests run with `--run-studies`.
- **Aerosol conditions:** `aerosol_condition_flags` applies each condition's own threshold, and
  raises on a null input. `aerosol_trial_recommendation` returns each of its three outcomes at the
  boundaries of the day, month, and smallest-effect thresholds.
- **Probabilistic scores:** `probabilistic_scores` sorts crossing quantiles, holds them at the cap,
  floors them at zero, and divides each row's scores by its own capacity (in
  `test_cross_validation.py`).
- **Importance summary:** gains scaled to sum to 1 per model, and a column the model never split on
  gets a share of 0 rather than going missing. The grouping of columns by rung is in the chart
  script and is checked by looking at the figure.

## The five complexity triggers (for sizing)

1. **Changes what gets stored:** yes, a published page and new files under `data/studies/`.
2. **Touches the production serving path:** no.
3. **Touches a degradation rule:** no.
4. **Admits more than one defensible design:** yes (ladder order, hour convention, the CAMS target
   definition, tiered download).
5. **Spans code whose callers could not be named without searching:** no, but the study imports
   `packages/studies` machinery that has to be read first.

Size: complex, as every study is. The study skill's process applies in full, with two Opus science
reviews, because the page will carry numbers.

## Decisions the maintainer has made

1. **Download span:** 2019-09 to 2026-09 for every variable. The experiments are expected to show
   little signal, so the maintainer chose the longest span available.
2. **CAMS target:** the clearness index, with W m⁻² reported beside it.
3. **Aerosol:** include CAMS EAC4 aerosol optical depth as the eleventh rung (G10).
4. **MARS-only variables:** keep all 12. The study's purpose includes telling the team whether
   fetching them from MARS is worth the effort, which planned contrast P4 and its decision rule
   answer.
5. **Smallest effect of interest:** 0.1 percentage points of capacity on the PV target, and 0.01 on
   the clearness index on the CAMS target.
6. **Daylight threshold:** top-of-atmosphere horizontal flux above 50 W m⁻².
7. **GitHub issue:** a `Spike` issue under the studies epic, carrying the full plan as its body.

## Review 1 (simplicity) triage

**Accepted:** `tisr` already in G0 through `add_solar_geometry`, so it is not downloaded and G3
keeps `ssrdc` only; derived features cut to the clearness index; grouped permutation importance cut;
the label-H sensitivity run cut; the CPU noise-floor refit cut (also the maintainer's instruction);
the 2026 aerosol splice cut; the negative control reuses `climatology_permutation`; the headline
absorbs the step-change and controls figures; the hour averaging names `hourly_from_snapshots`.

**Rejected, with the reason:**

- **Collapse the ladder to four rungs.** The maintainer asked for the ladder (minimal, then total
  cloud, then cloud layers, then more), so the intermediate rungs stay as exploratory rows.
- **Make the CAMS target exploratory.** The maintainer asked for both targets.
- **Cut `fdir` and the beam/diffuse features.** `fdir` is available from Open-Meteo's IFS and stays
  as a plain input in G5. The derived direct-normal-irradiance, diffuse-irradiance, and
  plane-of-array features are cut under the derived-feature item above.
- **Cut the 12 MARS-only variables.** The study exists to tell the team whether the effort of
  fetching these variables from MARS is worth it, so cutting them would remove the answer.

## Review 2 (correctness and testability) triage

**Accepted, all as text changes to this plan:** the row-set rules for `cbh` and `cin`; the `expver`
release rule; the G10 refit on its own rows and the EAC4 end month; the shared-input caveat for the
CAMS target; the snow censoring; the clearness-index threshold and CAMS top-of-atmosphere source;
the smallest effect of interest and the Bonferroni level; P0; the hour-convention table and the
seam families; the negative-control wording and single column group; the EAC4 interpolation with its
test; the pairing guard; the second-setting verdict rule; one device.

**Unverified claims the reviewer flagged** are labelled unverified here and checked before the page
states them: the 12 May 2026 date for IFS cycle 50r1, Heliosat-4's 3-hourly aerosol, the 0.1 to 0.2
aerosol optical depth, and the IFS availability lists. The Tegen climatology for ERA5 is verified by
the [Data sources](https://openclimatefix.github.io/nged-substation-forecast/roadmap/data-sources/)
page.
