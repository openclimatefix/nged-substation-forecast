# How the matched-lead study reads its extra products, and what they show at other lead days

**This page holds the parts of the [matched-lead study](matched-lead.md) that concern its exploratory
products: how each is read, and the two results sections that have no figure.** The main page
carries the planned contrasts, the blends, Figures 1 to 20, the discussion, and the
[limitations](matched-lead.md#limitations), which apply to every number here. Every result on this
page is exploratory and fitted at the primary setting only.

## Data and methods


### The extra lead days, day 0, and the products added later

**The extra arms are fitted on a graphics processing unit (GPU), in four batches, and every one is
exploratory.** Each extra arm is the same XGBoost model given one product's forecast, at the primary
setting only, on the same shared rows, folds, and eras as every other arm, except that IFS HRES 9 km
loses its gap days (below) and the AIFS arms have rows of their own ([How AIFS is
read](#how-aifs-is-read)). The batches are:

- **Batch 1:** the ENS mean at days 5, 10, and 14; the ENS control member at day 0; the GEFS mean at
  days 0, 5, 10, and 14; IFS 0.25° and GFS (Open-Meteo) at days 0, 5, and 7; ICON global at days 0
  and 5; and every other Previous Runs product at day 0.
- **Batch 2:** the ENS mean and the GEFS mean at day 7, the ENS control member at days 2, 3, 5, 7,
  10, and 14, and GPU refits of the published arms that batch 1 left on the CPU.
- **Batch 3:** GFS (native) at days 0, 1, 2, 3, 5, 7, 10, and 14.
- **Batch 4:** IFS HRES 9 km at days 0, 1, 2, 3, 5, and 7.

**Each product has the lead days below, and the horizons in this table come from public
specifications that the study did not confirm.** Where the archive holds no value, the table gives
the share of hours with a value.

| Product | Lead days fitted | Days not fitted, and why |
|---|---|---|
| ENS mean, ENS control member, GEFS mean, and GFS (native) | 0, 1, 2, 3, 5, 7, 10, and 14 | Days 4, 6, 8, 9, and 11 to 13 were not fitted. Each of these products reaches day 14. |
| IFS 0.25° and GFS (Open-Meteo) | 0, 1, 2, 3, 5, and 7 | Days 4 and 6 were not fitted. Open-Meteo's Previous Runs archive stops at day 7, so days 10 and 14 are not in the archive. |
| IFS HRES 9 km (Open-Meteo Single Runs) | 0, 1, 2, 3, 5, and 7 | Days 4, 6, 8, and 9 were not fitted. The runs end at lead 240 hours, so day 10 is beyond the runs. |
| ICON global | 0, 1, 2, 3, and 5 | Days 4 and 6 were not in archive. Day 7 is 0% non-null in the archive, so day 6 is the last day with values. |
| ICON-EU | 0, 1, 2, and 3 | Day 4 was not in archive. Day 5 is 0% non-null in the archive, so day 4 is the last day with values. |
| ARPEGE Europe (solar only) | 0, 1, 2, and 3 | Beyond the product's horizon from day 4. |
| UKV | 0 and 1 | Day 2 is 0% non-null in the archive, and the study could not verify UKV's horizon locally. |
| ICON-D2, AROME France (solar only), DMI HARMONIE-AROME, and KNMI HARMONIE-AROME | 0 and 1 | Beyond the product's horizon from day 2. |

**Day 0 reads the freshest run that covers each hour, and its served lead is checked for only
ICON-EU and ICON-D2, and only for radiation.** The past-weather studies matched ICON-EU's served
radiation to ICON's own files within 1 W/m² at 9 of 9 hours (one day, one place). For ICON-D2 the
freshest run was the closest match at 7 of 9 hours, and differed by up to 44 W/m². The wind lead of
0 to 2 hours for both products is inferred from where the hour-to-hour jumps of the served series
fall and from the 3-hourly run cycle, and the radiation lead is 1 to 3 hours. This study's day-0
series for the two products equals those studies' series on every shared hour (the largest absolute
difference is 0.0000). The day-0 lead of every other Previous Runs product is estimated from the
product's own run cycle, not measured. ENS, GEFS, and IFS HRES 9 km read the 00 UTC run of the
hour's own day at day 0, so their day-0 marks cover hours before that run is published. GFS (native)
reads the freshest of four runs a day, at a lead of 1 to 6 hours for solar and 0 to 5 hours for
wind, which ignores the hours GFS takes to publish a run.

**GFS (native) reads Dynamical.org's store of NOAA's GFS at each generator's nearest 0.25° cell, and
differs from GFS (Open-Meteo) in its source, run cycle, and lead.** The store holds four runs a day
(00, 06, 12, and 18 UTC), with hourly leads to 120 hours and 3-hourly leads after. Day N of 1 or
more reads the 00 UTC run issued N days before the hour's own day, at ENS's own leads. The store's
radiation is the mean since the last 6-hourly reset, and the build converts it to the mean over each
hour, or over each 3 hours after lead 120 hours. Days 5, 7, 10, and 14 lie on 3-hourly leads and are
upsampled to hourly as the ENS and GEFS arms are.

**IFS HRES 9 km is Open-Meteo's Single Runs archive of ECMWF's IFS HRES on its native O1280 grid,
the same forecasting system as IFS 0.25° on a finer grid.** The archive (`ecmwf_ifs`, on ECMWF's
O1280 grid) holds one 00 UTC run a day, with hourly leads 0 to 240 hours. Day N reads the 00 UTC run
issued N days before the hour's own day, at ENS's own leads. The archive publishes every 3 hours
after lead 90 and every 6 hours after lead 144, and Open-Meteo interpolates those steps to hourly,
so the hourly values at days 5 and 7, and at the last hours of day 3, are interpolated. Radiation is
clipped at zero (the archive holds 244 values below zero, the lowest -1.0 W/m²), and wind is the
served speed and the sine and cosine of the served direction. Each solar generator reads its nearest
cell and each wind generator its nearest land cell, and sites B and D share a source cell and carry
identical series. Open-Meteo's processing of HRES was not checked against a native archive. IFS
Cycle 50r1 began on 2026-05-12, inside the span, and these arms add no era feature, so the shared
rows' era code has no boundary at that date.

**IFS HRES 9 km is scored on the shared hours minus the target days its archive lacks.** The archive
holds 919 of 926 run days, so seven run days are absent. Those gap days are never filled from
another run. The six solar arms score 34,751 to 34,786 rows, and the six wind arms 36,957 to 37,008,
against 35,263 and 37,407 shared rows. Every contrast that involves IFS HRES 9 km is computed on the
rows both arms score, from the existing out-of-fold errors with no refit of the other arm, whose
training rows included the gap days. Dropping the gap days of the day-1 arm moves the ENS mean's
day-1 error from 8.771% to 8.792% for solar (+0.022 points) and from 8.350% to 8.379% for wind
(+0.029 points).

**Part of the rise in ENS's and GEFS's error with lead may come from coarser time steps.** ENS
serves 6-hour steps beyond 144 hours and GEFS beyond 240 hours. ENS at days 7, 10, and 14 and GEFS
at days 10 and 14 therefore fall on 6-hour steps, and ENS at day 5 touches a 6-hour step only in the
margin its upsampling reads beyond lead 144. Part of the rise in the error of those arms with lead
may come from the coarser steps and not only from an older run. `verify_extra_leads.py` checked,
before the build, that GEFS's radiation beyond 240 hours is a 6-hour window mean.

**Every mark in Figures 3 and 4 is a GPU fit, and a GPU fit is not bit-identical to a central
processing unit (CPU) fit.** Every contrast among the extra arms therefore uses reference arms (the
arms each contrast subtracts) refitted on the same device. The device noise floor is the difference
between a GPU refit and the published CPU fit of the same arm. For solar it lies between -0.024 and
+0.014 points, and every interval includes zero. For wind every point estimate is negative, between
-0.090 and -0.040 points, and one difference, ICON-EU at day 3 (-0.079 points [-0.167, -0.005]), is
statistically significant at the 5% level. A wind mark from a GPU fit may therefore sit slightly
below where a CPU fit would put it. A GPU fit repeats itself: the 23 solar arms and 23 wind arms
that both an earlier GPU run and batch 1 fitted have identical per-row errors, to 0.0 on every row
(a comparison of the two runs' saved losses that no committed script prints). The published headline
contrasts, the blends, and the per-generator contrasts (Figures 5, 6, and 9 to 12) rest on the CPU
fits of the published run.

**The ENS numbers at the extra leads are not comparable with the ENS horizons study.** Days 5, 7,
10, and 14 of ENS, and ICON's day 0, use this study's rows, which start after the IFS Cycle 49r1
change. The ENS numbers here are therefore not comparable with the [ENS horizons
study](ens-horizons.md), whose ENS rows span that change.
<!-- report (extra leads): Device noise floor; Absolute error of every arm fitted here -->

### How AIFS is read

**AIFS is read at the same lead as ENS, on 6-hourly steps.** For a target hour on UTC day D, the
day-`d` band reads the 00 UTC run of day D minus `d`, so the lead is 24`d` plus the hour, at days 1
and 2. AIFS Single and AIFS ENS serve one value every 6 hours, where ENS serves one every 3 hours to
144 hours, so an XGBoost model given ENS at full resolution would be favoured by its finer steps
alone. The like-for-like references are therefore ENS's control member (against AIFS Single) and
ENS's mean (against AIFS ENS's mean), both averaged to the same 6-hourly steps. Hourly IFS 0.25° is
a one-sided reference, because its hourly steps and its shorter lead both favour IFS 0.25°. The
positive control is ENS's mean on 6-hourly steps minus ENS's mean on 3-hourly steps, which should be
positive: it is +0.243 points [+0.153, +0.332] for solar and +0.321 points [+0.206, +0.419] for
wind.

**AIFS Single's radiation is a 6-hour mean that ends at the lead, and its wind is instantaneous.**
Against ERA5, the mean absolute difference of AIFS Single's radiation from ERA5's 6-hour mean is
smallest, at 16.36 W/m², when the window ends at the lead (39.55 W/m² one hour earlier and 37.97
W/m² one hour later). The correlation of AIFS Single's 100 m wind speed with ERA5's instantaneous
100 m wind speed is highest, at 0.9422, at the same hour. The units and the grid orientation also
passed their checks.
<!-- verification: aifs_steps.md -->

**Every AIFS arm reads the same spatial average that ENS reads.** Each generator's value is the mean
of the 0.25° cells under its H3 resolution-5 hexagon, weighted by overlap. A nearest-cell AIFS
Single arm shows how far the spatial read moves the result.

**AIFS arms have rows of their own.** AIFS Single carries radiation and 100 m wind from the
2025-02-24 06 UTC run, and AIFS ENS from 2025-07-02, so neither covers the hours every other product
is scored on. The `single` row set holds the hours whose 00 UTC run is inside one AIFS Single
version, and the `ens` row set is the part of that AIFS ENS also covers. Every arm on a row set is
scored on the same hours, and the study fits the reference arms again on each row set. The
Dynamical.org store carries no model-version marker, so the study assigns each run to a version by
its run date, taken from ECMWF's release pages. Folds are cut inside each version era, and the
months that hold a version switch (2025-08 and 2026-05) are dropped. The `single` row set holds
28,570 solar rows and 28,873 wind rows over 16 months, and the `ens` row set holds 16,911 solar rows
and 18,555 wind rows over 11 months. The `ens` row set starts in 2025-09, because July 2025 would
form a one-month version era. Each AIFS arm has the same seven columns as an ENS arm.

**One contrast was planned before any AIFS fit: AIFS Single against ENS's control member at day 1,
for solar and for wind.** The page calls this AIFS contrast a deciding contrast, and it is planned.
The study also refits it at the sensitivity setting, without day of year in either arm, and at a
97.5% interval (a Bonferroni correction across solar and wind), and drops each month in turn. The
AIFS ENS contrasts are descriptive only, because about 84% of the scored (generator, fold, calendar
month) cells of the `ens` row set have no training row of their calendar month (55 of 65 solar cells
and 27 of 33 wind cells). Every other AIFS number is exploratory and post hoc. Every AIFS fit is on
the GPU.
<!-- report (AIFS): header; row sets; deciding contrast -->

**The conventions of AIFS Single's radiation and wind were checked at days 1 and 2 only.** The
script `verify_aifs_steps.py` compares AIFS Single with ERA5 at several offsets to find the hour at
which each field is stamped. At days 7 and 14 the check fails, because AIFS Single's correlation
with ERA5's wind speed is only 0.43 at day 7 and 0.15 at day 14, so an offset test cannot tell one
offset from another. The checks pass at days 1 and 2, and the wiring check, which compares the built
columns with the raw store, passes at every day. This study therefore assumes, and did not check,
that the same radiation window and wind reading hold at days 7 and 14.
<!-- verification: aifs_steps.md, aifs_wiring.md (blends folder) -->

**AIFS Single is also read at days 7 and 14, and blended with ENS's mean at days 1, 2, 7, and 14.**
Days 7 and 14 read the 00 UTC run 7 and 14 days before the hour's own day, at ENS's own leads, so
the ENS control member and the ENS mean are natively 6-hourly there and share AIFS Single's steps.
A row is dropped when its run lies outside the AIFS version era of its hour. That drops 362 solar
and 454 wind hours at day 7 and 1,121 and 1,433 at day 14. The `single` row set then holds 28,208
solar and 28,419 wind rows at day 7, and 27,449 and 27,440 at day 14, all over 16 months. A blend
gives one XGBoost model ENS's mean and AIFS Single's forecast at the same day. A solar blend has 9
columns and a wind blend 11, against 7 for ENS's mean alone, so the study also fits two controls
with the blend's number of columns. The blend's control shuffles AIFS Single's weather
among hours of the same generator, year-month, and hour of day, and the mirror control shuffles
ENS's mean instead. Because two shuffles of the same weather can differ (see the
[AIFS section](matched-lead.md#aifs-single-has-a-lower-wind-error-than-enss-control-member-at-day-1-and-no-solar-difference-is-claimable)),
a shuffled arm's interval may understate the noise. The blends fit and the P4 second-seed refit
below were fitted on the GPU.

**Four contrasts per technology are deciding contrasts, and the page does not call them planned.**
Each was named before the fit of this section, but after the day-1 and day-2 AIFS results and the
ENS results at days 7 and 14 were known. They are:

- **H7 and H14:** AIFS Single minus the ENS control member at day 7 and at day 14.
- **B7 and B14:** the blend of ENS's mean and AIFS Single minus ENS's mean alone at day 7 and at
  day 14, with the blend minus its control as the guard.

A deciding contrast is claimable only if its interval is statistically significant at the 5% level
at both hyperparameter settings with the same sign, and every dropped month keeps the sign. For H7
and H14 the refit without day of year in either arm must also agree in sign. For B7 and B14 the
guard must also be statistically significant at both settings. The report also prints each deciding
contrast at 99.375%, the Bonferroni level across the eight deciding contrasts (four per technology).
Every other contrast in the section is exploratory and post hoc, and AIFS ENS rows are descriptive.

**A day-14 reading rule decides whether there is any skill to compare at day 14.** Skill exists at
day 14 only if AIFS Single or the ENS control member has an error that is statistically
significantly below both shuffled-AIFS arms (two seeds, primary setting, 95% interval). Where the
rule finds no skill, the page does not read H14 or B14, and prints their intervals only for
completeness. A second reading rule applies beside H7 and H14: AIFS Single having a lower error
than the control member but not than the ENS mean is consistent with smoothing, and is not
described as a better weather forecast.
<!-- report (blends): header; deciding contrasts; day-14 reading rule; smoothing reading -->

### How WeatherNext 3 is read

**WeatherNext 3 (WN3), Google's machine-learned ensemble weather model, is read at the same lead as
ENS, at days 1, 2, 7, and 14, on hourly steps.** For a target hour on UTC day D, the day-`d` band
reads the 00 UTC run of day D minus `d`, as for ENS and AIFS. The WN3 store holds 360 hourly leads
of the ensemble mean only, on a 0.1° grid, for runs from 2026-01-01. The study copies the 00 UTC
runs and the cells that cover the 6 solar farms and 3 wind farms, and averages the cells over
each farm's H3 cell in the same way as for ENS and AIFS. Radiation is the accumulation over the hour
ending at the target time, in joules per square metre, so the study divides it by 3,600 s to get
watts per square metre. For solar, the 2 m temperature at an hour's midpoint is the mean of the two
hourly values on either side of it, as ENS's is.

**WN3's wind speed is the length of the mean wind vector, which is biased low relative to ENS's.**
The store holds the ensemble mean of the eastward and northward wind components and no member
values, so WN3's speed is the length of the mean vector. ENS's speed is the mean of the members'
speeds, which is larger when the members disagree in direction. The study therefore also fits, for
wind at every day, a matched ENS reference: ENS's mean whose speed is the length of the mean of its
members' wind vectors. That reference, and not ENS's mean speed, is the one the wind contrasts
against WN3 use as the like-for-like comparison. ENS's members' speed and direction at 100 m and
10 m are on local disk for every day, so the reference needs no download. If a run lacked members,
the build would drop that run and the fit would stop, naming the run.

**The WN3 rows cover about 7 months, so every WN3 contrast is descriptive.** The row set holds
February to April and June to 10 September 2026. September holds 10 days only, and counts as a whole
month when the intervals resample months. Each calendar month occurs in one year only, so no scored
month has a training row of its own calendar month. The contrast of interest is WN3's mean minus
ENS's mean on the same rows, at days 1, 2, 7, and 14, for solar and wind, each with the sensitivity
setting and each read as descriptive. Each is paired with a negative control that shuffles WN3's
weather within farm, year-month, and hour of day, under two seeds. The day-14 gap minus the day-1
gap is exploratory, and is read from the two intervals, not tested. A difference whose 95% interval
spans 0 is reported with the interval's bounds, because a null at day 14 excludes only differences
larger than the interval, and is never described as no difference. AIFS Single and the AIFS ENS
mean keep the days they were fitted at (1, 2, 7, and 14), on their own row sets.

### What is known about WN3's lookahead

**Which WeatherNext 3 (WN3) weather-model version made the 2026 archive is not documented, so the
in-sample months (before July) may or may not overlap its training data.** [Rasp et al.
(2026)](https://arxiv.org/abs/2609.03582) state "For the production model, we train until June 30
2026". Their Appendix A.1.3 also lists versions trained until the start of 2025 and until the start
of 2026, and the paper does not say which version produced the archive. We found no statement from
Google of which version made the 2026 archive. Every WN3 result on this page is therefore reported
for three groups of rows: the months before July (February to June), which may overlap training; the
months from July (July to September), which lie after every training end the paper lists; and a
pooled group of every scored row from February to September 2026, including the rows that fall in
neither of the first two groups. The July to September rows are the check for a version change and
for overlap with training data, and the leaderboards draw the pooled group, whose 7 months clear the
study's minimum of 6 months for an interval. WN3 against AIFS or ENS on the months before July may
not be a fair comparison, because WN3's weights may have seen those months and the other products'
had not. The pooled group mixes both kinds of month, so its lead over ENS may include such overlap,
and the leaderboards' captions say so. The folds are unchanged, and each group selects rows that
were already scored out of fold. The July to September group is still not fully clean: the
out-of-fold models train on the February to June WN3 rows, which may overlap WN3's training data,
and the direction of that effect is unknown.

**Hypothesis, not documented and stated by no source we found: the archive for January to June 2026
was made by a WN3 weather model trained on data from before 2026, and the archive from July 2026
onwards by the production WN3 weather model, trained on data from before July 2026.** The reasoning
behind the hypothesis is that an archive from a weather model that had seen the months it forecasts
would overstate its skill. If the hypothesis holds, no archived forecast comes from a model that had
seen the period it forecasts. The three groups of rows are motivated by this hypothesis. The split
checks whether a version change at the start of July shows in the scores, and makes no claim about
which version made any row. A difference between the groups is weak evidence either way, because
every product's error differs between the months (see the absolute-skill paragraph in the results).

**A row is out-of-sample only if its valid month is July or later and the 00 UTC run it reads was
issued after 30 June.** A day-14 row that verifies in July from a run issued in the last two weeks
of June sits in neither group. The report gives the count of such rows for every technology and lead
day. The out-of-sample group holds 3 calendar months, fewer than the study's minimum of 6 for an
interval, so its intervals give the range of the months and are indicative only.

**WN3's archive is labelled real-time, and the copy cannot show when Google produced each run.**
Google's [dissemination guide](https://developers.google.com/weathernext/guides/dissemination)
gives a latency of 7 hours 45 minutes (Cloud Storage) to 8 hours 10 minutes (BigQuery) for the
6-hourly runs, and the [Cloud Storage guide](https://developers.google.com/weathernext/guides/gcs)
labels the archive "real-time operational data for 2026 to present". The store's `run_written` and
`source_init_time` records show that the 271 runs at 00 UTC from 2026-01-01 to 2026-09-28 are all
present, with no gap and no run whose `source_init_time` differs from its `init_time`. Our own
download wrote those records, so they show that the copy is complete and say nothing about when
Google produced each run. If Google produced the runs after the fact, WN3's error is optimistic.
The `--lookahead-cleared` flag of `fit_aifs.py --wn3` stops the fit from starting until whoever runs
it has read this section.

## Results

### Neither native GFS nor IFS HRES 9 km has an error detectably lower than the ENS mean at any lead day (exploratory)

**Native GFS has a higher error than the ENS mean at every lead day from 1 to 10 for both
technologies, and at day 0 for solar.** The two cannot be told apart at day 14 for either
technology, or at day 0 for wind. The native GFS arm at day N reads a 00 UTC run's leads from 24N
hours, which is ENS's own lead, so the day-N contrasts compare equal leads. Day 0 is the exception:
native GFS reads the freshest of four runs a day, a shorter lead than ENS's, which favours native
GFS. The contrasts are exploratory, and each product is read at a different spatial scale (native
GFS at the nearest 0.25° cell, ENS as an average over an H3 cell).

| Exploratory: native GFS minus ENS mean at the same day (primary) | Solar: native GFS (%) | Solar: ENS mean (%) | Solar difference (points) | Wind: native GFS (%) | Wind: ENS mean (%) | Wind difference (points) |
|---|---|---|---|---|---|---|
| Day 0 | 10.061 | 8.125 | +1.936 [+1.649, +2.207] | 7.567 | 7.426 | +0.141 [-0.019, +0.293] |
| Day 1 | 11.438 | 8.771 | +2.667 [+2.286, +3.021] | 9.622 | 8.350 | +1.272 [+1.003, +1.516] |
| Day 2 | 12.081 | 9.759 | +2.322 [+2.020, +2.629] | 11.117 | 9.473 | +1.644 [+1.372, +1.882] |
| Day 3 | 12.996 | 10.797 | +2.199 [+1.852, +2.541] | 12.798 | 11.089 | +1.709 [+1.302, +2.079] |
| Day 5 | 14.112 | 12.560 | +1.552 [+1.086, +2.059] | 16.233 | 14.251 | +1.981 [+1.141, +2.689] |
| Day 7 | 14.423 | 13.748 | +0.676 [+0.337, +1.058] | 17.946 | 16.884 | +1.062 [+0.188, +1.892] |
| Day 10 | 14.867 | 14.457 | +0.410 [+0.082, +0.727] | 19.129 | 18.223 | +0.907 [+0.318, +1.536] |
| Day 14 | 14.696 | 14.876 | -0.180 [-0.426, +0.052] | 19.184 | 18.867 | +0.318 [-0.156, +0.775] |

**Native GFS has a higher error than GFS (Open-Meteo) at days 1 and 3 for solar and at days 1, 2, 3,
and 5 for wind, and the difference mixes source and lead.** Open-Meteo serves the freshest run at
least N days old at day N, a shorter lead than the 24N hours of the native arm. The two cannot be
told apart at the other days compared.

| Exploratory: GFS (native) minus GFS (Open-Meteo) at the same day (points, primary) | Solar | Wind |
|---|---|---|
| Day 1 | +0.536 [+0.322, +0.746] | +0.634 [+0.452, +0.800] |
| Day 2 | +0.070 [-0.195, +0.319] | +0.687 [+0.437, +0.967] |
| Day 3 | +0.418 [+0.230, +0.634] | +0.688 [+0.274, +1.056] |
| Day 5 | +0.228 [-0.041, +0.508] | +0.509 [+0.142, +0.938] |
| Day 7 | +0.045 [-0.255, +0.365] | +0.160 [-0.291, +0.597] |

**IFS HRES 9 km has a higher error than the ENS mean at every lead day fitted, for both
technologies.** IFS HRES 9 km and the ENS mean read the same lead, the 00 UTC run at 24N hours, so
the difference is not a difference of lead. The difference may include the ensemble averaging that
ENS's mean has and a single run lacks. The study did not test that explanation, and did not check
Open-Meteo's processing of HRES against a native archive. Every arm here is scored on the shared
hours minus the seven run days the archive lacks, and each contrast on the rows both arms score.

| Exploratory: IFS HRES 9 km minus ENS mean at the same day, on the rows both score (primary) | Solar: IFS HRES 9 km (%) | Solar: ENS mean (%) | Solar difference (points) | Wind: IFS HRES 9 km (%) | Wind: ENS mean (%) | Wind difference (points) |
|---|---|---|---|---|---|---|
| Day 0 | 8.816 | 8.142 | +0.674 [+0.522, +0.822] | 7.636 | 7.434 | +0.202 [+0.045, +0.378] |
| Day 1 | 9.762 | 8.792 | +0.970 [+0.762, +1.161] | 8.879 | 8.379 | +0.500 [+0.305, +0.717] |
| Day 2 | 10.801 | 9.775 | +1.026 [+0.796, +1.249] | 10.179 | 9.506 | +0.673 [+0.419, +0.938] |
| Day 3 | 11.925 | 10.834 | +1.091 [+0.775, +1.416] | 12.047 | 11.129 | +0.918 [+0.606, +1.228] |
| Day 5 | 13.475 | 12.623 | +0.852 [+0.425, +1.346] | 15.810 | 14.323 | +1.487 [+0.987, +1.913] |
| Day 7 | 14.505 | 13.789 | +0.716 [+0.278, +1.106] | 17.864 | 16.996 | +0.868 [+0.166, +1.631] |

**IFS HRES 9 km has a higher error than IFS 0.25° from Previous Runs at days 1, 2, 3, and 5, and the
two cannot be told apart at day 7, but the two products read different leads.** Previous Runs serves
IFS 0.25° from the freshest run at least N days old, a shorter lead than IFS HRES 9 km's. A lead
difference that averages about 9 hours probably accounts for only part of gaps this large, and the
study did not test what accounts for the rest, such as Open-Meteo's processing of the 9 km archive.
Against ICON-EU, which also has a shorter lead, IFS HRES 9 km cannot be told apart at days 1 to 3
for solar, has a higher error at days 1 and 3 for wind, and cannot be told apart at day 2 for wind.

| Exploratory: IFS HRES 9 km minus the product named, at the same day, on the rows both score (points, primary) | Solar | Wind |
|---|---|---|
| IFS 0.25° at day 1 | +0.927 [+0.686, +1.152] | +0.488 [+0.314, +0.674] |
| IFS 0.25° at day 2 | +1.143 [+0.830, +1.445] | +0.938 [+0.680, +1.222] |
| IFS 0.25° at day 3 | +0.966 [+0.668, +1.257] | +1.203 [+0.876, +1.551] |
| IFS 0.25° at day 5 | +0.780 [+0.485, +1.055] | +0.982 [+0.579, +1.410] |
| IFS 0.25° at day 7 | -0.115 [-0.553, +0.208] | +0.383 [-0.143, +0.986] |
| ICON-EU at day 1 | +0.166 [-0.123, +0.418] | +0.340 [+0.156, +0.572] |
| ICON-EU at day 2 | +0.125 [-0.188, +0.410] | +0.211 [-0.047, +0.497] |
| ICON-EU at day 3 | +0.118 [-0.217, +0.413] | +0.494 [+0.168, +0.811] |

<!-- report (native GFS): Native GFS against the ENS mean; against Open-Meteo's GFS; (IFS HRES): against
the ENS mean, IFS 0.25 degree, ICON-EU; row-set diagnostic -->

### WN3 and AIFS at days 0, 3, 4, and 10 (exploratory)

**In the pooled rows and in the February to June rows, WN3's mean has a lower error than ENS's mean
(for wind, ENS's mean-vector reference) at days 3 and 4 for solar and wind, with intervals that
exclude 0, and in the July to September rows only solar days 3 and 4 and wind day 3 do the same.**
In every group, the solar day-0 and day-10 differences span 0. At wind day 0 the difference excludes
0 in the pooled and February to June rows and spans 0 in the July to September rows, and at wind day
4 it spans 0 in the July to September rows. At wind day 10, WN3's error is higher than ENS's, by an
interval that excludes 0 in the pooled and February to June rows and spans 0 in the July to
September rows. Day 0 is a hindcast: the row of an hour reads the 00 UTC run of the hour's own day,
a forecast no service could read. Solar day 0 omits the hours ending 01:00 to 06:00 UTC for every
arm, because those hours precede the first 6-hourly AIFS step, and the WN3 wind day 0 drops the hour
ending 00:00 UTC, for which the WN3 store holds no lead. Every number in this section is exploratory
and fitted at the primary setting only. Each WN3 error below is the mean absolute error of an
XGBoost model given WN3's mean, as a percentage of capacity. Each difference is WN3's error minus
ENS's error on the same rows, in points of capacity, so a negative difference means WN3 has the
lower error. The wind reference is ENS's mean-vector speed, which matches how WN3's speed is built.
<!-- report (WN3 extra days): all three row groups, ens_mean (solar) and ens_meanvec (wind) arms -->

| Row group, technology, day | WN3 error (%) | ENS error (%) | WN3 minus ENS (points) |
|---|---|---|---|
| Pooled, solar, day 0 | 8.796 | 8.878 | -0.082 [-0.226, +0.097] |
| Pooled, solar, day 3 | 10.791 | 12.268 | -1.478 [-2.196, -0.831] |
| Pooled, solar, day 4 | 12.035 | 13.120 | -1.086 [-2.059, -0.460] |
| Pooled, solar, day 10 | 14.993 | 15.293 | -0.300 [-0.816, +0.157] |
| Pooled, wind, day 0 | 6.416 | 6.720 | -0.305 [-0.663, -0.053] |
| Pooled, wind, day 3 | 10.107 | 11.787 | -1.680 [-2.672, -0.574] |
| Pooled, wind, day 4 | 11.721 | 13.370 | -1.650 [-3.162, -0.261] |
| Pooled, wind, day 10 | 19.003 | 17.352 | +1.652 [+0.014, +3.502] |
| February to June, solar, day 0 | 8.956 | 8.970 | -0.014 [-0.134, +0.089] |
| February to June, solar, day 3 | 11.523 | 13.204 | -1.681 [-2.562, -1.120] |
| February to June, solar, day 4 | 12.633 | 14.145 | -1.513 [-3.077, -0.408] |
| February to June, solar, day 10 | 16.783 | 17.216 | -0.433 [-1.279, +0.396] |
| February to June, wind, day 0 | 7.037 | 7.490 | -0.454 [-0.993, -0.105] |
| February to June, wind, day 3 | 11.700 | 14.153 | -2.453 [-3.225, -0.410] |
| February to June, wind, day 4 | 13.338 | 16.419 | -3.081 [-4.376, -1.991] |
| February to June, wind, day 10 | 24.268 | 21.652 | +2.616 [+0.625, +5.676] |
| July to September, solar, day 0 | 8.592 | 8.761 | -0.169 [-0.328, +0.634] |
| July to September, solar, day 3 | 9.957 | 11.066 | -1.109 [-1.878, -0.293] |
| July to September, solar, day 4 | 11.188 | 11.652 | -0.464 [-0.684, -0.120] |
| July to September, solar, day 10 | 13.150 | 13.225 | -0.075 [-0.214, +0.253] |
| July to September, wind, day 0 | 5.722 | 5.860 | -0.138 [-0.803, +0.066] |
| July to September, wind, day 3 | 8.172 | 8.817 | -0.645 [-1.379, -0.280] |
| July to September, wind, day 4 | 9.444 | 9.548 | -0.104 [-0.594, +0.483] |
| July to September, wind, day 10 | 12.667 | 12.313 | +0.354 [-1.121, +2.686] |

**The February to June rows cover 4 calendar months and the July to September rows 3, both fewer
than the study's minimum of 6, so their intervals are indicative only.** The pooled rows cover 7
months (February to April and June to September 2026). Google has not documented which WeatherNext 3
model version made the archive, so the pooled rows may include months that overlap WN3's training
data. The sensitivity setting was also fitted for the contrasts whose interval lies near the 5%
line, and the report gives it.

**At wind day 10, WN3's mean is not detectably better than WN3's own weather shuffled within site,
year-month, and hour of day.** The pooled WN3 error minus the first shuffled arm's is +0.202 points
[-0.847, +1.278], so the interval spans 0 and the XGBoost model given WN3's day-10 wind has no
detectable skill over the shuffled control. At solar day 10 the two shuffle seeds disagree in the
pooled rows: WN3 minus the first shuffled arm is -1.169 points [-1.910, -0.463], and minus the
second is -0.364 points [-0.952, +0.336].

**AIFS Single and the AIFS ENS mean at these days are read as point differences, because the fits
name no contrast and give no paired interval.** The absolute intervals overlap heavily wherever they
are given, so no AIFS difference below is claimed. Each row is on the rows of its own row set (16
months for AIFS Single, 11 for the AIFS ENS mean), against ENS's mean on the same rows. The AIFS
minus ENS mean column is the AIFS error minus the ENS mean error, taken from the tabulated errors,
because no report holds those differences. <!-- report (AIFS extra days): single and ens row sets
-->

| AIFS arm, technology, day | AIFS error (%) [95% interval] | ENS mean error (%) [95% interval] | AIFS minus ENS mean (points) |
|---|---|---|---|
| AIFS Single, solar, day 0 | 9.318 [8.595, 9.926] | 8.707 [8.027, 9.317] | +0.611 |
| AIFS Single, solar, day 3 | 11.005 [10.100, 11.808] | 11.319 [10.429, 12.230] | -0.314 |
| AIFS Single, solar, day 4 | 12.262 [11.241, 13.240] | 12.290 [11.355, 13.241] | -0.028 |
| AIFS Single, solar, day 10 | 14.850 [13.796, 16.015] | 14.698 [13.654, 15.912] | +0.152 |
| AIFS Single, wind, day 0 | 7.600 [6.926, 8.293] | 7.014 [6.308, 7.755] | +0.586 |
| AIFS Single, wind, day 3 | 10.864 [9.616, 12.237] | 10.896 [9.552, 12.424] | -0.032 |
| AIFS Single, wind, day 4 | 12.623 [11.088, 14.337] | 12.708 [11.157, 14.384] | -0.085 |
| AIFS Single, wind, day 10 | 17.568 [15.264, 19.761] | 17.484 [15.151, 19.916] | +0.084 |
| AIFS ENS mean, solar, day 0 | 9.073 [8.025, 9.882] | 8.584 [7.613, 9.286] | +0.489 |
| AIFS ENS mean, solar, day 3 | 10.616 [9.321, 11.834] | 11.548 [9.990, 13.129] | -0.932 |
| AIFS ENS mean, solar, day 4 | 12.100 [10.455, 13.731] | 12.393 [10.629, 14.333] | -0.293 |
| AIFS ENS mean, solar, day 10 | 14.536 [12.728, 16.626] | 14.641 [12.976, 16.415] | -0.105 |
| AIFS ENS mean, wind, day 0 | 8.162 [7.178, 9.069] | 7.391 [6.434, 8.277] | +0.771 |
| AIFS ENS mean, wind, day 3 | 11.695 [10.138, 13.197] | 12.096 [10.300, 13.939] | -0.401 |
| AIFS ENS mean, wind, day 4 | 12.814 [10.769, 14.738] | 13.796 [11.531, 15.884] | -0.982 |
| AIFS ENS mean, wind, day 10 | 19.715 [15.857, 23.114] | 18.915 [15.330, 22.112] | +0.800 |

**At day 0, AIFS has a higher point error than ENS's mean for both technologies, by 0.5 to 0.8
points, and at day 3 and day 4 the AIFS point error is lower in every row.** At day 10 the sign
changes with the row set and the technology. The day-4 ENS values come from a supplementary
extraction of ENS's leads 105, 108, and 111 hours, which the study's main ENS extract lacks.
