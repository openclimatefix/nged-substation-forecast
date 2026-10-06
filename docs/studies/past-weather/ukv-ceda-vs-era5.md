# At three wind farms ERA5's wind gave a slightly lower power error than CEDA's archived UKV, and at four weather stations UKV's temperature was closer than ERA5's

**At three wind farms in Lincolnshire, an XGBoost model (a gradient-boosted tree model) given ERA5's
100 m and 10 m wind had a lower power error than an XGBoost model given the Met Office's UK
variable-resolution (UKV) 10 m and 925 hPa wind from the archive of the Centre for Environmental
Data Analysis (CEDA), by 0.125 points of capacity [+0.033, +0.216] at the primary hyperparameter
setting and [+0.035, +0.217] at the second.** The difference is statistically significant at the 5%
level and smaller than the 0.16-point margin fixed before any result, so by the planned rule the
study recommends ERA5 for past wind. The page calls the archived UKV UKV-CEDA. UKV-CEDA is a
6-hourly archive read at leads of 0 to 5 hours, so the result says nothing about live UKV, which the
[past-wind page](wind.md) found ahead of ERA5. Two post hoc results sit beside it. The gap between
the two products grows with UKV-CEDA's lead, from -0.058 points at lead 0 to +0.323 points at lead
5, and it does not shrink when both XGBoost models are given 10 m wind alone (+0.505 points [+0.394,
+0.620]), so the difference in wind height does not explain it.

**At the four Met Office stations inside the UKV-CEDA crop, UKV-CEDA's 1.5 m air temperature was
closer to the readings than ERA5's 2 m air temperature, by 0.124 K [0.113, 0.135] after removing
each product's bias, so by the planned rule the study recommends UKV-CEDA for past temperature.**
The advantage falls from 0.250 K at lead 0 to 0.035 K at lead 5 (post hoc), which fits the stations'
readings entering UKV's own data assimilation, and the page does not test that explanation. How
large the advantage is away from those stations, and whether it helps a demand forecast, is
untested. The choice between the two temperatures moved the power error of an XGBoost model at six
solar farms by no more than 0.009 points of capacity in either direction.

![Figure 1: At four stations UKV-CEDA is closer than ERA5 for wind and temperature, but ERA5 gives
the lower wind-power error, and the choice does not move solar
power](../assets/ukv_ceda_vs_era5/fig01_headline.svg)

- **Wind history:** the planned rule gives ERA5, not UKV-CEDA's 6-hourly archive, for the 100 m and
  10 m wind that an XGBoost model reads. This is a statement about this archive at these three farms
  and not about UKV.
- **Temperature history:** the planned rule gives UKV-CEDA, with the conditions in
  [Discussion](#discussion-what-to-use): the licence, the five dropped months, and the 6-hourly lead
  pattern.
- **Training history from 2019:** the early-years test passes for UKV-CEDA temperature, and era 0 of
  the UKV physics holds only 3 months.

> **How this page was made.** The research question came from a human. Everything else — the code
> behind every result, the analysis, the figures, and the text — was written by Claude, Anthropic's
> AI model (for this page, Claude Sonnet 5.5, with shared code from earlier Claude Opus 5.5 and
> Claude Sonnet 5 sessions). Several independent Claude reviewers have checked the method, the
> evidence, and the prose adversarially.

## Key findings

**The planned wind contrast P3 favours ERA5 by 0.125 points of capacity, which is statistically
significant and below the margin ([wind](#era5-gave-the-lower-wind-power-error-by-0125-points)).**
The planned rule therefore gives ERA5. The page does not claim that ERA5's wind is better than
UKV's.

**The wind gap grows with UKV-CEDA's lead, and it differs from the past-wind page's result for
reasons the archives explain ([lead](#the-wind-gap-grows-with-the-served-lead)).** From 2024-08-12
the gap is +0.039 points [-0.115, +0.186], against -0.44 points on the past-wind page.

**With 10 m wind alone in both XGBoost models, ERA5 is ahead by +0.505 points, so wind height is
not the cause ([matched pair](#10-m-wind-alone-widens-the-gap)).**

**UKV-CEDA's temperature is closer at the four stations by 0.124 K, and the advantage falls from
0.250 K to 0.035 K across leads 0 to 5
([temperature](#ukv-cedas-temperature-is-closer-at-four-stations-and-less-so-at-longer-leads)).**
The planned rule gives UKV-CEDA. The advantage is not shown to help a demand forecast.

**For six solar farms the choice of temperature product moves the power error by no more than 0.009
points ([solar](#solar-power-does-not-depend-on-the-temperature-product)).** The planned veto could
not have fired.

**The shuffled-weather controls differ from zero by less than their own noise, which is as wide as
the wind margin
([controls](#the-controls-show-no-bias-but-too-much-noise-to-validate-small-differences)).**

**A UKV-CEDA training history has costs: five dropped months, 3.45% of runs missing or partial, a
6-hourly lead pattern, and three physics eras ([costs](#a-ukv-ceda-history-has-practical-costs)).**

**In 2026 the wind difference points towards UKV-CEDA at one hyperparameter setting only
([third era](#in-2026-the-wind-difference-points-towards-ukv-ceda-at-one-setting)).**

## Introduction

**The question is whether the main Flexpectation work should take past wind speed and past air
temperature from UKV-CEDA or from ERA5.** The main work already takes past irradiance from the CAMS
satellite retrieval, and this study does not reopen that choice. The other weather variables in the
main work's configuration (dew point, sea-level pressure, surface pressure, 500 hPa height,
precipitation type, and windchill) are not scored. No published page compares UKV with ERA5 before
August 2024, and CEDA's archive starts in September 2019, so the study tests 2019 and 2020 as well.

| Product | Grid | How read | Served lead |
|---|---|---|---|
| ERA5, from Copernicus (wind) and Open-Meteo's copy (temperature) | 0.25 degrees (about 25 km) | The nearest cell at each farm or station; an instantaneous value each hour | An hourly analysis |
| UKV-CEDA, from CEDA's archive of Met Office GRIB files | 2 km | The nearest cell, from the freshest run that has started at or before the hour | 0 to 5 hours, because the archive's runs start at 00, 06, 12, and 18 UTC |

**Lead equals the hour of the day modulo 6, so a lead is also an hour of day.** The 06 UTC hour
reads the 06 UTC run at lead 0, never the 00 UTC run at lead 6. The archive's runs are 6-hourly,
whereas Open-Meteo's archive, which the [past-wind page](wind.md) used, serves UKV's hourly
analysis. The repository's notes record that CEDA's UKV differs statistically from live UKV, so no
result here says how live UKV would perform.

## Data and methods

**Two sets of experiments answer the question, and the rule that combines them was fixed before any
result.** Set A scores each product against Met Office station readings, with no machine learning.
Set B fits one XGBoost model per metered generator and per product, and scores the model's power
error. Four contrasts were planned, each UKV-CEDA minus ERA5 with a negative value favouring
UKV-CEDA. A planned contrast was written into the study plan before any result existed. Every other
number on this page is exploratory, and a row added after a result was seen is also labelled
post hoc.

| Label | Set | What it compares | Margin |
|---|---|---|---|
| P1 (planned) | A | 10 m wind speed against the stations; context only, never decides | 5% of ERA5's error on the same rows (0.053 m/s) |
| P2 (planned) | A | Air temperature against the stations; decides temperature | 5% of ERA5's error (0.036 K) |
| P3 (planned) | B | Wind-farm power, as-available wind; decides wind | 0.16 points of capacity |
| P4 (planned) | B | Solar-farm power; can veto a UKV-CEDA temperature recommendation | 0.06 points of capacity |

**A contrast reads "clear" only if its 95% interval lies wholly on one side of zero and its point
estimate passes the margin.** Otherwise it reads "small, not clear" (the interval excludes zero but
the estimate is inside the margin) or "no clear difference". The set B margins come from the
half-widths of intervals published on earlier pages and from the number of scored months. P3 must
read the same at both hyperparameter settings. Temperature is decided by P2, provided that P2 is
also clear at leads 3 to 5 hours (P2-lead, planned), and a clear ERA5 reading of P4 at either
setting would veto a UKV-CEDA recommendation. The default is ERA5.

**Set A uses four stations, and the score removes each product's bias.** The stations are the Met
Office MIDAS Open stations whose nearest UKV-CEDA cell lies within 2 km (inside the archive's crop)
and that report wind speed and temperature at 90% or more of hours. Four stations qualify. The score
is the mean absolute error after removing each (station, product, calendar month, hour of day) mean
error, the closest set A analogue of an XGBoost model that learns each farm's bias. A 2 km cell
still represents a point better than a 25 km cell, so set A does not attribute a UKV-CEDA lead to
the product alone. The report also prints the raw error and the error after removing each (station,
product, calendar month) mean. ERA5 wind at one station ends in December 2023, so the wind rows
number 182,521 station-hours and the temperature rows 200,035.

**Set B gives each XGBoost model the same rows and the same number of columns.** The wind rows
number 145,758 over three farms (W1 to W3), the solar rows 122,890 over six farms (A to F), and each
holds 78 scored months. Each wind XGBoost model has 7 columns: the hour of day, the day of year, the
UKV era, and four weather columns. The ERA5 model reads the 100 m speed, the sine and cosine of the
100 m direction, and the 10 m speed. The UKV-CEDA model reads the 10 m speed, the sine and cosine of
the 10 m direction, and the 925 hPa speed, because the archive holds no 100 m wind. The wind
contrast therefore mixes product, height, and served lead, and the page claims no cause. Each solar
XGBoost model has 10 columns and differs only in the temperature: the solar geometry, the hour of
day, the day of year, the era, the temperature, and CAMS's global, beam, and diffuse irradiance.
Column subsampling is off, so a model with more columns gets no free advantage.

**The rows are decided by the target and by availability, never by a product's values.** An hour
is kept only if the target exists and every XGBoost model's input exists. The wind rows drop every
hour that holds an exactly zero half-hour, because two farms' feeds changed in 2026. The solar rows
drop the zero half-hours, one corrupt block of an unrelated product, the end of January 2026, and a
commissioning ramp. Of the solar rows, 398 are hours when the network operator had curtailed export;
the fit excludes them from training and still scores them. Wind power is the hour centred on the
label and solar power the hour ending at it. The study drops the 12 days from 2021-12-01 to
2021-12-12, when Open-Meteo's copy of ERA5 temperature disagrees with Copernicus by up to 2 degrees
Celsius, from every set.

**Five months are dropped from every set because UKV-CEDA's runs are missing or partial.** A month
that loses more than 25% of its hours to incomplete runs is dropped, and an incomplete run is never
replaced by an older run, because that would change the lead part-way through the record. The five
months are 2020-03 (33% of hours lost), 2022-09 (49%), 2022-10 (33%), 2022-12 (56%), and 2023-05
(26%). The decision to drop them, which keeps the planned limit, was taken by the maintainer's
delegate after the implementer had seen the set A tables in memory, and review judged the choice
neutral between the two products. Two more months straddle a change of UKV's physics and are dropped
too: 2019-12 and 2026-01.

**The folds are blocks of whole months inside each of three UKV eras.** The Met Office changed UKV's
physics on 2019-12-04 and 2026-01-21, so era 0 is September to November 2019 (3 scored months),
era 1 is 2020-01 to 2026-01 exclusive (67 months), and era 2 is 2026-02 onward (8 months). Each
era is cut into five folds of whole months, the era is a column, and the fold rotation covers every
calendar month with a training row. Whether a physics change happened that the repository does not
record (PS44) was not searched again: the repository's roadmap records that no date or content for
PS44 could be found.

**Each XGBoost model is fitted three times at each of two hyperparameter settings.** The primary
setting is `max_depth` 6, `learning_rate` 0.05, `subsample` 0.8, `min_child_weight` 20, `reg_lambda`
1, and 500 rounds. The second setting is `max_depth` 4, `learning_rate` 0.03, `subsample` 0.8,
`min_child_weight` 50, `reg_lambda` 5, and 1,200 rounds. Neither was tuned. Each fit uses seeds 0,
1, and 2, and every planned contrast is fitted at both settings. The error is the mean absolute
error as a percentage of the generator's capacity, each row divided by its own generator's capacity
before any mean. Each 95% interval resamples whole calendar months, paired across the two XGBoost
models, and one of the three fitting seeds, 2,000 times. The interval covers month-to-month weather
and the seed, and does not cover differences between generators or between places.

**Controls and checks.** The negative control shuffles a product's weather columns within a
generator, a year-month, and an hour of day (jointly for the four wind columns, and for temperature
only in the solar rows, which leaves CAMS's irradiance intact). Two shuffled XGBoost models carry no
weather information, so their difference should be about zero. The hour-ending pair builds the power
hour as the solar pages do, ending at the label. One ERA5 wind XGBoost model was refitted on the
CPU, to measure the difference between a GPU fit and a CPU fit.

**Every XGBoost model was fitted on one NVIDIA RTX A6000 GPU with XGBoost 3.4.1, apart from the CPU
refit.** The fits ran on 2026-10-06. GPU and CPU results are not bit-identical, so the page keeps
one device inside each planned contrast.

**The post hoc and exploratory rows are labelled, and the study does not correct for multiple
comparisons.** A row with no real effect behind it has a nominal 5% chance of reaching the 5% level,
the number of such rows is unknown, and the rows share their months, so spurious results cluster.
The page counts how many splits reach it in [Limitations](#limitations).

## Results

### The XGBoost models track measured power at every farm

**The out-of-fold predictions follow the measured output of every generator in all three weeks
chosen by rule.** The rule picks the week of highest mean output, the week of the largest spread of
output, and the week of lowest mean output, pooled over the generators. The wind weeks fall in
February 2020 (highest mean output and largest spread) and March 2022 (lowest mean output). The
solar weeks fall in May 2020 (highest), April 2025 (largest spread), and August 2021 (lowest).
Figures 2 and 3 show them, and gaps are hours dropped from the rows.

![Figure 2: Out-of-fold wind farm power as a share of capacity, in three weeks chosen by
rule](../assets/ukv_ceda_vs_era5/fig02_wind_weeks.svg)

![Figure 3: Out-of-fold solar farm power as a share of capacity, in three weeks chosen by
rule](../assets/ukv_ceda_vs_era5/fig03_solar_weeks.svg)

**Every XGBoost model's own error is reported, and the shuffled-weather models err about 12 points
more.** The wind XGBoost models given real weather have mean absolute errors of 7.174% (ERA5) and
7.299% (UKV-CEDA), against 19.589% and 19.637% when their weather columns are shuffled. The solar
models given real temperature err by 4.877% (ERA5) and 4.876% (UKV-CEDA), against 4.898% and
4.902% when the temperature is shuffled, so temperature alone adds little for solar power. Figure 4
shows the errors at the primary setting. The matched 10 m models err by 7.312% (ERA5) and 7.817%
(UKV-CEDA).

![Figure 4: An XGBoost model given weather that has been shuffled errs about 12 points more than one
given the real weather](../assets/ukv_ceda_vs_era5/fig04_absolute_errors.svg)

### ERA5 gave the lower wind-power error by 0.125 points

**P3, the planned wind contrast, is +0.125 points [+0.033, +0.216] at the primary setting and +0.125
points [+0.035, +0.217] at the second, so the XGBoost model given ERA5's wind has the lower
error.** Both intervals lie above zero, so the difference is statistically significant at the 5%
level, and both estimates lie below the 0.16-point margin, so the reading is "small, not clear".
The planned rule gives ERA5 for wind, and the page words the result as a small statistically
significant advantage for the XGBoost model given ERA5's wind, which cannot be attributed to the
product alone because the two XGBoost models differ in height and served lead as well. P3's early
window (2019-09 to 2020-12, 14 months) is +0.193 [+0.001, +0.430] at the primary setting and +0.230
[+0.043, +0.443] at the second, and its late window is +0.114 [+0.012, +0.210] and +0.109 [+0.010,
+0.205]. The keep-zero-hours replication, which refits both XGBoost models on rows that keep the
hours holding an exactly zero half-hour, gives 7.209% for ERA5 against 7.340% for UKV-CEDA.

**The advantage is concentrated in October to March and at one of the three farms (exploratory).**
From October to March the difference is +0.272 points [+0.167, +0.388], and from April to September
it is -0.015 points [-0.132, +0.113]. At W1 it is +0.213 [+0.128, +0.293], at W2 +0.077 [-0.046,
+0.202], and at W3 +0.081 [-0.063, +0.217]. By year it is +0.293 (2019, three months, no interval),
+0.176 (2020), +0.203 (2021), -0.004 (2022), +0.244 (2023), +0.150 (2024), +0.137 (2025), and -0.159
(2026), and only 2023 has an interval wholly above zero. A stable winter boundary layer, where
neither 10 m nor 925 hPa wind stands in for hub height, would fit the winter concentration. That is
a hypothesis, not a finding.

![Figure 8: ERA5's wind-power advantage is concentrated in October to March and at one of three
farms](../assets/ukv_ceda_vs_era5/fig08_power_splits.svg)

### The wind gap grows with the served lead

**The difference rises from -0.058 points at lead 0 to +0.323 points at lead 5, and it is post
hoc.** The six rows are -0.058 [-0.192, +0.070], +0.091 [-0.021, +0.202], +0.071 [-0.057, +0.185],
+0.173 [+0.049, +0.288], +0.149 [+0.030, +0.270], and +0.323 [+0.205, +0.437] for leads 0 to 5, at
the primary setting. A lead is the UTC hour modulo 6, so the lead split is confounded with the
diurnal cycle. The trend rises with two small dips, and the station temperature error of ERA5 stays
flat across the same hours (0.702 K to 0.718 K), so lead is the likelier reading, but the page
cannot separate lead from hour of day.

**From 2024-08-12, the first day of the past-wind page's window, the difference is +0.039 points
[-0.115, +0.186], and the past-wind page's -0.44 is not a difference in period.** The past-wind
page found Open-Meteo's UKV ahead of ERA5 by 0.44 points [0.24, 0.63] over the same days. The
two studies read different things. Open-Meteo serves UKV's analysis (lead 0) every hour with a
hub-height wind, whereas CEDA's archive is 6-hourly, with leads of 0 to 5 hours and only 10 m and
925 hPa wind. The CEDA archive also differs statistically from live UKV. The page cannot say how
much of the gap each difference explains.

![Figure 6: ERA5's wind-power advantage grows with UKV-CEDA's lead, and holds with 10 m wind
alone](../assets/ukv_ceda_vs_era5/fig06_wind_leads.svg)

### 10 m wind alone widens the gap

**With each product's 10 m wind alone, ERA5 is ahead by +0.505 points [+0.394, +0.620] at the
primary setting and +0.510 [+0.398, +0.624] at the second, so wind height is not the cause (post
hoc).** The two matched XGBoost models each have 6 columns: the 3 shared columns and the 10 m
speed, sine, and cosine. The pair is ahead in every lead (+0.292 at lead 0 to +0.714 at lead 5), in
the early window (+0.577) and the late window (+0.494), and from 2024-08-12 (+0.366 [+0.196,
+0.534]). The UKV-CEDA 10 m model errs by more (7.817%) than the as-available UKV-CEDA model that
also reads 925 hPa wind (7.299%), so the 925 hPa column helps UKV-CEDA, and the planned P3 is the
smaller of the two gaps. The matched pair is post hoc and does not change the planned decision.

### UKV-CEDA's temperature is closer at four stations, and less so at longer leads

**P2, the planned temperature contrast, is -0.124 K [-0.135, -0.113], so UKV-CEDA is closer than
ERA5 to the stations' readings.** The mean absolute error after removing bias is 0.711 K for ERA5
and 0.587 K for UKV-CEDA over 200,035 station-hours, and the margin is 0.036 K, so the reading is
"clear". The raw errors are 0.845 K and 0.626 K. P2's early window is -0.108 [-0.128, -0.088] and
its late window -0.128 [-0.141, -0.114], both planned. The pre-registered row P2-lead splits P2 by
lead: -0.187 K [-0.198, -0.176] at leads 0 to 2 and -0.061 K [-0.075, -0.047] at leads 3 to 5, and
the planned condition that P2 is clear at leads 3 to 5 is met. Set A's wind contrast P1 is -0.140
m/s [-0.156, -0.125] (0.925 m/s for UKV-CEDA against 1.066 for ERA5), and P1 never decides.

**The advantage shrinks steadily with lead, from -0.250 K at lead 0 to -0.035 K at lead 5 (post
hoc).** The six rows are -0.250 [-0.263, -0.237], -0.179 [-0.192, -0.167], -0.131 [-0.144, -0.118],
-0.097 [-0.111, -0.083], -0.050 [-0.065, -0.035], and -0.035 [-0.050, -0.021] K for leads 0 to 5,
while ERA5's error stays between 0.702 K and 0.718 K. At lead 5 the difference is inside the 0.036 K
margin. The wind advantage decays more slowly, from -0.213 to -0.106 m/s. The decay fits the four
stations' readings entering UKV's hourly data assimilation, with the analysis closest to them. The
page has not checked which stations UKV assimilates, so this is an explanation, not a finding. The
part of the advantage that a 2 km cell would keep at every lead is at most 0.035 K.

![Figure 5: UKV-CEDA's advantage at the four stations shrinks as the lead
grows](../assets/ukv_ceda_vs_era5/fig05_station_leads.svg)

**No single station decides the sign, but the stations differ a great deal.** UKV-CEDA's
temperature is closer at stations S1 (-0.227 K [-0.245, -0.211]), S3 (-0.194 K), and S2 (-0.077 K),
and level at S4 (+0.002 K [-0.012, +0.014]), so the planned contrast rests on three of four
stations. Leaving one station out moves P2 between -0.166 K and -0.090 K, and wind between -0.185
and -0.095 m/s. The interval on P2 (±0.011 K) covers month-to-month weather at four stations only.
By year P2 runs from -0.098 K (2020) to -0.158 K (2022), and from October to March and from April to
September it is -0.124 and -0.123 K. Figure 7 shows each split.

![Figure 7: UKV-CEDA is closer than ERA5 at three of four stations, and no one station decides the
sign](../assets/ukv_ceda_vs_era5/fig07_stations.svg)

### Solar power does not depend on the temperature product

**P4, the planned solar contrast, is -0.001 points [-0.009, +0.004] at the primary setting and
-0.002 points [-0.006, +0.002] at the second, which reads "no clear difference".** On these six
farms the choice of temperature product moves the error by no more than 0.009 points at either
setting. The planned veto of a UKV-CEDA temperature recommendation could never have fired: the
XGBoost model's whole gain from temperature is 0.021 points for ERA5's temperature (against its own
shuffled copy) and 0.026 points for UKV-CEDA's, and a clear ERA5 reading needed a difference above
0.06 points, 2.9 times ERA5's whole gain. P4's information is its bound, and the page does not count
it as evidence for the temperature recommendation. P4's early window is +0.001 [-0.015, +0.017] and
its late window -0.002 [-0.009, +0.004]. The solar XGBoost models were not given demand, so the
study cannot say how temperature helps a demand forecast.

### The controls show no bias, but too much noise to validate small differences

**The negative controls differ from zero by less than their own intervals, and the wind control's
interval is about as wide as the wind margin.** The shuffled UKV-CEDA model minus the shuffled ERA5
model is +0.049 points [-0.099, +0.190] for wind and +0.004 points [-0.005, +0.013] for solar.
Neither interval excludes zero, so the controls show no systematic bias between the two products'
pipelines. Two cautions follow. First, the two shuffled wind models err by about 19.6%, nearly three
times the real models' 7.2%, so their noise says little about a pipeline that runs at 7%. Second,
the wind control's interval half-width is about 0.14 points, close to the 0.16-point margin, so a
difference of P3's size (+0.125) would not stand out against the shuffled models' noise. P3 rests on
its own paired interval ([+0.033, +0.216], and [+0.035, +0.217] at the second setting) and on the
better noise floor, the refit of the same ERA5 wind XGBoost model on the CPU, which differs from the
GPU fit by -0.003 points [-0.014, +0.009]. Read P3 as a small difference that the study can resolve,
and not as one that the shuffled control validates.

**Nine of the 12 controls and replications are statistically significant, and that is expected.**
Each product's XGBoost model against its own shuffled copy is significant (wind -12.415 points for
ERA5 and -12.339 for UKV-CEDA, solar -0.021 and -0.026), because real weather carries information
that shuffled weather does not. The power-hour checks are also significant: ending the hour at the
label instead of centring it raises ERA5's wind error by +0.087 points [+0.064, +0.114] and
UKV-CEDA's by +0.231 [+0.203, +0.260], so the UKV-CEDA minus ERA5 gap is +0.269 [+0.185, +0.353]
under the hour-ending convention and +0.125 under the centred one. The centred hour is right for an
instantaneous wind value, and the convention moves the gap by about the size of P3. No control is
within 20% of its interval's width of zero, so no control needed the second-setting rerun.

![Figure 9: Shuffled UKV-CEDA and shuffled ERA5 differ by about zero, with a wind interval as wide as
the 0.16-point margin](../assets/ukv_ceda_vs_era5/fig09_controls.svg)

### In 2026 the wind difference points towards UKV-CEDA at one setting

**In era 2 (2026-02 onward, 8 months), the wind difference is -0.159 points [-0.340, +0.045] at the
primary setting and -0.192 points [-0.354, -0.009] at the second, so it favours UKV-CEDA and is
statistically significant at the second setting only (post hoc).** Era 2 is the closest era to
today's UKV, and the page names it as a lead for a follow-up. The era is exploratory, its folds
trained mostly on era 1, and set A holds no station data from it (the stations' record ends in
December 2025), so P1 and P2 say nothing about it. Era 1 gives +0.154 [+0.057, +0.247], and era 0,
which has 3 months, +0.293 with no interval.

### Monthly steps show no UKV-only change, and the two station series step together

**No step in UKV-CEDA minus ERA5 marks a change of UKV, but the two minus-station series step
together around August 2021 (post hoc).** `verify.md` lists the months that start the largest
differences between the mean of the following 6 months and the mean of the preceding 6, in units of
the series' month-to-month spread, for UKV minus ERA5, UKV minus the station, and ERA5 minus the
station, raw and with each calendar month's mean removed. No threshold was set before the series
was seen, so the table is exploratory. A step in a minus-station series with no matching step in
UKV minus ERA5 points at the station, as the 2021 steps do. Bias removal pools across years, so a
station-side step adds error to both products. Figure 10 draws the raw series and carries the
licence notice, because it plots UKV-derived monthly means.

![Figure 10: No step in UKV-CEDA minus ERA5 marks a change of UKV, but the station series step
together around August 2021](../assets/ukv_ceda_vs_era5/fig10_monthly_steps.svg)

### A UKV-CEDA history has practical costs

**A UKV-CEDA training history would need other weather for 357 of 10,338 runs, and the page's
XGBoost models were given the era.** Of the runs, 9,981 are complete, 15 are partial, 82 are
missing, and 260 (on 65 days) have no directory on CEDA. The re-check of those 65 days against
CEDA was not done. Five months lose more than 25% of their hours and are dropped, and two more
straddle a physics change, so the main work would fill them from another source, a mixed-source
history that this study did not test. The 6-hourly lead pattern means the error rises with lead and
drops at each run boundary, which a feature built over several hours (a lag or a rolling mean)
inherits. A history from 2019 spans three physics eras, with 3, 67, and 8 scored months, so the main
work would need an era column as the XGBoost models here had.

## Discussion: what to use

**For past wind, the planned rule gives ERA5, and the choice is between ERA5 and CEDA's 6-hourly
UKV store, not between ERA5 and UKV.** The XGBoost model given ERA5's wind has the lower error by
0.125 points, which is below the margin, so the gain from choosing ERA5 is small. The post hoc rows
say that most of the gap sits at longer leads, in winter, and at one farm. What would change the
recommendation: a UKV archive served at lead 0 with a hub-height wind (the past-wind page's
Open-Meteo archive), or evidence from 2026 that RAL3's UKV is ahead, which era 2 hints at but does
not establish.

**For past temperature, the planned rule gives UKV-CEDA, on the evidence of four stations that are
probably assimilated.** UKV-CEDA's 1.5 m temperature is 0.124 K closer to the stations' readings
than ERA5's 2 m temperature, and 0.035 K closer at lead 5. The page says nothing about temperature
away from those stations, nothing about how it helps a demand forecast, and nothing about the 2 km
cell against the 25 km cell, which favours a point observation by itself. For solar power the choice
makes no difference above 0.009 points. The recommendation carries the costs in [A UKV-CEDA history
has practical costs](#a-ukv-ceda-history-has-practical-costs) and the licence below.

**For training history from 2019, the early-years test passes for UKV-CEDA temperature.** The test
needs the early window's point estimate below zero and its upper bound below the margin: P2's
early window is -0.108 K [-0.128, -0.088], with margin 0.035 K. Era 0 holds only 3 months, so the
early window rests mostly on 2020.

**Whether the main work's use of UKV-CEDA as training history is non-commercial is the maintainer's
decision.** CEDA's catalogue record for the NWP-UKV data states the Creative Commons
Attribution-NonCommercial-ShareAlike 4.0 licence. This page publishes error scores and charts and
no UKV value, apart from Figure 10's monthly means, and cites the data as Met Office (2016):
NWP-UKV: Met Office UK Atmospheric High Resolution Model data. Centre for Environmental Data
Analysis, <https://catalogue.ceda.ac.uk/uuid/f47bc62786394626b665e23b658d385f>. Whether ShareAlike
reaches adapted material is not settled here.

## Limitations

- **Four stations are few.** The intervals cover month-to-month weather and the fitting seed, and do
  not cover differences between stations, between generators, or between places outside one box. The
  four stations and six solar farms share their weather.
- **Set A scores the product at a point that UKV may have assimilated.** The advantage falls with
  lead, and no result here says how large it is away from the stations.
- **A 2 km cell favours a point observation over a 25 km cell, by itself.** Bias removal removes
  offsets from exposure and orography and not that advantage.
- **Bias is removed in-sample and pooled across years and eras.** Both products' errors are debiased
  by their own means, taken over all years including the scored row. The in-sample optimism is about
  equal for the two products. Era 0's older physics, if anything, penalises UKV-CEDA in the early
  window.
- **The stations step together around August 2021.** The step adds error to both products, and
  slightly more in relative terms to UKV-CEDA, the product with the smaller error.
- **The wind contrast mixes product, height, and served lead, and a 2 km cell against a 25 km cell
  for the farms.** The page claims no cause. CEDA's archive also differs from live UKV.
- **ERA5 is read at the nearest 0.25 degree cell.** The temperature comes from Open-Meteo's copy of
  the Copernicus archive, which disagrees with Copernicus on 12 days that the study drops.
- **The significance statements cover month-to-month weather and the seed.** Of the 66 exploratory
  and post hoc splits of the UKV-CEDA against ERA5 contrasts at the primary setting, 28 are
  statistically significant at the 5% level; a split with no real effect has a nominal 5% chance of
  reaching it, and the page does not correct for that. Separately, 9 of the 12 controls and
  replications are significant, as the controls section explains.
- **The figures rest on the `effective_capacity` table at Delta version 1, read when the rows were
  built on 2026-10-06.** Each generator's capacity is its 99th percentile of output from that table,
  so a rebuilt table would move every figure.
- **Some choices followed a look at results.** The five dropped months were chosen after the
  implementer had seen the set A tables, and `verify.md` already showed UKV-CEDA ahead on raw error
  at lag 0 for wind and temperature. The margins, the rule, and P2-lead were fixed in the plan
  earlier, and nothing in the code was tuned to those looks.
- **Departures from the plan.** The 25% guard dropped five months; the re-check of the 65 unlisted
  days against CEDA was not done; a keep-zero-hours block covering all months was added; the wind
  row set is the intersection of the centred and hour-ending power targets; the search for PS44 was
  not repeated; and post hoc rows were added after the first results (each single lead and
  leave-one-station-out in set A, the lead, published-window, and era splits of P3, and the matched
  10 m pair).

## Scope

**The study covers wind speed and air temperature at four stations, and power at three wind farms
and six solar farms, in one box in Lincolnshire, from September 2019 to September 2026.** It does
not score dew point, sea-level pressure, surface pressure, 500 hPa height, precipitation type, or
windchill. It does not test live UKV, a UKV archive served at lead 0, UKV's 100 m or hub-height wind
(the archive holds none), a wider crop, a demand forecast, or a mixed-source history. It does not
reopen the choice of CAMS for irradiance.

## Data and code availability

**The inputs are public except the generators' metered output, their coordinates, the capacity
table, and the station-to-farm mapping, which is withheld because it would narrow where a metered
generator is.** The public inputs are the CEDA archive of UKV runs (licence above), ERA5 from the
Copernicus Climate Data Store and Open-Meteo's copy, CAMS irradiance, and the Met Office MIDAS Open
release `dataset-version-202607`. The generators appear only as W1 to W3 and A to F, and the
stations as S1 to S4 with no identifier or coordinate. The code that produced every figure is in
`studies/past_weather/` and `packages/studies/`, at the commit that merged this page. Every XGBoost
model used XGBoost 3.4.1, `tree_method` `hist`, the objective `reg:absoluteerror`, and four threads
per fit, and the hyperparameter settings are `studies.cross_validation.PRIMARY_HYPER_PARAMETERS` and
`SENSITIVITY_HYPER_PARAMETERS`. The plan for this study is in the repository's git history.

## Reproducing the figures

**Run the commands below in order, after the stores, ERA5, CAMS, and MIDAS downloads are on disk.**
`--check-only` runs the coverage check without reading a UKV value, and each script refuses to
overwrite an output, so a re-run first moves the earlier output to `superseded/`.

```bash
uv run python studies/past_weather/ukv_ceda_vs_era5_build.py --check-only
uv run python studies/past_weather/ukv_ceda_vs_era5_build.py
uv run python studies/past_weather/ukv_ceda_vs_era5_verify.py
uv run python studies/past_weather/ukv_ceda_station_scores.py
uv run python studies/past_weather/ukv_ceda_vs_era5_fit.py --verified
uv run python studies/past_weather/ukv_ceda_vs_era5_build.py --matched-10m
uv run python studies/past_weather/ukv_ceda_vs_era5_fit.py --fit-matched
uv run python studies/past_weather/ukv_ceda_vs_era5_fit.py --report-only
uv run python studies/past_weather/ukv_ceda_vs_era5_charts.py
npx svgo@4 --multipass --precision=1 --final-newline docs/studies/assets/ukv_ceda_vs_era5/*.svg
```
