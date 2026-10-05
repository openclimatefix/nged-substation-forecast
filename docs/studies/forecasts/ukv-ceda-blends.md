# Adding UKV to ECMWF's ensemble mean lowered the wind error at lead days 1 and 2 under every check, and the solar error by about 0.1 points

**At 3 wind farms in Lincolnshire, an XGBoost model (a gradient-boosted tree model) given the Met
Office's UKV weather forecast as well as the mean of the European Centre for Medium-Range Weather
Forecasts (ECMWF) ensemble forecast (ENS) had a lower power-forecast error than one given the ENS
mean alone at lead days 1 and 2, and the result survives every check we ran.** The gain is 0.24 to
0.27 points of capacity at day 1 and 0.19 to 0.21 points at day 2, across two hyperparameter
settings. At wind day 3 the gain rests heavily on February 2026. At 6 solar farms the gain is about
0.1 points of capacity at days 1 to 3, and the study cannot say that any one of those days differs
from the others. At day 4 the study is inconclusive for both technologies. The page cannot say how
much of any gain comes from UKV's weather and how much from UKV's run starting 3 hours after ENS's
run. The UKV data comes from the archive at the Centre for Environmental Data Analysis (CEDA), which
this page calls UKV-CEDA.

**The error is a mean absolute error in percentage points of the generator's capacity, and every
difference is the first model's error minus the second's.** A negative difference means the model
with UKV-CEDA forecasts better. Each bracketed pair is a 95% interval from resampling whole calendar
months and a fitting seed. Every number comes from the study's `report.md` or from its post hoc
`report_2.md`, and the two hyperparameter settings are the primary setting and a sensitivity setting
(the primary setting first, then the sensitivity setting in brackets or in a second row).

![Figure 1: For six solar farms, adding UKV-CEDA lowered the ENS mean's error by about 0.1 points of
capacity at lead days 1 to 3, and day 4 is inconclusive](../assets/ukv_ceda_blends_v2/solar_headline.svg)

![Figure 2: For three wind farms, adding UKV-CEDA's winds lowered the ENS mean's error at lead days 1
and 2 under every check, day 3 rests on February 2026, and day 4 is
inconclusive](../assets/ukv_ceda_blends_v2/wind_headline.svg)

> **How this page was made.** The research question came from a human. Everything else — the code
> behind every result, the analysis, the figures, and the text — was written by Claude, Anthropic's
> AI model (for this page, Claude Sonnet 5.5). Several independent Claude reviewers have checked the
> method, the evidence, and the prose adversarially.

## Key findings

- **At wind days 1 and 2, adding UKV-CEDA's winds lowers the error under every check we ran
  ([wind results](#wind-days-1-and-2-survive-every-check)).** The differences are -0.239 [-0.314,
  -0.170] (-0.269 at sensitivity) at day 1 and -0.210 [-0.313, -0.105] (-0.190) at day 2, from errors
  of 8.236% and 9.470% for the ENS mean alone. The planned rule is met, both results survive the
  Bonferroni correction at both settings, and both survive dropping any one calendar month.
- **At wind day 3, the gain rests on February 2026 ([wind day 3](#wind-day-3-rests-on-february-2026)).**
  The difference is -0.277 [-0.533, -0.081], and the planned rule is met. Without February 2026 and
  at the sensitivity setting, the interval reaches +0.022, and the correction fails at that setting
  even with every month.
- **At solar days 1 to 3, adding UKV-CEDA lowers the error by about 0.1 points, as one pattern
  ([solar results](#solar-about-01-points-at-days-1-to-3-as-one-pattern)).** The differences run from
  -0.082 to -0.163 across days and settings. The planned rule is met at day 3 only, and two
  shuffled controls that carry no information differ from each other by up to 0.079 points.
- **At day 4, the study is inconclusive ([day 4](#day-4-is-inconclusive)).** The intervals do not
  exclude a gain of 0.172 points for solar or 0.295 points for wind, as large as the gains at days 1
  to 3.
- **The study cannot separate UKV-CEDA's weather from its later run ([freshness](#the-freshness-test-cannot-say-how-much-of-the-gain-is-weather)).**
  Moving UKV-CEDA's run 21 hours older than ENS's removes most of the wind day-1 gain (the post hoc
  stale blend's gain is -0.072 [-0.167, +0.016]).

## Introduction

**The question is whether the Met Office's UKV, read at lead days 1 to 4, lowers the power-forecast
error of ECMWF's ensemble mean.** UKV is the Met Office's 2 km weather model for the United
Kingdom. The [blends page](blends-with-ens.md) tests UKV at day 1 only, from Open-Meteo's archive
of live UKV. CEDA's archive holds whole UKV runs out to 120 hours, so this page tests days 2 to 4
as well. The page covers solar and wind power separately, at the 6 solar farms and 3 wind farms
used by the [matched-lead page](matched-lead.md).

**The ENS mean and UKV-CEDA read different runs, and UKV-CEDA's run starts later.** For a valid hour
on day D at lead day N, ENS reads the 00 UTC run of day D−N, at a lead of 24N plus the hour of the
day. UKV-CEDA reads the 03 UTC run of the same day, which a service running at 09:00 UTC could read.
UKV-CEDA's lead is therefore 3 hours shorter at every hour.

| Product | Run read for lead day N | Lead of the hour of day h | Resolution in time |
|---|---|---|---|
| ECMWF ENS mean (51 members) | 00 UTC run of day D−N | 24N + h hours | 3-hourly, upsampled |
| UKV-CEDA | 03 UTC run of day D−N | 24N + h − 3 hours | hourly to lead 48 hours, then 3-hourly to 120 |

**Day 5 is not tested.** At day 5 the lead is 24 × 5 + h − 3 hours, which exceeds UKV-CEDA's
120-hour reach for every hour after 03:00 UTC.

## Data and methods

**A blend is an XGBoost model given ENS's mean weather and UKV-CEDA's weather, and the page compares
it with two other XGBoost models.** The padded ENS model is given ENS's mean weather and exact
copies of the same columns, so that it has the blend's column count (9 for solar, 11 for wind). The
shuffled-control model has the blend's columns, but its UKV-CEDA columns are shuffled among hours
that share a generator, a year-month, and an hour of day. The study fits two shuffled controls, with
shuffle seeds 0 and 1000. Equal column counts matter because an XGBoost model with more columns can
win without carrying more information. A check at wind day 1 and solar day 1 found the padded
ENS model's per-row losses identical to those of the unpadded ENS model.

**The solar models are given ENS's mean global irradiance and temperature, and UKV-CEDA's
global irradiance and temperature.** The wind models are given ENS's 100 m wind speed, the sine and
cosine of its 100 m direction, and its 10 m wind speed. UKV-CEDA's wind columns are its native 10 m
wind speed, the sine and cosine of its 10 m direction, and its 925 hPa wind speed. They are not 100
m winds. Every model also has the hour of day, the day of the year, and an era code, and the solar
models have the sun's elevation and azimuth. Each model is one XGBoost model per generator.

**The rows, folds, and intervals follow the matched-lead page.** The study scores 21 months of valid
hours, from December 2024 to September 2026, with January 2026 dropped because the Met Office
upgraded UKV on 21 January 2026. The eras are: era 0 before October 2025 (10 months), era 1 from
October to December 2025 (3 months, too few for an interval), and era 2 from February 2026 (8
months). Folds are blocks of whole months within each era. Each row is scored only if ENS and
UKV-CEDA both have values for it, so every model of a lead day is trained and scored on the same
rows: 38,253 to 38,287 solar rows and 41,526 to 41,541 wind rows per lead day. Each row's error is
divided by its own generator's capacity before any mean or difference. The intervals resample whole
calendar months and one of three fitting seeds, paired across models. They cover month-to-month
weather and fitting-seed variation, and not differences between generators.

**Two contrasts are planned.** A planned contrast was written into the study plan before any result
existed. P1 is the blend minus the padded ENS model. P2 is the blend minus each shuffled control,
with one contrast per shuffle seed. The reading rule is: the blend "lowers the error" at a
technology and lead day only if the upper 95% bound of P1 and of both P2 contrasts is below zero at
both settings. The study reports P1's Bonferroni-adjusted 99.375% interval, which corrects across
the 8 P1 intervals per setting, and it does not adjust P2. Where P1 is below zero at both settings
and a P2 bound is not, the reading is "unresolved: lower than padded ENS, control test not passed".
Every other number is exploratory, and the page labels the analyses made after the first
scientific-validity review as post hoc. Exploratory rows have a nominal 5% chance each of reaching
statistical significance at the 5% level with no real effect behind them. The page does not correct
them for multiple comparisons, the number of spurious rows is unknown, and the rows' shared months
make spurious results cluster.

**Every fit is on one graphics processing unit.** The page reports a CPU refit of one model as a
noise floor ([Limitations](#limitations)).

## Results

### The XGBoost models make sane forecasts

**Out of fold, the wind and solar models follow the measured output across the weeks chosen by
rule.** Figures 7 and 8 plot the day-1 blend's out-of-fold forecast against measured output, with
each generator's output as a percentage of its own capacity, over one week per era chosen from
measured output alone. A week is left out for an era in which no week has every generator covered
on all seven days, which is why solar has Figures 7a and 7c and no 7b.

![Figure 7a: Measured output of the six solar farms and the XGBoost model's out-of-fold forecast over
one week](../assets/ukv_ceda_blends_v2/solar_week1.svg)

![Figure 7c: Measured output of the six solar farms and the XGBoost model's out-of-fold forecast over
one week](../assets/ukv_ceda_blends_v2/solar_week3.svg)

![Figure 8a: Measured output of the three wind farms and the XGBoost model's out-of-fold forecast over
one week](../assets/ukv_ceda_blends_v2/wind_week1.svg)

![Figure 8b: Measured output of the three wind farms and the XGBoost model's out-of-fold forecast over
one week](../assets/ukv_ceda_blends_v2/wind_week2.svg)

![Figure 8c: Measured output of the three wind farms and the XGBoost model's out-of-fold forecast over
one week](../assets/ukv_ceda_blends_v2/wind_week3.svg)

**The mean absolute error of every model rises with the lead day, and the padded ENS model's error
is the reference level.** For solar, the padded ENS model's error is 8.820% of capacity at day 1 and
11.692% at day 4. For wind it is 8.236% at day 1 and 12.328% at day 4. The tables give every model's
own error, at both settings, from `report_2.md`. The intervals on these levels are wide, because
every model's error rises and falls together from month to month, and pairing cancels that shared
swing. Figures 5 and 6 draw the levels with their intervals.

| Solar lead day | Setting | Padded ENS | ENS + UKV-CEDA | ENS + shuffled, seed 0 | ENS + shuffled, seed 1000 |
|---|---|---|---|---|---|
| 1 | primary | 8.820 | 8.715 | 8.858 | 8.779 |
| 1 | sensitivity | 8.774 | 8.684 | 8.827 | 8.765 |
| 2 | primary | 9.849 | 9.762 | 9.896 | 9.855 |
| 2 | sensitivity | 9.782 | 9.700 | 9.823 | 9.815 |
| 3 | primary | 10.769 | 10.606 | 10.744 | 10.798 |
| 3 | sensitivity | 10.703 | 10.587 | 10.697 | 10.722 |
| 4 | primary | 11.692 | 11.638 | 11.621 | 11.699 |
| 4 | sensitivity | 11.526 | 11.490 | 11.503 | 11.570 |

| Wind lead day | Setting | Padded ENS | ENS + UKV-CEDA | ENS + shuffled, seed 0 | ENS + shuffled, seed 1000 |
|---|---|---|---|---|---|
| 1 | primary | 8.236 | 7.996 | 8.222 | 8.239 |
| 1 | sensitivity | 8.169 | 7.900 | 8.170 | 8.186 |
| 2 | primary | 9.470 | 9.260 | 9.441 | 9.450 |
| 2 | sensitivity | 9.344 | 9.154 | 9.389 | 9.372 |
| 3 | primary | 11.020 | 10.743 | 10.996 | 11.009 |
| 3 | sensitivity | 10.846 | 10.666 | 10.866 | 10.851 |
| 4 | primary | 12.328 | 12.222 | 12.345 | 12.374 |
| 4 | sensitivity | 12.168 | 12.034 | 12.194 | 12.252 |

![Figure 5: For the six solar farms, an XGBoost model given ENS's mean alone has a mean absolute error
of 8.8% of capacity at lead day 1 and 11.7% at lead day 4](../assets/ukv_ceda_blends_v2/solar_errors.svg)

![Figure 6: For the three wind farms, an XGBoost model given ENS's mean alone has a mean absolute error
of 8.2% of capacity at lead day 1 and 12.3% at lead day 4](../assets/ukv_ceda_blends_v2/wind_errors.svg)

### Wind days 1 and 2 survive every check

**At wind days 1 and 2, the planned rule is met, and the gain survives the Bonferroni correction
and the loss of any one month.** At day 1, P1 is -0.239 [-0.314, -0.170] at the primary setting and
-0.269 [-0.357, -0.192] at the sensitivity setting. P2 against the seed-0 control is -0.226 [-0.307,
-0.145] (-0.270) and against the seed-1000 control -0.243 [-0.321, -0.162] (-0.286). At day 2, P1 is
-0.210 [-0.313, -0.105] (-0.190 [-0.272, -0.103]), P2 is -0.181 [-0.296, -0.069] (-0.236) against
seed 0 and -0.190 [-0.285, -0.097] (-0.218) against seed 1000.

**The Bonferroni intervals stay below zero at both settings.** At day 1 they are [-0.342, -0.144]
and [-0.379, -0.173], and at day 2 [-0.338, -0.071] and [-0.298, -0.061]. In the exploratory,
post hoc leave-one-month-out check (nothing refitted), the highest upper bound over the drops is
-0.151 at day 1 (dropping January 2025) and -0.083 at day 2. The sensitivity setting gives -0.178 and
-0.088.

**The gain is similar in era 0 and era 2, and clear at two of the three generators.** In the exploratory era
rows, day 1 gives -0.260 [-0.367, -0.131] in era 0 and -0.228 [-0.348, -0.149] in era 2, and day 2
gives -0.215 [-0.386, -0.025] and -0.185 [-0.293, -0.057]. Era 1 has 3 months and no interval, with
-0.198 at day 1 and -0.249 at day 2. By generator, the day-1 differences are -0.273 [-0.394,
-0.178] at W1, -0.294 [-0.408, -0.191] at W2, and -0.146 [-0.329, +0.035] at W3. The day-2
differences are -0.147 [-0.257, -0.037], -0.231 [-0.361, -0.105], and -0.251 [-0.488, -0.049].
Figure 4 draws the generators.

**The shuffled controls differ from each other by no more than 0.058 points at any wind lead day.**
The largest gap is at day 4 and the sensitivity setting (-0.058 [-0.201, +0.056]), and no wind gap
is statistically significant at the 5% level. At wind days 1 and 2 the gap between the controls is
therefore far smaller than P1.

![Figure 4: At each of the three wind farms, adding UKV-CEDA to the ENS mean changes the error by a
different amount](../assets/ukv_ceda_blends_v2/wind_generators.svg)

### Wind day 3 rests on February 2026

**At wind day 3 the planned rule is met, and the gain depends on one month.** P1 is -0.277 [-0.533,
-0.081] at the primary setting and -0.180 [-0.384, -0.010] at the sensitivity setting, and both P2
contrasts are below zero at both settings. The Bonferroni interval at the primary setting is
[-0.625, -0.025], but at the sensitivity setting it is [-0.463, +0.035], so the gain does not
survive the correction at both settings. Dropping February 2026 gives -0.186 [-0.348, -0.046] at the
primary setting and -0.111 [-0.242, +0.022] at the sensitivity setting, so the sensitivity interval
reaches zero. February 2026 is the first month of era 2. In the exploratory era rows, era 2 gives
-0.437 [-0.989, -0.085] and era 0 gives -0.227 [-0.518, -0.002].

**By generator, wind day 3 is clear at W2 only.** The differences are -0.101 [-0.315, +0.076] at W1,
-0.452 [-0.765, -0.213] at W2, and -0.271 [-0.613, +0.052] at W3. The page therefore reports wind
day 3 as met by the planned rule and as dependent on one month, and does not place it beside days 1
and 2.

### Solar: about 0.1 points at days 1 to 3, as one pattern

**For solar, adding UKV-CEDA lowers the error by about 0.1 points of capacity at days 1 to 3, and
the study cannot say that day 3 differs from days 1 and 2.** P1 is -0.105 [-0.177, -0.017] (-0.091
[-0.155, -0.017]) at day 1, -0.086 [-0.155, -0.018] (-0.082 [-0.158, -0.006]) at day 2, and -0.163
[-0.250, -0.079] (-0.116 [-0.188, -0.044]) at day 3, so P1 is statistically significant at the 5%
level at both settings on each day. The planned rule is met at day 3 only. Two of day 3's P2 upper
bounds, -0.006 and -0.001, lie within 0.01 points of zero.

**At solar days 1 and 2 the reading is "unresolved: lower than padded ENS, control test not
passed".** The blend's error is lower than the padded ENS model's by 0.08 to 0.11 points at both
settings. The rule fails because one shuffled control's interval reaches zero: at day 1 the
seed-1000 control at the primary setting gives P2 of -0.064 [-0.142, +0.029], and at day 2 the
seed-0 control gives -0.134 [-0.269, +0.002]. After the Bonferroni correction the interval reaches
zero at the primary setting at both days ([-0.200, +0.020] at day 1 and [-0.187, +0.001] at day 2),
and at the sensitivity setting too ([-0.175, +0.006] and [-0.187, +0.020]). Solar day 3 survives
the correction at both settings ([-0.284, -0.052] and [-0.215, -0.016]).

**Two shuffled controls that carry no information differ from each other by about as much as the
solar gain.** At day 1 the seed-0 control minus the seed-1000 control is +0.079 [+0.025, +0.133] at
the primary setting and +0.062 [+0.018, +0.103] at the sensitivity setting, and at day 4 the
sensitivity gap is -0.067 [-0.131, -0.006]. The intervals resample months and fitting seeds, not
shuffle seeds, so they understate how far a solar model's error moves for reasons unrelated to its
inputs. The wind gaps are at most 0.058 points. The page therefore describes solar as one pattern,
not as four separate verdicts.

**Solar is also fragile to the loss of one month.** In the post hoc leave-one-month-out check, the
highest upper bound over the drops is -0.001 at day 1 (dropping May 2026, primary setting), -0.004
at day 2 (June 2026), and +0.008 at day 2 at the sensitivity setting. At day 3 it is -0.063 at the
primary setting (May 2026) and -0.033 at the sensitivity setting.

**By generator, every solar generator's point estimate is below zero at days 1 to 3.** At day 1 the
differences run from -0.045 [-0.197, +0.084] at A to -0.196 [-0.356, -0.030] at E. Figure 3 draws
them. By era, solar day 3 gives -0.154 [-0.254, -0.059] in era 0 and -0.159 [-0.312, +0.009] in era
2, and days 1 and 2 give similar point estimates with intervals that include zero.

![Figure 3: At each of the six solar farms, adding UKV-CEDA to the ENS mean changes the error by a
different amount](../assets/ukv_ceda_blends_v2/solar_generators.svg)

### Day 4 is inconclusive

**At day 4 the intervals do not exclude gains as large as those at days 1 to 3, so the study cannot
say whether the gain fades.** For solar, P1 is -0.054 [-0.172, +0.053] (-0.036 [-0.108, +0.034]),
and a gain of 0.172 points is not excluded at the primary setting and 0.108 at the sensitivity
setting. For wind, P1 is -0.105 [-0.268, +0.058] (-0.133 [-0.295, +0.031]), and a gain of 0.268
points is not excluded at the primary setting and 0.295 at the sensitivity setting. A gain larger
than those bounds is excluded. The wind day-4 P2 contrasts are below zero at the sensitivity setting
for both seeds (-0.160 [-0.315, -0.025] and -0.218 [-0.396, -0.056]), so wind day 4 is near the
5% line.

### The freshness test cannot say how much of the gain is weather

**UKV-CEDA's run starts 3 hours after ENS's, and the planned contrasts cannot separate UKV-CEDA's
weather from that later run.** The stale blend is a post hoc, exploratory analysis, run after the
first scientific-validity review had seen the planned results. It gives ENS day N plus UKV-CEDA day
N + 1, for N of 1 to 3, which is the 03 UTC run one day earlier. That run is 21 hours older than
ENS's run, where the planned blend's run is 3 hours newer, so the change is 24 hours. The stale blend
is compared with a padded ENS model trained on the same rows, because it loses 0.3% to 0.4% of rows
that have no UKV-CEDA day N + 1.

**No stale blend meets the reading rule, and the test can bound the effect of the 3 hours only from
above.** The 24-hour change is larger than the 3-hour advantage, so the change in error
bounds the share due to the 3 hours from above, and does not measure that share. The table gives
the stale blend minus its padded ENS model (P1, stale) and the stale blend minus the planned blend,
at the primary setting and the sensitivity setting.

| Technology and lead day | P1, stale (primary) | P1, stale (sensitivity) | Stale minus planned blend (primary) | Stale minus planned blend (sensitivity) |
|---|---|---|---|---|
| solar 1 | -0.018 [-0.071, +0.036] | -0.013 [-0.060, +0.026] | +0.085 [-0.023, +0.184] | +0.077 [-0.010, +0.157] |
| solar 2 | -0.035 [-0.135, +0.070] | -0.027 [-0.098, +0.052] | +0.038 [-0.064, +0.135] | +0.047 [-0.036, +0.130] |
| solar 3 | +0.042 [-0.057, +0.135] | +0.030 [-0.035, +0.092] | +0.215 [+0.095, +0.329] | +0.157 [+0.064, +0.252] |
| wind 1 | -0.072 [-0.167, +0.016] | -0.065 [-0.142, +0.009] | +0.188 [+0.120, +0.249] | +0.224 [+0.166, +0.275] |
| wind 2 | -0.137 [-0.294, -0.015] | -0.073 [-0.168, +0.010] | +0.064 [-0.054, +0.175] | +0.114 [+0.012, +0.208] |
| wind 3 | -0.135 [-0.267, -0.022] | -0.096 [-0.241, +0.012] | +0.137 [-0.015, +0.315] | +0.085 [-0.033, +0.215] |

**At wind day 1 and solar day 3 the stale blend is clearly worse than the planned blend, and the
other rows are not separable.** At wind day 1 the stale blend loses 0.188 [0.120, 0.249] of the
planned blend's 0.239 points, so most of that gain depends on UKV-CEDA's run being newer. The stale
blend's gain at wind day 1 is not statistically significant, and an effect of 0.167 points at the
primary setting is not excluded. At wind days 2 and 3 the stale blend's primary-setting gain is
below zero (-0.137 and -0.135), the sensitivity-setting gain is not, and the post hoc reading
stays "no detectable difference". For solar, the effect of 0.071, 0.135, and 0.057 points at days
1, 2, and 3 is not excluded at the primary setting. The stale blend is also lower than its shuffled
controls at wind day 1 at both settings and both seeds, so a stale UKV-CEDA still carries some
information that the shuffled copy lacks. The padded ENS model trained on all rows differs from the
one trained on the stale rows by at most 0.022 points (wind day 1, primary setting), so the
difference in training rows does not drive the contrasts.

### Why the gain might come from something other than UKV-CEDA's weather

**Five explanations other than UKV's skill remain, and the study does not test any of them.**

- **Time resolution at day 1.** UKV-CEDA is hourly to lead 48 hours, while ENS is 3-hourly and
  upsampled, so some of the day-1 gain may come from the finer time steps. At days 3 and 4 both
  products are 3-hourly, so this cannot explain the day-3 gains.
- **Wind heights.** The wind blend adds UKV-CEDA's 10 m and 925 hPa winds to ENS's 100 m and 10 m
  winds. Part of the gain may come from a second vertical level, which carries information about
  how wind changes with height, rather than from UKV's skill. The page's finding is about an XGBoost
  model given UKV-CEDA's 10 m and 925 hPa winds.
- **Spatial support.** The study reads UKV-CEDA at each generator's nearest 2 km cell, and ENS as
  the mean over the hexagon that contains the generator. Part of the gain may come from how well a
  nearest cell represents a site. A hexagon-mean read of UKV-CEDA was not tried.
- **Delivery time.** The 09:00 UTC availability of the 03 UTC run is an assumption that was not
  measured for CEDA.
- **A bias in ECMWF's solar forecast in February 2025.** A check run during review, and not part of
  the committed report, found that every ECMWF product over-forecast sunshine against CAMS (a
  satellite-based radiation service) in February 2025, and that UKV-CEDA did not. A second weather
  model can help in such a month, so part of the solar gain may come from that regime.

## Discussion: what to use

**For wind at days 1 and 2, UKV-CEDA's winds are worth adding to the ENS mean in an XGBoost model
trained on UKV-CEDA, subject to the limitations below.** The result survives every check, but the
freshness test shows that most of the day-1 gain may depend on UKV-CEDA's run being newer than ENS's,
so a service that reads an older UKV run should expect less. What would change this: a blend that
reads a hexagon mean of UKV-CEDA, the same blend fitted on live UKV, and more months of data.

**For solar and for wind at day 3, the page does not recommend adding UKV-CEDA.** The solar gain of
about 0.1 points is the size of the noise floor that two uninformative controls show, and the wind
day-3 gain rests on one month. At day 4 the study is inconclusive, and the page does not say that
UKV-CEDA does not help there.

## Limitations

**The results rest on one region, 21 months, and an archive that differs from the live service.**
The 6 solar farms and 3 wind farms are in Lincolnshire, the sample is weather episodes and not
generator-hours, and the intervals resample months. Every figure uses the effective-capacity table
built for the matched-lead study.

- **UKV-CEDA differs statistically from the live UKV.** An XGBoost model trained on UKV-CEDA must
  not be run on live UKV, and the page cannot say how a blend with live UKV would do.
- **The upgrade era is short.** The Met Office upgraded UKV on 21 January 2026. Era 1 has 3 months
  and no interval of its own, and the study drops January 2026.
- **The fitting noise floor is small.** A CPU refit of the day-1 blend at wind generator W1 differs
  from the GPU fit by 0.7513 points of capacity per row on average (7.8589 at most), and the two
  fits' mean errors are 6.6993 (CPU) and 6.6893 (GPU). Two GPU fitting seeds of the same blend
  differ by 0.7013 points per row on average (7.6923 at most), and the three seeds' mean errors span
  0.0140 points. The CPU-GPU difference is therefore no larger than the difference between seeds.
- **The scored months are shared across generators.** Neighbouring generators share their weather,
  so the effective sample is the 21 months and the intervals do not cover differences between
  generators.
- **The radiation timestamp handicaps the blend.** UKV-CEDA's hourly radiation is a mean of two
  snapshots and is centred slightly later than the power hour.
- **The licence of the UKV-CEDA data is not settled on this page.** The fetch script records the
  licence as Creative Commons Attribution-NonCommercial-ShareAlike 4.0 (non-commercial use only).
  The [weather products survey](../../background/weather-products-survey.md) lists it only as the "Met
  Office licence via CEDA", and the CEDA catalogue record does not state the terms. The page can say
  that the study is research use and cannot say that production use is permitted.

## Scope

**The study does not cover day 5, a 15 UTC UKV run, live UKV, a hexagon-mean read of UKV-CEDA,
probabilistic scores, other regions, or any product other than UKV-CEDA.** It tests the blend of
the ENS mean and UKV-CEDA only, and not UKV-CEDA alone.

## Data and code availability

**The inputs and fitted losses are in the private data store, and every output carries only the
anonymised `site` label.** The generators are labelled A to F for solar and W1 to W3 for wind. The
code is `studies/ukv_ceda_blends/`, with `fit_aifs.py` and `nwp_forecast_comparison.py` from
`studies/nwp_forecast_comparison/` and `packages/studies/`. UKV-CEDA is the Met Office's UKV from the
CEDA archive. XGBoost fitted every model on one RTX A6000 GPU.

## Reproducing the figures

Run each step only after the step before it exits 0.

```bash
D=data/studies/ukv_ceda_blends
uv run python studies/ukv_ceda_blends/check_arm_columns_unchanged.py
uv run python studies/ukv_ceda_blends/build_ukv_ceda_inputs.py
uv run python studies/ukv_ceda_blends/verify_ukv_ceda_inputs.py
uv run python studies/ukv_ceda_blends/fit_ukv_ceda_blends.py --check
uv run python studies/ukv_ceda_blends/fit_ukv_ceda_blends.py --verified
uv run python studies/ukv_ceda_blends/fit_ukv_ceda_blends.py --post-hoc-stale --report-name report_2
uv run python studies/ukv_ceda_blends/ukv_ceda_blends_charts.py \
  --figures-dir docs/studies/assets/ukv_ceda_blends_v2 \
  --intervals-name report_2_intervals.parquet
```

The numbers are in `$D/report.md` and `$D/report_2.md`.
