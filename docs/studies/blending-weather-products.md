# Does blending weather products beat the best single weather product?

**At six solar farms and three wind farms in Lincolnshire, an XGBoost model given several weather
products at once beats an XGBoost model given the best single product, even when the single product
is given its neighbouring hours as extra columns.** For solar, an XGBoost model given all six
products has a mean absolute error of 4.74% of capacity, against 4.87% for an XGBoost model given
CAMS, the satellite retrieval, with its neighbouring hours and its beam split: 0.13 points better
[0.10, 0.17]. For wind, an XGBoost model given all five products has an error of 5.76%, against
6.24% for an XGBoost model given UKV's wind with its neighbouring hours: 0.48 points better [0.40,
0.56]. A blend a live forecasting service could read beats the best single product as well: UKV with
ICON-EU, by 0.26 points [0.21, 0.31] for wind.

**The gain comes from the other products' hour-by-hour weather, not from giving the XGBoost model
more columns.** A control that keeps the extra columns but replaces their weather with other days'
values from the same month does no better than the single product. The evidence is six metered solar
farms from December 2022 to September 2026, 79,384 generator-hours, and three metered wind farms
from August 2024 to September 2026, 50,734 generator-hours.

![Figure 1: At these nine farms, an XGBoost model given several weather products beats an XGBoost model
given the best single product, enriched](assets/blend_headline.svg)

> **How this page was made.** The research question came from a human. Everything else — the code
> behind every result, the analysis, the figures and the text — was written by Claude, Anthropic's
> AI model (for this page, Claude Opus 5.5). Several independent Claude reviewers have checked the
> method, the evidence and the prose adversarially.

## Key findings

- **[The XGBoost models behind every comparison track the measured output at every
  generator.](#the-xgboost-models-track-the-measured-output)**
- **[For solar, an XGBoost model given all six products beats an XGBoost model given CAMS alone by
  0.13 points, and an XGBoost model given the four weather models beats an XGBoost model given
  ICON-D2 alone by 0.34 points.](#solar-a-blend-beats-cams-with-its-extra-columns)**
- **[For wind, an XGBoost model given all five products beats an XGBoost model given UKV alone by
  0.48 points, and an XGBoost model given UKV and ICON-EU beats UKV alone by 0.26
  points.](#wind-a-blend-beats-ukv-with-its-extra-columns)**
- **[The gain comes from the other products' weather: a control given the same columns without their
  weather does no better than the best single
  product.](#the-gain-comes-from-the-other-products-weather)**
- **[Every named blend beats its best single product at every generator and in every
  season.](#the-gain-holds-at-every-generator-and-in-every-season)**
- **[The wind gain is spread across every level of output; the report's "5% of hours carry 99% of
  the gain" reflects hours that improve and hours that worsen netting
  out.](#the-wind-gain-is-spread-across-every-level-of-output)**
- **[An XGBoost model given every product's columns beats a linear stack of single-product
  predictions, and averaging with CAMS makes the solar error far
  worse.](#an-xgboost-blend-beats-a-linear-stack-and-simple-averages)**
- **[The comparison detects a synthetic gain of 0.13 to 0.14 points, the size of the solar
  gain.](#the-comparison-detects-a-gain-of-this-size)**

## Introduction

**Two earlier pages ranked weather products one at a time, and neither measured whether an XGBoost
model given several products at once does better.** [Which weather product best describes past
sunshine?](weather-products-for-past-solar.md) found that CAMS, the Copernicus Atmosphere Monitoring
Service's satellite retrieval, describes past sunshine far better than any weather model tested.
[Which weather product best describes past wind?](weather-products-for-past-wind.md) found that
ICON-D2 and UKV describe past hub-height wind best of the five products tested. Both pages scored
each product through an XGBoost model fitted per generator to predict hourly output from that
product alone.

**Several parts of this project read past weather, and each could read a blend instead of one
product.** The solar page names these consumers of past weather, which are parts of the project, not
electricity customers. Training history is the years of past weather that pre-training a forecasting
model needs. Historical features give the live forecasting model the weather of hours already past.
Capacity estimation infers a generator's size from how its output tracks the weather. Disaggregation
separates hidden generation from demand at a substation. This page asks whether an XGBoost model
given several products beats an XGBoost model given the best single product, and says which blend,
if any, each consumer should read.

**The products are those of the two earlier pages.** The solar products are CAMS; ERA5, the
reanalysis of the European Centre for Medium-Range Weather Forecasts (ECMWF); the Met Office's UK
variable-resolution model (UKV); and three models of the German weather service (DWD): ICON-D2,
which covers Germany and its neighbours, ICON-EU, and ICON global. The four products other than CAMS
and ERA5 are weather models, run every few hours. The wind products are the same, less CAMS, which
publishes no wind. ICON-D2 does not cover South West England or South Wales. Its western edge runs
from about 2°W on the south coast to about 2.5°W in the Midlands.

**Each blend reads one set of products, named for the consumer that set could serve:**

| Set | Solar products | Wind products | Why this set |
|---|---|---|---|
| All products | all six | all five | the most any blend can read |
| CAMS and ICON-D2 | CAMS, ICON-D2 | – | history where ICON-D2 covers |
| CAMS and ICON-EU | CAMS, ICON-EU | – | history anywhere in Great Britain |
| CAMS and ERA5 | CAMS, ERA5 | – | history back to 2004 |
| ICON-D2 and UKV | – | ICON-D2, UKV | the two leaders on the wind page |
| UKV and ICON-EU | UKV, ICON-EU | UKV, ICON-EU | a live service anywhere in Great Britain |
| Four weather models | UKV, ICON-D2, ICON-EU, ICON global | the same | a live service where ICON-D2 covers |

## Data and methods

**Every comparison is scored on the two earlier studies' own hours, folds, and fitting seeds.** The
solar rows are 79,384 generator-hours at six solar farms, labelled A to F, from 1 December 2022 to
10 September 2026. The wind rows are 50,734 generator-hours at three wind farms, labelled W1 to W3,
from 12 August 2024 to 10 September 2026. Each hour is kept only if every product covers it. This
study refits every single-product XGBoost model the two earlier pages published: 22 solar and 16
wind combinations of weather columns and hyperparameter setting. Every refit reproduces the
published per-hour errors bit for bit, which shows that the hours and the folds are unchanged.

**Each XGBoost model is fitted per generator and given the same calendar columns as on the earlier
pages, plus the weather columns below.** The calendar columns are the time of day, the season, and
which side of the Met Office's upgrade of UKV on 21 January 2026 the hour falls on. The solar
XGBoost models are also given the sun's position and ERA5's air temperature. Each comparison is
between two such XGBoost models, differing only in the weather columns:

- **Plain single product:** the columns the earlier pages gave one product. Solar: the product's
  global horizontal irradiance. Wind: the product's hub-height speed, that height's direction as a
  sine and a cosine, and its 10 m speed.
- **Enriched single product:** the plain columns plus the extras the solar page found improve one
  product on its own. Solar: the irradiance in the hour before and the hour after; for CAMS, its
  beam and diffuse irradiance as well; for UKV, the hour rebuilt as the mean of the snapshots at its
  start and end. Wind: the hub-height speed 1 and 2 hours either side, and the 10 m speed 1 hour
  either side.
- **Blend:** every product in a set, each with its plain columns or each with its enriched columns.

**The best single product of a set is the single-product XGBoost model with the lowest error on the
set's products.** The candidates are every single-product XGBoost model the earlier pages published
and every enriched one. For every set containing CAMS the best single is CAMS enriched, with an
error of 4.87%. For the four weather models it is ICON-D2 enriched, at 7.41%. For every wind set it
is UKV enriched, at 6.24%, ahead of ICON-D2 enriched at 6.28%. The extra columns matter on their
own: enriching UKV lowers its wind error by 0.59 points, and enriching CAMS lowers its solar error
by 0.19 points.

**Each set is blended four ways:**

- **XGBoost given every product's columns**, the blend this page's headline figures describe.
- **XGBoost given the products' mean**: the mean of the products' values, as the same number of
  columns as one product has. The wind mean mixes ICON's 80 m speed with the 100 m speed of ERA5 and
  UKV, which a per-generator XGBoost model can absorb.
- **A linear stack**: a weighted mean of the single-product XGBoost models' predictions, with
  weights that are not negative and sum to 1. The weights are fitted per generator, per fitting
  seed, and per fold, on the other folds only.
- **The equal-weight mean** of the single-product XGBoost models' predictions.

**Each XGBoost blend has a control given the same columns without their weather.** The control keeps
the best single product's real columns. Every other product's columns are shuffled among the hours
sharing a generator, a calendar month, and an hour of the day. The shuffle keeps each month's mean
at each hour of the day, and removes the hour-to-hour weather. The blend minus its control measures
what the other products' weather adds. The control minus the best single product measures what the
extra columns add alone, with no weather in them.

**A synthetic product shows that the comparison can detect a gain of the size found.** The synthetic
product is each hour's measured output as a fraction of capacity, plus random noise drawn once with
a fixed seed. The noise has a standard deviation of 0.30 of capacity for solar and 0.45 for wind.
The synthetic product carries part of the answer by construction, and exists only to test the
comparison. The best single product of the all-products set is given the synthetic product as one
more column, and a control is given the same column shuffled as above.

**Every error is a mean absolute error as a percentage of each generator's own capacity.** Capacity
here is each generator's 99th percentile of metered output, not its registered or export capacity. A
difference between two errors is in percentage points of capacity, written "points". Each XGBoost
model is trained on some blocks of whole months and scored on the others; each held-out block is
called a fold, and there are five. Each XGBoost model is fitted with three seeds.

**A bracketed pair after a figure is its 95% interval, from resampling whole calendar months and one
of the three fitting seeds.** A difference whose interval excludes zero is statistically significant
at the 5% level. That test covers month-to-month variation in the weather and the choice of fitting
seed. It does not cover differences between generators, or between the XGBoost models trained on
different folds. Each headline figure therefore also carries a t-interval across the five folds'
differences and the range of the difference across the generators.

**A planned comparison is one written into the study plan before the run that measured it; every
other figure is exploratory.** The headline comparisons, of each enriched blend against the enriched
best single product, were written into the plan after the first science review and before the re-run
that measured them. The review found that much of the first run's gain was available from one
product given its extra columns. The plain comparisons were written into the plan before any result
existed, and are reported as secondary. With many comparisons, about 1 in 20 exploratory rows is
statistically significant at the 5% level by chance, so an exploratory result is a lead rather than
a finding. Every planned comparison is also rerun with a second hyperparameter setting for XGBoost.

## Results

### The XGBoost models track the measured output

**The XGBoost models given the best single product, and those given every product, follow the
measured output hour by hour at every generator, in weeks chosen by rule.** Figures 2 and 3 show
three weeks at each generator: the week with the highest mean output, the most variable week, and
the week with the lowest mean output. Each XGBoost model's prediction is out of fold, so the hours
it predicts were held out of its training. The prediction shown is the mean over the three fitting
seeds.

![Figure 2: XGBoost models given CAMS, or all six products, track measured solar
output](assets/blend_solar_weeks.svg)

![Figure 3: XGBoost models given UKV, or all five products, track measured wind
output](assets/blend_wind_weeks.svg)

**The blend's error is lower than the best single product's at every generator, for every set in
Figure 4.** The errors range from about 4% of capacity at the best solar farm to about 9% for the
four weather models at the worst. Generator W3's output swings for months at a time against every
product's wind, which the wind page attributes to turbine availability the feed does not record.

![Figure 4: Each blend's error is lower than its best single product's at every
generator](assets/blend_per_generator.svg)

### Solar: a blend beats CAMS with its extra columns

**An XGBoost model given all six products beats an XGBoost model given enriched CAMS by 0.13 points
[0.10, 0.17], 2.7% of CAMS's error, in a planned comparison.** The t-interval across the five folds
is [0.08, 0.18], and the gain at each generator ranges from 0.09 to 0.18 points. With the second
hyperparameter setting the gain is 0.12 points [0.08, 0.15].

**CAMS with ICON-EU alone gives most of that gain: 0.10 points [0.07, 0.13] over enriched CAMS, in a
planned comparison.** The t-interval across the folds is [0.05, 0.15], and the per-generator gain
ranges from 0.05 to 0.13 points. In exploratory comparisons, CAMS with ICON-D2 gains 0.12 points
[0.09, 0.15], and CAMS with ERA5 0.07 points [0.04, 0.09].

**Among the weather models alone, an XGBoost model given all four beats an XGBoost model given
enriched ICON-D2 by 0.34 points [0.25, 0.43], 4.5% of ICON-D2's error, in a planned comparison.**
The t-interval across the folds is [0.21, 0.48], and the per-generator gain ranges from 0.27 to 0.43
points. With the second hyperparameter setting the gain is 0.29 points [0.20, 0.38]. In an
exploratory comparison, UKV with ICON-EU beats enriched UKV by 0.52 points [0.44, 0.60].

**The plain comparisons, planned before the first run, show larger gains because their single
product lacks the extra columns.** Against plain CAMS, an XGBoost model given all six products'
plain columns gains 0.21 points [0.16, 0.25]. Against plain ICON-D2, an XGBoost model given the four
weather models' plain columns gains 0.47 points [0.40, 0.55].

### Wind: a blend beats UKV with its extra columns

**An XGBoost model given all five products beats an XGBoost model given enriched UKV by 0.48 points
[0.40, 0.56], 7.7% of UKV's error, in a planned comparison.** The t-interval across the five folds
is [0.33, 0.59], and the gain at each generator ranges from 0.41 to 0.53 points. With the second
hyperparameter setting the gain is 0.50 points [0.43, 0.58].

**UKV with ICON-EU beats enriched UKV by 0.26 points [0.21, 0.31], 4.2% of UKV's error, in a planned
comparison.** The t-interval across the folds is [0.16, 0.34], and the per-generator gain ranges
from 0.23 to 0.32 points. With the second hyperparameter setting the gain is 0.27 points [0.23,
0.32]. In exploratory comparisons, the four weather models beat enriched UKV by 0.46 points [0.39,
0.54], and ICON-D2 with UKV by 0.39 points [0.32, 0.46].

**The plain comparisons, planned before the first run, again show larger gains.** Against plain
ICON-D2, the best plain wind product, all five products' plain columns gain 0.82 points [0.74,
0.90]. Against plain UKV, UKV with ICON-EU gains 0.65 points [0.59, 0.72].

### The gain comes from the other products' weather

**For every named blend, the blend beats its control, and the control does no better than the best
single product.** Figure 5 splits each blend's gain in two. For all six solar products, the blend
beats its control by 0.20 points [0.17, 0.24], and the control is 0.07 points worse than enriched
CAMS [0.05, 0.09]. For all five wind products, the blend beats its control by 0.53 points [0.46,
0.62], and the control is 0.06 points worse than enriched UKV [0.02, 0.09]. For UKV with ICON-EU,
the control's difference from enriched UKV is not statistically significant at the 5% level (0.02
points worse [−0.01, +0.04]). All of these comparisons are planned.

**Extra columns without weather make the XGBoost model slightly worse, so the blend's gain over its
control is larger than its gain over the best single product.** The gain is therefore not an effect
of giving XGBoost more columns to split on. The control keeps each month's mean at each hour, which
carries a little real information, so a blend's gain over its control is if anything understated.

![Figure 5: The gain comes from the other products' weather, not from the extra
columns](assets/blend_decomposition.svg)

### The gain holds at every generator and in every season

**Each named blend beats its best single product at each of the nine generators, in each of the four
seasons, and on each side of UKV's January 2026 upgrade.** All 108 of these exploratory intervals
are statistically significant at the 5% level. The splits share hours and XGBoost models with the
whole-record figures and with one another, so they are not independent tests.

**The solar gain is smallest from June to August, and the wind gain holds in every season.** An
XGBoost model given all six products beats enriched CAMS by 0.09 points [0.03, 0.14] from June to
August, and by 0.19 points [0.10, 0.28] from December to February. All five wind products beat
enriched UKV by 0.39 to 0.58 points in each season. Since UKV's upgrade the solar gain is 0.08
points [0.05, 0.12] and the wind gain 0.40 points [0.33, 0.50], each resting on 8 months.

![Figure 6: Each named blend beats its best single product at every generator and in every
season](assets/blend_splits.svg)

### The wind gain is spread across every level of output

**The wind blends beat enriched UKV in every band of measured output, and most of the net gain sits
in hours below half of capacity.** All five products gain 0.26 points in the hours below a tenth of
capacity, which are 36% of the hours, and 0.73 to 0.83 points in each band above 0.3 of capacity.
The hours below 0.3 of capacity carry 52% of the net gain, and the hours above 0.9 carry 4%. The top
panel of Figure 7 shows each band.

**The report's statement that the most-improved 5% of hours carry 99% of the gain describes gains
and losses netting out, not a gain confined to a few hours.** A blend and a single product differ by
much more in any one hour than on average. Of the hours, 56% improve and 44% worsen. Sorting the
hours from most improved to most worsened, the running sum of the improvement passes the net gain
after about 5% of hours, rises to about three times the net gain where the improving hours end, and
falls back as the worsening hours are added. So the other 95% of hours net to about 1% of the gain.
For UKV with ICON-EU the most-improved 5% carry 143% of the net gain, because the other 95% of hours
are worse in total. The bottom panel of Figure 7 shows both running sums.

![Figure 7: The wind blends' gain is spread across every level of output](assets/blend_wind_bands.svg)

### An XGBoost blend beats a linear stack and simple averages

**An XGBoost model given every product's columns beats a linear stack of the single-product XGBoost
models, by 0.08 points for solar [0.05, 0.10] and 0.10 points for wind [0.05, 0.16], in planned
comparisons.** For wind the t-interval across the folds, [−0.01, +0.19], includes zero, so the wind
result rests on the month resampling alone. With the second hyperparameter setting the wind
difference is 0.11 points [0.06, 0.17], with a fold t-interval of [0.03, 0.18]. The stack still
beats the best single product in both domains: by 0.06 points [0.03, 0.09] for all six solar
products, and 0.38 points [0.32, 0.43] for all five wind products, in exploratory comparisons.

**Averaging CAMS with the other solar products makes the error far worse, where CAMS is far better
than every other product.** For all six solar products, the XGBoost model given the products' mean
is 1.64 points worse than enriched CAMS [1.47, 1.80], and the equal-weight mean of the predictions
is 1.75 points worse [1.58, 1.92]. For wind, both averages of UKV and ICON-EU beat enriched UKV, by
0.22 points each. All of these comparisons are exploratory.

![Figure 8: An XGBoost model given every product's columns has the lowest error of the four blends in
each named set](assets/blend_methods.svg)

**The linear stack weights CAMS heavily for solar and splits the wind weight between UKV and
ICON-D2.** For all six solar products the stack's mean weight on enriched CAMS is 0.85, and no other
product's exceeds 0.07. For all five wind products the mean weights are 0.43 on UKV, 0.37 on
ICON-D2, 0.12 on ICON-EU, 0.06 on ERA5, and 0.02 on ICON global. Across folds each weight moves by
0.03 or less on average.

**Fitting the stack's weights on the scored fold as well would flatter the stack by at most 0.02
points.** The report prints that in-sample gap for every stack, from 0.002 to 0.020 points. Stacking
one product's three fitting seeds, which isolates what averaging XGBoost models alone contributes,
gains about 0.01 points.

### The comparison detects a gain of this size

**Given the synthetic product, an XGBoost model beats the best single product by 0.13 points for
solar [0.11, 0.16] and 0.14 points for wind [0.10, 0.19].** Against its shuffled control the gain is
0.14 points for both. The noise was sized on a pilot fit so that the synthetic gain would be about
the size of the solar headline gains, and the comparison detects the synthetic gain with all five
folds agreeing. The synthetic product is the measured output plus noise, so the figure says only
that a gain of this size is visible to the comparison. A larger positive control, CAMS with ICON-D2
against ICON-D2 alone, shows a gain of 2.82 points [2.61, 3.03].

![Figure 9: The comparison detects a synthetic gain of 0.13 to 0.14 points, the size of the solar
blend's gain](assets/blend_synthetic.svg)

## What to use

**A blend waits for its slowest product, so the consumer decides which sets are available at all.**
The table gives each set's latency, which consumers can read it, and the first scored hour.

| Domain | Set | Available after | Live anywhere in Great Britain? | History only (CAMS or ERA5)? | Needs ICON-D2's domain? | Archive serves every product from | Scored hours start |
|---|---|---|---|---|---|---|---|
| Solar | CAMS and ICON-D2 | about 1 day | no | yes | yes | December 2022 | 1 December 2022 |
| Solar | CAMS and ICON-EU | about 1 day | no | yes | no | November 2022 | 1 December 2022 |
| Solar | CAMS and ERA5 | about 5 days | no | yes | no | January 2004 | 1 December 2022 |
| Solar | UKV and ICON-EU | about 4 hours | yes | no | no | November 2022 | 1 December 2022 |
| Solar | Four weather models | about 4 hours | no | no | yes | December 2022 | 1 December 2022 |
| Solar | All six products | about 5 days | no | yes | yes | December 2022 | 1 December 2022 |
| Wind | ICON-D2 and UKV | about 4 hours | no | no | yes | August 2024 | 12 August 2024 |
| Wind | UKV and ICON-EU | about 4 hours | yes | no | no | August 2024 | 12 August 2024 |
| Wind | Four weather models | about 4 hours | no | no | yes | August 2024 | 12 August 2024 |
| Wind | All five products | about 5 days | no | yes | yes | August 2024 | 12 August 2024 |

**These recommendations rest on six solar farms and three wind farms in one part of Lincolnshire,
over less than 4 years for solar and 2 years for wind.**

- **Historical features in the live service: UKV with ICON-EU, anywhere in Great Britain; the four
  weather models where ICON-D2 covers.** UKV with ICON-EU beats enriched UKV by 0.26 points for wind
  (planned) and 0.52 points for solar (exploratory). Inside ICON-D2's domain the four weather models
  beat the best single weather model by 0.34 points for solar (planned) and 0.46 points for wind
  (exploratory). A live blend needs every product at run time. Under this project's inherent
  stability rules, a missing product then needs a fallback that a single product avoids. Every
  figure here scores the archive's freshest run for each hour, and at run time the last few hours
  come from an older run.
- **Training history: CAMS with ICON-EU for solar anywhere in Great Britain from December 2022; all
  products where ICON-D2 covers.** For solar, all six products beat enriched CAMS by 0.13 points,
  and CAMS with ICON-EU by 0.10. For wind, all five products beat enriched UKV by 0.48 points, but
  UKV's hub-height wind starts only in August 2024, so the wind blends say nothing about training
  history before that. CAMS with ERA5 reaches back to 2004 for a gain of 0.07 points, measured only
  from December 2022.
- **Capacity estimation and disaggregation: no recommendation.** Both have to read a product's
  weather without an XGBoost model fitted to the generator's own output, and every gain here exists
  only after such a fit. The single-product recommendations on the solar and wind pages stand.

## Limitations

- **Region and period.** Six solar farms inside one 25 km by 23 km box, and three wind farms in flat
  country, all in Lincolnshire. The solar hours run from December 2022 and the wind hours from
  August 2024. Nothing here speaks to other regions, to terrain, or to offshore wind.
- **The intervals cover weather and fitting seeds, not generators or folds.** The per-generator
  range and the fold t-interval beside each headline show those two sources separately. With three
  wind farms, a fourth could rank the blends differently.
- **The headline comparisons were written after the first science review.** The review saw the first
  run's results before asking for the enriched comparisons, so the choice of extras was informed by
  results, though not by the enriched blends' own results.
- **The enriched best single product may not be the best single product possible.** The extras are
  the ones the solar page found useful. A single product given further columns, such as more
  neighbouring hours, might close part of the gap.
- **The linear stack's weights are fitted on out-of-fold predictions from XGBoost models trained on
  the scored fold's months.** This second-order leak is standard in cross-validated stacking, and
  would favour the stack, which still loses to the XGBoost blend. The interval treats the weights as
  fixed.
- **Every product is read from Open-Meteo's archive, at the served lead the earlier pages
  describe.** The weather models' values are forecasts made up to a few hours before each hour. A
  blend's gain at the leads a live forecast uses is not measured here.
- **The figures rest on the `effective_capacity` table as rebuilt in September 2026, and on version
  3 of the power Delta table.** A rebuilt capacity table would move every figure.

## Reproducing the figures

```bash
uv run python studies/beam_diffuse_split/weather_products.py
uv run python studies/beam_diffuse_split/wind_products.py
uv run python studies/beam_diffuse_split/blend_products.py
uv run python studies/beam_diffuse_split/blend_product_charts.py
```

The report lands in `data/studies/beam_diffuse_split/blend_products/report.md`, beside
`reproduction.md`, the per-hour errors and out-of-fold predictions of every XGBoost model, the stack
weights, and every interval as a table. `blend_products.py --resume` reuses the fits a crashed run
left behind. The chart script checks each chart's numbers against the report before drawing.
