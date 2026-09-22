# How far ahead is an ensemble forecast still worth having?

**ECMWF's ensemble forecast describes the next few hours better than a reanalysis does, matches the
reanalysis at day-ahead, and loses badly beyond two days.** Scored on 6 metered solar generators,
the ensemble beats ERA5 by 8.8% at leads under a day and is statistically indistinguishable from it
at day-ahead. By a week out its error is about 55% higher. The day-ahead result is the consequential
one: at the horizon flexibility is procured on, the weather input is already about as good as
perfect hindsight, so the error that remains sits in the model turning weather into power.

## Why the ensemble is not in the four-product table

**Every product in [the comparison of analysis products](comparison-of-analysis-products.md)
estimates an hour that has already happened, and a forecast of an hour that has not is not
comparable with any of them.** A forecast is penalised for being early; an analysis is not. Putting
an ensemble figure in that table would read as a statement about the ensemble's quality when most of
it would be a statement about lead time. So the ensemble is scored here against one reference —
ERA5, on the same rows — and against itself across horizons.

**ERA5 is the right reference precisely because it is not a forecast.** A reanalysis assimilates
observations of the hour it describes, so it stands in for the best description of the weather that
could be had after the fact. Comparing a forecast against it asks how much of the achievable skill
the forecast has already reached, rather than which of two forecasts is better.

## What was measured

- **The source** is the 51-member ECMWF ensemble, read at the generators' locations.
- **A horizon is a band of lead hours**, not a single lead, because one lead yields too few valid
  times to score. The bands are 3 to 24 hours, 24 to 48, 48 to 72, 168 to 192, and 336 to 360.
- **Scoring happens on the ensemble's own 3-hourly stamps.** Its radiation is period-ending over the
  preceding forecast step, so the hourly sources are averaged over the 3 hours ending at each stamp
  rather than the ensemble being interpolated down to an hour it never resolved. A window missing
  any of its 3 hours is dropped.
- **The instrument** is XGBoost fitted per generator on global horizontal irradiance plus solar
  geometry, temperature, and calendar features, validated on 5 contiguous blocks of whole months.
- **The metric** is mean absolute error in megawatts, with ERA5 scored on exactly the rows each
  horizon keeps.
- **Intervals** come from resampling whole calendar months with replacement, differencing the two
  variants before averaging so that shared weather cancels.

Four ways of turning 51 members into one power number were scored. Two of them reduce the members
before the power model and two after it:

| Variant | What it does |
|---|---|
| `control` | Uses the control member alone, which is what the production selection does |
| `mean_of_irradiance` | Averages the members' irradiance, then predicts once |
| `mean_of_power` | Predicts once per member, then averages the 51 power values |
| `median_of_power` | Predicts once per member, then takes the median of the 51 power values |

## The ensemble against ERA5, by horizon

| Lead band | Scored stamps | `control` | `mean_of_irradiance` | `mean_of_power` | `median_of_power` | ERA5 |
|---|---|---|---|---|---|---|
| 3 to 24 h | 15,987 | 0.7118 | **0.6841** | 0.6893 | 0.6957 | 0.7499 |
| 24 to 48 h | 15,964 | 0.7818 | **0.7430** | 0.7598 | 0.7593 | 0.7480 |
| 48 to 72 h | 15,941 | 0.8769 | **0.8263** | 0.8562 | 0.8533 | 0.7436 |
| 168 to 192 h | 7,935 | 1.1727 | **1.1451** | 1.1527 | 1.1536 | 0.7394 |
| 336 to 360 h | 7,855 | 1.1861 | 1.1891 | **1.1665** | 1.1692 | 0.7387 |

Mean absolute error in megawatts; lower is better; the best ensemble variant in each row is bold.

| Lead band | Contrast | Difference (MW) | 95% interval | Excludes zero? |
|---|---|---|---|---|
| 3 to 24 h | best ensemble against ERA5 | −0.0658 | [−0.0848, −0.0468] | **yes** |
| 24 to 48 h | best ensemble against ERA5 | −0.0050 | [−0.0266, +0.0180] | no |
| 48 to 72 h | best ensemble against ERA5 | +0.0827 | [+0.0602, +0.1057] | **yes** |
| 168 to 192 h | best ensemble against ERA5 | +0.4057 | [+0.3602, +0.4558] | **yes** |
| 336 to 360 h | best ensemble against ERA5 | +0.4277 | [+0.3820, +0.4806] | **yes** |

**ERA5's error barely moves down the table, which is the check that the horizon axis means what it
should.** A reanalysis has no lead time, so its figure varies only because each row scores a
different set of hours. It stays between 0.7387 and 0.7499 across all five rows, so the ensemble's
climb from 0.68 to 1.17 is lead time rather than a change in how hard the scored hours are.

## At day-ahead the weather is no longer the binding constraint

**Between 24 and 48 hours ahead, the ensemble and the reanalysis cannot be told apart: the
difference is −0.0050 MW with an interval running from −0.0266 to +0.0180.** The forecast has
therefore already reached the accuracy of a product that got to look at the observations afterwards.
Whatever error remains in the day-ahead power forecast is not waiting on a better description of the
weather; it is in the step from irradiance to megawatts, or in the generator's own behaviour. Effort
spent finding a sharper day-ahead weather input has little room to pay, and effort spent on the
power model has a great deal.

## Skill collapses after two days, rather than tapering

**The error rises by 11% between day-ahead and two days ahead, and by about 55% by a week.** From
168 hours onwards the figure barely changes — 1.1451 at a week against 1.1665 at a fortnight — which
is roughly what a forecast that has lost most of its information and fallen back on climatology
would do. A product covering 7 to 14 days is therefore a categorically weaker proposition than a
day-ahead one rather than a marginally weaker one, and should be built and described as a different
product.

## Using every member beats using the control member, at every horizon

| Lead band | `mean_of_power` against `control` | 95% interval | Excludes zero? |
|---|---|---|---|
| 3 to 24 h | −0.0225 | [−0.0320, −0.0137] | **yes** |
| 24 to 48 h | −0.0219 | [−0.0343, −0.0100] | **yes** |
| 48 to 72 h | −0.0207 | [−0.0310, −0.0103] | **yes** |
| 168 to 192 h | −0.0200 | [−0.0384, −0.0055] | **yes** |
| 336 to 360 h | −0.0196 | [−0.0318, −0.0091] | **yes** |

**The gain is small, consistent, and free.** The 51 members differ only in their irradiance column,
so they are 51 predictions from one fitted model rather than 51 models, and the cost is one
prediction pass rather than one training run. The production selection currently keeps the control
member and discards the other 50, which this measurement says costs about 0.02 MW at every horizon
tested.

## Where the members are reduced matters, and flips with horizon

| Lead band | `mean_of_power` against `mean_of_irradiance` | 95% interval | Excludes zero? |
|---|---|---|---|
| 3 to 24 h | +0.0052 | [−0.0031, +0.0134] | no |
| 24 to 48 h | +0.0169 | [+0.0082, +0.0249] | **yes** |
| 48 to 72 h | +0.0300 | [+0.0196, +0.0403] | **yes** |
| 168 to 192 h | +0.0076 | [−0.0137, +0.0271] | no |
| 336 to 360 h | −0.0227 | [−0.0379, −0.0077] | **yes** |

**Averaging the irradiance first wins at one and two days ahead, and loses at a fortnight, with the
crossover somewhere between.** Averaging inside a function and averaging outside it agree only where
the function is close to straight across the range being averaged. At short lead the members
disagree only a little, the fitted power curve is nearly straight across that narrow range, and
averaging 51 noisy estimates of the same weather first acts as a noise filter. At long lead the
members disagree enough to span the flat part of the fitted curve, where the generator's output is
capped, and an averaged irradiance then describes no member's weather and implies a power the
generator could not have produced.

**The practical reading is that neither reduction should be adopted as a convention.** The crossover
falls inside the 3-to-10-day band the product is aimed at, so the choice deserves an experiment that
blends the two, weighted by how far apart the members are rather than by lead time.

## The median does not beat the mean, which says the members are too similar to each other

**Mean absolute error is minimised by the median of the predicted distribution, so taking the
members' median ought to win, and it does not.** It is worse than the mean by 0.0064 MW at the
shortest lead band, with an interval from +0.0036 to +0.0092, and indistinguishable from it
everywhere else. The 51 members give an empirical predicted distribution for free, so the failure is
informative: their spread is narrower than the errors the forecast actually makes, and a median
taken from an under-dispersed distribution is not the median that minimises the loss. Widening the
member spread to match the measured error before reducing it is the arm that would test this
directly.

## What these numbers do not establish

- **The population is 6 solar generators sharing one region's weather**, so the number of
  independent weather episodes is far smaller than the number of scored stamps, and nothing here
  speaks to wind.
- **The per-member variants were fitted on the control member's irradiance and then shown every
  member's**, while `mean_of_irradiance` was fitted on the averaged irradiance it was later shown.
  The comparison between those two therefore mixes where the reduction happens with what the model
  was trained on, and a clean version would fit each variant on the input it is scored with.
- **No export-cap clamp and no commissioning-ramp exclusion were applied**, unlike the beam/diffuse
  split experiment, so a handful of hours are scored that the other study withholds.
- **One booster seed was used**, so the intervals carry month-to-month variation but not the
  variation between seeds, and are correspondingly narrower than the main experiment's.
- **The two longest bands are scored on a 3-hour window inside a 6-hour step**, because the ensemble
  halves its step frequency beyond 144 hours and one window definition was kept across every
  horizon.
