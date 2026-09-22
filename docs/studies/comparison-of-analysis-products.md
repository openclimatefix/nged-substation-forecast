# Which irradiance product best describes an hour that already happened?

**A satellite retrieval describes past sunshine about twice as well as any weather model we tested,
and among the weather models a finer grid does help — but the Met Office's UKV underperforms for the
grid it runs on.** Four products were scored on how well a gradient-boosted tree, given each
product's global horizontal irradiance, predicts what 6 metered solar generators actually produced.
The Copernicus Atmosphere Monitoring Service (CAMS) retrieval roughly halves the error of every
model. Among the models, DWD's ICON-D2 at 2 km beats ECMWF's ERA5 reanalysis at 31 km by about one
percentage point of capacity, and beats UKV — also at 2 km — by rather more than that.

## This is not the question of which forecast is best

**Every product on this page estimates an hour that has already passed, so none of them is a
forecast.** ERA5 is a reanalysis, CAMS is a satellite retrieval, and the two limited-area models are
read at their analysis step, where the model has just finished absorbing observations. The consumers
of that estimate are all offline: capacity estimation, training targets, historical features, and
the disaggregation work. Lead time never enters, so a product can win here and lose badly at
day-ahead.

Which model gives the best forecast at operational lead times is measured separately, in [how far
ahead an ensemble forecast is still worth having](ens-forecast-horizons.md).

## What was measured

Each product was built into the same modelling frame as the [beam/diffuse split
experiment](beam-diffuse-split.md), and scored the same way:

- **The instrument** is XGBoost fitted per generator, predicting half-hourly power from the
  product's global horizontal irradiance plus solar zenith angle, solar azimuth angle,
  extraterrestrial horizontal irradiance, temperature, hour of day, and day of year.
- **The population** is 6 metered solar generators in one region of the distribution network.
- **The metric** is mean absolute error as a percentage of each generator's effective capacity,
  averaged across generators and hours.
- **The validation** cuts each generator's span into 5 contiguous blocks of whole months and scores
  each block from a model fitted on the other 4.
- **The scope** of each comparison is the hours and generators that the pair of products both cover,
  because a product that drops an hour must not be credited with the hours it dropped.

## Measured on shared hours, a pair at a time

| Pair | Shared hours | Winner | Loser |
|---|---|---|---|
| CAMS against ICON-D2 | 73,566 | CAMS 5.24% | ICON-D2 8.22% |
| CAMS against UKV | 88,569 | CAMS 5.34% | UKV 9.62% |
| ICON-D2 against ERA5 | 85,965 | ICON-D2 7.23% | ERA5 8.28% |
| ICON-D2 against UKV | 86,113 | ICON-D2 7.22% | UKV 8.40% |

**These four rows do not compose into a single ranking, and reading them as though they did is the
mistake the table most invites.** ICON-D2 scores 7.22% in the row against UKV and 8.22% in the row
against CAMS. Those are the same model and the same generators; only the hours differ, because CAMS
withholds hours its own reliability flag rejects and those hours are not evenly hard. A ranking of
four products needs all four scored on one common set of hours, with each product's model refitted
on that set. That measurement has not been made, and it is the first task of the study this page
feeds.

## Observing cloud beats simulating it

**CAMS roughly halves the error of every weather model it was compared against, which points at how
the cloud field is obtained rather than at how finely it is resolved.** CAMS infers cloud from what
a geostationary satellite saw, so for an hour that has already happened it is closer to a
measurement than to a simulation. Every other product on this page solves the equations of motion
and derives cloud from the solution, and for solar power the cloud field is almost the whole of the
problem. The gap between a retrieval and a model here — roughly 3 percentage points of capacity — is
far larger than any gap between two models.

That reading has a limit worth stating. CAMS is served interpolated to a requested point rather than
on a grid, so "CAMS at 5 km" and "ICON-D2 at 2 km" are not two samples of one quantity at two
spacings, and the comparison cannot be turned into a clean statement about resolution.

## A finer grid does help, but UKV does not deliver what its grid promises

**ICON-D2 at 2 km beats ERA5 at 31 km by about one percentage point of capacity, which overturns an
earlier reading that a finer grid buys nothing.** That earlier reading came from comparing ERA5 with
UKV, where the 2 km model failed to beat the 31 km one. Adding a second 2 km model separates the two
explanations: if fine grids were useless, ICON-D2 would have lost too.

**UKV loses to ICON-D2 by 1.26 percentage points of capacity on the 86,113 hours the two share, at
the same grid spacing.** Whatever the deficit is, it is a property of that particular product rather
than of its resolution. Two candidate explanations are open and this page cannot choose between
them: how the archive was assembled, and how the product itself has changed over time.

### The January 2026 upgrade moved UKV and left the others alone

**The Met Office made UKV's PS47 upgrade operational on 21 January 2026, and UKV's behaviour on
these generators changes sharply across that date while the other products' behaviour does not.**
Splitting the span at February 2026 and measuring how much each product's published beam/diffuse
split improves on a split estimated from its own total irradiance, the UKV figure moves from −0.084
to −0.461 percentage points, a factor of 5.5. On the same rows and the same power data, CAMS moves
from −0.100 to −0.068 and ERA5 from +0.004 to +0.020, neither of which is a step.

Only the product that received an upgrade moved. National weather services improve their models at
intervals, so a long archive is not one homogeneous product, and a result measured across such a
boundary is an average of two different models. Any future UKV figure should say which side of
January 2026 it was measured on.

## ICON-D2 does not cover Great Britain

**ICON-D2's domain stops around 2.5°W, which excludes South West England and South Wales, so it
cannot serve a national forecast on its own.** Probing the archive at public place coordinates,
requests for Truro, Swansea, Cardiff, Exeter, and Bristol are refused outright, while Birmingham and
Nottingham return data. The generators in this study all sit inside the covered part, which is why
ICON-D2 could be scored at all.

That limitation is the reason ICON-EU matters. ICON-EU is the same modelling system on a wider
domain at about 7 km, it covers the whole of Great Britain, and it has not yet been downloaded or
scored.

## Two products together beat either one alone

**Showing the tree both ICON-D2's and UKV's irradiance moves the error by 0.28 percentage points of
capacity against ICON-D2 alone, which is roughly three times what the published beam/diffuse split
moves it.** On the 86,113 hours the two products share, across 46 months:

| Arm | Irradiance shown to the tree | Mean absolute error (% of capacity) |
|---|---|---|
| UKV alone | UKV total | 8.449 |
| ICON-D2 alone | ICON-D2 total | 7.224 |
| ICON-D2 twice | ICON-D2 total, under two column names | 7.224 |
| ICON-D2 with its split | ICON-D2 total, beam, and diffuse | 7.126 |
| Both products | ICON-D2 total and UKV total | 6.922 |
| Both products with both splits | both totals, beams, and diffuses | 6.785 |

| Contrast | Difference (percentage points) | 95% interval | Excludes zero? |
|---|---|---|---|
| ICON-D2 twice against ICON-D2 alone | +0.0000 | [+0.0000, +0.0000] | no |
| Both products against ICON-D2 alone | −0.2846 | [−0.3555, −0.2207] | **yes** |
| ICON-D2 with its split against ICON-D2 alone | −0.1008 | [−0.1382, −0.0671] | **yes** |
| Both products with both splits against both products | −0.1437 | [−0.1870, −0.1050] | **yes** |
| ICON-D2 alone against UKV alone | −1.2583 | [−1.4902, −1.0185] | **yes** |

**The duplicated-column arm is what makes the gain readable as information rather than as column
count.** An arm holding two products has more columns than an arm holding one, and a tree offered
more columns has more splits to choose between, so a naive reading of "both products win" cannot
tell the two explanations apart. Showing the tree ICON-D2's own irradiance twice, under two names,
scores exactly what showing it once scores — to every reported digit, because `colsample_bytree` is
1.0 and the duplicated column changes no tree. Column count therefore buys nothing here, and the
0.28-point gain is the second product's information.

**The two gains are close to additive, and combining products is worth about three times the
split.** Starting from 7.224, subtracting the 0.28 for the second product and the 0.14 for the
splits predicts 6.80 against a measured 6.785. UKV contributes this despite being the weaker product
by 1.26 points, which is the useful part: a product does not have to be better to be worth adding,
only to be wrong in different places.

**Both inputs here are analyses, so this result does not yet transfer to the forecast case.**
Combining two forecasts is the claim that matters for the live service, and two forecasts of the
same future hour may share more of their error than two analyses of the same past hour do.

## What these numbers do not establish

- **The population is 6 solar generators sharing one region's weather**, so the effective number of
  independent weather episodes is far smaller than the number of generator-hours, and nothing here
  speaks to wind.
- **Each product's model was fitted on that product's own rows**, so a surviving difference is a
  difference between two pipelines rather than between two descriptions of the weather alone.
- **No common row set exists yet**, so the pairwise figures do not compose, as set out above.
- **The two limited-area models are read at their analysis step**, and UKV's assimilation ingests a
  large volume of satellite-derived cloud from the same geostationary platform CAMS retrieves from,
  so "model against retrieval" is not a clean contrast for UKV.
