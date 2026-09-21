# Does a weather product's beam/diffuse split help a PV forecast?

**On a 5 km satellite retrieval, the product's own direct-beam field cuts PV power error by 1.5%
beyond what any separation model recovers from global irradiance alone; on a 31 km reanalysis it
adds nothing at all; and choosing the better irradiance source matters about thirty times more than
having the split at all.** Six metered solar farms inside one 34 km box in Lincolnshire, seven
years of hourly daylight readings, two irradiance products and two model families. The advantage,
where it exists, is information rather than a better-published correlation: replacing the 1982 Erbs
correlation with a separation model fitted on this data, 2.4 times more faithful to the published
direct fraction, buys nothing measurable, while the published field itself buys 0.09 percentage
points of error. That advantage disappears under a clear sky and concentrates where cloud makes the
split genuinely uncertain, which is the shape an information account predicts.

## The decision this feeds

**The ECMWF ensemble feed this project ingests carries global short-wave irradiance and no direct
component, so a physically-grounded PV model has to derive the beam/diffuse split rather than read
it.** The alternatives are to ask Dynamical.org to add the direct beam to the feed they build for
us, to take the split from a different source, or to run a separation model locally on the global
irradiance we already have. Each costs something, and none is worth paying before knowing what the
split is worth. This page measures that.

## What this measures, and what it does not

**Both irradiance products here are analyses of what the weather did, not forecasts of what it will
do, so what follows is the information content of the beam field rather than forecast skill.** A
forecast of the beam would carry its own error on top, and nothing measured here bounds that error.
A result saying the published beam helps is therefore an upper bound on what a forecast beam could
deliver, and a result saying it does not help rules the forecast case out as well.

Two further limits are worth stating before any number. The finding is about one micro-region over
2019 to 2026, and it is about models that predict power from irradiance at a single site. It is not
a statement about the physics of PV generation, where the beam/diffuse split is not in doubt: a
tilted panel sees the beam projected by the cosine of its incidence angle and the diffuse almost
unchanged, so a model that knows the split can do a transposition a model given one number cannot.
The question is only whether a *published* beam field tells a model something it could not have
worked out for itself.

## Data

### Six metered solar farms, seven years, hourly

The power readings are half-hourly metered output from six solar farms in NGED's Lincolnshire
licence area, running from September 2019 to September 2026. They are averaged to the hourly grid
the irradiance products use, and only daylight hours are kept: 127,882 site-hours on the satellite
source and 149,746 on the reanalysis, the difference being hours the satellite service flags as
unreliable.

**Every figure and table on this page is normalised, and the sites carry shuffled letters rather
than names, because a single solar farm's output is commercially sensitive.** NGED has asked that
generator data leave the project only anonymised. Error and power are therefore expressed as a
percentage of each site's own 99th percentile of absolute metered output — written **P99 output**
below — rather than in megawatts, and the site labels A to F come from a fixed permutation that no
published table can be used to invert. The unit is not registered capacity, which this experiment
never reads.

### Three irradiance sources, two of them the same reanalysis

| Source | What it is | Resolution |
|---|---|---|
| ERA5 via the Copernicus Climate Data Store | ECMWF's global reanalysis, carrying global short-wave (`ssrd`) and direct short-wave (`fdir`) | ~31 km grid, hourly |
| ERA5 via Open-Meteo's mirror | The same reanalysis, the same fields, served in about a minute rather than most of a night | identical grid and hours |
| CAMS radiation service | A satellite retrieval: cloud inferred from Meteosat, published at each meter's own coordinates | ~5 km cloud field, point delivery |

The reanalysis and the satellite retrieval are different instruments rather than two routes to one
answer. ERA5 averages its cloud field over roughly 31 km and lands each meter in a grid cell up to
17 km away; the CAMS service retrieves cloud at around 5 km and interpolates to the meter's own
position. Running the same comparison on both is what separates "the split carries no information"
from "this grid has already smoothed the beam away".

### The mirror really is ERA5

**Before any result could be read as an ERA5 result, the mirror had to be shown to serve ERA5's own
`fdir` rather than a separation model's estimate of it** — otherwise the arm given the "published"
split would be a copy of the arm given a derived one, and a null result would be guaranteed by
construction rather than measured. Comparing the two downloads hour by hour over every hour both
cover, 1,232,160 cell-hours, the mean absolute difference is 0.15 W m⁻² on global irradiance and
0.13 W m⁻² on the beam, against Open-Meteo's own rounding of 1 W m⁻². Fewer than 0.12% of hours
differ by more than that rounding. The mirror carries the same fields.

### Cleaning, and a half-hour question the data cannot settle

Three filters run before any model sees a row, each blind to the beam/diffuse split so that none of
them can favour an arm. Runs of exactly-zero power at least 24 hours long are treated as outages, on
the grounds that a solar farm cannot produce such a run by physics; that removes about 1.2% of
daylight rows. An hour containing an exactly-zero half-hour under more than 100 W m⁻² of global
irradiance is treated as a meter dropout rather than a dim hour, which removes a further 3.7%. And
on the satellite source, hours the service itself flags as less than 90% reliable are dropped, which
removes about 17% — those hours are darker and more diffuse than the ones kept, so every satellite
number here is conditional on discarding them.

**Three independent tests say the power stamps arrive half an hour later than the contract implies,
so every result below is computed twice, once under each reading.** Against the sun's own horizon
crossings the first and last generating half-hour of a clear day both fall 30 minutes late; the
power-weighted centroid of a clear day runs 0.45 hours late; and the correlation with global
irradiance peaks at a 30-minute shift for all six meters in every year. It is not a daylight-saving
error, because the offset does not step at the March and October boundaries. Whether the contract or
the feed is at fault is a question for NGED. Running both readings costs one extra column of results
and removes the worry, because a half-hour misalignment blunts the sharp beam signal more than the
smooth diffuse one and so would penalise exactly the arm under test.

## Methods

### The arms differ only in which irradiance columns the model sees

Every arm sees identical rows, identical folds, identical seeds, identical settings and identical
non-irradiance features — solar zenith and azimuth, extraterrestrial horizontal irradiance, air
temperature, hour of day and day of year. Only the irradiance columns change, and every one of them
is a flux onto a horizontal plane, so no two arms differ in how a quantity is encoded as well as in
what it knows.

| Arm | What the model is shown |
|---|---|
| A — global only | Global horizontal irradiance |
| B — Erbs | Global irradiance, plus the beam and diffuse fluxes the Erbs correlation derives from it |
| C — the product's own split | Global irradiance, the published beam, and the difference |
| D — direct fraction | Global irradiance and the published beam's share of it |
| B-DISC | Arm B with the DISC correlation in place of Erbs |
| B-LEARNED | Arm B with a separation model fitted on this data in place of Erbs |

**Arm C against arm B is the comparison the experiment exists for, and it was named before the run.**
Giving the geometry to every arm is deliberate: a fixed-tilt array's sensitivity to the split is
partly a function of sun position, which a tree can absorb from the geometry columns, so arm A is
made as strong as it can be and any advantage arm C shows is a lower bound.

**Arm B-LEARNED exists because arm C could beat arm B for two reasons that carry opposite
decisions.** The published beam may hold information no function of global irradiance and solar
geometry can recover, in which case the field is worth asking a supplier for. Or the product may
simply publish a better separation model than a correlation fitted in 1982, in which case the same
gain is available locally for nothing. Erbs alone cannot tell those apart. Arm B-LEARNED can: its
beam is a prediction of the product's own direct fraction from exactly arm A's feature set, so every
value it carries is a function of what arm A already holds. It is 2.4 times more faithful to the
published direct fraction than Erbs. If arm C still beats it, the advantage is information rather
than representation.

That arm has to be built carefully, because the obvious construction leaks. Folds are cut inside
each site's own span, so one fold number is a different calendar period at each site; a separation
model that merely dropped the rows carrying that fold number would still train on other sites' rows
at the scored fold's own hours, and on the reanalysis those other sites are the same grid cell. The
withholding is therefore by calendar month, and it covers the training rows as well as the scored
fold, through an inner cross-validation — a column sharper where the arm trains than where it is
scored would be over-trusted by the power model and would penalise the arm for a reason unrelated to
the split.

### Two instruments

The first instrument is XGBoost, one model per site, shown the irradiance columns as features and
left to discover what to do with them. The second is a five-parameter physical PV model — tilt,
azimuth, capacity, a clipping limit and a temperature coefficient — which is *given* the
transposition rather than having to learn it, and fitted per site on each training fold. The second
exists because a null result from a tree is ambiguous between "the split carries nothing" and "the
tree could not use it".

### Folds, and why the intervals are wider than the row count suggests

Each site's span is cut into five contiguous blocks of whole months, and each block is scored by a
model fitted on the other four. Training on both sides of the test block is right here, because the
question is about information content rather than about forecasting forward in time.

**Six meters inside a 34 km box share their weather, so the effective sample size is the number of
independent weather episodes rather than the number of site-hours.** On the reanalysis it is worse
than that: the six sites fall inside only two ERA5 grid cells, and within each group the global
irradiance is bit-identical, so the per-site rows are two irradiance series against six power
targets. Every interval quoted below is therefore a monthly block bootstrap: whole calendar months
are resampled, the same months for both arms so the comparison stays paired, with all six sites' rows
inside each block, and each resample also draws one of the three seeds. Widening the blocks to
quarters or to calendar years leaves every verdict below unchanged.

### Two controls

**Arm B is a negative control the experiment gets for free.** Erbs reads global irradiance and solar
geometry and nothing else, all of which arm A already holds, so arm B cannot carry information arm A
lacks. Whatever B−A comes out as is this pipeline's reading on a feature set known to be
uninformative.

**A positive control runs the same arms against a synthetic target built by transposing the true
split onto a tilted plane**, where the split must help by construction. It is built outside the
physical model's hypothesis class on purpose: each site gets its own tilt and azimuth, none of them
the values the optimiser starts from, and the sky diffuse is transposed by the Hay-Davies model where
the instrument assumes an isotropic sky. What it establishes is that the instruments detect an effect
of this kind when one exists — on that target, arm C beats arm B-LEARNED by 0.17 points.

## The models work

Before any contrast of a tenth of a percentage point is worth reading, the pipeline has to be shown
producing a sane forecast. These are out-of-fold predictions for one site across three weeks: the
clearest week in the record, the most variable, and the dullest, chosen by clearness index rather
than by eye.

![Predicted against measured PV power at site A](power_timeseries_site_a.svg)

The other five sites are below.

![Site B](power_timeseries_site_b.svg)

![Site C](power_timeseries_site_c.svg)

![Site D](power_timeseries_site_d.svg)

![Site E](power_timeseries_site_e.svg)

![Site F](power_timeseries_site_f.svg)

The error levels those predictions sit at, per site and per setup:

![Mean absolute error per site and setup](per_site_error.svg)

| Site | ERA5 → XGBoost | CAMS → XGBoost | ERA5 → physical | CAMS → physical |
|---|---|---|---|---|
| A | 9.14 | 6.25 | 9.35 | 7.10 |
| B | 8.71 | 5.93 | 9.31 | 6.82 |
| C | 7.97 | 5.15 | 8.43 | 6.22 |
| D | 9.21 | 6.05 | 10.01 | 7.30 |
| E | 11.08 | 10.16 | 10.44 | 9.64 |
| F | 8.41 | 5.04 | 8.91 | 6.06 |
| **Pooled** | **8.80** | **5.93** | **9.23** | **6.84** |

Mean absolute error as a percentage of each site's P99 output, each setup given the product's own
split. Site E is the worst everywhere and also the shortest series, at 7,588 satellite hours against
roughly 25,000 for the others.

**One site drifts across the record, and the paired design absorbs it.** At site A the mean signed
error moves from −2.3% of P99 output in 2023 to +4.3% in 2026, so a model fitted mostly on earlier
years increasingly overpredicts the later ones — visible as the gap in the clearest-week panel above,
which falls in April 2026. Panel degradation and rising curtailment would both look like this, and
the data here cannot separate them. It does not touch any contrast below, because every arm is
scored on the same rows and the bootstrap differences them row by row before resampling.

## The irradiance source matters far more than the split does

**On the 126,784 hours both sources cover, the satellite retrieval cuts XGBoost's error from 10.12
to 6.07% of P99 output — 4.1 points, or 40% relative.** Every contrast in the rest of this page is
between 0.01 and 0.13 points. The single largest lever available to a PV power model in this region
is which irradiance product it is fed, and it is not close.

| Instrument | ERA5 | CAMS | Difference |
|---|---|---|---|
| XGBoost, global only | 10.12 | 6.07 | −4.05 |
| XGBoost, product's own split | 10.07 | 5.94 | −4.13 |
| Physical model, Erbs split | 10.37 | 6.73 | −3.64 |

Mean absolute error as a percentage of P99 output, restricted to the hours both sources cover.

That gap is what a 5 km cloud field at the meter's own coordinates buys over a 31 km field averaged
across a cell the meter may sit 17 km from. It also explains why the beam question has different
answers on the two sources, which the next section takes up.

## XGBoost beats the fitted physical model, but not by much

**On the satellite source the tree reaches 5.93% of P99 output against the physical model's 6.84%,
so the tree is about 0.9 points better** — and it wins at every site on both sources except site E,
where the physical model is slightly ahead on the shortest record. The physical model is doing this
with five parameters per site against a gradient-boosted ensemble, and it is given the transposition
rather than having to learn it.

Two things make that comparison less lopsided than the numbers suggest. The tree has seven years of
half-hourly history per site to fit on, which a real deployment at a new site would not; the physical
model needs enough data to pin five parameters and no more. And the physical model produces
interpretable quantities — the fitted tilts land between 17 and 25 degrees and the azimuths within a
few degrees of due south, which is what these arrays plausibly are. Neither model is the production
design, and the comparison here exists to check that a null from one instrument is not an artefact of
that instrument.

## Does the published beam field add information?

**On the satellite retrieval, yes: the product's own split beats the Erbs split by 0.091 points of
P99 output, or 1.5% relative, with an interval excluding zero. On the reanalysis, no: the same
contrast is +0.007 points with an interval straddling zero.**

![The headline contrasts, both sources and both instruments](beam_diffuse_split_result.svg)

| Contrast | CAMS (5 km) | ERA5 (31 km) |
|---|---|---|
| C − B — the product's split against Erbs | **−0.0905** [−0.1099, −0.0725] | +0.0073 [−0.0137, +0.0263] |
| C − B-LEARNED — against the fitted separation model | **−0.0925** [−0.1108, −0.0754] | −0.0009 [−0.0240, +0.0218] |
| B-LEARNED − B — a better separation model, on its own | +0.0020 [−0.0039, +0.0079] | +0.0082 [−0.0097, +0.0252] |
| B − A — the negative control | −0.0334 [−0.0474, −0.0221] | −0.0523 [−0.0750, −0.0305] |
| C − A — the product's split against global alone | −0.1240 [−0.1461, −0.1023] | −0.0450 [−0.0711, −0.0199] |

Change in mean absolute error in percentage points of P99 output, XGBoost, shifted stamps. Negative
favours the first arm. Bold marks the two contrasts the page's conclusion rests on.

### The gain is information, not a better correlation

**The middle row of that table is the one that settles what the advantage is.** Arm B-LEARNED is a
separation model fitted on this data that reproduces the published direct fraction 2.4 times more
faithfully than Erbs does — it leaves 4% of that fraction's variance unexplained against Erbs's 9%.
All that extra fidelity buys +0.002 points, an interval spanning zero, and the same verdict on the
probabilistic score. Meanwhile the published field itself buys 0.09 points. So arm C's advantage is
not a better deterministic function of global irradiance and solar geometry; a model given arm A's
features cannot reach it however well it is fitted.

That reading is corroborated by a diagnostic run before the arms: out of fold, a model given arm A's
own features predicts 96% of the variance of the satellite product's direct fraction. The remaining
4% is where the advantage lives, and it is not recoverable from what arm A holds.

### What the result survives

The satellite finding holds under every check the design carries. It reproduces at both stamp
alignments (−0.091 and −0.077), at both hyperparameter settings (−0.091 and −0.072), on the
continuous ranked probability score as well as mean absolute error, in 5 of 5 folds, in 5 of 6 sites,
and at every solar-elevation band. Seed-to-seed spread is 0.003 points against a 0.091-point effect.

The reanalysis null is equally robust, and it is not an artefact of the mirror, the stamps or the row
set: the Copernicus download reproduces the Open-Meteo result to the third decimal (+0.003 against
+0.007 at shifted stamps), both alignments agree, and the null survives restriction to the hours the
satellite source also covers.

### The physical model disagrees, and is not a second opinion

On the same rows the physical model reports the opposite sign: the product's own split is *worse*
than the Erbs split, by 0.13 points on the satellite source. That looks like a contradiction and is
not one, because the physical model's two arms do not differ only in the beam field they are handed.
In every run and at every site, the arm given the product's split settles on a tilt 3 to 11 degrees
shallower than the arm given Erbs, so the arms differ in fitted geometry as well. A 30-minute stamp
shift is absorbed into the fitted azimuth — the shifted runs settle around 162 to 179 degrees and the
as-labelled runs around 200 to 212 — and the ordering of the arms changes sign with it, in sample as
well as out. The physical model also divides the horizontal beam by the cosine of the zenith angle to
transpose it, which magnifies a beam error near the horizon up to twentyfold, where the tree performs
no such division.

**So the physical instrument answers "which beam field lets a five-parameter isotropic-sky model with
a fitted azimuth fit best", and its answer moves with a timestamp convention.** It is a
misspecification probe rather than a second reading of the same quantity, and the right response is to
report what each instrument measured rather than to reconcile the signs.

## Where the advantage lives

**The satellite advantage vanishes under a clear sky and concentrates where cloud makes the split
uncertain**, which is what the information account predicts and a calibration difference would not.

![The headline contrast split by sky condition](sky_conditions.svg)

| Sky condition | Clearness index | C − B | Relative | Hours |
|---|---|---|---|---|
| Overcast | below 0.2 | −0.0588 [−0.0795, −0.0392] | −1.75% | 19,331 |
| Mostly cloudy | 0.2 to 0.4 | −0.1299 [−0.1569, −0.1045] | −2.47% | 34,411 |
| Broken cloud | 0.4 to 0.6 | −0.1344 [−0.1783, −0.0973] | −1.91% | 39,936 |
| Clear | above 0.6 | −0.0151 [−0.0459, +0.0171] | −0.21% | 33,076 |

CAMS, XGBoost, shifted stamps. The clearness index is the share of the extraterrestrial horizontal
irradiance that reached the ground.

Under a clear sky the direct fraction follows from the sun's position, so a separation model already
knows it, and the published field adds nothing measurable. As cloud thickens the direct fraction
becomes genuinely uncertain at a given global irradiance — two hours with the same total can carry
very different beam depending on whether the sun's disc happens to be covered — and that is where the
published field earns its advantage, at roughly 0.13 points. Under thick overcast the effect shrinks
again, as it must when there is almost no beam left to know about. On the reanalysis the contrast is
null in all four bins, consistent with its pooled result.

## What follows for the forecast feed

**Ask a supplier for the direct beam only where the source resolves cloud finely enough to carry beam
information its own global field lacks.** On a 31 km reanalysis the beam field is worth nothing beyond
a separation model run locally for free, and the mechanism is visible: at that resolution 90% of the
direct fraction is already implied by the global field and the sun's position. On a 5 km retrieval it
is worth about 1.5% of error. Whether that is worth paying for depends on what it costs, which this
page does not know.

**Spend the first effort on the irradiance source rather than on the split.** The gap between the two
products is 4.1 points of P99 output and the gap the split opens is at most 0.13, a ratio of about
thirty to one. Any decision that trades source quality for the split has the priorities backwards.

Two caveats attach to carrying this into a forecasting decision. Both products here are analyses, so
a forecast beam would arrive with its own error and the 1.5% is an upper bound. And "run the
separation locally for free" presumes an archive of the published beam to fit a separation model on,
which the forecast feed at issue does not carry.

## Limitations

The finding is about six meters inside one 34 km box in Lincolnshire between 2019 and 2026, and the
effective sample is the number of independent weather episodes rather than the 128,000 site-hours.
On the reanalysis the six sites resolve to two grid cells, so the per-site breakdown there is two
irradiance series rather than six replications.

The satellite numbers are conditional on discarding the 17% of hours the service flags as
unreliable, which are darker and more diffuse than the hours kept. The false-zero filter that removes
3.7% of rows was added after the first results existed; matched within irradiance bins it favours
neither arm by more than 0.004 of diffuse fraction, and the pre-cleaning run reaches the same verdict,
but it was not pre-registered.

The second hyperparameter setting exists to check that an arm ordering is a property of the features
rather than of the settings. On the satellite source it passes; on the reanalysis the headline is null
at the primary setting and excludes zero at the sensitivity setting, so that check did not pass there.

Arm B-LEARNED is one gradient-boosted model at the power model's own settings rather than a tuned
separation model, so a better-tuned one would shrink the contrast. What makes that unlikely to change
the conclusion is that B-LEARNED is already 2.4 times more faithful to the published fraction than
Erbs while buying nothing measurable, so power accuracy is close to flat in split fidelity over a wide
range.

## Reproducing this

The code that produced every number and figure on this page lives in a pull request that was
deliberately closed without merging:
[openclimatefix/nged-substation-forecast#785](https://github.com/openclimatefix/nged-substation-forecast/pull/785),
answering [issue #784](https://github.com/openclimatefix/nged-substation-forecast/issues/784). It is
throwaway by design — outside the Dagster asset graph, importing nothing and imported by nothing,
adding no package and changing no data contract. The conclusions are what the project keeps; the
scripts are there so the measurement can be audited and re-run.

Every number quoted here is printed by a script rather than transcribed by hand, and every figure is
drawn from the results files rather than redrawn from a table.
