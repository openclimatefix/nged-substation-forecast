# Throwaway experiment: does the beam/diffuse split help a PV forecast?

Everything in this directory is a one-off. It is not part of the Dagster asset graph, nothing else
in the repo imports it, it adds no package and it changes no data contract. It exists to answer one
question recorded in [issue
784](https://github.com/openclimatefix/nged-substation-forecast/issues/784), and the answer — not
the code — is what gets kept.

**The question: does a weather product's own direct-beam field carry information a PV power model
can use, beyond what global horizontal irradiance already tells it?** The decision it feeds is
whether to ask Dynamical.org to change the ECMWF feed we ingest, since the free open-data feed
carries no direct beam at all.

**Neither ERA5 nor the CAMS radiation service is a forecast, so this measures the information
content of the split rather than forecast skill.** A forecast of the beam field would carry its own
error, which this experiment says nothing about.

## Two questions the design has to keep apart

**Whether the split helps at all is a different question from whether a product's published beam
field is the best way to get it.** The physical route from irradiance to power runs through the
plane of array, where the beam is projected onto the panel by the cosine of its incidence angle and
the diffuse is not. A model given only global horizontal irradiance cannot do that projection. A
model given a split can, and it can get the split either from the product or from a separation
model run on the product's own global irradiance.

So the arms are built around two contrasts. Everything measured against arm A says what
*transposition* buys. The arm given the product's split, measured against the arm given a
separation model's estimate of the same split, says what the *published field* buys on top.

## Three sources, two instruments, two stamp alignments

| Dimension | Values | What changing it tests |
|---|---|---|
| Source | Copernicus ERA5, Open-Meteo's ERA5 mirror, CAMS radiation service | Whether an answer from ERA5's 31 km grid survives a 5 km satellite retrieval at the meter itself |
| Instrument | XGBoost, a fitted five-parameter PV model | Whether a null result means the split carries nothing or that the tree could not use it |
| Stamp alignment | as-labelled, shifted 30 minutes earlier | Whether the answer depends on a timestamp convention the power feed may have got wrong |

## The scripts, in the order they run

| Script | What it does |
|---|---|
| `era5_grid.py` | The grid and date range both ERA5 downloads share, so "the same cells" is checkable rather than a coincidence of two literals. |
| `fetch_era5.py` | Downloads `ssrd`, `fdir` and `t2m` from the Copernicus Climate Data Store, six months per request, into `data/ERA5/beam_diffuse/`. Resumable. |
| `fetch_era5_open_meteo.py` | Downloads the same fields from Open-Meteo's ERA5 mirror onto the same grid, in about a minute rather than most of a night. |
| `verify_era5_sources.py` | Compares the two ERA5 downloads hour by hour, which is what establishes that the mirror serves ERA5's own `fdir` rather than a separation model's estimate of it. |
| `fetch_cams.py` | Downloads the CAMS radiation service's global, beam and diffuse irradiances at each meter's own coordinates, into `data/CAMS/`. |
| `build_dataset.py` | Joins the PV power readings to one source, adds solar geometry, the separation-model estimates and the synthetic control target, and writes the one frame every arm reads. Takes `--source` and `--alignment`. |
| `run_experiment.py` | The XGBoost instrument: fits every arm at every fold, seed and hyperparameter setting, and writes per-row losses, per-site metrics and bootstrap intervals. |
| `physics_model.py` | The transposition and the temperature-corrected power curve the second instrument fits. |
| `run_physics_experiment.py` | The physical instrument: fits five parameters per site per training fold and scores the held-out fold, on the same rows and folds. |
| `report_results.py` | Prints the markdown tables the write-up quotes, so no number is transcribed by hand. Takes `--instrument`. |
| `compare_sources.py` | Compares two sources on the hours they both cover, which the per-source tables cannot do. |
| `elevation_breakdown.py` | Splits the headline contrast by solar elevation, to separate an amplified error from a missing one. |
| `make_chart.py` | Draws the anonymised result chart. |
| `sky_conditions.py` | Splits the headline contrast by clearness index, which tests whether the gain sits where cloud makes the split uncertain — the shape the information account predicts and a calibration difference would not. |
| `run_hybrid_experiment.py` | Feeds the physical model's out-of-fold prediction into XGBoost, to separate "the physical model is mis-calibrated" from "its five parameters are the wrong shape". Withholds each prediction by calendar month, the same way the learned separation arm does. |
| `make_figures.py` | Draws the per-site time series, the per-site error chart and the sky-condition chart the write-up publishes. |
| `inverter_clipping.py` | Splits the headline contrast by whether the meter was sitting on its inverter ceiling. A clipped hour cannot respond to irradiance, so this separates "clipping dilutes the effect" from "clipping manufactures it". |
| `anm_curtailment.py` | Reads NGED's `curtailment/` feed — the active-network-management log nothing else in the repository ingests — and tests it against the one curtailed site's output shortfall. Needs the NGED bucket credentials the other scripts do not. |
| `anm_setpoints.py` | Turns NGED's raw active-network-management setpoint export into a half-hourly export-cap series, and checks how the cap reads: a generator sitting at the largest cap it ever sees is unconstrained, not fully curtailed. Reaches 26 months where the bucket feed reaches five. |
| `export_cap.py` | Joins that export cap onto the modelling dataset and marks the hours the operator had moved it. Imported by all three runners: the curtailed hours are dropped from every training fold, and the predictions are held down to the cap at scoring time. |

## The arms

Identical rows, folds, seeds and non-irradiance features. Only the irradiance columns change.

| Arm | What the model is shown |
|---|---|
| A — global only | Global horizontal irradiance |
| B — separation model | Global irradiance, plus the beam and diffuse horizontal fluxes Erbs derives from it |
| C — the product's own split | Global irradiance, the published beam, and the difference |
| D — direct fraction | Global irradiance and the published beam's share of it |
| B-DISC — sensitivity | Arm B with the DISC separation model in place of Erbs |
| B-LEARNED — the discriminator | Arm B with a fitted separation model in place of Erbs: an out-of-fold prediction of the published direct fraction from arm A's own feature set |

**Arm B-LEARNED separates the two reasons arm C could win, which carry opposite decisions.** The
published beam may hold information no function of global irradiance and solar geometry can
recover, and that field is worth asking a supplier for. Or the product may merely publish a better
separation model than Erbs, and then the same gain is available locally for nothing. Erbs alone
cannot tell those apart, because Erbs is one fixed correlation from 1982 rather than the best
correlation this data supports. Arm B-LEARNED is a much more faithful one — on the satellite source
it leaves 0.04 of the published direct fraction's variance unexplained against Erbs's 0.09 — and
every value it carries is a function of what arm A already holds, so arm C beating it is evidence
of information rather than of representation.

**The withholding is by calendar month rather than by fold label, and it covers the training rows
as well as the scored fold.** Folds are cut inside each site's own span, so one fold number is a
different calendar period at each site; a model that dropped only the rows carrying that fold
number would still train on other sites' rows at the scored fold's own hours, and on the reanalysis
those other sites are the same grid cell. Training rows are withheld too, by an inner
cross-validation, because a column that is sharper where the arm trains than where it is scored
gets over-trusted by the power model and the arm is then penalised for a reason unrelated to the
split.

Two limits on what this arm settles. It is one XGBoost at the power model's own settings rather
than a tuned separation model, so a better-tuned one would shrink the contrast — what makes that
unlikely to matter is that B-LEARNED is 2.4 times more faithful to the published fraction than Erbs
and buys almost nothing in power error, so power accuracy is close to flat in split fidelity over a
wide range. And "the same gain is available locally" presumes an archive of the published beam to
fit a separation model on, which the forecast feed this decision concerns does not carry.

The physical instrument runs the same arms under the names `P-A` to `P-C`, plus `P-E`, which is
handed all three beam estimates at once and fits the weights of a convex combination over them. Arm
`P-A` is given no split, and for it the plane-of-array irradiance is the global horizontal
irradiance itself: with one number there is no transposition to do.

Every beam column is a flux onto a horizontal plane, never a direct-normal one, so no two arms
differ in how a quantity is encoded as well as in what it knows.

**Arm C against arm B is the comparison the experiment exists for, and arm B is a negative control
the experiment gets for free.** Erbs reads global irradiance and solar geometry and nothing else,
all of which arm A already holds, so arm B cannot carry information arm A lacks. Whatever B−A comes
out as is the pipeline's reading on a feature set known to be uninformative, and it is the band any
real effect has to clear.

**A second control runs the arms against a synthetic target built by transposing the true split onto
a tilted plane**, where the split must help by construction. A null result on the real meters means
nothing until the instrument has been shown to detect an effect it should detect, and what the
control reports is the size of the difference each instrument produces when the split genuinely
matters — the threshold below which a difference on the real meters says nothing. The target is
built outside the physical model's hypothesis class on purpose: each site gets its own tilt and
azimuth, none of them the values the optimiser starts from, and the sky diffuse is transposed by the
Hay-Davies model where the instrument assumes an isotropic sky.

## Anonymisation

These are metered generators, whose output is commercially sensitive, so no site name, no
`time_series_id` and no site coordinate may appear in any chart, comment or write-up. `_pv_sites`
in `build_dataset.py` relabels the sites `A`–`F` under a fixed permutation before anything is
written, and nothing downstream of it sees an identifier. `fetch_cams.py` sends each meter's
coordinates to the Atmosphere Data Store because the service is a point service, reads them at run
time from the private roster, and writes only the anonymised label.

## Reading the numbers

Six sites inside a 34 km box share their weather, so the effective sample size is the number of
independent weather episodes rather than the number of site-hours. The bar for a result is a
monthly block bootstrap interval on the arm-to-arm difference that excludes zero, and a point
estimate on its own is not a result. The seed-to-seed spread the runners report means different
things for the two instruments: for XGBoost it measures how much of a difference is fitting noise,
and for the physical model it only measures how far the optimiser's restarts wander, which is a few
parts in a million.

**What the positive control licenses is a statement about detection, not a threshold a real effect
has to clear.** The control's target is a transposition of the true split plus Gaussian noise at 2%
of each site's 99th-percentile output, so the noise alone fixes a floor of about 1.59 percentage
points on any arm's error, and the whole span between the global-only arm and perfection is about
0.38 points. The control's own arm-to-arm difference is therefore a *ceiling* — the largest
difference this instrument could report on a target that is a pure function of the split — and
requiring a real effect to exceed it would demand that real weather beat a synthetic target with no
physics in it but transposition. It would also make the bar a free parameter, because halving the
control's noise widens its arm-to-arm difference without anything about the real measurement
changing. The control answers one question: whether the instrument detects an effect of this kind
at all. Report the real effect against the error it removes, and say what fraction of the control's
difference it comes to, rather than treating that fraction as a pass mark.

Whatever the answer, it is about one micro-region of Lincolnshire over 2019 to 2026.

**"Percentage of capacity" here means percentage of the site's 99th-percentile metered output.**
The denominator is `effective_capacity_mw`, which `ml_core.metrics` computes as the 99th percentile
of each series' own absolute output over its whole history. It is not the registered capacity, and
because it is a statistic of the target it is not a data-independent unit. Every arm is divided by
the same number, so it cannot manufacture a contrast.

## What each reading does not settle

**The fitted physical model is a misspecification probe, not a second reading of the same
quantity.** Its two arms do not differ only in the beam field they are handed: the arm given the
product's own split settles on a tilt 3 to 11 degrees shallower than the arm given Erbs, in every
run and at every site, so the arms differ in fitted geometry as well. A 30-minute stamp shift is
absorbed into the fitted azimuth — the shifted runs settle around 162 to 179 degrees and the
as-labelled runs around 200 to 212 — and the ordering of the arms changes with it, in sample as
well as out. So the physical model answers "which beam field lets a five-parameter isotropic-sky
model with a fitted azimuth fit best", and its answer moves with a timestamp convention. Do not
reconcile its sign with the tree's; report what each instrument measured.

**The physical model also amplifies beam error where the tree does not.** `MIN_COS_ZENITH` floors
the divisor that converts a horizontal beam to a normal one, so a beam error near the horizon is
magnified up to twentyfold, and the tree never performs that division at all. The penalty is
largest in the lowest elevation band but it is present in every band, so low-sun amplification is
part of the physical model's penalty rather than all of it.

**The satellite run discards the hours the service itself flags, and the reanalysis run discards
none.** `MIN_CAMS_RELIABILITY` removes about 17% of the satellite source's daylight hours, and
those hours are darker and more diffuse than the ones kept — the regime where no arm can be far
wrong. The reanalysis has no equivalent flag, so the two sources' row sets differ for that reason
as well as through their own false-zero filters. `compare_sources.py` exists to settle whether a
source difference survives a common row set.

**On the reanalysis the six per-site rows are two irradiance series, not six replications.** The
roster's six meters fall inside two ERA5 grid cells, and within each group the global irradiance is
bit-identical, so the per-site table is two experiments run three times each against a different
power target. The satellite source is clean here, because it retrieves at each meter's own
coordinates. The pooled bootstrap is unaffected either way, because it resamples whole calendar
months across all six sites at once.

**The second hyperparameter setting did not do its job on the reanalysis.** It exists to check that
the arm ordering is a property of the features rather than of the settings, and on the reanalysis
the headline contrast is null at the primary setting and excludes zero at the sensitivity setting.
Both are reported; the check simply did not pass there.

**The false-zero filter was added after the first results existed.** It removes about 3.7% of
daylight rows, three times what the outage filter removes, and it was written once the first run
had already produced tables. Matched within irradiance bins the dropped rows' diffuse fraction
differs from the kept rows' by about 0.004, so it favours no arm, and the pre-cleaning run reaches
the same verdict — but a filter chosen after seeing results has to be declared rather than
defended.
