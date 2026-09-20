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

## The arms

Identical rows, folds, seeds and non-irradiance features. Only the irradiance columns change.

| Arm | What the model is shown |
|---|---|
| A — global only | Global horizontal irradiance |
| B — separation model | Global irradiance, plus the beam and diffuse horizontal fluxes Erbs derives from it |
| C — the product's own split | Global irradiance, the published beam, and the difference |
| D — direct fraction | Global irradiance and the published beam's share of it |
| B-DISC — sensitivity | Arm B with the DISC separation model in place of Erbs |

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
monthly block bootstrap interval on the arm-to-arm difference that excludes zero. A point estimate
on its own is not a result, and neither is a difference smaller than the positive control's own
arm-to-arm difference. The seed-to-seed spread the runners report means different things for the
two instruments: for XGBoost it measures how much of a difference is fitting noise, and for the
physical model it only measures how far the optimiser's restarts wander, which is a few parts in a
million.

Whatever the answer, it is about one micro-region of Lincolnshire over 2019 to 2026.
