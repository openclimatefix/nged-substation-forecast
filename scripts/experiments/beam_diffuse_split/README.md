# Throwaway experiment: does the beam/diffuse split help an XGBoost PV forecast?

Everything in this directory is a one-off. It is not part of the Dagster asset graph, nothing else
in the repo imports it, it adds no package and it changes no data contract. It exists to answer one
question recorded in [issue
784](https://github.com/openclimatefix/nged-substation-forecast/issues/784), and the answer — not
the code — is what gets kept.

**The question: does ERA5's own direct-beam field carry information an XGBoost PV forecast can use,
beyond what global horizontal irradiance already tells it?** The decision it feeds is whether to ask
Dynamical.org to change the ECMWF feed we ingest, since the free open-data feed carries no direct
beam at all.

ERA5 is the source because it publishes `ssrd` (global horizontal irradiance) and `fdir` (the direct
beam's flux onto a horizontal plane) from the same radiation scheme, on the same grid, at the same
time. That is what makes the comparison clean: the two arms differ in what the model is shown and in
nothing else.

**ERA5 is a reanalysis, so this measures the information content of the split, not forecast skill.**
A forecast of `fdir` would carry its own error, which this experiment says nothing about.

## The scripts, in the order they run

| Script | What it does |
|---|---|
| `era5_grid.py` | The grid and date range both downloads share, so "the same cells" is checkable rather than a coincidence of two literals. |
| `fetch_era5.py` | Downloads `ssrd`, `fdir` and `t2m` from the Copernicus Climate Data Store, six months per request, into `data/ERA5/beam_diffuse/`. Resumable. Source of the headline result. |
| `fetch_era5_open_meteo.py` | Downloads the same fields from Open-Meteo's ERA5 mirror onto the same grid, in about a minute rather than most of a night. |
| `verify_era5_sources.py` | Compares the two downloads hour by hour, which is what establishes that the mirror serves ERA5's own `fdir` rather than a separation model's estimate of it. |
| `build_dataset.py` | Joins the PV power readings to ERA5, adds solar geometry, the separation-model estimates and the synthetic control target, and writes the one frame every arm reads. Takes `--source`. |
| `run_experiment.py` | Fits every arm, at every fold, seed and hyperparameter setting, and writes per-row losses, per-site metrics and bootstrap intervals. Takes `--source`. |
| `report_results.py` | Prints the markdown tables the write-up quotes, so no number is transcribed by hand. |
| `make_chart.py` | Draws the anonymised result chart. |

**Two ERA5 downloads, one experiment.** The Copernicus Climate Data Store is the source of the
headline result, and Open-Meteo's mirror of the same reanalysis is run beside it as a replication.
The mirror exists here because the Climate Data Store runs one of an account's jobs at a time and
takes around five minutes per month of hourly fields, so seven years is most of a night. Reporting
both costs one extra run and says whether the answer survives a change of delivery route.

## The arms

Identical rows, folds, seeds, hyperparameters and non-irradiance features. Only the irradiance
columns change.

| Arm | Irradiance features |
|---|---|
| A — global only | `ssrd` |
| B — separation model | `ssrd`, plus the beam and diffuse horizontal fluxes Erbs derives from `ssrd` |
| C — the model's own split | `ssrd`, `fdir`, and `ssrd − fdir` |
| D — direct fraction | `ssrd` and `fdir / ssrd` |
| B-DISC — sensitivity | arm B with the DISC separation model in place of Erbs |

Every beam column is a flux onto a horizontal plane, never a direct-normal one, so no two arms
differ in how a quantity is encoded as well as in what it knows.

**Arm C against arm B is the comparison the experiment exists for, and arm B is a negative control
the experiment gets for free.** Erbs reads global irradiance and solar geometry and nothing else,
all of which arm A already holds, so arm B cannot carry information arm A lacks. Whatever B−A comes
out as is this pipeline's reading on a feature set known to be uninformative, and it is the band any
real effect has to clear. Arm C adds a quantity the radiation scheme computed and `ssrd` alone does
not carry, and has the same column count as arm B.

**A second control runs every arm against a synthetic target built by transposing the true split
onto a tilted plane**, where the split must help by construction. A null result on the real meters
means nothing until the instrument has been shown to detect an effect it should detect.

## Anonymisation

These are metered generators, whose output is commercially sensitive, so no site name and no
`time_series_id` may appear in any chart, comment or write-up. `build_dataset.py` relabels the sites
`A`–`F` before writing anything, and nothing downstream of it sees an identifier.

## Reading the numbers

Six sites inside a 34 km box share their weather, so the effective sample size is the number of
independent weather episodes rather than the number of site-hours. The bar for a result is a
monthly block bootstrap interval on the arm-to-arm difference that excludes zero. A point estimate
on its own is not a result, and neither is a difference smaller than the seed-to-seed spread that
`run_experiment.py` reports alongside it.

Whatever the answer, it is about one micro-region of Lincolnshire over 2019 to 2026.
