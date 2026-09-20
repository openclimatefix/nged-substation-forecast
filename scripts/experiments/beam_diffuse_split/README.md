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
| `fetch_era5.py` | Downloads `ssrd`, `fdir` and `t2m` from the Copernicus Climate Data Store, one request per calendar month, into `data/ERA5/beam_diffuse/`. Resumable. |
| `build_dataset.py` | Joins the PV power readings to ERA5, adds solar geometry and the separation-model estimates, and writes the one frame every arm reads. |
| `run_experiment.py` | Fits every arm, at every fold, seed and hyperparameter setting, and writes per-row losses, per-site metrics and bootstrap intervals. |
| `make_chart.py` | Draws the anonymised result chart. |

## The arms

Identical rows, folds, seeds, hyperparameters and non-irradiance features. Only the irradiance
columns change.

| Arm | Irradiance features |
|---|---|
| A — global only | `ssrd` |
| B — separation model | `ssrd`, plus the direct-normal and diffuse-horizontal estimates Erbs derives from `ssrd` |
| C — the model's own split | `ssrd`, `fdir`, and `ssrd − fdir` |
| D — direct fraction | `ssrd` and `fdir / ssrd` |
| B-DISC — sensitivity | arm B with the DISC separation model in place of Erbs |

**Arm C against arm B is the comparison the experiment exists for.** Arm B only re-expresses
information arm A already holds, because Erbs is a deterministic function of the clearness index and
the sun's position, both of which arm A can reach. Arm C adds a quantity the radiation scheme
computed and `ssrd` alone does not carry.

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
