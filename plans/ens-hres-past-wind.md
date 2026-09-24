# Plan: ECMWF ENS and ECMWF IFS HRES wind in the past-wind study (phase 1)

**Problem.** [Which weather product best describes past wind?](../docs/studies/weather-products-for-past-wind.md)
scores ERA5, UKV, ICON-D2, ICON-EU and ICON global at three wind farms (W1 to W3). It does not
score ECMWF's own forecast products, and the project's live service reads ECMWF ENS. The past-solar
page gained an ENS section in PR #879. Wind has no ENS section, and the page scores no ECMWF IFS
HRES at all (checked: no HRES arm in `wind_products.py`, no HRES text on the page).

**Solution.** Add one page section, one study script and one chart script, copying #879's design
and reusing its tested machinery: an XGBoost model per farm given the mean of ENS's 51 members'
wind, and one given HRES's wind, each scored on the same rows as ERA5, UKV, ICON-EU and ICON-D2,
with five planned contrasts fixed in this file before the first fit.

The issue is not named in the request; the parent studies are #826 (wind) and #784 (ENS).

## Verdict, size and the five trigger answers

Worth doing as described. **Size: complex**, so both plan reviews and both diff reviews run, then
the extra reviews the maintainer asked for (science, personas, prose, pre-merge check).

1. **What gets stored:** yes. A published page section, new outputs under
   `data/studies/beam_diffuse_split/past_weather_v2/ens_hres_past_wind/` and a chart. No Patito
   contract or Delta table.
2. **Production serving path:** no. Study scripts only.
3. **A degradation rule:** no.
4. **More than one defensible design:** yes. The three-hourly-to-hourly treatment, how members are
   combined, and the speed-only versus speed-and-direction column sets each admit alternatives, so
   the choices are fixed below and the alternatives run as exploratory arms.
5. **Callers not nameable without searching:** no. Only new scripts under `studies/`. A small
   addition to `packages/studies/` happens only if a helper turns out reusable; if it does, the
   mutation pass runs on it.

## Planned contrasts (written before any fit)

Every planned contrast uses: ENS band `T+3` (leads 3 to 21 hours from the 00 UTC run, all 51
members, Dynamical.org's archive at 0.25 degrees); members combined by the wind treatment below
before any XGBoost fit ("ensemble mean"); HRES as Open-Meteo's `ecmwf_ifs` hourly wind; every arm
given **speed only**, the hub-height speed and the 10 m speed, because the HRES file carries no
direction. Hub height: 100 m for ENS, HRES, ERA5 and UKV (the page's own choice for those products).

| # | Contrast (first minus second) | Question it answers |
|---|---|---|
| P1 | `ens_mean_t3_wind` − `era5_wind` | Does ENS's ensemble mean describe past wind better than the reanalysis a service could read instead? |
| P2 | `ens_mean_t3_wind` − `ukv_wind` | Does ENS's ensemble mean describe past wind better than the page's product for historical features (UKV)? |
| P3 | `hres_wind` − `era5_wind` | The same question for ECMWF's deterministic forecast. |
| P4 | `hres_wind` − `ukv_wind` | The same, against UKV. |
| P5 | `ens_mean_t3_wind` − `hres_wind` | Ensemble mean against deterministic forecast, both from ECMWF. |

Each is rerun at the second hyperparameter setting (`SENSITIVITY_HYPER_PARAMETERS`). Everything
else is exploratory and labelled so.

**What no planned contrast can separate.** P1 to P5 mix several differences at once, and the page
says so beside each result: served lead (ENS's `T+3` scores leads 3 to 21 hours from a 00 UTC run;
the archived ERA5 and UKV values are an analysis or a T+0 field; HRES's served lead is not
measured here), step width (ENS is 3-hourly, rebuilt to hourly; HRES, ERA5 and UKV are hourly),
native and served resolution (facts to be verified against ECMWF documentation, see below), IFS
model version (the cycle changed during the window), height (all 100 m, but each product's height
is a model diagnostic), and member averaging (P5 confounds ensemble-mean smoothing with HRES being
a single higher-resolution run).

## The wind-specific treatment (fixed before the fit)

**Wind is instantaneous at its label, so the solar clear-sky-index reconstruction does not apply.**
The rebuilt hourly wind comes from the treatment `ens_forecast_horizons.py` selected by its own
pre-registered rule for wind (`COMBINATIONS["wind"]`): per member, speed and direction are turned
into eastward and northward components at each 3-hourly step, the components are interpolated
linearly to the hourly leads 3 to 21, and speed is the magnitude of the interpolated vector. The
ensemble mean speed is the mean of the 51 members' speeds; the direction is the direction of the
mean wind vector (`ens_forecast_horizons.reduce_members`). **Direction is never averaged or
interpolated as a plain number.** Planned arms use speed only, so direction enters no planned arm.

**Exploratory alternatives** (each an XGBoost model on the same rows and the same five columns
unless stated):

- `ens_mean_t3_linear_speed`: interpolate speed linearly instead of via components.
- `ens_mean_t3_on_steps`: scored only on hours that are ENS steps (3-hourly), no interpolation, on
  its own row set (every arm refit on it).
- `ens_control_t3`: the control member alone.
- `ens_mean_t3_dir`, `era5_wind_dir`, `ukv_wind_dir`, `icon_eu_wind_dir`, `icon_d2_wind_dir`:
  speed, direction as sine and cosine, and 10 m speed (seven columns); shows what the speed-only
  restriction costs, which is what HRES gives up.
- `icon_eu_wind`, `icon_d2_wind`, `icon_global_wind` speed-only, as reference rows.

**Hub-height mismatch.** The three farms' hub heights are unknown. As the page's existing arms do,
each arm takes its product's native hub-height speed (100 m) plus its 10 m speed and lets the
per-farm XGBoost model absorb the height mismatch. No shear extrapolation in phase 1.

## Rows, folds, fairness

- **Row set.** Hours where the 00 UTC run's `T+3` band covers the hour (valid hours 03 to 21 UTC,
  the day of the run), every product covers it, and the page's own row rules hold (zero half-hour
  hours dropped, the post-upgrade tail of January 2026 dropped, folds cut inside each UKV era).
  ENS coverage starts on 2024-04-01, the page's window on 2024-08-12 (UKV), and the row set ends
  at the same date as the page (`era5_grid.LAST_DATE`). ENS wind at 22 to 02 UTC is absent from
  the row set by construction, and the page says so.
- **Every arm on exactly the same rows;** equal column counts within a group (speed-only arms
  carry 5 columns, direction arms 7); `colsample_bytree=1`; per-arm column lists printed in the
  report; month-block folds; month-resampled paired bootstrap.
- **Deterministic row fingerprint** (Float32-cast floats before hashing) saved beside the losses,
  checked on `--report-only`.
- **Servable-hours split:** rows ending before the run could have been read (before the
  Dynamical.org archive typically has the 00 UTC run, about 09:00 UTC) versus after, for ENS. HRES's
  own availability delay is quoted from the fact-check and split the same way if it differs.
- **Printed-number guard:** every number on the page traces to `report.md` (a script check fails
  when a number in the new section's text is missing from the report).
- **Three farms.** Every pooled interval on the page carries "three wind farms are few independent
  sites" beside it.

## Product facts to verify before writing (fact-check agent)

Verified against ECMWF, Dynamical.org and Open-Meteo documentation and written to a fact report:
ENS native grid and served grid, runs per day and step widths in the open-data subset, when the 00
UTC run is disseminated and archived, HRES native grid, how Open-Meteo builds and archives hourly
HRES, and IFS cycle dates inside the window. Limits that belong to ECMWF's open-data subset or to
the two archives are attributed to them, not to ENS.

## Files

- `studies/beam_diffuse_split/ens_hres_past_wind.py` (new): build rows, jobs, fingerprint, report;
  imports `ens_forecast_horizons` and `wind_products` public functions where public, and
  `run_experiment.run_all`. Copies the structure of `ens_past_solar.py`.
- `studies/beam_diffuse_split/ens_hres_past_wind_charts.py` (new): charts, in the same pattern as
  `ens_past_solar_charts.py`, days 1 to 7 on the time axis, % of capacity, `aria=False` on marks.
- `docs/studies/weather-products-for-past-wind.md`: a new results section, one "What to use" bullet,
  "Data and methods" and "Limitations" additions, the reproducing commands.
- `docs/studies/assets/`: the new SVGs (optimised with svgo).
- `packages/studies/`: unchanged unless a shared helper is worth moving; tests then added.

## Tests and verification

Study scripts are not unit-tested; their check is their own report. Any helper moved into
`packages/studies/` gets tests, and the mutation pass. Verification: `ruff check`, `ruff format`,
`ty check`, `pytest`, `pymarkdown scan`, `mkdocs build --strict`, `check_docs_links.py`,
`pydoclint`, and reading the built HTML under `site/`.

## Risks and open questions

- **HRES has no direction.** Recommendation: planned arms all speed-only, direction arms
  exploratory (as above).
- **HRES grid cell.** The file is a 0.05-degree grid of the trial-area box without coordinates.
  The script recovers the coordinates from the box definition, reads the nearest point, and prints
  no coordinate. Its value is checked against the neighbouring points.
- **HRES's served lead is unmeasured** unless the fact-check finds documentation. The page says so.
- **T+3 covers 19 hours a day.** Night-time hours (22 to 02 UTC) are absent, so wind's diurnal
  shape is partly missing from the row set.

## Reviews this plan buys

Both plan reviews (simplicity, then correctness); after implementation both diff reviews (Opus code
review, then Opus mutation review), then the maintainer's further list.
