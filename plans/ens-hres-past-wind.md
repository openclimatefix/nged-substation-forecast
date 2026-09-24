# Plan: ECMWF ENS and ECMWF IFS HRES wind in the past-wind study (phase 1)

**Problem.** [Which weather product best describes past wind?](../docs/studies/weather-products-for-past-wind.md)
scores ERA5, UKV, ICON-D2, ICON-EU and ICON global at three wind farms (W1 to W3). The page scores
no ECMWF IFS HRES (checked: no HRES arm in `wind_products.py`, no HRES text on the page) and no
ECMWF ENS. The past-solar page has both. The ENS horizons page compares ENS with ERA5 for wind at
day 0 and day 1, but not with UKV, ICON, or HRES.

**Solution.** One new study script that reads the page's own row set, adds an HRES arm and an ENS
ensemble-mean arm, and refits every product on the rows all of them cover. It scores five
contrasts fixed in this file before any fit. It reuses the horizons study's ENS wind inputs and the
`wind_icon_dream.py` structure, so no new ENS code is written. One results section, one "What to
use" bullet and one chart script follow.

## Verdict, size and the five trigger answers

Worth doing, with the design changes below (adopted from the first plan review). **Size: complex**,
so both plan reviews and both diff reviews run, then the reviews the maintainer asked for.

1. **What gets stored:** yes. A published page section and new outputs under
   `data/studies/beam_diffuse_split/past_weather_v2/ens_hres_past_wind/`. No Patito contract, no
   Delta table.
2. **Production serving path:** no. Study scripts only.
3. **A degradation rule:** no.
4. **More than one defensible design:** yes. Which ENS band, how the 3-hourly steps become hourly
   wind, and which HRES source. The choices are fixed below and the alternatives run as
   exploratory arms.
5. **Callers not nameable without searching:** no. New scripts only. `packages/studies/` changes
   only if a helper proves reusable; the mutation pass then runs on it.

## Departures from the request, for the coordinator to veto

- **HRES source.** The request named the 9 km grid file, which holds wind speed only, in km/h. The
  Previous Runs file `data/studies/weather/ECMWF-IFS-HRES/previous_runs/combined.parquet` holds each
  farm's HRES wind speed and direction at 10 m and 100 m, hourly, from the same `ecmwf_ifs` model
  (bare columns are the freshest run). Using it keeps every arm at the page's seven columns and
  needs no grid-coordinate recovery. The script checks that its speed agrees with the grid file at
  the nearest point, and the page cites both files. The grid file is otherwise unused.
- **ENS band.** The request listed the `T+3` wind file. That file covers leads 3 to 21, so valid
  hours 03 to 21 UTC only, and its download script is not in the repository. The ENS horizons study
  already holds the ensemble-mean wind for the same download at day 0 (leads 0 to 23, every hour
  of the day), rebuilt to hourly by the treatment its own pre-registered rule chose, with the
  tested code and saved inputs (`data/studies/ens_forecast_horizons/wind_inputs.parquet`). The
  plan reads day 0 from there. The `T+3` wind file goes unused.
- **P1 replicates a published number.** The horizons page already reports the day-0 ensemble mean
  against ERA5 for wind. That contrast is labelled a replication, not a planned contrast.

## Planned contrasts (written before any fit)

Every planned contrast uses: ENS band **day 0** (valid hours 00 to 23 UTC of the run's own day,
leads 0 to 23 hours from the 00 UTC run, Dynamical.org's archive, 0.25 degrees, all 51 members);
members combined as the horizons study does (`reduce_members`: the mean of the members' speeds, the
direction of the mean wind vector) after each member's 3-hourly steps are rebuilt to hourly by the
wind treatment below; HRES as Open-Meteo's freshest-run `ecmwf_ifs` hourly wind; every arm given
the page's seven columns (`SHARED_FEATURES` plus `_wind_columns`), at hub height 100 m for HRES, ENS,
ERA5 and UKV.

| # | Contrast (first minus second) | Question it answers |
|---|---|---|
| P1 | `hres_wind` − `era5_wind` | Does ECMWF's deterministic forecast describe past wind better than the reanalysis? |
| P2 | `hres_wind` − `ukv_wind` | Does it describe past wind better than UKV, the page's product for historical features? |
| P3 | `ens_mean_day0_wind` − `ukv_wind` | Does ENS's ensemble mean describe past wind better than UKV? |
| P4 | `ens_mean_day0_wind` − `hres_wind` | Ensemble mean against deterministic forecast, both from ECMWF. |

Reported as an exploratory replication: `ens_mean_day0_wind` − `era5_wind`. Every contrast is
rerun at the second hyperparameter setting (`SENSITIVITY_HYPER_PARAMETERS`).

**What no planned contrast can separate.** Each contrast mixes: served lead (ENS day 0 spans leads
0 to 23 from a 00 UTC run; HRES's freshest-run leads are 1 to 12 hours before 1 October 2025 and
1 to 6 after, per the solar page; ERA5 is an analysis; UKV is T+0); step width (ENS is 3-hourly,
rebuilt to hourly); native and served resolution (both about 9 km native; ENS served at 0.25
degrees, HRES from 1 October 2025 on the native grid, before then not established); IFS cycle (49r1
until 12 May 2026, then 50r1); the source of HRES's archive (changed on 1 October 2025); height;
and, for P4, ensemble averaging against a single run. The page says so beside each result.

## The wind-specific treatment (fixed before the fit)

**Wind is instantaneous at its label, so the solar clear-sky-index reconstruction does not apply.**
The planned ENS input is the horizons study's `speed_components` combination: each member's speed
and direction become eastward and northward components at each 3-hourly step, the components are
interpolated linearly to hourly, and the hourly speed is the magnitude of the interpolated vector;
direction is interpolated as an angle, handled circularly, and enters as sine and cosine. The horizons
study chose that combination by a rule written before its results. Direction is never averaged or
interpolated as a plain number. Exploratory alternatives, each an XGBoost model on the same rows
and columns: `components` (direction also through components), `direction_components`, and
`linear`; each reads the same saved horizons inputs.

**Hub-height mismatch.** The three farms' hub heights are unknown. As the page's existing arms do,
each arm takes its product's native hub-height speed and its 10 m speed, and the per-farm XGBoost
model absorbs the mismatch. Phase 1 adds no shear extrapolation.

## Rows, folds, fairness

- **Row set.** The page's rows (`wind_products.common_rows(joined(...))`) joined to ENS day 0 and
  HRES, from 1 December 2024, the first whole month after IFS Cycle 49r1 (12 November 2024), to the
  page's end date. The row set starts later than the page's (12 August 2024) because HRES's served
  grid before 2025 is not established, as on the solar page. Every product is refit on this row
  set.
- **Arms.** Every product's wind arm at the primary setting (ERA5, UKV, ICON-D2, ICON-EU, ICON
  global, HRES, ENS); the second setting for the four products in the planned contrasts and ERA5.
- **Fairness.** Every arm has exactly the same rows and seven columns; `colsample_bytree=1`;
  per-arm column lists in the report; folds cut inside each UKV era (as the page); month-resampled
  paired bootstrap of whole months and seeds.
- **Row fingerprint.** Deterministic, floats cast to Float32 before hashing, saved beside the
  losses, and checked on `--report-only`.
- **Servable-hours split.** ENS scored hours ending before the run could have been read (before
  about 09:00 UTC) against after. HRES's archive delay is not established, so HRES gets no split.
- **Period splits (exploratory):** before and after 1 October 2025 (HRES's archive source
  changed), and before and after 12 May 2026 (IFS Cycle 50r1), from the saved losses.
- **Printed-number guard.** A check fails when a number in the new section is missing from
  `report.md`.
- **Three farms.** Every pooled interval on the page states that three wind farms are few
  independent sites.

## Product facts to verify before writing

A fact report (`scratch/facts-report.md`) covers the ENS and HRES grids, runs, step widths,
dissemination and archive timing, and IFS cycle dates. Its quotes were summarised by a small
model, so an Opus science reviewer re-checks every fact against ECMWF's, Dynamical.org's and
Open-Meteo's pages. Limits belonging to ECMWF's open-data subset or to either archive are
attributed to them, not to ENS.

## Files

- `studies/beam_diffuse_split/ens_hres_past_wind.py` (new): build rows, jobs, fingerprint, report,
  copying `wind_icon_dream.py`'s structure; imports the Float32 fingerprint and printed-number
  check from `ens_past_solar.py` and `ens_past_solar_charts.py` where importable.
- `studies/beam_diffuse_split/ens_hres_past_wind_charts.py` (new): leaderboard, planned contrasts,
  and a days-1-to-7 out-of-fold check for the two new arms (% of capacity, `aria=False`).
- `docs/studies/weather-products-for-past-wind.md`: one results section, a Key-findings bullet, a
  "What to use" bullet, two product-table rows, "Data and methods" and "Limitations" additions,
  and the reproducing commands.
- `docs/studies/assets/`: the new SVGs, optimised with svgo.
- `packages/studies/`: unchanged unless a helper proves reusable.

## Tests and verification

Study scripts are not unit-tested; the report is their check. Any helper moved into
`packages/studies/` gets tests and the mutation pass. Verification: `ruff check`, `ruff format`,
`ty check`, `pytest`, `pymarkdown scan`, `mkdocs build --strict`, `check_docs_links.py`,
`pydoclint`, and reading the built HTML under `site/`.

## Risks and open questions

- **HRES's archive source changed on 1 October 2025**, so HRES's lead and grid differ across the
  row set. The period split reports it; the page names it as a confound.
- **Day 0 is a best case.** Most day-0 hours end before the 00 UTC run can be read, so the ENS arm
  is past weather delivered late, as the horizons page says.
- **The IFS cycle changes inside the row set** on 12 May 2026.

## Reviews this plan buys

Both plan reviews, then both diff reviews, then two science reviews, the persona reviews, a prose
sweep and a pre-merge check of the rendered HTML.

## Findings rejected

- Reading ERA5, UKV and the ICON arms from the page's published losses: rejected because the row
  set starts later, so their losses are refit.
