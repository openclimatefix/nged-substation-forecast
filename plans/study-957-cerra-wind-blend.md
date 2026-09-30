# Plan: does blending CERRA's wind levels beat CERRA 100 m wind alone? (#957)

**The problem.** CERRA, the Copernicus regional reanalysis for Europe, is on disk at five wind heights (10 m, 50 m, 75 m, 100 m, and 150 m) at 190 grid cells around the trial area, every 3 hours from 2019-09-01 to 2026-06-30. Nobody has tested which of those heights an XGBoost model should be given to predict metered wind power. The maintainer also asked how much skill 100 m wind adds over 10 m wind in a simple XGBoost model.

**The plan.** Fit one XGBoost model per wind farm per arm, where an arm is one choice of CERRA wind columns, on the same 3-hourly farm-hours for every arm, with column subsampling off and every arm padded to the same column count. Four contrasts are written down here before any fit: 100 m against 10 m, a mean of the near-100 m levels against 100 m, all four heights of 50 m to 150 m as separate columns against 100 m, and all five heights against 100 m. The shared loader for CERRA wind is owned by #968 and lands first, so implementation starts only after that loader's pull request has merged, and this pull request stays plan-only until then. The result is a new study page, `docs/studies/past-weather/cerra-wind-levels.md`, so the page `docs/studies/past-weather/wind.md` that #968 is editing stays untouched.

## Verdict, size and departures

**Verdict: worth doing, roughly as described.** The data preconditions hold: #969 is closed with all five files validated (3,792,400 rows each, no gaps, duplicate keys, or nulls), and `origin/main` is at 45becc59. No open pull request or branch mentions #957. The question is distinct from #968, which scores CERRA's 100 m wind as one more product in the wind page's ranking and does not test blends.

**Size: complex.** The five triggers, answered one by one:

- **What gets stored:** fires. A published study page counts as stored, and the study saves per-row out-of-fold losses and interval tables under `data/studies/`.
- **Production serving path:** does not fire. Throwaway study scripts only; nothing under `src/` or the served forecaster changes.
- **A degradation rule:** does not fire. R&D code, which fails fast.
- **More than one defensible design:** fires. The definition of "a blend", how the 10 m-only arm is padded to the same column count, and the row set each admit several defensible answers (see Risks).
- **Code whose callers cannot be named without searching:** does not fire. New scripts only, with no importers, and no edit to `packages/`.

One trigger is enough, and two fire. The issue therefore gets the full routine: both plan reviews, then both diff reviews from `implement-issue`, and the `study` skill's two Opus scientific-validity reviews, the persona reviews, and a `prose-review` before the page merges. The mutation pass runs only if `packages/studies/` changes; this plan changes nothing there, so the PR body must say so.

**Departures from the issue body.**

- The issue names "the three wind farms the matched-lead and past-wind studies already cover" as the test set; this plan keeps that, labelled W1 to W3.
- The issue says "a blend of levels" without defining it; the plan defines two (a mean and a learned combination), because one definition could hide the effect of the other.
- The issue does not mention the maintainer's 10 m question, which the brief adds; the plan makes it planned contrast 1.

## Facts about the data that shape the design

- **CERRA's wind files hold speed only, with no direction.** Every arm therefore carries speed columns and no direction, unlike the wind page's arms, which carry a direction as sine and cosine. Contrasts on this page are not comparable in level with the wind page's errors, and the page says so.
- **The values are 3-hourly analyses (00, 03, ..., 21 UTC), instantaneous at their label.** The row set therefore has one hour in three, at most 8 per farm-day. Power for the hour labelled T is built from the half-hours ending at T and T + 30 minutes, as `wind_products._hourly_power` does with `centred=True`. The power-hour offset is scanned for CERRA before the first fit, as the `study` skill requires for every product.
- **Timestamps are timezone-naive UTC** (`validation_wind.json`, `lineage_*.json`).
- **The grid is 5.5 km, read at each farm's nearest cell, and the wind farms' cells are not yet derived.** `generator_cells.parquet` holds 6 rows, for the solar farms only, so the wind farms' cells come from the loader (owned by #968), which derives them from `cerra_grid.parquet` (columns `y_index`, `x_index`, `latitude`, `longitude`). The implementation asserts that each farm's cell lies strictly inside the 190-cell crop, not on its edge, so a farm outside the crop cannot silently read an edge cell. The report prints only pooled distance ranges, never a coordinate or a cell index.
- **`cerra_grid.parquet` has no land-sea flag**, so the plan makes no claim about whether a cell is land or sea. A nearest cell near the coast could be influenced by the sea, as the wind page found for ICON global. Checking that needs CERRA's land-sea mask, a small download that the Data DOWNLOAD COORDINATOR owns; the plan asks the maintainer before any fetch, and the page states the gap as a limitation if no mask is fetched.
- **The 10 m level is a different CERRA product from the other four.** The 10 m speed comes from `reanalysis-cerra-single-levels`, a surface diagnostic, and the 50 m to 150 m speeds come from `reanalysis-cerra-height-levels`, which reads model levels. Planned contrast 1 therefore compares a diagnostic with a model-level value as well as two heights, and the page says so wherever it quotes that contrast.
- **`hour_of_day` takes only 8 values (0, 3, ..., 21 UTC) on the 3-hourly rows.** The page states this in "Data and methods".
- **An era check is a gate before the first fit.** The CDS record may join two production streams (likely around 2021). Before any fit, the implementation reads the CDS dataset documentation and computes each height's monthly means at the farms' cells, and looks for a step at any candidate date. If a join exists, the folds are cut inside each era (`assign_folds(by=("site", "era"))`), every arm gets an era column (its width counted in the padding), and the part-month at the join is dropped. The yearly means in `validation_wind.json` show no jump, which is why the plan calls this a gate rather than a finding.

## What changes, file by file

All new files. Nothing under `packages/`, `src/`, or `docs/studies/past-weather/wind.md` is edited.

- **`studies/beam_diffuse_split/cerra_wind_levels.py`** (new). The fit script, in the style of `wind_products.py` and `cerra_past_solar.py`:
  - It imports the shared CERRA wind loader from `packages/studies/`, which #968 adds with tests. The loader reads the five parquet files, derives the wind farms' nearest cells, pivots the heights into columns, and builds the centred power hour. This script writes no loader of its own. It joins each farm's `effective_capacity_mw`, and relabels farms W1 to W3 with `studies.anonymise.site_labels_for` before anything is written. If the loader's interface differs from what this plan assumes, the plan is updated after that pull request merges.
  - `common_rows()` reuses the rules of `wind_products.common_rows` (drop hours holding an exact-zero half-hour), applied to the target only, so every arm scores exactly the same rows. It drops no hours by any CERRA value.
  - `arm_columns()` returns one fixed-length tuple per arm from one function, and the report prints every arm's column list (the "silently lost column" rule). A `check_column_counts` raises if any two arms of a planned contrast differ in width.
  - `jobs()` builds the arm list for `run_experiment.run_all`, with `colsample_bytree=1` and `device="cpu"` (both checked by assertions), for every arm and control, and the report records the device. At `PRIMARY_HYPER_PARAMETERS` for all arms and `SENSITIVITY_HYPER_PARAMETERS` for every planned contrast and any result near the 5% line.
  - The report, `report.md` under `data/studies/beam_diffuse_split/`, prints every table the page quotes: each arm's absolute error, each contrast with its interval from `studies.bootstrap`, the arms' column lists, the row counts per farm and year, the power-hour offset scan, and the controls.
- **`studies/beam_diffuse_split/cerra_wind_levels_charts.py`** (new). Figure 1 is a leaderboard of each arm's own error (`studies.charts.leaderboard_panel`), Figure 2 the paired contrasts (planned rows labelled per `studies.charts.planning`), then the "method working" figures for W1 to W3. The SVGs go through `svgo` before commit.
- **`docs/studies/past-weather/cerra-wind-levels.md`** (new). The page, in the `study` skill's section order, with the disclaimer, a Summary with the headline figure, Key findings, Introduction, Data and methods, Results, Discussion, Limitations, Scope, Data and code availability, and Reproducing the figures.
- **`mkdocs.yml`** (one nav line under Past weather). A new entry after the wind page's line.
- **`studies/beam_diffuse_split/README.md`** (one row per new script in the scripts table, and a description of each saved file).

## How #957 avoids editing the pages #968 edits

- **`docs/studies/past-weather/wind.md` is untouched by this PR**, so the two branches cannot conflict there. A one-line "see also" link from `wind.md` to the new page is a follow-up commit made only after #968 has merged, and only if the maintainer wants it.
- **`docs/studies/past-weather/methods.md` is untouched too.** The new page states its own row set and planned contrasts and links the shared methods page for the folds, the normalisation, and the intervals. If #968 adds a wind row-set table to `methods.md` and merges first, a follow-up adds one row for this page.
- **The one file both branches may touch is `mkdocs.yml`.** #968 may add nothing there (it adds row sets to an existing page); if it does, the conflict is one nav line.
- **The CERRA wind loader is owned by #968.** The maintainer has decided that the shared loader (reading the files, deriving the nearest cells, pivoting the heights, building the centred power hour) goes into `packages/studies/` with tests, is owned by #968 (which needs it for CERRA and NORA3), and merges first. This plan imports it, so both pages share one nearest-cell rule and one power-hour rule.
- **`data/studies/` writes go to a new directory** (`data/studies/cerra_wind_levels/`), so no published output is overwritten. Only one agent may run study scripts at a time, so the run needs the runner slot from the Study MAIN COORDINATOR.

## Study design

**Question.** At three wind farms in Lincolnshire, does an XGBoost model given a blend of CERRA's wind levels predict hourly wind power better than one given CERRA's 100 m wind alone, and how much does 100 m wind add over 10 m wind?

**Row set.** Every 3-hourly farm-hour from the start of the power record (checked before the first fit) to 2026-06-30 21:00 UTC, dropping hours that hold an exact-zero half-hour, on which all five CERRA levels have a value at each farm's cell. The count is printed in the report before any fit.

**Folds.** `studies.cross_validation.assign_folds`, contiguous blocks of whole months, per farm, with `search_fold_offsets`-style coverage so no calendar month is without training rows. Intervals resample whole calendar months and one of three seeds (`studies.bootstrap.bootstrap_difference`, 2,000 resamples). Three farms sharing weather means these intervals describe these farms only.

**Metric.** Mean absolute error as a percentage of each farm's capacity, normalised per row before any mean or difference, absolute errors reported for every arm as well as the contrasts.

**Features.** Every arm carries `hour_of_day` and `day_of_year` plus its wind columns. No UKV era column is needed, since no arm reads UKV.

**Arms and planned contrasts.** Every arm is padded to the widest arm's column count (five wind columns, one for each of the five heights) so the comparison is not decided by width.

| Arm | Wind columns (real) | Padding to five |
|---|---|---|
| `speed_10m` | 10 m | four monotone transforms of 10 m |
| `speed_100m` | 100 m | four monotone transforms of 100 m |
| `speed_10m_100m` | 10 m and 100 m | three monotone transforms of 100 m |
| `mean_near_100m` | mean of 75 m, 100 m, 150 m | four monotone transforms of the mean |
| `levels_50_to_150` | 50 m, 75 m, 100 m, 150 m | one monotone transform of 100 m |
| `levels_all` | all five | none |

A monotone transform of a column adds no information a tree can use, so the padding is deterministic and carries nothing new. That also means a padded arm cannot show whether width alone moves the error, which the negative control below measures directly.

Planned contrasts (each labelled "(planned)" on the page, each also run at the second hyperparameter setting):

1. `speed_100m` minus `speed_10m`: what 100 m wind adds over 10 m wind.
2. `mean_near_100m` minus `speed_100m`: does a simple mean of the near-100 m levels beat 100 m alone.
3. `levels_50_to_150` minus `speed_100m`: does a learned combination of four hub-region heights beat 100 m alone.
4. `levels_all` minus `levels_50_to_150`: does adding the 10 m level to the four higher heights change the answer.

Every other contrast, including `speed_10m_100m` against `speed_100m` and any per-farm or per-season split, is exploratory, and any analysis added after the first run is labelled post hoc.

**Controls.**

- **Negative control:** `speed_100m` (100 m plus four monotone transforms of it) against `speed_100m_noise`, which is 100 m plus four information-free columns: the other levels shuffled by month block, so each column keeps a realistic distribution and carries no information about the hour it sits on. The difference shows the size a change in width produces from nothing. Monotone-transform padding alone would be inert at `colsample_bytree=1` and would show nothing.
- **Positive control:** a synthetic target built from a fixed power curve applied to a log-height interpolation of CERRA's speeds to 120 m (linear in log-height between the 100 m and 150 m levels), plus noise, with the same folds. A target built from the 150 m speed alone would be trivial, since one arm column would reproduce it. On the interpolated target, `levels_50_to_150` must beat `speed_100m` by a margin the interval excludes, because the model has to learn the blend; otherwise a null result on the real target cannot be read as "no effect".

**A null result is stated with its bound**, as in "an effect as large as X points is not excluded".

**Not done.** No column subsampling, no direction, no comparison with other weather products (that is #968), no forecast lead (CERRA is an analysis, so the study says how well CERRA's levels describe past wind, and makes no claim about forecasting).

## Design-philosophy check

This is R&D code under `studies/`, off the production path, so `inherent-stability.md`'s degrade-and-record rules do not apply and the scripts fail fast on any violated check (column counts, uncovered months, unequal rows). No asset check is added. It touches no hypothesis label in `engineering-hypotheses.md`. Principle trade: none. The study adds a page that a later decision on which reanalysis wind height to buy or ingest can cite.

## Tests

Study scripts carry no unit tests, per the `study` skill: the check is the script's own printed report, and each table the page quotes is printed by the committed script. `packages/studies/` does not change, so no mutation pass. The self-checks in the script that would fail on a wrong build:

- `check_column_counts` fails if two arms of a planned contrast have different widths, which is the failure the wind page's "lost 10 m column" hit.
- A row-set check fails if any two arms are scored on different (farm, time) keys.
- An assertion fails if any fit uses `colsample_bytree` other than 1.
- The positive control fails the run if the blend does not beat 100 m on the synthetic target.
- An assertion fails if any wind farm's nearest cell lies on the edge of the 190-cell crop. The loader's own tests, in `packages/studies/`, cover the derivation.

If a reusable helper is promoted into `packages/studies/` during implementation, it comes with tests and a mutation pass, and the PR body says so.

## Docs to update

- The new page (above), with the "How this page was made" disclaimer whose model list is taken from the commits' `Co-Authored-By` trailers, and without the human-review sentence until the maintainer has reviewed it.
- `mkdocs.yml` nav, and `docs/studies/past-weather/index.md` gets one line linking the new page if the index lists its pages (checked at implementation).
- `studies/beam_diffuse_split/README.md`.
- `docs/studies/past-weather/wind.md` and `methods.md`: not until #968 has merged.

## Verification commands

```bash
uv run ruff check . && uv run ruff format --check . && uv run ty check
uv run pytest
uv run pymarkdown scan -r docs README.md CLAUDE.md packages/*/README.md
uv run mkdocs build --strict
```

The run itself is `uv run python studies/beam_diffuse_split/cerra_wind_levels.py`, then the charts script, after the runner slot is granted. Memory says the CI has steps the skill's set omits (pydoclint, the docs link checker), so the implementer runs every step in `.github/workflows` locally. Before the run, a CPU load check. Every fit, control included, uses `device="cpu"`, set explicitly and recorded in the report and on the page; the 3-hourly rows are few enough that a GPU adds nothing worth the device-mixing risk.

## Risks and open questions

**Findings from the plan review that were taken into the plan** (each checked against the code and the data):

- **Contrast 4 mixed two changes.** Confirmed by reading the arm table: `levels_all` adds the 10 m level and the 50, 75 and 150 m levels at once. Contrast 4 is now `levels_all` minus `levels_50_to_150`.
- **Contrast 1 compares two CERRA products.** Confirmed: `lineage_10m_wind_speed_surface.json` names `reanalysis-cerra-single-levels`, and the other four name `reanalysis-cerra-height-levels`. The plan and the page now say so.
- **The wind farms are not in `generator_cells.parquet`.** Confirmed: the file has 6 rows (`site`, `y_index`, `x_index`, `distance_km`), for the solar farms. The cells are derived from `cerra_grid.parquet`, asserted strictly inside the crop.
- **`cerra_grid.parquet` has no land-sea flag.** Confirmed: its columns are `y_index`, `x_index`, `latitude`, and `longitude`. The plan drops the claim and asks the maintainer before any mask download.
- **The negative control was inert at `colsample_bytree=1`.** Accepted: it now pads with information-free shuffled columns.
- **The 150 m positive control was trivial.** Accepted: it now uses a 120 m log-height interpolation.
- **Device.** Accepted: CPU explicitly for every fit, recorded.
- **The era check.** Accepted as a gate before the first fit. `hour_of_day` taking 8 values is stated on the page.
- **Reviewer findings rejected:** none.
- **The shared loader (decided by the maintainer, not a review finding).** #968 owns it, and #957's implementation starts only after the loader's pull request has merged. This pull request stays plan-only until then.

1. **How to pad the narrower arms to equal column count.** The plan uses monotone transforms of an existing column, which a tree cannot exploit. Recommendation: keep this, with the negative control showing it is inert; the alternative is unequal widths at `colsample_bytree=1`, which the skill says still favours the wider arm by about 0.4% of mean absolute error on synthetic data.
2. **Definition of "a blend".** The plan tests a plain mean and a learned combination. A hub-height-interpolated speed (linear in log-height between the levels either side of each farm's hub height) is a third defensible blend, but it needs each farm's hub height, which is farm-identifying private data. Recommendation: leave interpolation to a follow-up unless the maintainer supplies the heights as a range.
3. **The row set is one hour in three.** This cuts the row count by about two-thirds against the wind page's hourly rows and widens every interval. Recommendation: accept it; rebuilding hourly wind by interpolating CERRA's 3-hourly analyses would make every hourly value a model, as `cerra_past_solar.py` had to do for solar.
4. **No direction.** Wind power depends on direction through wake and terrain effects that speed alone cannot see. Recommendation: state this in Limitations; direction is not in the download and #969 is closed.
5. **Possible source change inside CERRA's record.** To be checked against the dataset documentation before the first fit. Recommendation: cut folds by era and add an era column to every arm if a change is found.
6. **Dependency on #968's loader.** Implementation is blocked until that pull request merges; if the loader's interface differs from this plan's assumptions, revise the plan then.
7. **Whether the second Opus science review is needed.** The study skill requires two before publishing, so the plan assumes both; the brief says a second only if the result is scientifically important, which the skill's floor overrides.
