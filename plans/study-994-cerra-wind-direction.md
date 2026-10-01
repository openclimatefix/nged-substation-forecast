# Plan: CERRA wind direction in the wind-levels study (issue 994)

**The problem.** The CERRA wind-levels study fitted XGBoost models to wind speed at several heights and gave them no wind direction. Operational wind-power forecasting normally uses speed and direction together, and the wind page's arms carry direction as sine and cosine, so the wind-levels errors are not comparable in level with the wind page's. The study therefore cannot say whether direction adds anything to CERRA's speed, or whether direction at several heights adds anything beyond direction at one height. CERRA's direction files for 10, 75, and 100 m are on disk. The 50 m and 150 m files are still downloading (the 150 m file finishes at about 04:00 UTC on 2026-10-02).

**The plan.** A new script, `studies/beam_diffuse_split/cerra_wind_direction.py`, reuses the wind-levels study's row set, folds, seeds, bootstrap, gates, and negative-control design, and adds the direction columns. Three contrasts are written down before any fit: 100 m speed against 100 m speed plus direction, 10 m speed against 10 m speed plus direction, and 100 m against 10 m with direction at each. A second family of arms asks the veer question (direction at several heights against direction at one height), and every contrast in that family is exploratory. Small tested helpers (the sine and cosine encoding, the veer, the month shuffle) go into `packages/studies/`. Nothing is fitted until the plan and the script have been reviewed, and the full run waits for the 50 m and 150 m files.

## Verdict, size and departures

**Verdict: worth doing as described.** No pull request or branch mentions issue 994. The prerequisite study (pull request 983) is merged and its saved outputs are on disk. The data precondition holds for 10, 75, and 100 m, with 3,792,400 rows each, no bad values, and keys identical to the speed files (`--check` prints this).

**Size: complex, by the study skill's rule that a published page counts as what gets stored.** The five triggers:

- **What gets stored:** a new write-once folder `data/studies/cerra_wind_direction/` (per-row losses, intervals, report) and, later, a page under `docs/studies/past-weather/`. This fires the trigger.
- **The production serving path:** untouched. The code is under `studies/` and `packages/studies/`, and nothing in `src/` imports either.
- **A degradation rule:** none touched. R&D code fails fast, which the script does at every check.
- **More than one defensible design:** yes, in the column budget (equal width across all arms against two width-matched families), the direction encoding, and which veer parametrisations to test. The plan states its choice and the open questions below ask for approval.
- **Callers I could not name without searching:** none. The new `packages/studies/` module is imported only by the new script.

The study skill requires all reviews: two plan reviews, two Opus scientific-validity reviews, a diff review, and a mutation pass because `packages/studies/` changes. The coordinator dispatches the fresh Opus reviewers.

**Departures from the issue body.** There are two.

- **The veer arms hold the 10 m and 100 m speeds in every arm.** The issue says "direction at several heights" without saying what speed columns the arms carry. Holding speed fixed makes the veer contrasts about direction alone.
- **The 10 m against 100 m contrast is written as 100 m minus 10 m, each with direction,** the same sign as the prerequisite study's planned contrast.

## What the data says about direction

CERRA direction is in degrees clockwise from north, the direction the wind blows from (the meteorological convention). The files' circular mean at 100 m is 227 degrees, which is south-westerly and matches the UK's prevailing wind, so the convention is confirmed from the data. A direction enters an arm as its sine and cosine. That encoding gives the same columns up to sign whether the angle is the direction the wind comes from or goes to, so the convention affects only the synthetic positive controls, which are built on the "from" convention.

The veer from 10 m to 100 m at the 190 cells has a median of 3.7 degrees, a 95th percentile of 33 degrees, and 20% of cell-hours at 10 degrees or more (a pooled count over all cells, read without any generator's location). Veer is small, so a study that finds nothing from it needs its positive control to say how large a veer effect the instrument can see.

## Arms

Every arm carries `hour_of_day` and `day_of_year` plus its wind columns, with `colsample_bytree` at 1. Arms sit in two families, and every arm within a family has the same number of columns. A contrast never crosses families (`check_family_widths` raises if one does). Padding uses strictly monotone transforms of the 100 m speed (or the 10 m speed for the 10 m arms), which a tree cannot use. The first four transforms are the prerequisite study's, so the two speed-only arms equal that study's arms column for column.

**Core family: 5 wind columns, 7 features in all.**

| Arm | Real wind columns | Padding |
|---|---|---|
| `speed_10m` | 10 m speed | 4 transforms of 10 m speed |
| `speed_100m` | 100 m speed | 4 transforms of 100 m speed |
| `speed_10m_dir` | 10 m speed, sine and cosine of 10 m direction | 2 transforms of 10 m speed |
| `speed_100m_dir` | 100 m speed, sine and cosine of 100 m direction | 2 transforms of 100 m speed |
| `speed_100m_dir_noise` (negative control) | 100 m speed, sine and cosine of another month's 100 m direction | 2 transforms of 100 m speed |

**Veer family: 12 wind columns, 14 features in all.** Every arm holds the 10 m and 100 m speeds.

| Arm | Direction columns (real) | Padding |
|---|---|---|
| `veer_speed_10_100` | none | 10 transforms of 100 m speed |
| `veer_dir_100` | sine and cosine at 100 m | 8 |
| `veer_dir_10_100` | sine and cosine at 10 m and 100 m | 6 |
| `veer_angle_10_100` | sine and cosine at 100 m, and of the veer from 10 m to 100 m | 6 |
| `veer_dir_all5` | sine and cosine at 10, 50, 75, 100, and 150 m | 0 |
| `veer_dir_100_noise` (negative control) | sine and cosine at 100 m, and of another month's 10 m direction | 6 |

`veer_dir_10_100` and `veer_angle_10_100` hold the same information in two parametrisations. A tree can build the veer from two raw directions only with many splits, so the contrast between them says whether explicit veer helps the tree. The script prints every arm's column list into the report, and the dry-run prints them before any run.

## Row set, folds, metric, and intervals

**Row set.** Exactly the prerequisite study's: 53,107 three-hourly farm-hours between 17 September 2019 and 30 June 2026, minus every hour holding an exactly-zero half-hour. The script stops unless its row keys equal the saved keys of `data/studies/cerra_wind_levels/rows.parquet` (`--check` shows 53,107 of 53,107). That also means the power-hour offset scan and the era step gate are inherited and not rerun: direction is an instantaneous analysis value at its label, like speed, and the zero rule reads the target only. Every arm scores the same rows by construction, and `check_no_missing` stops on a null or NaN in any column an arm reads.

**Folds.** `assign_folds`: each farm's span cut into 5 contiguous blocks of whole months, with `raise_on_uncovered_months`. **Metric.** Mean absolute error as a percentage of each farm's capacity, normalised per row before any mean or difference. **Intervals.** `studies.bootstrap`: whole calendar months and one of three fitting seeds, paired across arms, 2,000 resamples. The three farms share their weather, so intervals describe these three farms only. Every arm's absolute error is reported, and each planned contrast also runs at the second hyperparameter setting.

## Contrasts and multiplicity

**Planned contrasts** (written here before any fit, each labelled "(planned)" on the page, each also run at the second setting):

1. `speed_100m_dir` minus `speed_100m`: what direction adds at 100 m.
2. `speed_10m_dir` minus `speed_10m`: what direction adds at 10 m.
3. `speed_100m_dir` minus `speed_10m_dir`: 100 m against 10 m, each with direction.

**Exploratory contrasts.** The veer family: `veer_dir_100` and `veer_dir_10_100` against `veer_speed_10_100`; `veer_dir_10_100`, `veer_angle_10_100`, and `veer_dir_all5` each against `veer_dir_100`; `veer_dir_all5` against `veer_dir_10_100`; and `veer_dir_10_100` against `veer_dir_100_noise`. Also `speed_100m_dir` against `speed_100m_dir_noise`, and the three planned contrasts split by farm and by half of the year. Any analysis added after the first run is post hoc and labelled so.

**Multiplicity handling.** The three planned contrasts get an interval adjusted for three comparisons (Bonferroni, 98.33%, from `bootstrap_difference_at_level`) beside the unadjusted 95% interval. A planned contrast counts as a finding only if its adjusted interval excludes zero and the second hyperparameter setting agrees in sign and significance. The exploratory contrasts get unadjusted intervals and no correction, because the number of real effects among them is unknown. The page states the count of exploratory rows (the veer contrasts, the noise contrast, and the planned contrasts' per-farm and per-half splits, which are 8 plus 15 = 23 rows at the primary setting), says that an exploratory row with no real effect has a nominal 5% chance of reaching significance, says that spurious results cluster because all rows share months, and states no exploratory row as a finding. The issue's done criterion (whether direction at several heights adds anything beyond direction at one height) is answered by the exploratory `veer_dir_10_100` and `veer_dir_all5` contrasts against `veer_dir_100`, so the page's answer is labelled exploratory, with its interval and the positive control's bound.

## Controls

**Negative controls.** The prerequisite study's design, with direction in place of speed: the shuffled column takes another month's value, so it keeps its distribution and carries no information about the hour. `speed_100m_dir_noise` against `speed_100m` shows what two unusable columns do to the error. `speed_100m_dir` against `speed_100m_dir_noise` is width-matched. `veer_dir_100_noise` against `veer_dir_100` and `veer_dir_10_100` against `veer_dir_100_noise` do the same for a second height's direction. The month shuffle is of the direction in degrees, then encoded, so each sine and cosine pair stays consistent.

**Positive controls.** Two synthetic targets use a fixed power curve of the 100 m speed (cut-in 3 m/s, rated 12 m/s, a cubic between) and 1% noise.

- **Sector target:** power is cut by 40% when the 100 m direction is within 30 degrees of 255 degrees (26% of rows). The run stops unless `speed_100m_dir` beats `speed_100m` with an interval below zero.
- **Veer target:** power is cut by 40% when the veer from 10 m to 100 m is at least 10 degrees (18% of rows). The run stops unless `veer_angle_10_100` beats `veer_dir_100` with an interval below zero. The report also prints whether `veer_dir_10_100` recovers the rule, which bounds how a null on the real target can be read.

Both rules' shares of rows lie in [10%, 50%] (`check_synthetic_shares` stops otherwise). The thresholds were set from the pooled marginal distributions above, without reference to any model's output. A null result on the real target is stated with its bound ("an effect as large as X points is not excluded") and is read against the positive controls.

## What changes, file by file

- **`packages/studies/src/studies/wind_direction.py`** (new): `sine_cosine` (a direction in degrees to its sine and cosine, as Polars expressions), `veer_degrees` (the wrapped turn from a lower to an upper height, in [-180, 180)), and `shuffled_by_month` (the month shuffle, promoted from `cerra_wind_levels.py`, where it is private). `studies.reanalysis_wind.read_cerra_direction` and `CERRA_DIRECTION_FILES` already exist on `main` and are reused.
- **`packages/studies/tests/test_wind_direction.py`** (new): see Tests.
- **`studies/beam_diffuse_split/cerra_wind_direction.py`** (new): the script. It imports the prerequisite script's public pieces (`Contrast`, `check_column_counts`, `check_settings`, `check_same_rows`, `read_half_hourly_power`, and its constants) and, like that script, `_wind_sites`, `_fingerprint`, `_arm_columns_lines`, and `_add_time_features`. It does not edit `cerra_wind_levels.py`. Its `main` has `--dry-run`, `--check`, and `--report-only`.
- **`studies/beam_diffuse_split/README.md`**: one table row for the script.
- **Later, in the implementation pull request** and not this plan's: a charts script, and the page `docs/studies/past-weather/cerra-wind-direction.md` with nav and index entries.

## The script's safeguards

- **Write-once output** in a new folder, `data/studies/cerra_wind_direction/`. `refuse_to_overwrite` stops on any existing output file, and `--check` fails if one exists.
- **Anonymity.** Farms are W1 to W3 from `_wind_sites`. The report prints no coordinate, name, identifier, or cell index, and `--check` prints pooled counts only.
- **The full run stops while any direction file is missing** (`FileNotFoundError` naming the files). `--dry-run` reports them and exits 0, and `--check` runs on the heights present (10, 75, and 100 m now).
- **Arm columns** come from one function, `arm_columns`, and the report prints each arm's list. `check_family_widths` stops if any family's arms differ in width or repeat a column. `check_settings` stops unless every fit is on the CPU with no column subsampling, on at most 8 cores (2 fits of 4 threads).
- **Agreement with the prerequisite study.** `speed_10m` and `speed_100m` should reproduce that study's per-row losses exactly on the same device. The report prints the largest absolute difference, which is a free end-to-end check that the build and the fit loop are unchanged.
- **The fingerprint** of the row set, columns, seeds, and settings guards `--report-only`.

## How the script is checked before it runs

The script is not unit-tested, so the study skill makes its own output the check. Before the first fit:

1. `--dry-run` prints the arms, every arm's columns, the number of fits, the output paths, and the missing files.
2. `--check` reads the three direction files present and the private power table, and verifies the file checks (keys equal the speed files, no bad values, circular mean in the south-westerly half), the arm and setting checks, the row set against the prerequisite study's keys, the synthetic rules' shares, and that no output exists.
3. A fresh Opus review of the script, before its first run, as the study skill requires.
4. The 50 m and 150 m files are validated with `--check` and the `data-validation` skill when they land.

Output of steps 1 and 2 as of this plan: 11 real arms, 27 (arm, setting) fit groups, 1,215 XGBoost fits; row set 53,107 rows, equal to the prerequisite study's; synthetic rule shares 26.3% and 18.2%.

## Runtime estimate

**About 1 hour of wall time on 8 cores, which is an estimate and not a measurement.** The basis is the prerequisite study's file times: its main fits (720 fits, 7 columns) ended 5 minutes after its power-hour scan, and its bootstrap and report took 23 minutes. This run has 1,215 fits, and the veer family's arms have 14 columns (about 1.5 times the cost per fit), which gives about 15 to 25 minutes of fitting. The bootstrap covers about 50 contrast rows and 110 absolute rows, about 1.3 times the prerequisite study's, which gives about 30 minutes. `uptime` is checked before the run, and one agent runs it at a time.

## Design-philosophy check

This is R&D code off the production path, so the fail-fast rules apply and no degradation path is touched. No asset or asset check changes. No hypothesis label is touched. The study makes a past-weather descriptive claim only: CERRA is an analysis of the past, so the page says how well CERRA's speed and direction explain power that has already been generated, and makes no forecast-skill claim.

## Tests

`packages/studies/tests/test_wind_direction.py` (each assertion fails on a plausible bug; none exists on `main` because the module is new):

- `sine_cosine`: 359 and 1 degrees are close together on the circle while 359 and 180 degrees are far apart (fails if degrees went in raw or the encoding used one column); 0 and 360 degrees give equal columns; 90 degrees gives sine 1 and cosine 0 (fails if the two columns were swapped).
- `veer_degrees`: 10 over 350 is +20, 350 over 10 is -20, 200 over 100 is +100, 0 over 180 is -180 (fails if the wrap is missing or uses the wrong half-open end).
- `shuffled_by_month`: each month's rows hold one other month's values, never its own (fails if a month keeps its own values or the donors are mixed); a short donor month is repeated to fill a long month; fewer than two months raises.

No test is added for `read_cerra_direction`, which `main` already tests. The mutation pass runs over `wind_direction.py` and its tests, because `packages/studies/` changes.

## Docs to update

None in this pull request beyond the README row. The page is written after the run and the two scientific-validity reviews. It carries the disclaimer with the model names from the commit trailers, and states in its Limitations which `effective_capacity` table the figures rest on.

## Verification commands

```bash
uv run ruff check . && uv run ruff format --check . && uv run ty check
uv run pytest packages/studies
uv run pymarkdown scan -r docs README.md CLAUDE.md packages/*/README.md
uv run python studies/beam_diffuse_split/cerra_wind_direction.py --dry-run
uv run python studies/beam_diffuse_split/cerra_wind_direction.py --check
```

Every step in `.github/workflows` is also run locally before the pull request leaves draft, including `pydoclint` and the docs link checker.

## Risks and open questions

1. **The veer arms hold only the 10 m and 100 m speeds.** A veer family that also holds all five speeds (the prerequisite study's best arm) would ask whether direction helps once speed is as informative as it gets, and would need a third family of width 15. Recommendation: not in this study. If `veer_dir_all5` against `veer_dir_100` is non-null, a follow-up can test it.
2. **Run now without the 50 m and 150 m files?** Only `veer_dir_all5` needs them, so the other 10 real arms could run now. The script runs all arms in one write-once folder and stops while any file is missing. Recommendation: wait for the 150 m file (about 04:00 UTC on 2026-10-02), because splitting a run across two folders breaks the single fingerprint and the report's tables.
3. **Should the issue's done criterion be a planned contrast?** The issue asks for the veer arms to stay exploratory, so `veer_dir_10_100` against `veer_dir_100` is exploratory. A reader may expect it to be the study's headline. Recommendation: keep it exploratory, as the issue says, and report its interval and the veer positive control beside it.
4. **Duplicated shuffle code.** `shuffled_by_month` now exists in `packages/studies/` and, privately, in `cerra_wind_levels.py`. Recommendation: leave the merged study's script untouched, because its saved outputs fingerprint its code path, and tidy it in its own pull request if wanted.
5. **No direction era check.** The prerequisite study checked CERRA's speed for a production-stream join and found no step above its gate. Direction is a circular quantity, and the script has no equivalent test. Recommendation: accept, because a step in direction would show as a step in the veer arms' effect by half of the year, which the per-half split reports, and state the gap in Limitations.
6. **The device is the CPU,** as the prerequisite study's, so the two speed-only arms reproduce its losses. The page states the device.
