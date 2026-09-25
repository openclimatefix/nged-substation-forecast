# Plan: add ECMWF AIFS Single and AIFS ENS to the matched-lead comparison (issue #923)

**The matched-lead page compares weather forecast products for solar and wind power at six solar
farms and three wind farms, and it does not mention ECMWF's machine-learned weather model, AIFS.**
The data is now on disk under `data/studies/weather/ECMWF-AIFS/` (AIFS Single) and
`data/studies/weather/ECMWF-AIFS-ENS/` (AIFS ENS): four runs a day, leads 0 to 360 h in 6-hour
steps, a 3 by 4 crop of 0.25° cells (12 cells: latitude
indices 0 to 2, longitude indices 0 to 3). The extra longitude column comes from a wider re-fetch,
because the original 3 by 3 crop held only 65% of the area of some sites' H3 hexagons. The two
products are harder to compare fairly than the products already on the page. Each serves 6-hourly steps where ENS serves 3-hourly steps to 144 h, so a
product with finer time steps is favoured by resolution alone. AIFS Single carries radiation and
100 m wind only from the 2025-02-24 06 UTC run, and AIFS ENS starts on 2025-07-02, so neither
product covers the page's 35,263 solar and 37,407 wind shared rows. Both products change version
inside the record, and the store carries no version marker.

**The plan builds AIFS's inputs beside the code that builds GEFS's, scores AIFS on its own row set,
and adds paragraphs to the page without any claim that AIFS improves.** A new `--aifs` flag on
`build_forecast_inputs.py` reads each AIFS run at the same lead as ENS (the 00 UTC run, day `d`
after the run's day, so lead `24d + h`, days 1 and 2), and upsamples 6-hourly steps to hourly
targets with the ENS-horizons page's own rules. The AIFS arms read the same H3 area-mean weights the
ENS arms read, so the spatial read matches.
The like-for-like references are ENS's control member (for AIFS Single) and ENS's mean (for the AIFS
ENS mean), at the same lead and the same 6-hourly steps as AIFS. The already-fitted hourly IFS 0.25°
day-1 column is a one-sided reference, because its hourly steps and its lead both favour IFS 0.25°.
Every reference is refitted on the AIFS rows, on the same graphics processing unit (GPU) as the AIFS
arms, at the primary hyperparameter setting only. Folds are cut inside the AIFS version eras, whose
edges are whole months. One contrast, `aifs_single − ens_control6` at day 1 (solar and wind), is
named deciding before any fit, and every other AIFS number, including every AIFS ENS number, is
labelled exploratory and post hoc. The output is a new write-once folder,
`data/studies/nwp_forecast_comparison_aifs/`, a `report.md` that prints every
number the page quotes, and one new results section on the page. Nothing under `packages/` changes,
and no new module is added: the plan reuses `geo.h3.compute_h3_grid_weights` and the helpers named
below, so the scope stays the one plan review 1 trimmed it to.

## Verdict, size, and departures

**Verdict: worth doing as described, with the departures below.** No pull request or branch is open
against #923 other than this one (`gh pr list --search 923` and `git branch -a --list '*923*'`
found none before it). The issue's premises match the data on disk: I re-derived them from the two
`README.md` files, the two `lineage.json` files, and the parquet files (see "What the data holds").

**Size: complex, by the five triggers in `plan-issue` step 3.**

1. *What gets stored:* fires. A new folder under `data/studies/`, and a published page, count as
   stored (the `study` skill sizes every study as complex). No Patito contract, Delta table, or
   Dagster asset changes.
2. *Production serving path:* does not fire. Nothing under `src/`, `packages/ml_core/`, or
   `packages/xgboost_forecaster/` changes.
3. *Degradation rule:* does not fire. This is R&D code, which fails fast on purpose
   (`docs/design-philosophy/inherent-stability.md`).
4. *More than one defensible design:* fires. The lead convention, the era edges, the IFS
   reference, and the row set each admit alternatives, which "Risks and open questions" lists.
5. *Callers not nameable without a search:* does not fire. The one shared function this plan
  touches,
   `ens_member_arms`, gains a defaulted argument, and its callers are in
   `build_forecast_inputs.py` (named by grep).

Two triggers fire, so the issue is complex. Both plan reviews have run (see "Findings of plan
review 2" below). It also gets the two Opus scientific-validity reviews the `study` skill requires
before any number reaches `docs/`, and one diff review. **The mutation pass is dropped:**
`packages/studies/` no longer changes, so there is no
tested package code to mutate, and the scripts have no unit tests. The PR body says so. The Opus
review of the fit code finishes before the first fit runs.

**Departures from the agreed design input (`/tmp/claude-1000/aifs-design.md`):**

- **Era edges are whole months, not init-time instants.** The design keys eras on init time. A cut
  at an instant inside a month needs a new era cutter in `packages/studies`. Whole months reuse
  `studies.cross_validation.cut_eras` unchanged, the same way the published study drops January 2026
  around the UKV upgrade. A check in `fit_aifs.py` proves every kept row's run lies in the run-date
  range of the era its month gave, so "keyed on init time" still holds. Three of 19 valid months are
  dropped (2025-02, 2025-08, and 2026-05), and the AIFS ENS set also loses 2025-07.
- **IFS 0.25° is one hourly reference at day 1, not a bracketed control.** The archive's lead for
  IFS 0.25° is `24d + (h mod n)` with the run cycle `n` unmeasured, so a matched-lead IFS 0.25° arm
  is not available. See "The references".
- **The Single arm reads its `control` reduction, not `mean`.** The mean-wind-vector direction in
  `reduce_members` divides by the vector norm, which is 0 for a calm hour when there is one member.
- **The 2026-05-12 boundary reads 06 UTC, not 00 UTC.** The design says AIFS Single v2 starts
  "2026-05-12 00 UTC". Both `README.md` files and `docs/roadmap/data-sources.md` say 06 UTC. The
  rows do not depend on it, because 2026-05 is dropped whole and only 00 UTC runs are read, but the
  era constants use the later, better-sourced time.

**Findings of plan review 1 that were modified or rejected, and why:**

- *S1 (drop the IFS 0.25° arms), modified as the lead decided:* the middle option is taken. The
  6-hourly
  IFS 0.25° control, day 0, the bracket, `studies/stamps.py`, and any `band_steps` change are
  dropped.
  The hourly `ifs025_day1` stays as a one-sided reference. The `band_steps` wind-first-stamp fix
  ships in its own pull request (branch `band-steps-wind-first-stamp`) and is not part of this one.
  Without day 0, no arm in this plan reads the dropped stamp, so this plan does not depend on it.
- *S2, accepted:* `model_versions.py` is replaced by a short check in `fit_aifs.py`. I verified the
  margins the review states: the nearest kept row's run sits at least 6 hours from the excluded
  window and days from every other switch.
- *S3, accepted:* the extract is a sibling of `_gefs_members_frame`. I verified that function drops
  `NaN` radiation (`is_not_nan`, line 741) and that `check_no_missing` counts `NaN` as missing. The
  script still prints a `NaN` count after the run filter, which must be 0.
- *S4, accepted:* `--aifs` beside `--extra-leads`, and neither rename is made.
- *S5, accepted, with one modification:* day 2 is kept for the AIFS, ENS mean, and ENS control arms
  (it adds fits and no code, and shows whether the gap moves with lead). It is dropped for
  `ifs025`, which has no matched day 2.
- *S6, accepted:* no `published` noise-floor set. The page cites the extra-lead run's floor.
- *S7, accepted:* no resume, unless the `--check` timing estimate says the full run takes hours.
- *S8, accepted:* no complete-run filter and no "runs complete" check. The row counts below were
  measured with a complete-run rule, and every 00 UTC run in range is complete on today's data, so
  they are unchanged.
- *S9, accepted:* the chart calls `studies.charts.leaderboard_panel` (verified to exist) and
  `interval_panel`.

**Findings of plan review 2 (correctness and testability), all verified and applied unless noted:**

I re-ran the reviewer's `rev2_folds.py` and confirmed its figures (solar `single` 37 of 95 and
`ens` 55 of 65 cells uncovered by calendar month, wind 18 of 48 and 27 of 33, 13 and 4 fold designs
found, identity first). I confirmed that `geo.h3.compute_h3_grid_weights`,
`studies.bootstrap.bootstrap_difference_at_level`, and `_leave_one_month_out_lines` exist, and that
`fit_extra_leads.py` refuses to overwrite `report.md`.

- *M1, accepted:* AIFS is read with the H3 area-mean weights ENS uses, and a nearest-cell arm is
  kept as a sensitivity arm. Risk 6 is corrected.
- *M2, accepted:* the coverage check is stated so it can fail (see "Fold offsets"), and the
  `ens` set is exploratory throughout.
- *M3, accepted:* the tests are rewritten so each fails on the bug it exists for.
- *M4, accepted:* the era check reads the run that fed each row.
- *M5, accepted:* one deciding contrast per technology, and one invocation, so `report.md` is
  written once.
- *S1 to S10, accepted:* each is applied in the section it names. S8 removes the device-floor
  sentences.
- *N1 to N6, accepted:* the 2025-02-26 counts, one leaderboard panel per row set, the `git diff
  --stat` check, the wording in place of "cost", a lead filter in the lazy scan, and the `ens`
  arms with no listed contrast now feed the leaderboard only.

Nothing in review 2 is rejected.

## What the data holds (read-only checks, run while writing this plan)

- **AIFS Single:** 3,629 runs from 2024-04-01 00 UTC to 2026-09-25 00 UTC, 61 leads (0 to 360 h in
  6-hour steps). To be checked once the wide ENS store lands and both wide stores are validated: 12 cells and 732 rows per run for every 00 UTC run. Every 00 UTC run from 2025-02-26
  to 2026-09-25 is present. The 06, 12, and 18 UTC runs are present except 2026-09-25's, which had
  not run when the download finished. Shortwave, longwave, and 100 m wind are `NaN` (not null)
  before the 2025-02-24 06 UTC run. Shortwave and longwave are `NaN` at lead 0 in every run.
  There is no `ensemble_member` column.
- **AIFS ENS:** 1,801 runs from 2025-07-02 00 UTC to 2026-09-25 00 UTC, 51 members (0 to 50, member
  0
  the control), 61 leads. To be checked once the wide ENS store lands: 12 cells (37,332 rows per 00 UTC run), all fields present beyond radiation's lead 0. No 00 UTC run is
  missing.
- **Coverage of the published shared rows** (35,263 solar, 37,407 wind) by a 00 UTC run at the
  site's nearest 0.25° cell, at day 1: Single covers 31,769 solar and 31,315 wind rows (from init
  2025-02-26, the set's own rule); AIFS ENS covers 22,333 solar and 22,797 wind rows in 14 valid
  months (from init 2025-07-02). Day 2 differs from day 1 by under 0.5%.
- **Wind at 10 m only:** the store holds 10 m wind and temperature from 2024-04-01, so wind alone
  could be scored on 6,220 shared rows in 2024-12, 2025-01, and 2025-02, from runs of unknown
  version (before AIFS Single v1.0 went operational on 2025-02-25 06 UTC). Solar has no equivalent,
  because radiation is `NaN` there. The plan does not score them (see open question 2).

## The row sets, eras, and folds

**The AIFS row set is the published shared rows restricted to hours whose 00 UTC run is inside a
version era.** Two nested sets follow, and each arm is fitted on the sets it belongs to.

| Set | Arms | Run rule | Valid months kept | Solar rows (day 1) | Wind rows (day 1) |
|---|---|---|---|---|---|
| `single` | AIFS Single, every reference | 00 UTC init from 2025-02-26 (the first run after v1.0 went operational) | 16: 2025-03 to 2025-07, 2025-09 to 2025-12, 2026-02 to 2026-04, 2026-06 to 2026-09 | 28,570 | 28,873 |
| `ens` | those, plus AIFS ENS | as `single`, and AIFS ENS's own start (init 2025-07-02) | 11: 2025-09 to 2025-12, 2026-02 to 2026-04, 2026-06 to 2026-09 | 16,911 | 18,555 |

Day 2 counts equal day 1's on both sets. The published shared rows already drop 2026-01 (the UKV
upgrade) and much of spring 2026, so AIFS inherits those gaps: 2026-04 and 2026-05 hold only 272 and
224 wind rows.

**Three eras for `single` and two for `ens`, each a version of AIFS.** The dated table is from
ECMWF's release pages as recorded in `docs/roadmap/data-sources.md` and both `README.md` files, and
is not verified in the data.

| Era | AIFS Single | AIFS ENS | Months in era (`single` / `ens`) |
|---|---|---|---|
| 0 | v1.0, init 2025-02-25 06 UTC to 2025-08-26 (the 2025-07-31 06 UTC to 2025-08-01 18 UTC window is excluded) | not yet running (v1 starts 2025-07-01 06 UTC) | 5 / not present |
| 1 | v1.1, init 2025-08-27 06 UTC to 2026-05-11 | v1, init 2025-07-01 06 UTC to 2026-05-11 | 7 / 7 |
| 2 | v2, init from 2026-05-12 06 UTC | v2, init from 2026-05-12 06 UTC | 4 / 4 |

The `ens` set starts in 2025-09, and not with ENS's first month (2025-07), because AIFS Single
changes version from v1.0 to v1.1 inside 2025-07 to 2025-08. Keeping 2025-07 would put one month of
Single v1.0 into an era whose other 7 months are v1.1, or make a one-month era that the XGBoost
model could never train on when that month is held out. Both sets therefore use an era code that is
a version of every arm on the set. The month cut is `cut_eras(first_months=("2025-09", "2026-06"))`
on `single` and `("2026-06",)` on `ens`. Era 0's last month is 2025-07, because 2025-08 holds the
reverted v1.1 attempt and the v1.0 to v1.1 switch, and it is dropped whole. Era 2 starts in 2026-06,
because 2026-05 holds the v2 switch.

**The era check reads the run that fed each row, not a reconstruction.** The era table is a dict
of `(first_month, last_run_date)` constants with their source cited. `_aifs_frame` keeps each arm's
run as `<arm>_init_time`, joined from the band's `Steps.keys` before `reduce_members` (which drops
`init_time`). The build's written-column schema allows that column, and it is not a feature. After
the row filter, `fit_aifs.py` raises unless every row's `<arm>_init_time` lies inside the run range
of its row's era, for every AIFS arm. The same check asserts `<arm>_init_time.date() ==
day_start(time) − day`, where `day_start` is `time − 30 min` for solar and `time` for wind
(`efh._day_start`; a solar row is labelled by the end of its hour). The check replaces a separate
module. It never prints an `<arm>_init_time` column.

Era 2 is short (4 months), and the IFS Cycle 50r1 change of the same date is a confound the page
states: a v2 change cannot be told apart from a change in IFS itself, and the ENS references sit in
the same era.

**Every arm on a set gets the same era code, and the code is the AIFS era.** `arm_columns` puts
`era_code` into every arm's calendar columns. `cut_eras` sets `era_code` itself, so the recut on the
AIFS rows overwrites the published one, the references refitted here see the same era feature as the
AIFS arms, and every arm keeps seven columns.

**Fold offsets are found before any fit and re-checked on every run.** I ran
`studies.cross_validation.search_fold_offsets` read-only on both sets and both technologies. The
identity rotation is the first result: 13 rotations cover `single` and 4 cover `ens`, all found from
which hours exist, with no error read. The plan sets `AIFS_FOLD_OFFSETS = {0: 0, 1: 0, 2: 0}` for
`single` and `{0: 0, 1: 0}` for `ens`, and the fit script reruns `search_fold_offsets` and stops if
the constant is no longer in its result.

**The coverage check can fail, and on `ens` it mostly cannot bind.** `coverage_table` re-checks the
folds on every run and raises on an uncovered calendar month that occurs in more than one year. That
check is nearly vacuous here, because on `ens` 10 of 11 calendar months occur once. Under the
identity rotation, 55 of 65 solar and 27 of 33 wind (site, fold, calendar month) cells on `ens` have
no training row of their calendar month (37 of 95 and 18 of 48 on `single`). The report prints these
counts for each set, and the page states them beside the published page's 20 of 124. The plan
handles this in two ways. Every `ens` result is scored descriptively only, with no deciding label
(the maintainer's decision, Risk 13), and the page states the limit plainly beside the absolute
skill: the XGBoost model extrapolates `day_of_year` on about 85% of scored `ens` cells. The deciding
contrast (see "Fit design") is also fitted with `day_of_year` removed from every arm, so each arm
has six columns, and the page reports whether the sign and the 5% significance of the contrast
survive. Fewer folds
would not help, because 4 months per era is already the minimum.

## How the AIFS build reuses the published build

**AIFS is read as a whole-run archive, at the same lead as ENS.** For a target hour on UTC day `D`
the band for day `d` reads the 00 UTC run of day `D − d`, so the lead is `24d + h`. That is
`ens_forecast_horizons.band_steps`'s rule, and it is what the published ENS and GEFS arms do. Only
the 00 UTC run is read, so the three fresher AIFS runs a day are unused; a live service could read
them, and the page says the study did not test that. Open-Meteo reports a 5.7 h delay for AIFS
Single (survey table), so at 09:00 UTC the 00 UTC run is
the freshest AIFS Single run a live user could read. AIFS ENS's delay is not established.

**The build is a `--aifs` flag on `build_forecast_inputs.py`, beside `--extra-leads`.** `build_aifs`
sits next to `build_extra_leads` (line 926), reads the published inputs' `(site, time)` keys only,
and writes `<domain>_aifs_inputs.parquet` to the output folder. Like `build_extra_leads`, it refuses
an output folder equal to the published folder, and calls `studies.guards.refuse_to_overwrite`.
Every written column is checked against a schema that holds no
coordinate, cell id, weight, or generator name; the only site column is the `A` to `F` and `W1` to
`W3`
label.

**The extract is `aifs_members_frame`, a sibling of `_gefs_members_frame`** (line 676), about 40
lines. The function scans the parquet lazily (`pl.scan_parquet`, because the AIFS ENS file holds 50
million rows), filters to 00 UTC runs from the set's first run, to leads of at most `24 *
max(AIFS_DAYS) + 30`
hours, and to the crop's 12 cells before collecting. It computes each site's H3 resolution-5 cell
weights over those 12 cells with `geo.h3.compute_h3_grid_weights`, from the same H3 indices ENS
reads, and raises unless each site's weights sum to 1 within 1e-6 (which proves the crop covers the
hexagon). The weighted mean of the cells is the site's value, and the function raises if any (site, run,
member, lead) group lacks a row for one of the site's weighted cells. A nearest-cell variant, using
`_gefs_cell_selection`, feeds the sensitivity arm. It computes wind speed and
the from-direction (`arctan2(-u, -v) % 360`, the expression `_gefs_members_frame` uses), casts
`lead_time` to whole hours, drops lead 0's radiation, and sets `ensemble_member = 0` where the store
has none. It returns the shape `ens_forecast_horizons.members` returns (`site`, `init_time`,
`ensemble_member`, `lead_hours`, `ghi_w_m2`, `temp_c`, `speed_100m`, `direction_100m`, `speed_10m`,
`direction_10m`).

**The extract then enters through the same four functions as ENS and GEFS.** `ens_member_arms`
calls `band_steps`, `upsampled_fields`, `combine`, `reduce_members`, and `prefixed`. The upsampling
combination is read from `UPSAMPLING_METHODS` (`clear_sky` for solar, `speed_components` for wind),
never re-chosen. So radiation is rebuilt through the clear-sky index, temperature is interpolated
linearly, wind speed is interpolated as vector components, and each hourly target is read from one
run, with no interpolation across runs. Two settings differ from the GEFS call:

- *No window inversion.* AIFS radiation is already a 6-hour window mean ending at the lead
  (`README.md`), so it needs no `gefs_step_means`. `verify_aifs_steps.py` checks that reading
  against
  ERA5, as `verify_extra_leads.py` checks GEFS's.
- *All steps are 6-hourly.* `band_steps` gets `fine_step_last_lead=0`, which gives every step width
  6. AIFS Single gets `ensemble_size=1`.

**Missing values fail loudly.** The run filter comes first, so no `NaN` from before 2025-02-24 06
UTC
is read. The script prints the count of `NaN` in the extract after the filter, which must be 0, and
the fit script calls `studies.guards.check_no_missing` on every arm's columns over the joined rows.
A missing run then shows up as null columns and raises, rather than being filtered out.

**Site to cell uses the H3 resolution-5 overlap weights ENS uses (see Risk 6).** The cell ids,
the weights (which encode the site's position within its cell), the roster's coordinates, and
`_grid_cells.parquet`'s contents are never printed or written. IFS 0.25° stays a point read, and the
page says so.

## The references

**The fair IFS-physics comparison is ENS's mean and control member at AIFS's lead and steps.**
`ens_member_arms` gains one defaulted argument, `six_hourly: bool = False`, passed to `band_steps`,
which calls `coarsen_to_six_hourly`. That keeps ENS's steps at multiples of 6 h and averages each
pair of 3-hourly radiation steps into their exact 6-hour mean. Arm names are `ens_mean6_day<d>` and
`ens_control6_day<d>`. The full-resolution `ens_mean_day1` (published column, refitted) shows how
much the 6-hourly steps raise ENS's error. The helpers stay private, because the new code sits in
the same module.

**What each contrast allows the page to say.** `aifs_single − ens_control6` compares two single,
unperturbed forecasts from the same 00 UTC analysis time, at the same lead, steps, and spatial read.
It still differs in the weather model's native grid (AIFS about 0.25°, ENS about 9 km), so the page
says "AIFS Single's forecast against ENS's control member", and never "machine learning against
physics". `aifs_single − ens_mean6` mixes the weather model with 51-member averaging, and the page
says so. `aifs_ens_mean − ens_mean6` is the closest pair: 51 members each, the same run, lead,
steps, and read. `aifs_ens_mean − aifs_single` measures ensemble averaging inside the AIFS family.

**`ifs025_day1` is a one-sided reference.** IFS 0.25° steps every 3 hours to 144 h, which Open-Meteo
interpolates to hourly. Its lead, `24 + (h mod n)`, is never longer than AIFS day 1's `24 + h`. Both
of those differences favour IFS 0.25°. If AIFS Single beats `ifs025_day1`, the step width and the
lead cannot explain the result. If AIFS Single loses, the result is unresolved. The page states
either outcome in one sentence, with the spatial read and Open-Meteo's serving named as confounds
that run in unknown directions. The plan runs no bracket, so it needs no `check_ens_monotonicity`
re-runs.

**ENS's control member as "the IFS physics" needs a check before the page states it.** ENS is
believed to run at the same 9 km resolution as ECMWF's single high-resolution run since IFS Cycle
48r1 (June 2023). The plan does not carry that claim. Before the page says it, check it against
ECMWF's release notes, and describe the arm as "ENS's unperturbed control member", never as HRES.

**A climatology reference and a null.** `aifs_single_day1_permuted` shuffles AIFS Single's day-1
columns among hours that share a site, a year-month, and an hour of day
(`studies.blending.climatology_permutation`, seed 0). `aifs_single_day1_permuted_b` repeats the
shuffle with seed 1000. `aifs_single − permuted` shows what AIFS adds over its own climatology.
`permuted − permuted_b` is the null: the difference the pipeline produces between two arms that
carry
the same information. Its interval is the size of difference the page treats as indistinguishable
from nothing.

**Bands are days 1 and 2 (`AIFS_DAYS = (1, 2)`).** Day 1 is the day-ahead product and the deciding
band. With no bracket, day 0 has no use.

## Fit design

**The fit script `fit_aifs.py` follows `fit_extra_leads.py`.** One row set and one technology at a
time, `device="cuda"` on every fit, at the primary setting, exploratory, and one fit job at a time.
Every reference is refitted on the same device on the AIFS row sets, so no contrast mixes a GPU fit
with a CPU fit. The Opus review of the fit code finishes before the first fit.

- *Arms per set.* `single` (11): `aifs_single` at days 1 and 2; `aifs_single_nearest` at day 1 (the
  spatial-read sensitivity arm); `ens_mean6` and `ens_control6` at days 1 and 2; `ens_mean` at day
  1; `ifs025` at day 1; `aifs_single_permuted` and `aifs_single_permuted_b` at day 1. `ens` (13):
  those, plus `aifs_ens_mean` at days 1 and 2. On `ens`, `ens_control6`, `ens_mean`, `ifs025`, and
  the permuted arms feed no listed contrast, and appear in the leaderboard only. That is 24 arms per
  technology, or about 215 arm-and-site fits across 6 solar and 3 wind sites, before the second
  setting and the `day_of_year`-removed refits of the deciding pairs.
- *Rows and folds.* `aifs_rows(input_dir, aifs_dir, domain, row_set)` loads
  `nwp_forecast_comparison.rows` on the published inputs, joins the AIFS columns, drops rows whose
  run is outside the set's rule or whose valid month is dropped, requires every arm's columns on
  every row (`check_no_missing`), recuts folds with
  `cut_eras(first_months=..., fold_offsets=AIFS_FOLD_OFFSETS[row_set])`, calls `coverage_table`, and
  runs the era assertion.
- *Column counts.* Every arm is `arm_columns(domain, prefixes=(prefix,))` with seven columns (six in
  the `day_of_year`-removed refits). The report prints every arm's column list (the `study` skill's
  "an arm can silently lose a column").
- *Determinism and timing.* `--check` fits `aifs_single_day1` at one wind site twice on the GPU and
  stops unless `fingerprint` agrees, and it prints an estimate of the full run's time (fit time
  times
  the arm-and-site count). It is run before the full fit, as `fit_extra_leads.py --check` is. Before
  either run, check CPU load (`uptime`; one spinning core oversubscribes an all-core XGBoost fit).
- *Resume only if needed.* There is no per-arm cache. If the `--check` estimate says the full run
  takes hours, the plan adds one before the run, and says so in the PR.
- *Second setting.* One deciding contrast is named before any fit, for each technology:
  `aifs_single_day1 − ens_control6_day1` on `single`. Its two arms are always fitted at the
  sensitivity setting too. In the
  same invocation, before `report.md` is written, the script finds every other contrast near the 5%
  line (an interval bound within 20% of the interval's width from zero), fits both of its arms at
  the sensitivity setting, and then writes the report once. There is no `--setting` flag, so
  `report.md` stays write-once. The report lists which pairs got the second setting and why. A
  verdict needs both settings to agree, and a disagreement is stated.

**The contrasts are listed here before any fit.** No AIFS contrast was written into the published
plan, so none of these is "planned" in the published page's sense. One (for each technology) is
named deciding here, before any fit. The page may claim a difference for a deciding contrast when
its interval excludes
zero (statistically significant at the 5% level) at both settings, with the leave-one-month-out
range and the `day_of_year`-removed refit agreeing in sign. It states that these are the only AIFS
contrasts named before the fit, and that the spatial read and native grid remain confounds. Every
other AIFS number is exploratory and post hoc, and the page may describe it but may not build a
claim on it. Each line of the report carries its row count and month count (from
`bootstrap_difference`) and its label.

- *Deciding, on `single`:* `aifs_single_d1 − ens_control6_d1`. The page may claim which of the two
  forecasts has the lower error at day 1, on `single`'s rows.
- *Descriptive, on `ens`, with no deciding label:* `aifs_ens_mean_d1 − ens_mean6_d1`, the closest
  pair. About 85% of scored `ens` cells have no training row of their calendar month, so the page
  gives the point estimate and interval, states that limit beside the absolute skill, and builds no
  claim on the contrast.
- *Single against ENS on `single`, exploratory:* `aifs_single − ens_control6` at day 2 and
  `aifs_single − ens_mean6` at days 1 and 2.
- *Single against IFS 0.25° on `single`, exploratory:* `aifs_single_d1 − ifs025_d1`, read one-sided
  as above.
- *AIFS ENS on `ens`, exploratory:* `aifs_ens_mean − ens_mean6` at day 2, and `aifs_ens_mean −
  aifs_single` at days 1 and 2.
- *Spatial read, exploratory:* `aifs_single_d1 − aifs_single_nearest_d1`.
- *How much the 6-hourly steps raise ENS's error, exploratory:* `ens_mean6_d1 − ens_mean_d1`. This
  is also a positive control (below).
- *Climatology reference and null:* `aifs_single_d1 − aifs_single_d1_permuted` and
  `permuted − permuted_b` (see "A climatology reference and a null").
- *Each version era on its own:* `aifs_single_d1 − ens_mean6_d1` in each of the three `single` eras
  and `aifs_ens_mean_d1 − ens_mean6_d1` in each of the two `ens` eras. Each era's line prints the
  point estimate, its rows and months, and how many of its months share the pooled sign. An
  interval is printed only for an era with at least 6 months. The eras differ in season (era 0
  spring and summer, era 1 autumn to spring, era 2 summer), and IFS Cycle 50r1 changes ENS at the
  start of era 2. The page therefore does not read a difference between eras as a version effect,
  and does not describe these lines as evidence on whether AIFS improves.
- *Absolute error of every arm,* with its 95% interval, rows, and months, for each set. This is the
  report's first table, because which input is best can matter more than any contrast
  (`report-absolute-skill-not-only-contrasts`).

**Multiplicity and robustness are reported, not gated.** About 32 intervals print across both
technologies, and about 1 or 2 will exclude zero by chance at the 5% level. The report says so in
its header. It prints the deciding contrast for both technologies at 95% and, through
`studies.bootstrap.bootstrap_difference_at_level`, at the Bonferroni level across the two (97.5%).
For each deciding contrast it prints the leave-one-month-out range (the pattern of
`_leave_one_month_out_lines`), and the result of the `day_of_year`-removed refit. None of these
changes a verdict by itself. The page reports them beside the contrast.

## What changes, file by file

**Edit: `studies/nwp_forecast_comparison/build_forecast_inputs.py`.** Add `aifs_members_frame`
(H3-weighted, plus a nearest-cell variant for the sensitivity arm),
`_aifs_frame` (which also keeps `<arm>_init_time`), `build_aifs`, and a `--aifs` flag (with
`--aifs-dir` for the AIFS store, as the
existing flags name their inputs), plus the defaulted `six_hourly` argument on `ens_member_arms`.
Update the module docstring's list of modes.

**New: `studies/nwp_forecast_comparison/verify_aifs_steps.py`** (read-only checks, below) and
**`studies/nwp_forecast_comparison/fit_aifs.py`** (fits and report, above). `fit_aifs.py` imports
`METRIC`, `SETTINGS`, `TARGET`, `arm_columns`, `assert_equal_rows`, `coverage_table`, `difference`,
`fingerprint`, `leaderboard`, `predictions_from_losses`, and `rows` from `nwp_forecast_comparison`,
and `out_of_fold_losses`, `cut_eras`, and `search_fold_offsets` from `studies.cross_validation`. It
does not copy `fit_extra_leads.py`'s report code; the shared table formatters (`interval_text`,
`error_text`, `contrast_line`) move to `nwp_forecast_comparison.py` as public functions only if the
diff review finds two copies would be needed.

**Edit: `studies/nwp_forecast_comparison/nwp_forecast_charts.py`.** One new caller of
`studies.charts.leaderboard_panel` (per technology, one panel per row set: AIFS Single, AIFS ENS,
and the references' absolute errors on their own rows, day 1, with intervals; two row sets never
share an axis) and of `interval_panel` for
the contrasts, with the row and month counts in the axis title so the marks cannot be read against
Figure 1. Loaded with the `dataviz` skill, and it refuses any site label that is not an anonymised
label, like the other charts. Output `docs/studies/assets/nwp_forecast_{solar,wind}_aifs.svg`,
optimised with `svgo`.

**Edit: `studies/nwp_forecast_comparison/README.md`.** Add the two scripts, the `--aifs` flag, the
output files, and what each holds (the `study` skill's reproducibility rule).

**Edit: `docs/studies/nwp-forecasts-at-matched-leads.md`** (see "Docs to update").

**Not touched, and reported:** `studies/README.md` line 65 still says "In progress — no fit has run
yet" for this study. That row is stale for reasons unrelated to AIFS, so it belongs in its own pull
request off `main`. The `band_steps` wind-first-stamp fix is in the pull request off branch
`band-steps-wind-first-stamp`.

## The verification script

`verify_aifs_steps.py` reads only and writes `verification/` in the new folder:

- **Radiation window.** The script reads ERA5's hourly global irradiance for all 24 hours at the
  site's own ERA5 point from `data/studies/weather/ERA5/` (hour-ending; the path and convention are
  named in the script). For each candidate end offset from -6 to +6 h, it computes the mean absolute
  difference between AIFS Single's radiation at lead `L` and ERA5's mean over the six hours ending
  at
  `L + offset`, pooled over sites, at days 1 and 2. The script fails unless the minimum falls at
  offset 0 and the 06 UTC valid-time subset, taken alone, also has its minimum at 0.
- **Wind is instantaneous.** For each offset from -3 to +3 h, the script correlates AIFS 100 m speed
  at lead `L` with ERA5's 100 m speed (`wind_era5.parquet`) at `L + offset`, and also with ERA5's
  6-hour mean ending at `L`. It fails unless the instantaneous offset 0 correlates best.
- **Units.** AIFS temperature in °C (a mean between -30 and 50), wind in m/s (10 m speed mean under
  15, 100 m above 10 m), radiation between 0 and 1,100 W/m².
- **`NaN` after the run filter.** The count of `NaN` in the extract after the 00 UTC and first-run
  filter, which must be 0.
- **Grid orientation.** The script asserts that `_grid_cells.parquet` latitude rises strictly with
  `lat_index` and longitude with `lon_index`. For each site whose chosen cell is not the crop's
  centre of the 3 by 3 block that AIFS shares with GEFS's crop (which it asserts is indexed 0 to 2),
  it correlates the chosen cell's 2 m temperature anomaly (the cell's value minus the mean of the 9
  shared cells at the same run and lead) with GEFS's control-member anomaly at the same latitude and
  longitude. The script fails unless that correlation is above the correlation with the mirrored
  cell's anomaly. It prints correlations only.
- **Operational runs.** Before the fit, check Dynamical.org's catalogue pages for both products,
  and record in the new folder's `README.md` whether every run from 2025-02-26 was archived as ECMWF
  disseminated it (Risk 12).

## Design-philosophy check

**This is R&D code, and it fails fast.** No asset, serving path, or degradation rule changes, so
`inherent-stability.md` does not apply beyond its own carve-out: "R&D is the opposite: the CV,
training and metrics assets fail fast". The build raises on a missing value in an arm's column, an
uncovered month, an era mismatch, and an existing output file. None of the hypotheses `H1` to `T5.1`
is delivered or affected. The change trades away no principle in `design-principles.md`: the shared
helper gains a defaulted argument, and no new package module is added.

## Tests

No `packages/` code changes, so there are no new unit tests, and the check is the printed report
(`study` skill). Each read-only check below fails on the defect it exists for:

- *Default path unchanged.* `ens_member_arms(six_hourly=False)` for days 1 and 2, on the published
  keys, equals the published parquet's `ens_mean_day{1,2}_*` and `ens_control_day{1,2}_*` columns
  (maximum absolute difference 0). This is a regression guard, and it passes on `main`.
- *The 6-hour path is wired* (fails if `six_hourly` is ignored). For wind, at hours whose UTC hour
  is
  a multiple of 6, `ens_control6_day1_speed_100m` equals `ens_control_day1_speed_100m` exactly, and
  at hours ≡ 3 (mod 6) the two differ on at least 90% of rows. For solar, `ens_mean6_day1_ghi`
  differs from `ens_mean_day1_ghi` on at least 90% of rows.
- *The AIFS time and lead wiring* (fails on a 6-hour offset, a wrong run day, or a time-zone slip).
  With the nearest-cell arm, at wind hours whose UTC hour is a multiple of 6,
  `aifs_single_nearest_day1_speed_100m` equals `sqrt(u² + v²)` of the raw store row at the same
  cell, init `time.date() − 1` 00 UTC, and lead `24 + hour`, to Float32 precision.
- *Positive control, printed with its expected sign.* On each AIFS set, `ens_mean6_day1 −
  ens_mean_day1` is positive. The ENS-horizons report found +0.230 (solar) and +0.236 (wind) points
  with a different feature set. A negative or null result stops the write-up until it is explained.
- *Weights.* Each site's H3 weights sum to 1 within 1e-6, or the build raises.
- The AIFS extract holds no `NaN` after the run filter; every arm on a set has the same row count
  and seven columns; `check_no_missing` and `assert_equal_rows` pass; the era check passes.
- The fold coverage report prints the calendar-month coverage counts for each set.

## Docs to update

**`docs/studies/nwp-forecasts-at-matched-leads.md`.** Present-tense, no history.

- *Title block and first summary paragraph:* add one sentence saying the page also reads ECMWF's
  AIFS Single and AIFS ENS, on rows of their own, and that those results are exploratory. The
  headline sentence is unchanged, because the AIFS comparison cannot alter a published verdict.
- *Key findings:* one bullet, labelled exploratory and post hoc, that states each AIFS number with
  its interval, rows, and months and says AIFS numbers do not share an axis with Figures 1 and 2.
  No bullet says AIFS improves.
- *Data and methods:* a new subsection, "How AIFS is read", after "How the leads are matched". It
  covers the 6-hourly steps and how each is mapped to hourly targets, the like-for-like ENS
  references,
  the one-sided hourly IFS 0.25° reference, the own-row-set rule, and the version eras. The existing
  sentence "Scoring is on the same hours for every arm" gains "except the AIFS arms, which have
  their
  own row set". The "Folds" paragraph gains the AIFS era edges in two sentences.
- *Results:* a new section "AIFS on its own rows (exploratory)", after "Among the other products,
  IFS 0.25° comes closest to ENS", with the two new charts and the tables' figures, each number
  followed by a `<!-- report: ... -->` marker naming its report section.
- *Discussion: what to use:* one paragraph on what the AIFS numbers do and do not change, ending in
  a scoped sentence that has no placeholder. Draft: "The literature we surveyed gave [no
  like-for-like evidence that AIFS improves faster than the physics-based
  IFS](https://openclimatefix.github.io/nged-substation-forecast/background/weather-products-survey/#aifs-has-not-been-shown-to-improve-faster-than-the-physics-based-ifs),
  and each AIFS version is [scored
  separately](https://openclimatefix.github.io/nged-substation-forecast/background/weather-products-survey/#score-each-aifs-version-separately).
  This study's rows span AIFS Single v1.0, v1.1, and v2 and AIFS ENS v1 and v2, one version at a
  time, so its data cannot measure a rate of improvement." The anchors are the headings on `main`
  today (`docs/background/weather-products-survey.md`, lines 228 and 291); recheck them and the
  version dates before the page ships. The page does not merge with any placeholder in it.
- *Limitations:* the 3 by 4 crop and three spatial representations; the 6-hourly steps; the hourly
  IFS 0.25° reference favouring IFS 0.25°; the version blend; the IFS Cycle 50r1 confound; the
  unmeasured AIFS ENS publication time; the short Single v2 era; the 6,220 wind rows at 10 m that
  are not scored; that every fit is on the GPU; and the fold-coverage limit on `ens`. ENS's 9 km
  resolution is stated only
  after the release-notes check above.
- *Scope, Data and code availability, and Reproducing this page:* the AIFS inputs, their start dates
  (2024-04-01 for Single, 2025-07-02 for ENS), the new commands in order, the new folder's
  write-once
  rule, and the commit hash of the last code change (the page's last-change line is updated at the
  end, as the existing hash is).

**`docs/background/weather-products-survey.md`.** The two rows for AIFS Single and AIFS-ENS say
"Not scored". They change to "Scored, exploratory, on the matched-lead page" with a link to the
published page's new section.

**`studies/nwp_forecast_comparison/README.md`** (above). **`docs/studies/index.md`:** the
matched-lead entry, if it lists what the page compares, gains AIFS.

No ship-time roadmap triage applies: this issue completes no roadmap item.

## Verification commands

Run every CI step locally, including the two the `implement-issue` set omits (`pydoclint`, the docs
link checker):

```bash
uv run ruff check . && uv run ruff format --check .
uv run ty check
uv run pytest packages/studies
uv run pymarkdown scan -r docs README.md CLAUDE.md packages/*/README.md
studies/nwp_forecast_comparison/README.md
uv run --isolated --no-project --with pydoclint==0.9.1 pydoclint --quiet --style=google .
uv run python scripts/lint/check_docs_links.py
uv run mkdocs build --strict     # then read the rendered new sections
```

Study runs, in order, one fit job at a time, each a write-once output (after the Opus review of the
fit code):

```bash
uv run python studies/nwp_forecast_comparison/verify_aifs_steps.py --published-dir
data/studies/nwp_forecast_comparison --output-dir data/studies/nwp_forecast_comparison_aifs
uv run python studies/nwp_forecast_comparison/build_forecast_inputs.py --aifs --published-dir
data/studies/nwp_forecast_comparison --output-dir data/studies/nwp_forecast_comparison_aifs
uv run python studies/nwp_forecast_comparison/fit_aifs.py --check --published-dir
data/studies/nwp_forecast_comparison --output-dir data/studies/nwp_forecast_comparison_aifs
uv run python studies/nwp_forecast_comparison/fit_aifs.py --published-dir
data/studies/nwp_forecast_comparison --output-dir data/studies/nwp_forecast_comparison_aifs
--workers 1
uv run python studies/nwp_forecast_comparison/nwp_forecast_charts.py --input-dir
data/studies/nwp_forecast_comparison --extra-dir data/studies/nwp_forecast_comparison_leads
--aifs-dir data/studies/nwp_forecast_comparison_aifs --output-dir docs/studies/assets
```

After the chart command, `git diff --stat docs/studies/assets/` must show only the two new SVGs,
because the command redraws every existing chart.

Only one agent may run these against the shared `data/` folder at a time
(`worktrees-share-one-data-folder`).
Every published number is checked against the printed `report.md`, and the page's marker comments
name
the report section. The published folder's checksums (`sha256sum
data/studies/nwp_forecast_comparison/*.parquet`)
are recorded before the run and compared after it, to prove nothing was written there.

## Review schedule

Both plan reviews (simplicity, then correctness and testability), then the Opus review of the fit
code before the first fit, then the study's two Opus scientific-validity reviews after the first
fit,
then one diff review, then a `prose-review` of the page. There is no mutation pass (see "Size").
Per the `study` skill, the maintainer's authority is needed to merge.

## Risks and open questions

1. **Which bands: days 1 and 2, or more?** Day 3 adds about 25% more fits and no contrast the page
   can use. *Recommendation: 1 and 2.*
2. **The 6,220 wind rows at 10 m only (2024-12 to 2025-02).** Scoring them needs a 10 m-only column
   set for every arm (10 m speed, sine, and cosine, plus the calendar columns), a fourth era for
   runs of unknown version, and its own folds, for 3 months of one technology. *Recommendation: do
   not score them; the page states they are excluded and why.*
3. **Whole-month era edges versus mid-month cuts.** Whole months lose 3 of 19 valid months and need
   no new machinery. A mid-month cut keeps them and needs a new era cutter, `search_fold_offsets`
   variant, and tests in `packages/studies`. *Recommendation: whole months.*
4. **The 2026-05-12 boundary time.** The design says 00 UTC and the READMEs and roadmap say 06 UTC.
   The rows do not depend on it. The era constants use 06 UTC and cite
   `docs/roadmap/data-sources.md`.
   *Question: is the design's 00 UTC from a different source the maintainer wants cited?*
5. **A 6-hour and 3-hour asymmetry remains.** ENS at full resolution, the published reference, keeps
   3-hourly steps. The report shows `ens_mean` at day 1 beside `ens_mean6`, and the page's headline
   comparison uses the 6-hourly one. A reader could read the full-resolution row as the fair one.
   The page's lead states which comparison is which.
6. **Spatial representation.** ENS is the overlap-weighted mean of the 0.25° cells under each
   generator's H3 resolution-5 cell, which is what the live service reads. The AIFS arms use the
   same read: `aifs_members_frame` computes each site's H3 resolution-5 cell weights over the
   crop's 12 cells with `geo.h3.compute_h3_grid_weights`, and raises unless the weights of each site
   sum to 1 within 1e-6. A nearest-cell AIFS Single arm at day 1 (`aifs_single_nearest_day1`) is
   fitted as an exploratory sensitivity arm, so the page can say how much the read moves the result.
   IFS 0.25° stays a point read, and the page says so.
7. **Multiplicity.** About 32 exploratory intervals print. The deciding contrast, for each technology,
   is named before the fit, and the report prints it at 95% and at the Bonferroni level across
   the two (97.5%). For every other contrast, the page quotes only intervals the text needs,
   labels each exploratory and post hoc, and never builds a claim on a single interval near zero.
   With 11 to 16 month clusters, a month bootstrap interval is likely to be somewhat too narrow, so
   each deciding contrast also prints its leave-one-month-out range.
8. **Device.** Every fit is on the GPU, so no contrast mixes devices, and the extra-lead run's
   CPU-to-GPU floor does not bound any AIFS contrast. The page states the device in Limitations. It
   does not print that floor beside AIFS contrasts, because a reader would take it for a noise level
   those contrasts carry.
9. **Sensitivity setting.** Fitted for the two deciding pairs (solar and wind) and every near-line
   result, in the same invocation as the primary fits, so `report.md` is written once. The
   near-line rule is applied mechanically by the script, not by eye.
10. **Dynamical.org's access ends on 2026-09-30.** The AIFS inputs are on disk, and the plan reads
    nothing that cannot be re-run from disk. `README.md` in the new folder records that.
11. **Stale row in `studies/README.md`** (not part of this issue): see "What changes, file by file".
12. **Operational runs, not reforecasts.** Before the fit, check Dynamical.org's catalogue pages for
    both products, and record in the new folder's `README.md` whether every run from 2025-02-26 was
    archived as ECMWF disseminated it, or whether any period was backfilled. If any period was
    backfilled, drop it and cite the source.
13. **Resolved by the maintainer: the `ens` set is scored descriptively only.** On `ens`, about 85%
    of scored (site, fold, calendar month) cells have no training row of their calendar month, so
    `aifs_ens_mean − ens_mean6` carries no deciding label. The page states the limit plainly beside
    the absolute skill. The `single` contrast stays the deciding one.

## AIFS blends and AIFS at long leads (planned before any fit)

**NGED's interest is longer-lead forecasts, so this section extends the AIFS comparison from days 1
and 2 to days 7 and 14, and tests whether blending ECMWF's ensemble mean (ENS mean) with an AIFS
forecast lowers the error at those leads.** The AIFS Single and AIFS ENS stores reach lead 360 h, so
both leads can be scored. The section adds three things to the plan above. It names two tests of
the maintainer's hypothesis, one per lead, of whether AIFS Single has a lower error than the ENS
control member at day 7 and at day 14. It adds blends of the ENS mean with AIFS Single, and with
the AIFS ENS mean, at days 1, 2, 7, and 14, each with a permutation control. It states which
contrasts are deciding, which are exploratory, and why. No fit for any arm named here has run, and
the section is written before one does.

**The wording rule stands.** The survey page's stance is that [AIFS has not been shown to improve
faster than the physics-based
IFS](https://openclimatefix.github.io/nged-substation-forecast/background/weather-products-survey/#aifs-has-not-been-shown-to-improve-faster-than-the-physics-based-ifs)
and that [each AIFS version is scored
separately](https://openclimatefix.github.io/nged-substation-forecast/background/weather-products-survey/#score-each-aifs-version-separately).
The hypotheses below are worded as "has a lower error than" and never as "improves". Every result
sentence carries its scope: 6 solar farms, 3 wind farms, the months of the row set, and the AIFS
versions in those months. The same anonymised labels (A to F, W1 to W3) are used everywhere, and no
output holds a generator's name, identifier, or coordinates.

### Verdict, size, and the five triggers

**Verdict: worth doing, as an addition to the AIFS plan and the same pull request.** The AIFS data
is
on disk, the fit machinery exists in `fit_aifs.py`, and the long-lead question is the reason NGED
would use the result. No other branch or pull request covers it.

**Size: complex, by the same five triggers as above.** The answers change from the AIFS plan in one
place.

1. *What gets stored:* fires. A new write-once folder,
   `data/studies/nwp_forecast_comparison_aifs_blends/`,
   and new page text.
2. *Production serving path:* does not fire. Nothing under `src/`, `packages/ml_core/`, or
   `packages/xgboost_forecaster/` changes.
3. *Degradation rule:* does not fire. This is R&D code, which fails fast.
4. *More than one defensible design:* fires. The row rule at long leads, the blend's control, the
   reference for a blend, and which contrasts are deciding each admit alternatives, listed under
   "Risks and open questions" below.
5. *Callers not nameable without a search:* does not fire. No shared function changes signature.
   `check_runs` in `fit_aifs.py` has one caller, `aifs_rows`, and the build's `--aifs` mode has one
   entry point.

Two triggers fire, so the section gets the plan's reviews, the two Opus scientific-validity reviews,
and one diff review. The mutation pass stays dropped, because `packages/studies/` does not change.
This section has had no plan review yet.

### What the read-only checks found

**Both AIFS wide stores reach lead 360 h, with no gaps, so days 7 and 14 can be read.** I ran these
checks read-only, on 00 UTC runs only, and printed no coordinates.

| Store | 00 UTC runs | First and last run | Leads | Cells | Members | Rows per run |
|---|---|---|---|---|---|---|
| `ECMWF-AIFS` (AIFS Single) | 908 | 2024-04-01 to 2026-09-25 | 61, 0 h to 360 h in 6-hour steps | 12 | 1 | 732 |
| `ECMWF-AIFS-ENS` | 451 | 2025-07-02 to 2026-09-25 | 61, 0 h to 360 h in 6-hour steps | 12 | 51 | 37,332 |

- **Completeness.** Every 00 UTC run holds every lead to 360 h. Rows per lead are 6,924 (577 runs of
  12 cells) for AIFS Single from 2025-02-26 and 276,012 (451 runs of 51 members of 12 cells) for
  AIFS
  ENS, at each of the leads 174, 192, 342, 354, and 360 h I counted.
- **Missing values.** From the first usable Single run (2025-02-26 00 UTC) to the last, no run has a
  `NaN` in shortwave radiation (leads 6 h and beyond), 2 m temperature, or the 10 m and 100 m wind
  components. AIFS ENS holds none from its start. The 330 Single runs before 2025-02-25 hold `NaN`
  radiation and 100 m wind, as the existing plan says.
- **Month coverage of the 00 UTC runs.** Single has every day of every month from 2024-04 to 2026-08
  and 25 days of 2026-09. AIFS ENS has every day from 2025-07-02 to 2026-08 and 25 days of 2026-09.
  The published shared rows end before the last run, so no scored row lacks a run at day 14.

**The row sets shrink at days 7 and 14, because a version era's first target hours need an earlier
run that belongs to the previous era.** Hour `t` at day `N` reads the 00 UTC run `N` days before the
hour's own day. The era check of the AIFS plan asks that run to lie inside the run dates of the
hour's era (`ROW_SETS[...].era_runs`). Two months break the rule at long leads: 2025-03, whose first
target hours need a run before AIFS Single's first usable run (2025-02-26), and 2025-09, whose first
target hours need a run from before the v1.1 start (2025-08-28). The plan drops those rows one row
at
a time and keeps the month. I counted the results on the published shared rows with the plan's own
month rule and the era windows in `fit_aifs.py`:

| Row set | Technology | Day 1 and day 2 rows | Day 7 rows | Day 14 rows | Months (all days) |
|---|---|---|---|---|---|
| `single` | solar | 28,570 | 28,208 (362 dropped: 176 in 2025-03, 186 in 2025-09) | 27,449 (1,121 dropped: 510, 611) | 16 |
| `single` | wind | 28,873 | 28,419 (454 dropped: 240, 214) | 27,440 (1,433 dropped: 735, 698) | 16 |
| `ens` | solar | 16,911 | 16,725 (186 dropped, all 2025-09) | 16,300 (611 dropped) | 11 |
| `ens` | wind | 18,555 | 18,341 (214 dropped) | 17,857 (698 dropped) | 11 |

The day 1 and day 2 counts equal the existing AIFS fit's, and the report prints all of them. The
drop is 1.3% to 5.0% of `single` rows. The fold design of the AIFS plan (`AIFS_FOLD_OFFSETS`, the
identity rotation) still covers every calendar month at days 7 and 14: `search_fold_offsets` returns
it on all 12 (row set, technology, lead) frames, with 13 designs for `single` and 4 for `ens`. The
uncovered (site, fold, calendar month) cells are the same as at day 1: 37 of 95 (solar) and 18 of 48
(wind) on `single`, 55 of 65 and 27 of 33 on `ens`. The `ens` set stays descriptive for the reason
already stated.

**Each lead gets its own frame, because its rows differ.** The frame for (row set, lead) is the
shared rows, the month rule, and the run-in-era rule. The folds are recut on that frame. Every arm
scored at a lead, blend or reference, is fitted on that lead's frame, so every contrast at a lead
compares arms on the same rows. XGBoost's row subsampling depends on row order, so each frame is
sorted by `(site, time)` as `aifs_rows` does.

### The served lead at each day, and which arms have it

**Day `N` reads the 00 UTC run `N` days before the hour's own day, for AIFS Single, the AIFS ENS
mean, the ENS control member, and the ENS mean alike.** That is the rule `band_steps` applies and
the
rule the extra-lead batches state for ENS. The leads follow.

| Day | Solar leads (hour ending at the label) | Wind leads (instant at the label) |
|---|---|---|
| 1 | 25 h to 48 h | 24 h to 47 h |
| 2 | 49 h to 72 h | 48 h to 71 h |
| 7 | 169 h to 192 h | 168 h to 191 h |
| 14 | 337 h to 360 h | 336 h to 359 h |

The last solar lead at day 14 is exactly the store's maximum (360 h), and the band's margin
(`24 * day + 30`) clips there. ENS steps are 6-hourly beyond 144 h and AIFS steps are 6-hourly
throughout, so **at days 7 and 14 AIFS and ENS have the same step width without the `six_hourly`
coarsening the day 1 and day 2 references need**, and ENS's radiation is already a 6-hour mean. The
lead chart therefore mixes `ens_control6_day1` and `ens_control6_day2` (coarsened) with
`ens_control_day7` and `ens_control_day14` (native), and the page says so under the chart. The ENS
control member's own leads at days 7 and 14 are those in the table. The AIFS build makes the ENS
columns at days 7 and 14 itself (see "What changes, file by file"), stamps each with its run's
`_init_time`, and `check_runs` verifies the ENS run dates as it verifies the AIFS run dates.

**The AIFS ENS mean at days 7 and 14 is read as the ENS mean is.** Each of the 51 members is
upsampled
from 6-hourly steps to hourly targets with `UPSAMPLING_METHODS` (radiation through the clear-sky
index, temperature linearly, wind as vector components), and then `reduce_members` averages the
members. AIFS radiation is a mean over the 6 hours ending at the lead, so no window inversion
applies
(`README.md` of the store), and `verify_aifs_steps.py` re-runs its ERA5 offset check at days 7 and
14. The build must not filter leads above `24 * 2 + 30`: the `aifs_members_frame` lead cap becomes
`24 * max(AIFS_DAYS) + 30 = 366 h`, which is above the store's last lead. The scan also filters
from below to each band, `[24 N − 6, 24 N + 30]` h for each day `N` requested, because the AIFS ENS
file holds 50 million rows and a scan of every lead reads about 10 times the rows used.

### Design: the P4 pattern with a permutation control

**A blend follows the published P4 pattern: a blend arm, a control arm with the same columns in
which the second product is shuffled, and a verdict that needs both.** The published page defines
P4a and P4b (`BLEND_ARMS`, `add_blend_guard_columns`, `blend_verdict` in
`nwp_forecast_comparison.py`). The blend gives one XGBoost model ENS's columns and the second
product's columns. The control gives the same model ENS's real columns and the second product's
columns shuffled among the hours that share a site, a year-month, and an hour of day
(`studies.blending.climatology_permutation`), so the control keeps the column count and removes the
second product's information. A blend gain is believed only if the blend beats its control as well
as
ENS alone.

**Each blend reads the full-resolution ENS mean and one AIFS forecast at the same day.**

- *Blend of ENS mean and AIFS Single* (`single` row set, days 1, 2, 7, 14): solar 9 columns (the 5
  calendar and sun columns, ENS's 2 weather fields, AIFS Single's 2), wind 11 columns (3 calendar
  columns, then 4 weather fields from each product).
- *Blend of ENS mean and AIFS ENS mean* (`ens` row set, days 1, 2, 7, 14): the same columns from the
  AIFS ENS mean.
- *ENS mean at days 1 and 2 is the full-resolution column* (3-hourly to 144 h), because a blend
  reads the ENS mean as a user gets it. The AIFS-alone reference for the same blend has 6-hourly
  steps.

**The blend's row set is the intersection of both products' rows, and that intersection is the AIFS
row set.** Every published shared row already holds every ENS value (the extra-lead reports print a
0.0000% missing share for the ENS mean and control at every day used here, and `check_no_missing`
raises on any null in an arm's columns). So the intersection with ENS removes nothing from `single`
or `ens`, and the references are fitted on that same frame.

**Three references for each blend, all fitted on the blend's frame, on the GPU.**

1. *ENS mean alone* (7 columns). The gain the blend is judged against. It has fewer columns than the
   blend, so the blend minus this reference is favoured by the extra columns, and the control
   removes
   that favour.
2. *AIFS alone* (7 columns): `aifs_single_dayN` or `aifs_ens_mean_dayN`. The plan adds a **mirror
   control** as its column-matched reference:
   AIFS real and ENS's columns shuffled (seed 0, which moves ENS's columns as the blend control
   moves
   AIFS's, and does no harm), on `single` only, at all four days. The mirror control is not in the
   published P4 pattern, which guards only the ENS side, and the maintainer has approved keeping it.
   The report gives "blend against AIFS alone" only as the blend minus its mirror control, because
   the
   extra columns favour the blend minus the 7-column AIFS-alone arm, and that interval adds nothing
   the
   mirror contrast lacks. AIFS alone's absolute error stays in the leaderboard.
3. *The control* (ENS real, AIFS shuffled), for the guard contrast.

**Shuffle seeds, folds, and rows.** The shuffle uses seed 0, the seed of `aifs_single_day1_permuted`
in the existing fit, applied to each lead's own frame (the groups are within site, year-month, and
hour of day, so a shuffled value never crosses a fold). The pure AIFS climatology arm
(`aifs_single_dayN_permuted`, 7 columns) and the blend control's shuffled columns are therefore the
same values. The null arm (`..._permuted_b`, seed 1000) is fitted at days 7 and 14 only. Folds are
the
era-aware folds of `aifs_rows` (three eras for `single`, two for `ens`), recut on each lead's frame.
Fitting seeds are XGBoost's 0, 1, and 2, as in every fit of this study.

**The verdict for one lead is a four-line function in `fit_aifs.py`.** It returns "lowers the error
at day N" when the blend minus ENS alone and the blend minus its control both have an upper 95%
bound
below 0. Otherwise it returns "no detectable difference", with the largest gain the blend's lower
bound leaves open. `combine_setting_verdicts` joins the two settings, as the published blend does.
`studies.bootstrap.blend_verdict` is not reused, because its label, "lowers the day-ahead error",
names the day-ahead lead and would print at days 7 and 14. As in the published report, a control
that
is itself significantly worse than ENS alone makes the guard uninformative, and the report lists
such
controls.

**Two departures from the published P4 pattern are stated on the page.** The shuffle seed is 0,
where
the published guard uses 20260920 plus the product's index, and the verdict label names the lead.

### Design: AIFS at long leads, and the references at days 7 and 14

**The maintainer's hypothesis needs one contrast per lead, on `single`, between two forecasts of one
kind.** AIFS Single is a single deterministic forecast. Its like-for-like reference is the ENS
control member, an unperturbed forecast of the same physics as IFS, from the same 00 UTC analysis
time, at the same lead and, beyond 144 h, the same 6-hourly steps. The contrast at day `N` is
`aifs_single_dayN − ens_control_dayN`, N = 7 and 14, both arms fitted on the same GPU on the same
frame. The page may say "AIFS Single's forecast against the ENS control member's". It may not say
"machine learning against physics" or that AIFS "improves", because the two forecasts differ in
native
grid and in what they were trained on. Before the page states the resolution difference, the
resolution of the control member is checked against ECMWF's release notes, as for days 1 and 2.

**AIFS Single may have the lower error because it is smoother, not because its weather is better.**
At day 7 the ENS mean beats the ENS control member by about 0.7 points on the published rows
(`ens_control_day7 − ens_mean_day7` is +0.656 [+0.292, +0.998] for solar and +0.764 [+0.155, +1.485]
for wind). AIFS Single is a deterministic forecast from a machine-learned weather model trained on a
squared-error loss, and such models are widely reported to blur their fields as lead grows. H7's
reference, the control member, is the sharp single physics forecast, so H7 can come out significant
in
AIFS's favour because AIFS Single is smoother. The day-1 wind results already fit that pattern: AIFS
Single against the control member is −0.581 [−1.126, −0.150] and against the 6-hourly ENS mean is
−0.182 [−0.514, +0.100]. H7 and H14 are therefore always reported beside
`aifs_single_dayN − ens_mean_dayN`, which compares AIFS with a physics forecast that averaging has
smoothed. If AIFS Single has a lower error than the control member but not than the ENS mean, the
page says the result is consistent with smoothing, and does not describe AIFS as the better weather
forecast. The report prints, for each arm and lead, the standard deviation of the arm's weather
columns over the scored rows (no fit needed), so that a reader can see which forecast is smoother.
`aifs_single − ens_mean` stays exploratory: it is the interpreting companion, and adds nothing to
the
deciding count.

**The reference that lets the page say "IFS" at day 14 is the ENS control member only.** The archive
limits the other IFS references, and the page states the limits plainly:

| Reference | Day 7 | Day 14 |
|---|---|---|
| ENS control member (`ens_control_dayN`, native steps) | the AIFS build's own column; refitted here | same |
| ENS mean (`ens_mean_dayN`) | the AIFS build's own column; refitted here | same |
| IFS 0.25° (`ifs025_dayN`, Open-Meteo Previous Runs) | joined from `nwp_forecast_comparison_leads_day10` (`_previous_day7`); refitted here; hourly, lead `24 * 7 + (h mod n)` | **none**: the API stops at `_previous_day7` |
| IFS HRES 9 km (`ifs_single_dayN`, Open-Meteo Single Runs) | joined from `nwp_forecast_comparison_leads_day10d`; refitted here as an exploratory reference | **none**: the archive's leads end at 240 h |

The AIFS build's ENS columns at days 7 and 14 (`ens_mean6_dayN`, `ens_control6_dayN`) equal the
native `ens_mean_dayN` and `ens_control_dayN` of the extra-lead folders, because ENS has no step to
coarsen beyond 144 h. This section writes `ens_mean_dayN` and `ens_control_dayN` for them, and a
check
asserts the equality on every row (see "Tests").

At day 14, the AIFS-versus-IFS test rests on the ENS control member. The control member is the same
physics model as IFS HRES, read from ECMWF's ensemble product, and the plan does not present it as
HRES. Neither IFS 0.25° nor IFS HRES 9 km reaches day 10 or day 14 in the archives read, and the
page
says the day-14 result cannot be checked against either. At day 7 the two IFS references are
secondary: IFS 0.25° serves hourly values and a lead that is never longer than AIFS's 24N + h, and
both differences favour IFS 0.25°, so the one-sided reading of the day 1 plan applies (a result in
AIFS's favour survives the bias, and a result against AIFS is unresolved). IFS HRES's missing target
days (batch 4 report) are left missing in its arm, and its contrasts run on the rows both arms
score,
as batch 4 did.

**A reading rule fixed before any fit protects the day-14 test from its likely null.** The published
extra-lead reports show that, on the published shared rows, the ENS mean's day-14 error is not lower
than climatology's (+0.418 points [-0.325, +1.108] for solar, +0.406 [-0.571, +1.341] for wind), and
neither is the ENS control member's (+0.307 and +0.349). The reference in that evidence,
`studies.baselines.climatology`, is the out-of-fold median power per site, calendar month, and hour,
with no XGBoost model. The rule's reference is stricter: an XGBoost model given AIFS's weather
shuffled within each year-month and hour, which keeps each month's mean AIFS value at each hour. The
page names the reference that way. If neither AIFS Single nor the ENS control member has an error
significantly below **both** shuffled-AIFS arms (`aifs_single_day14_permuted` and `_permuted_b`,
seeds
0 and 1000, primary setting, 95%) on the day-14 frame, the day-14 hypothesis is reported as "no
skill
to compare at day 14", and H14's interval is still printed. Two shuffle seeds gate the rule because
the existing AIFS fit found the two seeds of the same shuffle differ significantly (+0.453
[+0.223, +0.696] for solar at day 1), which is about the size of the quantity the rule reads. A wide
interval that spans zero is then bounded, not read as equal skill: the page states the largest
difference the interval does not exclude, following the study's rule for null results.

**The intervals at days 1 and 2 give a sense of the power.** The existing fit's contrasts at day 1
and day 2 (solar) have 95% intervals 0.69 and 0.62 points wide. The plan expects intervals at least
as
wide at day 14, where error is larger, so a gap under about 0.3 points is unlikely to be resolved
for solar, and a gap under about 0.5
points for wind (the day-1 and day-2 wind intervals are about 0.98 and 1.0 points wide). The page
says so where it reports a null.

**Positive control and instrument check.** The ENS control member minus the ENS mean at day 7 must
be
positive, as in the published reports (+0.656 points [+0.292, +0.998] solar, +0.764 [+0.155, +1.485]
wind). If either is not positive on the new frames, the write-up stops until the difference is
explained. Day 14 has no expected sign (published -0.110 and -0.057, both unresolved), so the
day-14 instrument check is the climatology contrast above.

### Fit design: devices, sizes, and resuming

**Device rules.** Every arm is fitted with `device="cuda"`, one device per contrast, and every arm
is
refitted in this folder, days 1 and 2 included, so no contrast reads a fit from another run. The
build writes days 1, 2, 7, and 14 into the new folder. An optional check asserts that the refitted
`aifs_single_day1` losses equal the saved ones in `nwp_forecast_comparison_aifs`, a free determinism
check across runs.

**The GPU-versus-CPU difference is stated, not corrected.** GPU XGBoost gives a different fit from
CPU:
in the extra-lead batches solar differed by within ±0.02 points, and wind by 0.04 to 0.09 points
lower
on GPU. Every arm in every contrast here is on the GPU, so the difference cancels within a
contrast, and the page states the device once, in Limitations. The report does not print the
CPU-versus-GPU figures beside a contrast, because a reader would take them for a noise level the
contrast carries. Absolute
errors here are not compared with the published CPU-fitted page's, because the rows differ.

**Sizes.** The AIFS fit's speed is about 9 s per arm-site fit at the primary setting (the setting
with
500 boosting rounds; the sensitivity setting has 1,200 rounds and is slower).

| Stage | Arms per technology (primary) | Arm-site fits, solar (6 sites) | Arm-site fits, wind (3 sites) |
|---|---|---|---|
| `single`, primary | 30 (AIFS Single, ENS mean, and ENS control at days 1, 2, 7, 14; 12 blends, controls, and mirror controls; `ifs025` and IFS HRES at day 7; 4 climatology and null arms at days 7 and 14) | 180 | 90 |
| `single`, sensitivity and `day_of_year`-removed refits of the deciding contrasts | 14 (10 sensitivity, 4 refit) | 84 | 42 |
| `single`, near-line extras, upper estimate | 10 | 60 | 30 |
| `ens`, primary | 16 (AIFS ENS mean and ENS mean at days 1, 2, 7, 14; the AIFS ENS blend and its control at the same days) | 96 | 48 |
| Total | 70 | 420 | 210 |

That is 630 arm-site fits, about 1.6 hours at 9 s each and at most 3 hours after the slower
sensitivity
fits. `--check` prints the measured estimate before the full run. `--workers 1`.

**Resuming.** The script writes each stage's losses file
(`<domain>_<row_set>_<stage>_losses.parquet`
plus its `.json`) and its `_predictions.parquet` (so a later chart needs no refit) when the stage
finishes, and skips a stage whose file exists, so a crash costs at most one stage of about 30
minutes. `report.md` refuses to be overwritten and is written once, after the last stage.

**Folders the run reads but must not write.** These are the published folder,
`nwp_forecast_comparison_aifs`, and the extra-lead folders `nwp_forecast_comparison_leads_day10` and
`nwp_forecast_comparison_leads_day10d`. The existing baseline `/tmp/claude-1000/pub-sha-before.txt`
is never overwritten. A new baseline, written to a new file under absolute paths, covers every file
of
every one of those folders (see "Verification commands").

**Determinism check.** `--check` fits `aifs_single_day7` at one wind site twice on the GPU and stops
unless the two fingerprints agree. Check CPU load with `uptime` first. Only one agent runs a
published
study script at a time.

### Contrasts named before any fit

**Four contrasts per technology are deciding, eight in all, and every other contrast is
exploratory.** The deciding contrasts are:

- **H7:** `aifs_single_day7 − ens_control_day7`, on `single`.
- **H14:** `aifs_single_day14 − ens_control_day14`, on `single`, read under the day-14 reading rule
  above.
- **B7:** the blend of ENS mean and AIFS Single at day 7, minus ENS mean alone, with the guard (the
  blend minus its control), on `single`.
- **B14:** the same at day 14.

**These four are deciding contrasts named before any fit of this section, not planned contrasts in
the page's sense.** The day-1 and day-2 AIFS results and the ENS extra-lead results at days 7 and 14
were known when they were named: AIFS Single minus the ENS control member at day 2 is −0.608
[−0.925, −0.302] (solar) and −0.643 [−1.195, −0.188] (wind), and the ENS control member minus the
ENS mean at day 7 is +0.656 (solar) and +0.764 (wind). The page says so beside each result.

**Each deciding contrast is claimable only if the study's full rule agrees.** Its 95% interval is
statistically significant at the 5% level at both hyperparameter settings, with the same sign. The
range of the leave-one-month-out point estimates keeps that sign. For H7 and H14, the refit without
`day_of_year` in either arm agrees in sign. For B7 and B14, the guard is significant too at both
settings. The report also prints each deciding contrast's interval at the Bonferroni level across
all
eight (99.375%), as robustness that changes no verdict on its own. The near-the-line rule applies to
every other contrast: any with an interval bound within 20% of the interval's width from zero gets
its
two arms fitted at the sensitivity setting, in the same invocation.

**Reasons for the deciding set.** H7 and H14 are the maintainer's hypothesis, one per lead, on the
row
set (`single`) where the fold-coverage limit is smaller than on `ens`. B7 and B14 are the blends at
the two leads that NGED's longer-lead interest names. Days 1 and 2 are exploratory for blends to
hold the deciding set to four per technology. `ens` results are descriptive for the reason in the
AIFS plan (about 85% of scored
(site, fold, calendar month) cells have no training row of their calendar month).

### What changes, file by file

**Edit: `studies/nwp_forecast_comparison/build_forecast_inputs.py`.** `--aifs` gains `--aifs-days`
(default `1 2`, so the existing behaviour is unchanged), and `AIFS_DAYS` becomes an argument of
`aifs_members_frame`, `_aifs_frame`, and `build_aifs`. The lead cap becomes `24 * max(days) + 30`,
with the lower bound per band described above. The build writes `aifs_single_dayN`,
`aifs_ens_mean_dayN`, `ens_mean6_dayN`, and `ens_control6_dayN` for every requested day, each with
its
`_init_time` (`keep_init_time=True` on the ENS `ens_member_arms` call), to
`<domain>_aifs_inputs.parquet` in the new output folder, and builds the nearest-cell arm only when
day 1 is requested. The run uses `--aifs-days 1 2 7 14`. `build_aifs` refuses an output folder equal
to `nwp_forecast_comparison_aifs`.

**Edit: `studies/nwp_forecast_comparison/verify_aifs_steps.py`.** The ERA5 window-offset, wind-lag,
unit, grid-orientation, and `NaN` checks run at days 7 and 14 as well as 1 and 2, and a new check
recomputes the H3-weighted mean from the raw store for a sample of (site, run, lead) and compares.

**Edit: `studies/nwp_forecast_comparison/fit_aifs.py`.**

- `check_runs`: derive the checked prefixes from the row set's arms (every arm that carries a run
  stamp and is not a shuffled copy), and parse each prefix's day with
  `re.fullmatch(r".*_day(\d+)", prefix)`, raising if there is no match. Today `int(prefix[-1])`
  reads
  4 for `aifs_single_day14`, and the hard-coded `AIFS_ARM_PREFIXES` tuple would skip the day-7 and
  day-14 arms without a word, because the loop skips any prefix not in the tuple. The ENS arms'
  `_init_time` stamps are checked the same way.
- `fit_jobs`: the expected column count comes from the arm's kind (7 for a single product, 6 without
  `day_of_year`, 9 for a solar blend or control, 11 for a wind blend or control), not a fixed 7.
  Today the fixed check raises on every blend.
- `RowSet` and `aifs_rows` gain a lead argument `day`, and `aifs_rows` drops rows whose run lies
  outside
  the era window (a row filter, before the fold recut).
- A `--blends` mode reads the days-1/2/7/14 inputs from the new folder, joins only `ifs025_day7`
  (from
  `nwp_forecast_comparison_leads_day10`) and `ifs_single_day7` (from
  `nwp_forecast_comparison_leads_day10d`) by `(site, time)` (raising if a shared row is missing),
  and
  builds the arms, controls, mirror controls, and stages of the table above. It also asserts that
  the
  build's `ens_mean6_day{7,14}` and `ens_control6_day{7,14}` equal the extra-lead folders'
  `ens_mean_day7`, `ens_control_day7`, `ens_mean_day14`, and `ens_control_day14` on every row, to
  Float32 precision.
- The contrast list, the one-lead verdict function, the day-14 reading rule, the smoothness
  diagnostic, the Bonferroni line, the leave-one-month-out lines, and the printed feature columns of
  every arm are written into `report.md`, with rows and months on every line and the absolute error
  of every arm first.

**Edit: `studies/nwp_forecast_comparison/nwp_forecast_charts.py`.** Per technology: one panel of
absolute error against lead (days 1, 2, 7, 14) for AIFS Single, ENS control, ENS mean, the blend,
and
the day-7 IFS references, with 95% intervals; a second panel of the four deciding contrasts and the
exploratory contrasts of the same kind, drawn with `interval_panel`, each row marked deciding or
exploratory; row and month counts in the axis titles; `ens` rows in a separate
panel. Output `docs/studies/assets/nwp_forecast_{solar,wind}_aifs_leads.svg`, optimised with `svgo`.
The chart code refuses a site label that is not an anonymised label.

**Edit: `studies/nwp_forecast_comparison/README.md`** (the new flag, mode, folder, and what each
file
holds) and **`docs/studies/nwp-forecasts-at-matched-leads.md`** (see "Docs to update"). No file
under
`packages/` changes and no Patito contract changes.

### Design-philosophy check

This is R&D code and fails fast. The build and the fit raise on a missing value in an arm, an era
mismatch, an existing output file, and a shared row missing from a reference folder. No asset,
serving path, or degradation rule changes,
and no hypothesis label (`H1` to `T5.1`) is delivered or affected. The change trades away no
principle in `design-principles.md`.

### Tests

The scripts have no unit tests, so each check below is a read-only run whose output goes into the
report, and each names the defect it would fail on. The first two fail on today's code.

- *Day-14 run check (fails on today's `int(prefix[-1])` parsing).* `check_runs` on a frame whose
  `aifs_single_day14_init_time` is 14 days before the hour's day passes, and with 13 days raises. On
  today's code the correct 14-day frame raises, because the expected date is computed from 4 days,
  and
  a 4-day frame passes. The same test asserts that `check_runs` inspects `aifs_single_day7` and
  `aifs_single_day14`: a frame whose `aifs_single_day7_init_time` is wrong raises. On today's code
  the
  hard-coded prefix tuple skips day 7, so the wrong frame passes. The negative half cannot live in
  `report.md`, so it runs as a throwaway invocation whose output goes in the PR body.
- *Blend columns.* `fit_jobs` accepts a 9-column solar blend and an 11-column wind blend, and still
  raises on an arm whose column count differs from its kind. Today it raises on every blend.
- *Days 7 and 14 exist.* `build_forecast_inputs.py --aifs --aifs-days 1 2 7 14` writes
  `aifs_single_dayN`, `aifs_ens_mean_dayN`, `ens_mean6_dayN`, `ens_control6_dayN`, and their
  `_init_time` columns for N in 1, 2, 7, and 14. Today the flag does not exist and the day-7 and
  day-14 columns are absent.
- *ENS columns equal the extra-lead folders'.* The build's `ens_mean6_day{7,14}` and
  `ens_control6_day{7,14}` equal `ens_mean_day7`, `ens_control_day7`, `ens_mean_day14`, and
  `ens_control_day14` from the extra-lead folders on every row, to Float32 precision. A wrong run
  day, lead offset, or step handling in the ENS read fails it.
- *Wiring.* For wind hours whose UTC hour is a multiple of 6, the weighted 100 m speed at day `N`
  equals an independent recomputation from the raw store at init date `− N` days and lead `24 N +
  h`,
  to Float32 precision, for N = 7 and 14. It fails on a wrong run day, a 6-hour lead offset, or a
  time-zone slip.
- *Radiation window.* The ERA5 offset check has its minimum at offset 0 at days 7 and 14. It fails
  if
  AIFS's window convention changes with lead.
- *Row counts.* The report's row and month counts equal the table above (for example 28,208 rows at
  day 7 and 27,449 at day 14 for solar `single`), and every kept row's run lies in its era window.
- *Shuffle and column counts.* Each control has the same column count as its blend (9 solar, 11
  wind);
  its shuffled columns hold the same multiset of values as the real columns within every (site,
  year-month, hour) group and differ from them on at least 90% of rows; no shuffled value moves
  between year-months. The seed-0 blend control and `aifs_single_dayN_permuted` hold identical
  shuffled values.
- *Same rows.* `assert_equal_rows` passes for every pair in a contrast, and `check_no_missing`
  passes
  on every arm except IFS HRES, whose missing values are left missing on purpose.
- *Positive control.* `ens_control_day7 − ens_mean_day7` is positive on both technologies.
- *Determinism.* `--check` fingerprints agree across two GPU fits of `aifs_single_day7`.
- *Published folders untouched.* The old baseline `/tmp/claude-1000/pub-sha-before.txt` passes
  `sha256sum -c` before the run, and the new baseline
  `.claude/worktrees/scratch/matched-lead/sha-before-aifs-blends.txt` passes after it. The new
  baseline lists every file of every folder the run reads but must not write.
- *Smoothness diagnostic.* The report prints the standard deviation of each arm's weather columns
  per
  lead over the scored rows, for every arm of H7, H14, and their `ens_mean` companions.

### Docs to update

**`docs/studies/nwp-forecasts-at-matched-leads.md`**, in the present tense and scoped to the sites,
months, and AIFS versions scored.

- *Data and methods:* extend "How AIFS is read" with the days 7 and 14 leads and the row-level era
  rule, the blend and its control (the columns, the shuffle, and the seeds), the references at each
  day
  and their limits (IFS 0.25° and IFS HRES stop before day 14), the day-14 reading rule, the
  smoothing reading
  rule, and the list of deciding contrasts with what was already known when they were named. The
  page's definition of "planned" is not used for them.
- *Results:* a section "AIFS at days 7 and 14 and blends of AIFS with the ENS mean", with the two
  new
  charts. The bolded lead of each subsection states the finding for the row set and technology, with
  its interval, rows, and months. Nothing says AIFS improves.
- *Key findings:* one bullet per deciding contrast, each labelled deciding and each carrying its
  interval.
  Exploratory bullets are labelled exploratory and post hoc.
- *Discussion: what to use:* what the day 7 and day 14 result means for a user reading forecasts at
  long leads, limited to the rows scored, with the survey page's stance quoted and linked (the two
  anchors above, rechecked before the page ships).
- *Limitations and Scope:* the day-14 limit on IFS references, the version blend and the IFS Cycle
  50r1 confound in the last era, the device, the `ens` fold-coverage limit, the missing HRES days,
  and
  that AIFS Single's version varies by month.
- *Data and code availability and Reproducing the figures:* the new commands, the folder, and the
  commit hash of the last code change.

**`docs/background/weather-products-survey.md`** and **`docs/studies/index.md`**: the AIFS rows and
the matched-lead entry mention days 7 and 14 and blends where they list what was scored.
**`studies/nwp_forecast_comparison/README.md`:** above. No roadmap item completes.

### Verification commands

The AIFS plan's verification set applies unchanged (`ruff`, `ty`, `pytest packages/studies`,
`pymarkdown`, `pydoclint`, `check_docs_links.py`, `mkdocs build --strict`). The run commands, in
order, one job at a time after the Opus review of the fit code:

```bash
D=/home/jack/dev/nged-substation-forecast/data/studies
B=/home/jack/dev/nged-substation-forecast/.claude/worktrees/scratch/matched-lead
sha256sum -c /tmp/claude-1000/pub-sha-before.txt   # the existing baseline must still pass first
find $D/nwp_forecast_comparison $D/nwp_forecast_comparison_aifs \
  $D/nwp_forecast_comparison_leads_day10 $D/nwp_forecast_comparison_leads_day10b \
  $D/nwp_forecast_comparison_leads_day10d -type f -print0 | sort -z | xargs -0 sha256sum \
  > $B/sha-before-aifs-blends.txt
uv run python studies/nwp_forecast_comparison/build_forecast_inputs.py --aifs \
  --aifs-days 1 2 7 14 --published-dir $D/nwp_forecast_comparison \
  --output-dir $D/nwp_forecast_comparison_aifs_blends
uv run python studies/nwp_forecast_comparison/verify_aifs_steps.py \
  --published-dir $D/nwp_forecast_comparison --output-dir $D/nwp_forecast_comparison_aifs_blends
uv run python studies/nwp_forecast_comparison/fit_aifs.py --blends --check \
  --published-dir $D/nwp_forecast_comparison --output-dir $D/nwp_forecast_comparison_aifs_blends
uv run python studies/nwp_forecast_comparison/fit_aifs.py --blends --workers 1 \
  --published-dir $D/nwp_forecast_comparison --output-dir $D/nwp_forecast_comparison_aifs_blends
sha256sum -c $B/sha-before-aifs-blends.txt
uv run python studies/nwp_forecast_comparison/nwp_forecast_charts.py \
  --input-dir $D/nwp_forecast_comparison --aifs-dir $D/nwp_forecast_comparison_aifs_blends \
  --output-dir docs/studies/assets
git diff --stat docs/studies/assets/   # only the two new SVGs
```

Every number on the page is checked against the printed `report.md`, and the page's marker comments
name the report section. Flag names above (`--aifs-days`, `--blends`) are proposals for the
implementer, and the chart command's flags follow the AIFS plan's chart command.

### Risks and open questions

1. **Mirror control and row-level era drop: decided.** The maintainer has approved keeping the
   mirror
   control and the row-level era drop (1.3% to 5.0% of `single` rows).
2. **Day 14 may hold no skill to compare.** The published reports found no ENS skill over
   climatology
   at day 14. The reading rule makes a bounded null a stated result rather than a hidden one.
   *Recommendation: run day 14 regardless, because a bounded null at the lead NGED cares about is
   itself the finding.*
3. **IFS HRES at day 7 as an exploratory reference.** It adds one arm per technology, and the finest
   IFS available. *Recommendation: keep.*
4. **The ENS control member's resolution.** The page states a resolution difference between AIFS and
   the control member only after ECMWF's release notes confirm it.
5. **Blends read the full-resolution ENS mean at days 1 and 2.** The blend then has finer ENS steps
   than
   the AIFS reference. The report prints `ens_mean_dayN` and the blend on the same rows, and the
   page
   states which references share step width. *Recommendation: keep, because a user blends the ENS
   product as served.*
6. **Multiplicity.** The report header prints the exact count of printed intervals, computed from
   the
   contrast list, and the expected number significant at the 5% level by chance (the review counted
   about 45 per technology before the sensitivity lines, so about 4 to 5 by chance across both). The
   page quotes only the intervals its text needs
   and never builds a claim on a single exploratory interval near zero.
7. **Versions.** AIFS Single's version changes by month inside the row set, and the last era is
   confounded with IFS Cycle 50r1. The exploratory by-era contrasts print a point estimate and an
   interval only for eras of at least 6 months, as in the AIFS plan.
8. **Data access ends on 2026-09-30.** Everything the plan reads is on disk, so no download is
   needed.

### Deciding and exploratory contrasts

Differences are first minus second, in percentage points of capacity, and a negative difference
means
the first arm has the lower error. Every contrast prints its rows, its months, and the absolute
error
of both arms.

| Contrast | Row set | Status | Reason |
|---|---|---|---|
| H7: `aifs_single_day7 − ens_control_day7` | `single` | **deciding** | The maintainer's hypothesis at day 7, between two single forecasts of the same lead and steps |
| H14: `aifs_single_day14 − ens_control_day14` | `single` | **deciding**, under the day-14 reading rule | The same at day 14, where only the control member is an IFS reference |
| B7: blend of ENS mean and AIFS Single at day 7 − `ens_mean_day7`, and its guard | `single` | **deciding** | NGED's long-lead interest; the P4 pattern names the blend and its control |
| B14: the same at day 14 | `single` | **deciding** | The same at day 14 |
| `aifs_single_dayN − ens_mean_dayN`, N = 7 and 14 | `single` | exploratory, reported beside H7 and H14 | Interprets H7 and H14: a lower error than the control member but not the ENS mean is consistent with smoothing |
| `aifs_single_day7 − ifs025_day7`, and `− ifs_single_day7` | `single` | exploratory, one-sided | The IFS references favour IFS, so only a result in AIFS's favour survives |
| `aifs_single_dayN − aifs_single_dayN_permuted` and `− _permuted_b`, and permuted minus permuted_b (N = 7, 14) | `single` | exploratory | Climatology references and the pipeline's null |
| `ens_control_day14 − aifs_single_day14_permuted` and `− _permuted_b` | `single` | exploratory (day-14 reading rule) | Read by the day-14 reading rule, which needs both shuffle seeds |
| `ens_control_day7 − ens_mean_day7` | `single` | exploratory, positive control | Must be positive at day 7 |
| Blend of ENS mean and AIFS Single at days 1 and 2, minus `ens_mean_dayN` and minus its control | `single` | exploratory | Kept exploratory to hold the deciding set to four per technology |
| Blend at days 1, 2, 7, 14 minus its mirror control | `single` | exploratory | The column-matched comparison with AIFS alone |
| `aifs_single_dayN − ens_control_dayN` by version era, N = 7 and 14 | `single` | exploratory | Interval only for eras of at least 6 months; not read as a version effect |
| The gap at day 14 minus the gap at day 7, on the day-14 rows | `single` | exploratory | Does the gap widen with lead (built as a difference of summed per-row losses, then `bootstrap_difference`) |
| `aifs_ens_mean_dayN − ens_mean_dayN`, N = 7 and 14 | `ens` | descriptive | About 85% of scored cells have no training row of their calendar month |
| Blend of ENS mean and AIFS ENS mean at days 1, 2, 7, 14 minus ENS mean alone and minus its control | `ens` | descriptive | The same limit. The AIFS Single blend is scored on `single` only |
| Absolute error of every arm | both | reported | The first table of the report, since which input is best can matter more than a contrast |

### Review findings triaged

An Opus review of this section found six must-fix and seven should-fix items, and every one was
checked against the code and the data before it was applied.

- *M1 to M6 (deciding label and what was known; both shuffle seeds; the smoothing confound; the
  checksum command; the `check_runs` and `fit_jobs` traps; the verdict label):* applied.
- *S1 (one source for the ENS columns), S2 (refit days 1 and 2 instead of reusing the old fit), S3
  (drop the `ens` arms that feed no listed contrast), S4 (interval count), S5 (wind power figure),
  S6
  (lead filter, predictions files, chart command, mirror seed), S7 (blend against AIFS alone through
  the mirror control only):* applied.
- *Rejected:* the reviewer's alternative, blending saved out-of-fold forecasts with fold-wise
  weights
  (`studies.blending.simplex_weights`), which the reviewer did not recommend. It departs from the P4
  pattern and answers a different question. At most it is one exploratory line after the fits.
- *Rejected in part:* S2's optional determinism assertion is an optional check, not a gate.

## P4 second-seed control refit (planned before any fit)

**The published page's blend claim rests on one shuffle of each control, and the AIFS fit found two
seeds of one shuffle can differ significantly.** In that fit the null contrast between two seeds of
the shuffled AIFS climatology was +0.453 [+0.223, +0.696] points for solar at the primary setting.
A P4 guard is a contrast against one shuffled control, so a second control under another seed shows
whether the guard's verdict depends on the seed. This section refits the published P4a and P4b
blends with a second control, on the GPU, in the same run slot as the blends fit, and adds nothing
to the published page's numbers.

### What is refitted, and on which rows

- **Arms, per technology, at both settings (primary and sensitivity):** ENS's day-1 mean
  (`ens_mean_day1`), `blend_p4a`, `blend_p4b`, each blend's published control
  (`blend_p4a_control`, `blend_p4b_control`), and each blend's second-seed control
  (`blend_p4a_control_b`, `blend_p4b_control_b`). That is 7 arms.
- **Columns:** solar 11 for each blend and control, and wind 15 (ENS's mean plus two other products,
  three columns of calendar and sun position for wind, five for solar). ENS's day-1 mean has 7.
- **Rows and folds:** the published run's own, from `nwp_forecast_comparison.rows` and its
  `add_blend_guard_columns`, in the published row order (row subsampling depends on the order).
  The published script is not modified, and the published folder is only read.
- **Second-seed shuffle:** the published guard's rule with the seed `1000` plus the product's index
  in place of `20260920` plus that index. A value moves only among the hours that share a site, a
  year-month, and an hour of day, so it never crosses a fold, and a product's direction sine and
  cosine move together. Its column names carry `_permuted_b`.
- **One device per contrast:** every arm is refitted on the GPU. The published blends are CPU fits,
  so no contrast here reads a published loss, and the page states the device once.

### Reading rule, fixed before any fit

**If the two controls disagree on the guard's verdict, the page calls the blend claim unresolved.**
The report computes the published blend verdict (`studies.bootstrap.blend_verdict`) twice per
setting, once with the published control as each blend's guard and once with the second-seed
control, joins the two settings with `combine_setting_verdicts`, and then joins the two seeds with
`seed_agreement_verdict`, which returns the shared verdict where the two agree and
`unresolved: the two shuffle seeds disagree on the guard` where they do not. The report prints, at
each setting, every contrast with both arms' absolute errors beside it: each blend minus ENS's
day-1 mean, each blend minus each control, each control minus ENS's day-1 mean, and the seed-to-seed
gap (the published control minus the second control) for P4a and P4b. Every contrast is exploratory.

### Code changes

- `fit_aifs.py`: a `--p4-controls` mode (`p4_frame`, `add_second_seed_guard_columns`, `fit_p4`,
  `p4_verdicts`, `seed_agreement_verdict`, `p4_lines`, `check_p4`, `run_p4`), and `arm_prefixes` and
  `expected_column_count` learn the P4 arm names (`blend_p4a`, `blend_p4b`, and their `_control` and
  `_control_b`), which hold three products.
- The output folder is new and write-once: `data/studies/nwp_forecast_comparison_p4_seeds/`. The
  mode refuses the published folder, the day-1 and day-2 AIFS folder, the blends folder, and the
  extra-lead folders as its output, and `report.md` refuses to be overwritten.
- No file under `packages/` changes, no Patito contract changes, and no published script changes.

### Tests, and the assertion each would fail on

All are in `packages/studies/tests/test_fit_aifs_blends.py`, on small synthetic frames.

| Test | Fails if |
|---|---|
| The refit holds ENS's day-1 mean and each blend with both controls | An arm is missing or renamed |
| A blend and both controls hold three products' columns (11 solar, 15 wind) | `expected_column_count` or `arm_prefixes` treats a P4 blend as a two-product blend |
| The first control's columns equal the published control's own (`jobs` in `nwp_forecast_comparison.py`) | The refit's control shows different columns from the published run's, so its fit is not a refit |
| The second control shows the second seed's columns and no others | A second control reads the first seed's shuffled columns |
| The second seed never crosses a site, a year-month, or an hour (solar and wind) | The shuffle groups drop `hour_of_day` or `month`, or a value leaks between sites |
| The second seed differs from the first on at least 90% of rows and is deterministic | Both controls use one seed, or a seed depends on call order |
| Adding the second seed leaves the published controls' columns unchanged | The second shuffle overwrites or reorders the published guard's columns |
| A wind second seed keeps direction sine and cosine on the unit circle | The two are shuffled apart |
| The seeds' verdicts must agree or the claim is unresolved | `seed_agreement_verdict` returns one seed's verdict |
| Blends that beat both controls lower the error and the seeds agree | The verdict function reads the wrong control or the wrong sign |
| A second control as good as the blend makes the seeds disagree | The second control is not used as a guard |
| The report recomputes both guards at both settings, with errors beside each contrast | A guard, a setting, the seed-to-seed gap, or an error column is missing |
| The stamp names the GPU and both seeds | A saved fit from another device or seed would be accepted |
| The refit never writes beside the blends or the published folders | The output folder guard omits a folder |

### Commands, in order, and the runtime of the refit

```bash
D=/home/jack/dev/nged-substation-forecast/data/studies
uv run python studies/nwp_forecast_comparison/fit_aifs.py --p4-controls --check \
  --published-dir $D/nwp_forecast_comparison --output-dir $D/nwp_forecast_comparison_p4_seeds
uv run python studies/nwp_forecast_comparison/fit_aifs.py --p4-controls --workers 1 \
  --published-dir $D/nwp_forecast_comparison --output-dir $D/nwp_forecast_comparison_p4_seeds
```

**The refit is 63 fits at each of the two settings** (7 arms at 6 solar and 3 wind sites).
`--check` fits `blend_p4b` at one wind site twice, stops unless the two fingerprints agree, and
prints the estimate. At about 9 s per fit at the primary setting the primary fits take about 10
minutes, and the sensitivity fits (2.4 times the boosting rounds) take at most about 25 minutes, so
the refit is about 35 minutes on one worker, separate from the blends fit's 2 to 3 hours.

### Review chain

The same chain as the rest of this plan: an Opus code review of the fit code before any fit, and
the diff reviews the size of the change calls for. The refit reads the published inputs only, so it
adds no download and no cost.
