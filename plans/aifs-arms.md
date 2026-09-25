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
