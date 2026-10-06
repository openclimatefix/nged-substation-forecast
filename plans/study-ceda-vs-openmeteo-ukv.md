# Plan: UKV from CEDA or from Open-Meteo for temperature, wind power, and solar power (issue #1051)

**The question is how different the two archives of the Met Office's UKV that the project can read
are, and whether the difference matters for power forecasts and for training history.** CEDA's UKV
archive reaches back to September 2019 and is the only UKV history we hold for 2019 to 2024. The UKV
that Open-Meteo serves matches the Met Office's live feed (its irradiance agrees with the Met
Office's own files to within 0.55 W m⁻² for hours since 12 August 2024), so it is the lineage a
live service would read. The CEDA download notes record that CEDA's archive differs statistically
from the live feed, and that a model trained on one should not be used on the other. Nothing yet
measures that difference in temperature, wind power, or solar power.

**The plan makes two measurements on the 23 whole months the archives share.** A model-free
comparison reports how far apart the two archives' values are. A power comparison fits an XGBoost
model per generator on each archive's values, and a transfer comparison trains an XGBoost model on
CEDA's values and scores it on Open-Meteo's values, a test of whether history in one archive can
feed inputs from the other. Three planned contrasts (section 3) answer the power questions, each
with a margin frozen here. **Before any fit, the plan states that "unresolved" leads to "do not mix
the two archives"**, and that on #1024's numbers an "interchangeable" verdict is probably out of
reach (section 3).

## 1. Verdict, size and departures

**Verdict: worth doing, with five departures from the issue body.**

- **The overlap is 23 whole months.** They are 2024-09 to 2025-12 (16, era 0) and 2026-02 to 2026-08
  (7, era 1). 2024-08 holds 20 days after Open-Meteo's backfill boundary and 2026-09 holds under 25
  days of data, and 2026-01 straddles UKV's Parallel Suite 47 (PS47) upgrade of 2026-01-21. PS43,
  PS44, and PS45 precede Open-Meteo's UKV, so the page says the study cannot test them. PS46 (May
  2025, a move to new computers) lies inside era 0, and the model-free comparison reads before and
  after it.
- **Question 4 (closeness to station readings) is dropped.** It needs Open-Meteo's UKV at the
  stations, which is not on disk. At lead 0 both archives are the same UKV analysis, so a station
  score mostly shows lead-0 against lead-1-to-5 values, which #1024 measured. The page says
  question 4 is answered only through power error, and that a station check would need a fetch.
- **The solar contrasts are planned on era 0 only.** After the PS47 upgrade Open-Meteo's hourly
  irradiance is built differently, so the two archives' irradiance columns are not like for like in
  era 1 (section 2). The solar rows of era 1 are scored as exploratory, labelled "irradiance
  construction differs after PS47", and wind and temperature keep the whole overlap.
- **Beam and diffuse irradiance cannot enter any arm,** because CEDA's files hold one downward
  shortwave field. The archives share temperature, 10 m wind, and global horizontal irradiance.
- **Wind compares the matched 10 m columns only.** CEDA has no 100 m wind, and a model cannot be
  scored on columns it was not trained on, so the matched pair carries both the archive contrast
  and the transfer.

**Size: complex.** All five triggers from `plan-issue` step 3 are answered.

| Trigger | Answer |
|---|---|
| Changes what gets stored | Yes. A published page, a write-once results folder, and a change to `packages/studies` |
| Touches the production serving path | No. Production never imports `studies` |
| Touches a degradation rule | No |
| More than one defensible design | Yes. The lead each contrast reads, the wind-height matching, the margins, and the transfer design |
| Callers not nameable without searching | Yes. `cross_validation.out_of_fold_losses` has seven callers across studies |

**Reviews: all four, plus the study skill's.** The diff reviews include the mutation pass, because
`packages/studies/` changes. The study skill adds a fresh Opus review of each script before its
first run, two Opus scientific-validity reviews, and the persona and prose reviews.

## 2. Data, arms, and reading each archive

**Most CEDA columns come from #1024's built frames.** The folder
`data/studies/per_study/ukv_ceda_vs_era5/` holds `wind_rows.parquet`, `solar_rows.parquet`, and
`station_hours.parquet`. All 23 months are present in the wind and solar frames, and none of
#1024's five dropped months falls in the overlap. The wind frame carries `ukv_ceda_speed_10m` (m/s),
its sine and cosine, power centred on the label, `effective_capacity_mw`, and `constrained`. The
solar frame carries the geometry, CAMS irradiance, and `ukv_ceda_temp` (the mean of the instants
at the hour's two ends). The build script reads these frames, records their hashes and #1024's
`effective_capacity` source in its stamp (the page names that table and its build date), and
replaces #1024's `era_code`, `era`, `fold`, and shuffled columns. **The frames hold no instants,
and no CEDA temperature or irradiance at the wind farms.** So the build also reads CEDA's
`temperature_1p5m`, `wind_speed_10m`, `wind_direction_10m`, and `shortwave_down` at lead-0-to-5
instants for all nine sites with `open_ukv_stores`, `nearest_ukv_cells`, and `ukv_at`, imported
from `ukv_ceda_vs_era5_build.py` in the same folder, so no store reader moves into
`packages/studies`. They go into a separate model-free frame on every hour both archives cover.
**Issue #1024's out-of-fold predictions are not reused,** because those models trained on 2019 to
2026 across three eras. Every arm refits on the 23-month folds.

**Open-Meteo's values come from `previous_runs/combined.parquet`.** It holds temperature, 10 m and
100 m wind, wind direction, and irradiance at the nine generator sites from 2024-01-01. Its wind
and global irradiance equal the `site_points/` extracts bit for bit on every shared row (checked
twice: 54,438 and 141,696 rows). No download is planned. Open-Meteo's backfill before 2024-08-12
comes from an unnamed source, and the build asserts that no earlier row enters. **Two builds make
Open-Meteo's values comparable with CEDA's, and a guard checks each at lead-0 hours:**

- **Units.** Open-Meteo's wind is in km/h and CEDA's in m/s. The build divides Open-Meteo's by 3.6
  and stops unless the median ratio of the two archives' 10 m speeds lies within 0.9 to 1.1. A
  unit test feeds a factor of 3.6 and must fail. A similar guard covers temperature (degrees
  Celsius from both).
- **Irradiance.** Open-Meteo's `shortwave_radiation` is UKV's snapshot at the hour's end multiplied
  by the ratio of the hour's mean cosine of the solar zenith angle to the cosine at the hour's end
  (computed with the apparent zenith, which matches no worse than the true zenith, checked). The
  median of that value over the snapshot runs from 0.52 at 05 UTC to 1.42 at 19 UTC on the overlap
  (checked). The build applies the same ratio to CEDA's `shortwave_down` at the label, and does not
  invert Open-Meteo's value to a snapshot, because the inverted `ghi_instant_w_m2` spikes after
  sunrise and dropping the spikes would select rows by one archive's values. The build stops unless,
  at lead-0 hours, the median ratio of CEDA's rebuilt value to Open-Meteo's lies within 0.97 to 1.03
  at every hour of day in era 0 (before PS47). Without the rebuild the guard fails. After PS47 the
  ratio does not hold, and that mismatch is a reported diagnostic and not a failure (see "Changed
  after implementation").

**Which lead each archive stands in for.** Open-Meteo serves each hour's freshest run analysis
(lead 0). CEDA's 6-hourly archive is read from the freshest run at or before the hour, so its lead
is 0 to 5 hours. **The archives can agree exactly only at the four lead-0 hours (00, 06, 12, and 18
UTC).** The planned contrasts read all hours, which is what a consumer of each archive gets. **Only
the model-free comparison separates a product difference from a lead difference,** by its lead-0
agreement of the raw values. The lead-0 subset of the fits is an exploratory read: the CEDA-trained
model still learned from leads 0 to 5, `hour_of_day` fixes the lead, and a solar row's temperature
mean reads the lead-5 instant at every lead-0 hour.

**Model-free comparison (question 1, no contrast, no margin).** For temperature, 10 m wind speed, 10
m wind direction (the circular difference, because CEDA's direction may be grid-relative and show as
a constant rotation), and global horizontal irradiance at the nine generator sites, the script
reports the mean difference, the mean absolute difference, the 99th percentile of the absolute
difference, and the correlation, by CEDA lead, hour of day, calendar month, and era. It uses all
hours both archives cover. The two means carry intervals that resample whole months
(`bootstrap_row_difference`), and the percentile and correlation are point values. A
before-and-after row for 2025-05 covers PS46. Per-site series stay unlabelled. Two diagnostics run
first and only report:

- **Cell match.** At lead-0 hours, which of the nine CEDA cells around the nearest has the value
  closest to Open-Meteo's? The study always reads the nearest cell, because choosing the
  best-matching cell would select on the outcome and shrink the difference by construction.
- **Elevation.** Open-Meteo may lapse-rate-adjust 2 m temperature to the point's elevation, which
  shows as a near-constant offset at lead 0. The offset cannot move P1 or P2, which fit per
  generator, but it does move P3, because a CEDA-trained model scored on Open-Meteo's temperature
  inherits it. An exploratory read scores the same model on Open-Meteo's temperature minus its
  per-site lead-0 mean offset.

**Wind arms (three wind farms, W1 to W3), 6 columns each.** The shared columns are `hour_of_day`,
`day_of_year`, and `era_code`. `ceda_wind_10m` adds 10 m speed and the sine and cosine of the 10 m
direction from CEDA, and `om_wind_10m` adds the same three columns from Open-Meteo (in m/s). One
function per arm type returns a fixed-length tuple, `check_column_counts` checks the pair (and the
shuffled pair), and the report prints every arm's columns. Power is the hour centred on the label,
because both archives' wind is instantaneous.

**Solar arms (six solar farms, A to F), 8 columns each.** The shared columns are `solar_zenith_deg`,
`solar_azimuth_deg`, `extraterrestrial_horizontal_w_m2`, `hour_of_day`, `day_of_year`, and
`era_code`. `ceda_ghi_temp` adds CEDA's global irradiance (rebuilt as above) and CEDA's temperature.
`om_ghi_temp` adds Open-Meteo's `shortwave_radiation` and **the mean of Open-Meteo's
`temperature_2m` at the hour's two ends** (`studies.neighbouring_hours.with_neighbouring_hours`),
the construction `ceda_ghi_temp` uses, so the row set also needs Open-Meteo's hour before. A solar
hour needs both instants' CEDA runs to be complete. Beam and diffuse irradiance are excluded from
both arms. The page answers the temperature-only question from #1024's finding (temperature moved
solar error by -0.001 points [-0.009, +0.004]) and from the model-free comparison.

## 3. Planned contrasts, margins, and the decision rule, frozen before any result

**A negative contrast favours CEDA, and the transfer penalty is read one-sided.** Verdicts for P1
and P2 have three outcomes: **interchangeable** (the 95% interval lies wholly inside the margin),
**differ** (the interval excludes zero and the estimate lies beyond the margin), and **unresolved**
(anything else, including a significant difference smaller than the margin). Each planned contrast
runs at both hyperparameter settings, and a verdict stands only if both agree.

| Label | Contrast | Margin |
|---|---|---|
| P1 (planned) | Wind power: `ceda_wind_10m` minus `om_wind_10m`, points of capacity | 0.16 |
| P2 (planned) | Solar power on era 0, fitted on both eras and on era 0 alone: `ceda_ghi_temp` minus `om_ghi_temp` | 0.06 |
| P3 (planned) | Transfer penalty: (trained on CEDA, scored on Open-Meteo) minus (trained on Open-Meteo, scored on Open-Meteo), read for the wind arms (all months) and for the solar arms (era 0, both fits) | 0.16 wind, 0.06 solar |

**P3 is read one-sided, because only a penalty is actionable.** "No penalty" means the interval's
upper bound is below the margin. "Penalty" means the lower bound is above zero and the estimate
exceeds the margin. Everything else is "unresolved". P3 swaps every Open-Meteo column of an arm at
once, so it cannot say which column carries a penalty. Exploratory partial swaps, with no refit,
score the CEDA-trained solar model on CEDA irradiance with Open-Meteo temperature and the reverse,
and the wind model on CEDA speed with Open-Meteo direction.

**The margins are inherited from #1024 and were resolution-based there.** Issue #1024 took 0.16
and 0.06 from the half-widths of earlier pages' intervals and the number of scored months, and froze
0.06 for a temperature-only contrast, not for irradiance and temperature together. This plan keeps
the numbers, so that the pages read alike, and does not claim they are a decision quantity. For
scale, 0.16 is about 2% of #1024's 7.2% wind error. The plan does not widen them after seeing a
result. #1024's intervals for nearly this window show how little the design resolves:

- **Wind.** Since 2024-08-12 the as-available CEDA-minus-ERA5 wind contrast was +0.039 [-0.115,
  +0.186] (standard error about 0.077) and the matched 10 m contrast +0.366 [+0.196, +0.534]
  (standard error about 0.086). "Interchangeable" needs the estimate plus the half-width below 0.16,
  so with a standard error of 0.077 it needs an estimate within about 0.01 of zero, and with 0.086
  it cannot happen.
- **Solar.** The different-product irradiance contrasts since August 2024 on the solar page have
  half-widths of 0.17 to 0.24 points, so P2's margin of 0.06 is probably unreachable. Reading solar
  on era 0 alone (16 months, against 23) widens the intervals by about 20% (the square root of 23
  over 16), and the margins are unchanged.
- **Why these figures may overstate the error.** They compare different products, and two copies of
  one weather model should give more strongly correlated errors, so the standard error may be much
  smaller. P3's paired errors share the scored inputs, so its interval may be narrower than P1's.
  The study cannot know before the fit.

**If the standard errors land near #1024's, P1 and P2 report an estimate and a bound ("an effect as
large as 0.17 points is not excluded") and the recommendation is "do not mix" by default.** The page
presents that as an unresolved difference and never as a measured one. The outcome costs nothing
extra, because P3 adds no fit.

**Rules by question.**

- **Question 2 (does the difference matter for power).** P1 and P2 each report their verdict.
- **Question 3 (eras and years).** Wind contrasts are reported for era 0 and era 1 separately. The
  solar contrasts are planned on era 0, and the solar rows of era 1 and every solar scope that
  includes them are exploratory and carry the note "irradiance construction differs after PS47
  (ratio 1.11 at 06 UTC, 0.86 at 18 UTC)", with the ratios read from the build. The model-free
  comparison reports the irradiance ratio by era and hour. The half-years and
  sites appear as charts only. Era 1 holds 7 months, so its intervals are approximate.
  The change across 2026-01-21 is exploratory. Calendar years are not read, because 2024 holds four
  months.
- **Question 5 (training history).** P3 measures transfer to Open-Meteo's lead-0 analysis only. The
  study says nothing about leads beyond 5 hours, and Open-Meteo's wind and temperature have not been
  checked against the Met Office's files (only irradiance has). The study commits the project to
  nothing:

| P3 reads (each arm type separately) | The page recommends |
|---|---|
| No penalty | A model trained on CEDA's leads 0 to 5 can be scored on Open-Meteo's UKV analysis for that input, at the precision tested, subject to the licence |
| Penalty | For that input, train on Open-Meteo's UKV history only (from 2024-08), or let a calibrator absorb the difference |
| Unresolved (the default) | Treat as "penalty": do not mix the two archives until a longer overlap exists |

**The decision is subject to the licence.** CEDA's licence is Creative Commons
Attribution-NonCommercial-ShareAlike 4.0. The study is non-commercial research, publishes only
error scores and charts with no UKV values, and cites "Met Office (2016): NWP-UKV: Met Office UK
Atmospheric High Resolution Model data. Centre for Environmental Data Analysis". Whether the main
work's use is non-commercial is the maintainer's decision.

## 4. Row set, gaps, and eras

**Rows are the intersection across every arm, decided from the target and availability, never from a
value.** An hour stays only if the target exists and both archives have every column every arm
reads. Wind and solar rows keep #1024's own drops (zero half-hours, the commissioning ramp, and the
other drops its row builders apply) and its export-cap `constrained` flag, which
`out_of_fold_losses` reads (those hours are scored and never trained on). A CEDA hour whose freshest
run is missing or partial is dropped for every arm and never filled from an older run. The 94 hours
from 2024-11-09 16:00 to 2024-11-13 13:00, a gap in the `site_points/` wind extract that
`combined.parquet` fills from an unchecked lineage, are dropped for every arm, and so are the 26
hours with a null 10 m wind direction (2024-12-22 07:00 to 2024-12-23 08:00). The build prints the
loss per month and stops if any month loses more than 25% of its hours to either archive.

**Eras and folds.** `cerra_past_solar.with_covering_folds` cuts at `FIRST_MONTHS =
(UKV_UPGRADE_MONTH,)`, which gives the two eras here. A synthetic check for this plan found that
`search_fold_offsets` returns covering rotations for the 23-month set, and the build re-runs it on
the real rows and stops if it returns none. Every arm gets `era_code`. January occurs only in 2025,
so one scored month has no January in training, which the coverage check exempts. The page reports
an exploratory row without it.

## 5. Intervals, settings, metric, hardware

- **Intervals:** `studies.bootstrap.bootstrap_difference` resamples whole calendar months, paired
  across arms, with one of three fitting seeds, 2,000 resamples. They cover month-to-month weather,
  not differences between generators.
- **Hyperparameters:** `PRIMARY_HYPER_PARAMETERS` and `SENSITIVITY_HYPER_PARAMETERS` on every
  planned contrast. Controls run at the primary setting only.
- **Metric and hardware:** mean absolute error as percent of capacity, each row divided by its own
  generator's `effective_capacity` before any mean, with every arm's absolute error reported. Check
  `uptime` for a spinning core, then `nvidia-smi`, and use `device="cuda"` with column subsampling
  off (`colsample_bytree` of 1). One arm refits on the CPU for the noise floor.
- **Negative control.** Each archive's weather columns shuffled within (site, year-month, hour of
  day) with `climatology_permutation`, each archive as its own column group so the two get
  independent permutations, at the primary setting. Shuffled-CEDA minus
  shuffled-Open-Meteo should differ by about zero. #1024 found shuffled models run at about three
  times the real error, so the control says little about a pipeline at 7%, and P1 to P3 rest on
  their own intervals and the CPU refit. No positive control runs: every null reading
  carries its bound instead, as section 3 states.

## 6. Scripts, files, and outputs

Scripts go in `studies/past_weather/` and import only that folder, `studies.*`, and reviewed
packages.

| Script | Job |
|---|---|
| `ukv_ceda_vs_openmeteo_build.py` | Reads #1024's frames, adds Open-Meteo's columns and CEDA irradiance, builds the arm-column functions; `--check-only` prints coverage and stops on a failed guard; `--dry-run` builds one month |
| `ukv_ceda_vs_openmeteo_compare.py` | Model-free comparison and the diagnostics; `direct_report.md` |
| `ukv_ceda_vs_openmeteo_fit.py` | Fits, transfer scoring, shuffled controls; `--dry-run`, `--verified`, `--report-only`; writes `report.md` and `decision.md` |
| `ukv_ceda_vs_openmeteo_charts.py` | Figures; every shared number is read from the reports |

**One change in `packages/studies/`, with tests.** `cross_validation.out_of_fold_losses` gains an
optional `scoring_site_rows`, a mapping from a `scoring_archive` name to a frame with the same rows
and another archive's values under the training columns' names. One booster per (fold, seed)
predicts the training frame and every scoring frame, so P1 and P3 share the identical CEDA-trained
booster and no model is refitted for the transfer. Rows align by `time`, the target, `constrained`,
capacity, and `fold` come from `site_rows`, and the function raises if a scoring frame's times or
`fold` differ. Losses carry a `scoring_archive` column. A script-local loop would have to copy the
private `_losses`. `sources.py` gains the study's output
directory as a `study_dir_for(study=...)` constant.

**Output.** `data/studies/per_study/ukv_ceda_vs_openmeteo/` is write-once (`refuse_to_overwrite`); a
re-run after a review moves the earlier output to `superseded/` first. The folder holds the frames
and build stamp, every arm's per-row losses and out-of-fold predictions (site, time, fold, seed,
arm, setting, scoring archive, signed error; the actual value is joined from the rows, because
`_losses` writes no prediction), the interval tables, `report.md`, and `decision.md`. One session
owns running the scripts, because every worktree shares the main checkout's `data/`.

## 7. Compute estimate

**The fits total 78 fit-sets, plus one CPU refit.** Wind planned arms take 12 (2 arms, 2 settings, 3
farms), solar planned arms 24 (2 arms, 2 settings, 6 farms), the era-0-only solar refit 24 (the same
four jobs on the rows of era 0 alone, 6 farms), and shuffled controls 18 (one shuffled pair per
type: 6 wind, 12 solar). The transfer scoring and the lead-0 reads add none. Issue #1024's
fit-sets took 8 to 10 seconds on 6,000 to 14,000 rows, and this study has about 15,000 wind rows and
5,000 to 8,000 daytime solar rows per farm, so plan **5 to 20 seconds per fit-set, 5 to 20 minutes
in all** for the first 54 and about 5 more minutes for the era-0 refit.

## 8. The page and the docs

**The page** is `docs/studies/past-weather/ukv-ceda-vs-open-meteo.md`, under Studies > Past weather,
following the study skill's twelve sections. Figure 1 is the planned contrasts with 95% intervals,
the margins as bands, each row labelled with both archives' absolute errors, and the second setting
as a marker. Each results section has a chart: the XGBoost models work (out-of-fold power against
measured, A to F and W1 to W3, over weeks chosen by a stated rule); model-free differences by lead
and by month with the PS47 boundary marked; the transfer penalty; and the controls. The title is
written after the results.

**Cross-references:** `docs/studies/index.md`, `docs/studies/past-weather/index.md`,
`past-weather/methods.md`, `studies/README.md`, `studies/past_weather/README.md`,
`packages/studies/README.md`, and the `mkdocs.yml` nav line. `docs/roadmap/training-history.md` and
`docs/roadmap/data-sources.md` change when the result lands, in a separate small PR, so the study PR
stays inside `studies/`, `packages/studies/`, `docs/studies/`, and `mkdocs.yml`.

## 9. Risks, open questions, and verification

1. **January 2025 has no January in training.** Recommendation: keep it and print the row without
   it.
2. **Two archives of one weather model may differ by less than the design resolves.**
   Recommendation: keep the margins and report the bound (section 3). 3. **Multiple comparisons.**
   Three planned contrasts, no correction. About one in 20 exploratory rows with no real effect
   reaches the 5% level, and the page says so.

The plan commits only to this study. It does not commit to serving live UKV, to training on either
archive, to fetching data, or to a decision about the licence.

```bash
uv run ruff check . && uv run ruff format --check . && uv run ty check
uv run pytest packages/studies
uv run pymarkdown scan -r docs README.md CLAUDE.md packages/*/README.md studies/README.md
uv run mkdocs build --strict      # and read the rendered page and its figures
uv run python studies/past_weather/ukv_ceda_vs_openmeteo_build.py --check-only   # must exit 0
```

Also run `pydoclint`, the docs-link checker, and the mutation pass. Each test fails on a mutant, not
only on `main`, where any use of the new keyword fails:

- Predictions on a scoring frame equal `fit_one_fold(train=..., test=<scoring rows>)` for the same
  seed.
- A scoring frame in shuffled row order gives the same per-time losses.
- A scoring frame whose target, `constrained`, or capacity differs changes no loss, and one whose
  `fold` or times differ raises.
- Constrained training rows stay excluded.
- With `scoring_site_rows` omitted, the losses are bit-identical to a call before the change.
- The build's unit guard fails on a factor of 3.6, and its irradiance guard fails on a snapshot
  that skips the zenith ratio.
- The build asserts that the scoring frame's columns equal the Open-Meteo arm's values under the
  CEDA arm's names, and that no Open-Meteo row before 2024-08-12 enters.

The fit script's capacity normalisation test uses unequal capacities.

## Considered and rejected

**Accepted from the simplicity review.**

- Reading #1024's frames and importing its store helpers: the frames hold every CEDA column but
  irradiance, and no package move needs a bit-for-bit proof.
- Cutting the station contrast and fetch: lead-0 values are the same analysis, so the score mostly
  shows lead, which #1024 measured.
- Cutting the temperature-only pair, the lead-0 refits, the noise control, the hour-ending pair, and
  the two-ends pair, and making P1 the matched wind pair: each added fits for a read that another
  planned fit or the model-free comparison gives (the lead-0 subset needs no refit).
- Fixing the helper name to `studies.solar_product_frames.without_sunrise_spikes`, and dropping
  the 2022 backfill row and the calendar-year read.
- Cell match reports only, and the stated reachability of "interchangeable" with #1024's numbers.

**Accepted from the correctness review.** The irradiance rebuild and guard, the two-instant
Open-Meteo temperature, the km/h conversion and guard, the multi-frame scoring API, the reworded
question-5 table, the inherited-margin wording, the lead-0 subset as exploratory, independent
shuffles, the gap-file and null-direction drops, PS45 and PS46, the named mutant tests, and the
wind-direction, partial-swap, and capacity-table additions.

**Rejected or kept.**

- Widening the margins to make "interchangeable" reachable: changing a frozen margin after seeing
  Issue #1024's intervals would be tuning to the outcome, so the plan keeps them and states the
  limit.
- Scoring the transfer in a second `out_of_fold_losses` call: it forces 18 more fits and an
  unchecked bit-identity on the GPU, so the mapping design scores every frame from one booster.
- Dropping P3 (transfer): it adds no fit and is the only measurement of the train-on-history,
  serve-live case.
- Training on 2019 to 2024 CEDA and scoring on Open-Meteo: it changes history length and era with
  the archive.
- Dropping the negative control: the study skill requires one.

## Changed after implementation

**Open-Meteo builds its hourly irradiance differently after the PS47 upgrade.** The first run of the
irradiance guard on the real data found that CEDA's snapshot, rebuilt with the zenith-cosine ratio,
matches Open-Meteo's `shortwave_radiation` at lead-0 hours before 2026-01-21 (median ratios 0.99 to
1.01 at 06, 12, and 18 UTC) and does not afterwards (1.11 at 06 UTC and 0.86 at 18 UTC; the
unscaled snapshot gives 1.45 and 0.74). The decision, made before any fit or result, is:

- **The solar planned contrasts P2 and P3 are read on era 0 only** (16 months), and the margins are
  unchanged. Wind and temperature keep the whole overlap.
- **The solar rows of era 1, and every solar scope that includes them, are exploratory** and
  labelled "irradiance construction differs after PS47 (ratio 1.11 at 06 UTC, 0.86 at 18 UTC)",
  with the ratios read from the build.
- **The era-0 guard stays fatal, and the era-1 mismatch is a reported diagnostic.** The model-free
  comparison reports the ratio by era and hour (`direct_irradiance_ratios.parquet`).
- **The row counts are** 45,948 wind rows, and 42,544 solar rows of which 27,766 are era 0.

**The third review (before any fit) changed five more things.**

- **The planned solar fits are refitted on era 0 alone.** A model that trains on all rows and is
  scored on era 0 has still trained on 14,778 era-1 rows whose Open-Meteo irradiance is built
  differently, so the two archives' training sets differ in construction. The solar jobs therefore
  run twice: once on the rows of both eras (scored on era 0), and once on the rows of era 0 alone
  (`solar_rows_era0.parquet`, folds cut inside era 0, 24 more fit-sets, no controls). **A solar
  verdict (P2 or P3) stands only if all four readings agree** (two fits, two settings), and
  otherwise it is `unresolved`. The confound is a limitation on the page: if the era-1 rows degrade
  the Open-Meteo-trained reference model, the transfer penalty moves towards `no_penalty` and the
  contrast towards CEDA, and the era-0-trained fit is the check on that. Fit-set count: 78.
- **The sun-elevation view of the irradiance rebuild.** Below 5 degrees of sun elevation the
  rebuilt value does not reproduce Open-Meteo's (ratio 0.02 below 2 degrees and 0.68 from 2 to 5
  degrees in era 0, against 1.000 above 10 degrees), and the 50 W/m2 cut of the guard hides it. The
  build and the compare script print the ratio by elevation bin, and the solar fits add two
  exploratory scopes, "sun above 5 degrees" and "era 0, sun above 5 degrees". The scope reads the
  geometry and neither archive's value.
- **The `--verified` gate checks the build.** The build stamps `guards_passed` and the hash of each
  row file into `build.json`, and the fit refuses unless the stamp says every guard passed, each row
  file hashes to its stamped value, and `direct_report.md` exists.
- **The build asserts that every row frame is sorted by site and time with unique keys,** because
  row order is part of an XGBoost fit.
- **The partial-swap temperature offset is learned from lead-0 instants on the training folds
  only.** It was the mean of the two-end temperature means at the hours where CEDA's lead is 0,
  which averaged in a lead-5 instant and read the scored fold. The fit also checks every output path
  before it writes the hardware stamp.

## Changed after the first fit

**Open-Meteo's 10 m wind speed is built differently in two spans, so every wind arm drops them.**
The first review of the fit found that Open-Meteo's 10 m speed against CEDA's, at lead 0, reads
about 0.97 in 20 of the 23 months and 1.01 to 1.06 in 2024-11, 2025-01, and 2025-02. The lineage
check (committed in the build) finds two spans with sharp edges, 2024-11-07 to 2024-11-30 and
2025-01-16 to 2025-02-18, in which Open-Meteo's speed is about 6% higher against CEDA's (median
ratio 1.064 against 0.968) and about 5% higher against ERA5's, while CEDA's speed against ERA5's is
steady. Open-Meteo's 100 m to 10 m speed ratio falls from 1.95 to 1.78 in them, and its
temperature, wind direction, and irradiance do not step. `combined.parquet` equals the
`site_points/` extracts bit for bit in those spans, so the file is the one the other studies read,
and the step is in what Open-Meteo served. The decision is by the served series and not by any
power result.

- **The spans are dropped from every wind arm,** and the three months they empty (2024-11, 2025-01,
  and 2025-02 lose 84%, 52%, and 65% of their rows, over the 25% limit) are dropped whole. The wind
  rows fall from 45,948 to 40,287 (13 months of era 0 and 7 of era 1). Solar reads no wind, and its
  inputs do not step, so its rows are unchanged.
- **The model-free comparison keeps every hour,** flags the spans (`om_wind_step`), reports wind
  speed and direction outside them, and reports the speed inside them separately.
- **The first run's outputs moved to `superseded/first_run/`.** In that run wind P1 read unresolved
  (+0.162 [+0.080, +0.241] at the primary setting and +0.146 [+0.073, +0.216] at the second) and
  wind P3 read penalty (+0.313 [+0.163, +0.459]). The rule is the same in the rerun.
- **A wind calibrator test was added at no cost in fits.** The CEDA-trained model also scores a frame
  holding Open-Meteo's speed rescaled to CEDA's level (each site's median CEDA-to-Open-Meteo speed
  ratio at lead-0 instants, learned on the training folds). All 78 fit-sets and the CPU refit
  were rerun from one build, so every output comes from one run.
- **New report rows,** all exploratory: each planned contrast by CEDA lead (0 to 5) and for the
  era-0 sensitivity, an "era 0, lead 0 only" scope, "without 2025-01", each arm's absolute error and
  mean signed error on the planned scope, and the setting names `primary` and `second`.
- **The two settings of the model are named `primary` and `second`,** replacing the earlier names.

## Changed after the second science review

**The wind P1 verdict depends on a post hoc data decision, so the page reports both runs.** The
plan's scope for wind was all rows, which kept the two spans. The first run read wind P1 as
+0.162 [+0.080, +0.241] at the primary setting and +0.146 [+0.073, +0.216] at the second
(unresolved), and the rerun without the spans reads +0.270 [+0.209, +0.333] and +0.243 [+0.192,
+0.294] (differ). The decision to drop the spans came from the ratio of Open-Meteo's speed to
CEDA's and not from a power result. Wind P3 is a penalty in both runs.

- **A committed script reads the saved intervals and losses and fits nothing.**
  `ukv_ceda_vs_openmeteo_extra_reads.py` prints the first run's wind P1 and P3 (in all hours and at
  lead 0) from `superseded/first_run/intervals.parquet`, and wind P1 and P3 by UTC hour. Its output
  is `extra_reads_report.md`.
- **The title, Summary, and Figures 1 and 8 no longer rest on exploratory lead-0 rows.** Lead-0 wind
  holds only after the spans are dropped. For solar the era-0-trained fit reads +0.127 [+0.021,
  +0.257] at lead 0, so Figures 1 and 8 draw that fit too.
- **The Discussion says that P3 holds training length equal,** so it does not weigh CEDA's extra
  history of about 5 years, and that the recommended Open-Meteo history contains the two dropped
  spans.
- **Smaller corrections:** the calibrator residual (+0.142) reads "unresolved" under the plan's
  rule; the era-0-trained solar penalty is part of the transfer penalty; lead is the UTC hour modulo
  6, and the wind gap is largest after the 12 UTC run; the reachability paragraph states the plan's
  prior expectation; the noise wording is consistent; Figures 7 and 8 titles are no longer clipped;
  the wind-step paragraph uses the CEDA-minus-Open-Meteo sign; and the figures are renumbered in
  reading order.
