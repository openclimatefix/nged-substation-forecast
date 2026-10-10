# Plan: do lagged power features improve a solar power forecast? (issue #1138)

## Problem

Issue #1138 theorises that an XGBoost model forecasting solar power should gain from lagged power
features. The lags could calibrate a forecast to soiling, panel degradation, a failed string, part
of a plant offline, aerosol episodes, and changes in the weather forecast product. In a model
trained across all plants, the lags could also tell the model each plant's quirks (such as its
DC:AC ratio) without a physical model of the plant. The maintainer also asked whether lags help the
forecast quantify its uncertainty, and asked for a wide sweep of ideas, wacky ones included, before
any careful comparison. Nothing in the repository tests any of this. The persistence baselines read
lagged power, but no study gives an XGBoost model lagged power next to weather.

## Planned solution

A two-phase study on the 6 NGED utility-scale solar plants, driven by ECMWF IFS HRES (Open-Meteo
Single Runs), reading the matched-lead study's inputs that are already on disk.

- **Phase 1, a broad shallow sweep:** about 22 ideas for lagged or lag-derived inputs, each fitted
  once per plant at the day-ahead lead and scored as its own absolute error, with no pairwise
  comparison and no significance claim. The page shows one ranked chart. The sweep is for
  generating hypotheses.
- **Phase 2, a rigorous comparison:** five planned paired contrasts, fixed before the sweep runs,
  one of which names "the best sweep arm", chosen by a rule written before the sweep. Phase 2 uses
  the primary and sensitivity hyperparameter settings, three seeds, equal column counts, a
  positive control, and the global model.

Both a point forecast and a probabilistic forecast (nine quantiles) are scored. The page is
figure-led.

## Verdict, size and departures

**Verdict: worth doing, with the changes below to the issue's wording.**

**Size: complex.** The five triggers:

- **What gets stored:** a published `docs/studies/` page and files under `data/studies/`; no Patito
  model, Delta table, or asset. Fires under the `study` skill's rule that a published page counts as
  what gets stored.
- **Production serving path:** not touched. Nothing under `src/` or any package except
  `packages/studies/` changes.
- **Degradation rule:** none touched.
- **More than one defensible design:** yes. The lag definition, the two-step leak control, the
  control arms, the global model's inputs, and the sweep's selection rule each admit several
  designs.
- **Callers not nameable without searching:** none; the plan adds functions to
  `packages/studies/` and edits no existing one.

**Reviews:** simplicity review 1 and correctness review (done, triaged below), simplicity review 2
(after this revision), two Opus scientific-validity reviews, the diff review, the mutation pass
(the package changes), and the prose reviews and personas the `study` skill sets. The study skill's
merge conditions apply; the diff stays inside `studies/`, `packages/studies/`, and `docs/studies/`.

**Departures from the issue body:**

1. **"24-hour lag" becomes "the same clock hour on the latest fully observed day".** A forecast
   issued for a target hour after the run starts cannot read power 24 h before the target at every
   hour. The lag of a lead-day-`N` target is `24 (N + 1)` hours (24, 48 and 72 h at lead-days 0, 1
   and 2), the lag `baselines.diurnal_persistence` reads; the correctness review confirmed the two
   agree on every row where both exist. The page defines "lag" this way once.
2. **The sweep covers ideas the issue does not list** (see the sweep table), at the maintainer's
   request.
3. **No new PV dataset** (next section).

## Do we need the UK PV dataset?

**Recommendation: no, for this study.** The target use is utility-scale plants; the
`openclimatefix/uk_pv` data is domestic rooftop PV, whose failures and soiling average out across
many small systems. It would also need a second weather download, and whether its time coverage
overlaps the IFS archive is unchecked. The shared rows hold **21 scored months** (December 2024 to
September 2026, with January 2026 dropped), 35,263 rows, and 4,246 to 6,976 rows per plant. The 6
plants sit in two grid cells and one 25 km box, so the effective sample size is weather episodes
(months), not plants. An effect is detected only if it is large; a null is read as "an effect as
large as X is not excluded", and a positive control sized to realistic soiling losses measures the
detection limit. The six similar plants are weak evidence for the global-model "plant quirks"
hypothesis, and the page scopes that result narrowly. The Scope section says domestic PV and other
regions are untested.

## Design

### Rows and inputs (reused from the matched-lead study)

- **Rows and weather:** the matched-lead inputs on disk under
  `data/studies/per_study/nwp_forecast_comparison/`: `original/solar_forecast_inputs.parquet`
  joined on `(site, time)` to `leads_day10d/solar_extra_lead_inputs.parquet`
  (`ifs_single_day{0,1,2}_{ghi,temp}`). Folds and `era_code` come from
  `nwp_forecast_comparison.rows()` and are joined, never recomputed after any row drop. The
  baseline arm has seven columns: hour of day, day of year, an era code, the sun's elevation and
  azimuth, and the product's global horizontal irradiance and 2 m air temperature at its own lead.
- **Origin convention:** the existing one (`baselines.issue_time`): the 00 UTC run is issued at its
  init time for lead-day 0 and at 09:00 UTC on the run's own day for later lead days. The lag of a
  lead-day-`N` target is `24 (N + 1)` hours. Lead-day 1, the day-ahead lead the live service uses,
  is the headline; lead-day 0 is not a forecast anyone could have used for its early hours, and
  the page says so.
- **Targets and scaling:** per-plant fits use `power_mw` as the target, exactly as the published
  arm does, so `out_of_fold_losses` and `clamp_to_cap` work unchanged. The global fit divides
  `power_mw`, `cap_mw`, and every power-derived input (lags, weekly statistics, control columns) by
  `effective_capacity_mw`, then sets `effective_capacity_mw` to 1.0, so the existing loss columns
  stay in fractions of capacity. Without that, the global model would learn each plant's capacity
  from the lags in megawatts.
- **Lags are read from the full cleaned hourly power table** (not from the daylight-filtered
  inputs), strictly, with no tolerance, unlike `diurnal_persistence`, which falls back up to 7 days.
  NGED timestamps are already corrected at ingest and are not shifted again. The script asserts
  `lag_time <= issue_time` on every row and that the strict lag equals
  `diurnal_persistence_day<N>` wherever both exist.
- **Row filter:** a row is dropped for every arm if any arm's required column is null, with these
  exceptions: the weekly and rolling statistics need at least a stated number of days, otherwise
  they are null and XGBoost's native missing-value handling applies (the correctness review found
  a strict 7-of-7 rule would drop about 24% of rows). The retained row count and each drop's count
  are printed. A target hour with no IFS run is dropped (7 run days are missing).
- **Cleaning removes the case the issue names.** Outage hours, commissioning-ramp hours and
  export-cap hours are dropped from targets and lags, so "part of a plant offline" is outside what
  the study measures. The page says so.

### Phase 1: the sweep

**One fit per arm and plant at lead-day 1, primary hyperparameter setting, seed 0, scored out of
fold on the plant's folds 0, 1 and 2 (about the first 13 months).** No pairwise contrast and no
significance claim: each arm gets its own mean absolute error and CRPS (from nine quantiles) with a
month-resampled interval (`bootstrap_absolute`), the baseline's value drawn as a reference line,
and the page states that arms have different column counts and widths, so the order is a screen
and not a ranking. Folds 3 and 4 are not scored in phase 1, so phase 2's verdict on the best sweep
arm uses months the sweep never saw.

| Family | Arms (extra columns) |
|---|---|
| Reference, no lag | B0 baseline (0); persistence and diurnal persistence with no ML (existing columns) |
| Single lag | L1 latest whole day, same clock hour (1); L2d the day before that (1); L7 seven separate lags, days 1 to 7 (7) |
| Lag with weather | L2 L1 plus the forecast global irradiance and temperature at the lag hour, same lead day as the target (3) |
| Weekly statistics | W7 minimum, maximum, mean of the same clock hour over the latest 7 whole days, at least 5 days present (3) |
| Slow trackers | Q30 the 30-day 90th percentile and median of the same clock hour (2); TR the slope of the daily peak over 30 days plus the 30-day 90th percentile (2) |
| Daily summaries | DS yesterday's peak power, energy, and mean clear-sky index (3); CSI the lag hour's observed clear-sky index (1) |
| Two-step | S2 stage-1 prediction at the target hour and the lag hour, the observed lag, and their difference (4); S3 mean stage-1 residual over the last 1, 7, and 30 days (3) |
| Cross-plant | NB the other five plants' lag at the same hour (5); FL the fleet's mean lag (1) |
| Outage | OF L1 plus yesterday's share of daylight hours with zero power (2) |
| Drift without lags | T1 days since 2024-03-01 (1): a time index that lets the trees learn drift, the alternative explanation of any lag gain |
| Post-model | R1 the baseline's prediction multiplied by the last 7 days' observed-over-predicted energy ratio, with no change to the model |
| Combination | KS the kitchen sink: L1, W7, Q30, S3, NB, OF (about 14) |
| Controls | N1 one observed hour from days 8 to 28 before the issue day at the same clock hour, never the L1 day (1); N2 a lag from a uniformly random observed day of the whole record (1) |

All stage-1 predictions (S2, S3, R1) come from the stage-1 model that withheld the scored fold and
the predicted hour's own fold. A lag hour outside the shared rows is in no fold, so it is predicted
out of sample by construction. Stage 1 predicts the lag hour from the lag hour's weather at the
target's lead day (`ifs_single_day<N>`), available at the issue time.

**The shortlist rule, fixed before the sweep:** X is the arm with the lowest phase-1 mean absolute
error other than B0, L1, L2, S2, N1, and N2 (which phase 2 carries regardless); Y is the arm with
the lowest phase-1 CRPS if it differs from X. Phase 2 carries X, and Y if it exists.

### Phase 2: the rigorous comparison

Every phase-2 arm has the same column count (the widest arm's, with narrower arms padded with
duplicates of their own columns), `colsample_bytree=1`, three seeds, both hyperparameter settings,
all five folds.

- **Per-plant scope:** B0, L1, L2, S2, X, Y, N1 at lead-day 1; B0 and L1 at lead-days 0 and 2
  (primary setting, three seeds, exploratory).
- **Global scope:** B0, L1, X at lead-day 1, one XGBoost model across the 6 plants with no plant
  identifier, coordinates or capacity column. The global fit is a loop around `fit_one_fold` in
  the fit script. For each scored fold it excludes from training every calendar month that any
  plant holds in that fold, because the shared rows' folds differ between plants, and neighbouring
  plants share their weather. The report asserts that no training row shares a calendar month with
  a scored row. The two IFS grid cells partly identify the plant, which the page says.
- **Quantiles:** every phase-2 fit also fits the quantile model, in the fit script's own wrapper
  around `fit_one_fold(with_quantiles=True)`, returning the sorted, capped quantiles. CRPS is
  `crps_capped_mw` divided by `effective_capacity_mw`, labelled "CRPS approximated from nine
  quantiles (0.1 to 0.9)". Coverage is stated as the out-of-fold share of rows inside the 10% to
  90% interval, against a nominal 80%. Interval widths are split by the forecast clear-sky index
  into clear, mixed, and cloudy hours.
- **No purge.** A training row whose input is a test label does not put that label in the model's
  output path, and B0 already trains next to test-month boundaries. The page says so in one
  sentence.
- **Compute:** see "Compute estimate". Fits need a slot from the Study MAIN COORDINATOR and
  checkpoint after every few arm-settings, because a reboot is planned.
- **Reproduction checks:** step 1 refits unpadded B0 on the published row set
  (`arm_rows(drop_gap_rows=True)`) on the GPU, as the published arm was, and asserts its
  fingerprint and error match the published 9.762 points. Step 2 reports padded minus unpadded B0
  on the lag rows. Step 3 reports B0 on the lag rows as its own figure.

### Controls

- **Positive control:** the same pipeline on a synthetic target in which power after a known date
  is multiplied by (1 − s), applied to the hourly power table before the lags are built, for s of
  2%, 5% and 10%. B0 and L1, per-plant, lead-day 1, primary setting, one seed. "Recovers" means the
  contrast meets the decision rule, and the smallest s that L1 recovers is the detection limit.
- **Null and level-only controls:** padded minus unpadded B0 shows what the column count alone
  does. N1 keeps the plant's slow level but drops yesterday's own weather, and N2 keeps neither, so
  L1 minus N1 measures the value of day-specific information beyond the month's level.

### Deciding contrasts (planned, written before any result)

All at lead-day 1, at both hyperparameter settings, with 99% intervals (Bonferroni over five,
through `bootstrap_difference_at_level`).

- **P1:** L1 minus B0, mean absolute error, per-plant, all 21 months.
- **P2:** X minus L1, mean absolute error, per-plant, on folds 3 and 4 only (months the sweep never
  scored), where X is the shortlist rule's arm.
- **P3:** S2 minus L2, mean absolute error, per-plant, all months (both arms carry lagged-weather
  information; the two-step design is the question).
- **P4:** CRPS of L1 minus CRPS of B0, per-plant, all months.
- **P5:** L1 minus B0, mean absolute error, global, all months.

**Decision rule, fixed before any result:** a contrast "helps" if, at both hyperparameter settings,
its 99% interval lies wholly below zero and its point estimate is at least 2% of the baseline's own
value of that metric (about 0.2 points for mean absolute error, the smallest effect worth acting
on); a smaller gain is reported as detectable but not worth the complexity. Every other number is
exploratory and labelled so: the sweep, S2 minus L1, Y, N1 and N2 comparisons, lead-day 0 and 2,
global against per-plant, coverage and sharpness (descriptive), an ENS-mean rerun of B0 and L1
(the live service's input; the columns are on the same rows), and any analysis added after the first
run (post hoc).

### Charts (the story, as figures)

1. **Figure 1, the headline:** P1 to P5 with their 99% intervals, both settings as markers, under
   the bottom line.
2. **The data:** forecast irradiance against measured power for each anonymised plant over one
   stated week per rule (clearest, most variable, dullest), showing steps, gaps and outages; the
   lag hour against the target hour as a scatter per lead-day.
3. **The techniques with known answers:** how each lag column is built for one example day; the
   synthetic level shifts and which arms recover them; N1 and N2 beside L1; the withheld folds of
   the two-step design drawn as months.
4. **The XGBoost models work:** out-of-fold predictions against measured output for each plant
   over the same weeks; each arm's absolute error per plant; B0 beside the published figure.
5. **The sweep:** one ranked chart of all arms' absolute error and CRPS, the baseline as a line,
   labelled as a screen.
6. **Results:** error against lead-day for B0 and L1; monthly paired differences; per-plant against
   global; reliability, coverage and width; feature importance (descriptive) by feature group.

Plants are anonymised A to F (`studies.anonymise.site_labels_for`); nothing prints an identifier,
name, or coordinate, and power is shown only as a share of capacity.

## Compute estimate

Measured on the workstation today: one XGBoost fit of 7,000 training rows with 11 columns, 500
rounds, took 0.7 s (point) and 6.2 s (nine-level quantile) on 4 CPU threads at the primary
setting, 1.1 s and 8.9 s at the sensitivity setting (1,200 rounds). At 44,000 rows (the global
fit) the same fits took 1.7 s and 15.4 s, and 3.0 s and 25.7 s. The quantile model costs about 9
times the point model. On the RTX A6000 the same fits were 3 to 7 times slower (3.7 s and 34.2 s at
7,000 rows), and the GPU was already at 99% utilisation from another job, so that comparison is
contended and an uncontended GPU run is still worth one measurement. For fits this small the GPU's
launch overhead outweighs its speed-up.

| Stage | Fits | Time on CPU, 4 threads each |
|---|---|---|
| Phase 1 sweep (about 24 arms x 3 folds x 6 plants, primary, quantiles) | about 430 | about 50 min of fit time, about 10 min wall at 6 concurrent fits |
| Phase 2 per-plant, lead-day 1 (7 arms x 2 settings x 3 seeds x 5 folds x 6 plants) | 1,260 | about 3 h of fit time, about 40 min wall |
| Phase 2 lead-days 0 and 2 (2 arms x 2 days x 3 seeds x 5 folds x 6 plants) | 360 | about 40 min, about 10 min wall |
| Phase 2 global (3 arms x 2 settings x 3 seeds x 5 folds) | 90 | about 30 min, about 10 min wall |
| Stage-1 models, positive control, ENS-mean rerun | about 600 | about 1 h, about 15 min wall |

**Total: about 5 to 6 h of single-fit time, about 1.5 h of wall time** at six concurrent fits, if
the machine is otherwise idle (load average is 6 now, so allow 2 to 3 h). Dropping the quantile
model from the sensitivity setting would cut about 40% of the time. The GPU would not help at these
sizes, so the plan uses the CPU for every fit except the reproduction check in step 1, and reports
the GPU-versus-CPU difference of B0 as the noise floor.

## Walk through

**The happy path.** `build_lag_frame.py` loads the matched-lead rows and IFS day columns, reads the
full cleaned hourly power (`studies.power`), and adds every sweep arm's columns with strict joins,
asserts the lag arithmetic, applies the row filter, and writes the frame under
`data/studies/lag_features/`. `fit_lag_arms.py` fits stage 1 with withheld folds, fits phase 1 and
then (after the shortlist rule runs and prints X and Y) phase 2, each fit with a point and a
quantile model, checkpointing; it writes per-row losses, out-of-fold predictions and normalised
quantiles. `report_lag_features.py` computes intervals (`bootstrap_difference_at_level`,
`bootstrap_absolute`) and prints every table the page quotes, with each arm's column list.
`lag_features_charts.py` draws the figures from the saved files and the report. The page
`docs/studies/lag-features.md` quotes the report.

**Branches.** The per-plant run calls `out_of_fold_losses` per plant (through `run_all`); the global
run is a separate loop with month exclusion. The positive control modifies the hourly power table
before the lags are built. Lead-day 0 and 2 runs differ in the lag and the IFS day columns.

**Error handling.** A study fails fast: an arm whose column list differs in length raises; an
uncovered calendar month raises (`raise_on_uncovered_months`); a failed lag-arithmetic assertion
raises; a shared-row check that does not hold raises. Dropped rows are counted and printed.

## What changes, file by file

- `studies/lag_features/build_lag_frame.py` — the strict lags, every sweep arm's columns (one
  function per arm family returning a fixed-length tuple), N1, N2, the positive control, the row
  filter, the assertions.
- `studies/lag_features/fit_lag_arms.py` — stage 1, phase 1, the shortlist rule, phase 2, the
  global loop, the quantile wrapper.
- `studies/lag_features/report_lag_features.py`, `lag_features_charts.py`, `README.md`.
- `packages/studies/src/studies/baselines.py` — added functions for the statistics that have a
  silent-failure risk: the weekly and rolling statistics (minimum, maximum, mean, quantile of the
  same clock hour over the latest whole days before the issue day, with the minimum-count rule),
  with tests in `packages/studies/tests/test_baselines.py`.
- `docs/studies/lag-features.md`, an entry in `docs/studies/index.md` and `mkdocs.yml`, and
  `docs/studies/assets/lag_features_*.svg` (optimised with `svgo`).

## Tests

- **Rolling and weekly statistics:** a fixture at lead-days 0 and 1 gives each day, including the
  issue day's clock hour, a distinct value, plus other clock hours and other sites; asserts only
  the latest whole days of the same clock hour and site are used, the issue day is excluded, and
  the minimum-count rule returns null below the count. Each assertion fails on the bug it names
  (including the issue day, the wrong clock hour, a mixed site).
- **Lag arithmetic, equal column counts, the global month exclusion, and the fingerprint of B0:**
  assertions inside the scripts, since study scripts have no unit tests; each raises on violation
  and the report prints the result.

## Design-philosophy check

R&D code: fail fast, as the inherent-stability page requires of training and metrics code. Nothing
runs in production. The study does not choose the production model; a follow-up issue would carry
any result into `ml_core` feature code. The page cites no hypothesis label unless one on
persistence or lag features exists in `engineering-hypotheses.md` (checked at write-up).

## Docs to update

The new study page, the studies index and nav, the `studies/README.md` row, and the family README.
No roadmap item completes, so there is no ship-time triage.

## Verification

`uv run ruff check .`, `uv run ruff format --check .`, `uv run ty check`, `uv run pytest
--run-studies -n auto packages/studies`, `uv run pymarkdown scan -r docs README.md CLAUDE.md
packages/*/README.md`, `uv run mkdocs build --strict` and reading the rendered page, pydoclint and
the docs-link checker.

## Risks and open questions

1. **Sample:** 21 scored months at 6 plants in two grid cells. P2 rests on about 8 months. Intervals
   from a percentile bootstrap over 21 month clusters tend to under-cover, plants are pooled and
   weighted by row count, and the test does not cover differences between plants. The page says all
   three. No new PV data.
2. **Baseline product:** IFS HRES via Open-Meteo scores 9.76 points at day 1 against 8.79 for the
   ENS mean, so a lag may correct this product's own errors. The exploratory ENS-mean rerun
   addresses that. The question for the maintainer: should it be a planned contrast instead?
3. **Device:** the plan uses CPU for every fit, since the GPU was slower in the benchmark. The
   question for the maintainer: is a GPU run of the whole study wanted anyway?
4. **Sweep size:** about 24 arms in phase 1. Cutting the combination, cross-plant and post-model
   arms would remove 8 columns of work but the wackiest ideas.

## Triage of the reviews

**Simplicity review 1.** Accepted: no cycle-change eras (shared folds); reuse of the on-disk
inputs; the existing origin convention; no leave-one-plant-out fit; a smaller phase-2 grid; the
positive control sized to a realistic shift; S2 stage 1 at one setting and seed. Modified: N1 stays
as a level-only control, with N2 added as the true null; the lag is read strictly. The maintainer's
two-phase request reverses the suggestion to run only five arms.

**Correctness review.** Accepted: the global fit with month exclusion across plants (finding 1);
`power_mw` as the per-plant target and capacity-normalised global inputs (2); N1 drawn from days 8
to 28 before the issue day (3); no purge (4); a quantile wrapper and capacity-normalised CRPS (5);
the reproduction check on the published row set and device (6); the data facts, including 21
months and 7 missing runs, and the empty 2024-11-12 split removed (7); stage-1 predictions that
withhold the hour's own fold, and lagged weather at the target's lead day (8); the minimum-count
rule for the weekly statistics, and lags from the full cleaned table (9); P3 as S2 minus L2; 99%
intervals; a threshold for the CRPS contrast; a multiplicative positive control applied before the
lags; script assertions for the lag arithmetic; the "partly identify the plant" wording; the ENS
rerun (as exploratory); the cleaning caveat; the interval caveats; the unspecified details.
Modified: the single device is CPU for the study and GPU for the reproduction step, with the
choice put to the maintainer. Rejected: none.
