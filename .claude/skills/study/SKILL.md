---
name: study
description: >-
  How to run a one-off scientific study in this repository and publish it under `docs/studies/`,
  so the result is scientifically valid before anyone reads it: where a study's code, data, and page
  live; the order of work, from contrasts named before the run to two Opus scientific-validity
  reviews and a re-run whenever a reviewer asks; the design rules that keep a comparison fair
  (shared rows, equal column counts, month-block folds, month-resampled intervals, leakage through
  neighbouring generators); the data traps that fail silently (served lead, derived fields,
  timestamp conventions, grid cells, steps in served data); how to chart a study so the charts tell
  the story alone; how to write the page without overclaiming; and anonymisation. Load before
  planning, running, re-running, charting, writing up, or reviewing any study under `studies/`,
  before changing `packages/studies/`, and before any experiment whose numbers will reach `docs/`.
---

# Running a study

A study answers one question with an experiment, such as "which weather product best describes past
sunshine?", and publishes the answer as a page under `docs/studies/`. The maintainer does not review
study code line by line, so **the adversarial reviews in this skill are the main defence of every
number a study publishes.** The priority, above speed and above tidiness, is a result that is
scientifically valid and says no more than its evidence supports.

**Work autonomously: the maintainer has asked for studies to run with as little waiting on a human
as possible.** Plan, run, review, re-run, chart, and write up without stopping to ask, and post short
progress updates while working and a full report at the end. So a study does not stop for human
review after `plan-issue`'s plan. Three decisions still belong to the maintainer:

- **Spending money.**
- **Ordering data or accepting a data licence** in someone's name.
- **Merging.** A study merges only if the maintainer has said so in the current session; otherwise
  it stops at a reviewed PR, as `implement-issue` does.

## Where a study's pieces live

| What | Where |
|---|---|
| Tested machinery shared by studies: the out-of-fold fit loop, the paired bootstrap, grid sampling, anonymisation, solar geometry, power aggregation, served-column checks, the Fractions Skill Score | `packages/studies/src/studies/` |
| The study's own scripts: fetch, build, run, report, chart | `studies/<study>/`, run with `uv run python studies/<study>/<script>.py` |
| Downloaded weather, one directory per product | `data/studies/weather/<PRODUCT>/` |
| NGED's active-network-management exports | `data/studies/anm/` |
| A study's datasets and results, and a `superseded/` directory for outputs a later run replaced | `data/studies/<study>/` |
| The published page and its charts | `docs/studies/<page>.md`, `docs/studies/assets/` |

**`packages/studies/` is Claude's to own.** Keep it tidy, and restructure it whenever a study needs
to. Every function in it carries tests, and each test must be able to fail on the bug it exists to
catch; run a mutation pass (the `implement-issue` skill, step 7) whenever the package changes. When
two studies need the same code, move that code into the package with tests. `wind_products.py`
still imports `weather_products.py`'s private helpers; do not copy that pattern into a new study.

**A study script is not unit-tested, so its check is its own output.** Every table the page quotes
is printed by a committed script into a `report.md`, never transcribed by hand, and every number on
the page is checked against that report. A number from a diagnostic run during review goes on the
page only after a committed script prints it into the report.

## The order of work

1. **Plan the study with the `plan-issue` skill, and state the five trigger answers.** A published
   page counts as what gets stored, so a study is complex. The plan names the question, the arms,
   the row set, the folds, the metric, and the interval method. The mutation pass
   (`implement-issue` step 7) runs only when `packages/studies/` changes, because study scripts
   have no tests; say so in the PR body.
2. **Name the deciding contrasts in the plan, before any result exists.** Name two to five contrasts
   that answer the question. Every other number the study produces is exploratory, and is labelled
   so on the page. An analysis added after the first run is post hoc, and is labelled so too.
3. **Build the datasets, run, and write the report.** Print every arm's feature columns into the
   report (see "An arm can silently lose a column", below).
4. **First Opus scientific-validity review**, of the design and the first results. Triage it, fix,
   and **re-run whatever the reviewer asks to be re-run.** Reasoning about what a re-run would show
   is not a substitute for running it.
5. **Draft the page and its charts, then run the second Opus scientific-validity review** with a
   fresh reviewer, judging the revised study and the draft page as an outside expert would before
   publication. Triage, fix, and re-run. Keep reviewing until a reviewer finds nothing that must be
   fixed.
6. **Finish the charts and the page** (sections below).
7. **Diff review and prose review**, with the `implement-issue` and `prose-review` skills.
8. **Merge, only as the maintainer has authorised,** after checking the PR's body and commit
   messages for closing keywords (see "GitHub hygiene").

Never publish a study with fewer than two scientific-validity reviews. If the process is shortened
anywhere, for example one plan review instead of two, say so in the PR body and in the final report,
with the reason.

## Design rules that keep a comparison fair

**Score every arm on exactly the same rows.** Keep an hour only if every arm's input covers it, and
decide which hours to drop from the target, never from any one arm's input. A filter that reads one
product's values (a "reliable" flag, a zero) hands that product a row set the others do not get. The
weather-products study reads the satellite retrieval in full for this reason, and reports the
flagged-hours-only figure beside it.

**Give every arm the same number of feature columns, and set XGBoost's column subsampling to 1.** With
`colsample_bytree` below 1, an arm with more columns wins without carrying more information: on this
repository's PV data, 7 features padded with 2 duplicate columns scored 10% better at
`colsample_bytree=0.8`, and exactly equal at 1.0. About 0.4% of mean absolute error still favours
the wider arm at 1.0 on synthetic data, so equal column counts matter even then. Keep row
subsampling (`subsample`), which is what makes seeds differ. Where one arm genuinely needs more
columns, give the other arm the same count, built from its own features: a separation model's
estimate of the beam is a deterministic function of the narrow arm's global irradiance and sun
position, and doubles as a negative control.

**An arm can silently lose a column.** In the wind study the planned "served 100 m" arm lost its 10 m
speed, and the draft page claimed ICON-EU loses to ERA5 at 100 m. The second science review caught
it; with the column restored, ICON-EU wins at both heights. No code in `wind_products.py` checks
this. Build each arm's columns from one function per arm type that returns a fixed-length tuple
(`_wind_columns` and `_served_100m_columns` do this), and print every arm's column list into the
report, so a reviewer can check it against the plan.

**Neighbouring generators share their weather, so the effective sample size is weather episodes, not
generator-hours.** The six solar farms in the trial area sit in one 25 km by 23 km box and in two
ERA5 grid cells. Three consequences:

- **Folds are contiguous blocks of whole months** (`studies.cross_validation.assign_folds`).
- **Each interval resample draws whole calendar months, paired across arms, and one of the three
  fitting seeds** (`studies.bootstrap.bootstrap_difference`, 2,000 resamples). A row-level bootstrap
  would be far too narrow.
- **A model that "generalises to an unseen generator" leaks unless the scored months are withheld at
  every generator.** A leave-one-site-out arm trained on the other five generators at the scored
  months learns each day's outcome from the neighbours. Before the fix, that arm appeared to beat
  the per-generator model; after it, transfer adds 0.06 to 0.18 points to the error.

**Cut the folds on both sides of any change in an input's version, and tell the model which side each
hour falls on.** The Met Office upgraded UKV on 21 January 2026. With month-block folds, every
post-upgrade row landed in a fold whose model had trained on pre-upgrade UKV alone. A contrast
measured after the upgrade came out five and a half times its size before, and most of that jump was
the model meeting an input it had never trained on. The fix cuts the folds within each era
(`assign_folds(by=("site", "era"))`), adds an era feature, and drops the part-month that straddles
the change. **Treat a change of source as an era boundary too:** Open-Meteo's UKV before 12 August
2024 is a backfill from a source it does not name, and CEDA's UKV archive differs statistically
from the live UKV.

**Normalise each row's error by its own generator's capacity before any mean or difference.** A test
fixture where every row had the same capacity passed on the bug it existed to catch; give fixtures
unequal capacities.

**Report every arm's absolute error, not only the contrasts.** Which input is best can matter far
more than the contrast the issue asked about: in the beam/diffuse study, the choice of weather
product moved the error by about 4 points and the choice of beam/diffuse split by about 0.1.

**Run the second hyperparameter setting (`studies.cross_validation.SENSITIVITY_HYPER_PARAMETERS`) on
every deciding contrast.** A second setting shows whether an ordering belongs to the features or to
the settings. Where a contrast changes sign or significance under the second setting, the page says
so.

**Build in controls.** A negative control (an arm known to carry no new information) shows the size
of difference the pipeline produces from nothing. A positive control (a synthetic target where the
effect must exist) shows the instrument can detect an effect at all. Before a null result is read as
"no effect", either a positive control must have passed or the interval must bound the effect, as in
"an effect as large as 0.08 points is not excluded".

**Clean the target, not the inputs, and never correct NGED's timestamps twice.**
`PowerTimeSeries.correct_late_timestamps` already moves NGED's readings to the right half-hour at
ingest, and shifting again undoes the repair on 93% of rows (`studies.power` warns about this). The
target also needs cleaning of hours the weather cannot explain: the export cap, active network
management curtailment, a commissioning ramp, and, for wind, turbine availability nobody records.

## Know what the archive actually serves

**A weather archive's value for an hour is a forecast made some hours earlier, and that served lead
is part of what a consumer gets.** Open-Meteo's historical-forecast archive keeps, for each hour, the
freshest run that covers it, so the lead follows each model's run cycle:

| Product | Served lead | How it was established |
|---|---|---|
| UKV | T+0, the analysis | `verify_ukv_lineage.py`: within 0.55 W m⁻² of the Met Office's files, for hours since 12 August 2024 only |
| ICON-EU | 1 to 3 h for radiation | `verify_icon_lineage.py --model icon-eu`: within 1 W m⁻² at 9 of 9 hours (one day, one place) |
| ICON-D2 | 1 to 3 h for radiation | the same check: the freshest run was the closest match at 7 of 9 hours, and differed by up to 44 W m⁻², so the 3-hour pattern in ICON-D2's own errors is the stronger evidence |
| ICON global | 1 to 6 h for radiation | inferred from its 6-hourly cycle; the check cannot read its grid |
| ERA5 | radiation from 1 to 12 h forecasts; wind an hourly analysis | ECMWF's documentation, and where the hour-to-hour jumps fall |

**Measure the lead; do not read it off documentation.** Compare the archive's value against the
source's own files for several hours and runs. Where that is impossible, use where the hour-to-hour
jumps fall in the served series, and say on the page which method each lead rests on.

**Radiation is a mean over the hour before its label; wind is an instantaneous value at its label.**
For a model run every `n` hours, an hourly-mean radiation value at label hour `h` has served lead
`((h−1) % n) + 1`, and an instantaneous wind value has lead `h % n`. So solar power is aggregated to
the hour ending at the label, and wind power to the hour centred on it: shift the stamps back 30
minutes before `studies.power.hourly_from_half_hourly`.

**Scan the power-hour offset for every product.** Applying the solar convention to wind handicaps UKV
by 0.27 points.

**A served value can be built differently from its neighbours in the table.** UKV publishes radiation
as a snapshot, and Open-Meteo builds UKV's hourly value from the snapshot at the hour's end.
Averaging the snapshots at both ends of the hour cut UKV's error by 0.60 points, and erased a gap
the draft page reported as ICON-EU beating UKV.

**A served field can be derived rather than native.** Open-Meteo's ICON "100 m" wind is the 120 m
speed multiplied by about 0.98, so the wind study reads each ICON product's native 80 m wind. A
"direct" radiation field can be a separation model's output rather than a retrieval. Run
`studies.served_column_checks.check_hourly_value_is_a_backward_mean` and
`check_direct_is_not_a_separation_model` at fetch time, as `fetch_open_meteo_point.py` does.

**Read each product from a grid cell that represents the generator.** At one wind generator the
nearest ICON global cell is influenced by the sea, with a 10 m speed 16% higher than the land cell's.
The wind study therefore asks Open-Meteo for each product's nearest land cell
(`cell_selection="land"` in `fetch_wind_point.py`); the solar study reads the nearest cell.
`studies.grid_sampling.sample_nearest_cell`, which samples a projected grid directly, silently
returns the edge cell for a site outside the grid, so check a regional product's domain first.

**Look for steps in the served data before attributing a gap to a model.** ICON global's served wind
at one generator steps down, relative to ICON-EU's, in early June 2025 and back up in early June
2026: about 15% at 10 m and 8% at 80 m. No other product shows the steps. They explain about half of
ICON global's deficit, which the draft page attributed to its coarser grid. Plot each product's
series against a sibling product's, per generator, before interpreting a per-generator gap.

**A feed's meaning can change mid-record.** From April 2026 two wind generators' feeds stop
publishing exact zeros, and their calm half-hours go missing instead. The wind study drops every hour
holding an exact zero, so the two periods match, and reports that keeping them moves no contrast by
more than 0.03 points.

**Check which variables a feed carries on a recent run.** Providers add fields over time, so a
listing from an old date gives a confident wrong answer.

**Know each source's access limits before planning around it:**

- **CDS ERA5:** one request at a time, about 2.5 minutes per month of three single-level variables
  at hourly resolution (about 2,200 fields), and a ceiling of 121,000 fields per request.
  Open-Meteo's ERA5 mirror is much faster. `verify_era5_sources.py` shows the two agree on global
  irradiance, direct beam, and air temperature; ERA5 wind from the mirror is unchecked.
- **Open-Meteo's Previous Runs API** offers only whole-day lead offsets (`_previous_day1` to
  `_previous_day7`), with no hour-level offset. ICON-D2 and UKV fill only `_previous_day1`. The
  offsets start on 19 January 2024 for the ICON models and on 6 August 2024 for UKV.
- **UKV's hub-height wind** on Open-Meteo starts on 12 August 2024.
- **SARAH-3** is not on the Copernicus Climate Data Store. EUMETSAT's Data Store API cannot subset a
  region, so the data needs a manual order, with an area subset, through the CM SAF web user
  interface in the maintainer's name.

## Reproducibility

- **Keep every intermediate result on disk unless it is truly huge, so a later chart or analysis
  needs no refit.** Save each model's out-of-fold predictions beside its per-row losses, with the
  actual value, the site label, the time, the fold, the seed, the arm, and the setting, and save each
  study's intervals as a table as well as in `report.md`. A plot or a blend of existing models added
  months later should read these files, not rerun hours of fitting. Record in the study's README
  what each file holds.
- **After a refactor, prove the per-row outputs are unchanged, bit for bit.** Moving the fit loop
  into `packages/studies` was accepted only after 5,289,312 per-row losses and 504 intervals came
  out identical to `main`'s. Per-row outputs are the level to compare: a per-site summary built from
  a frame concatenated in completion order is not bit-stable even between two runs of `main`, and
  XGBoost's row subsampling depends on row order.
- **When a published number drifts because the data changed, file an issue; do not fold the change
  into unrelated work.** A rebuilt `effective_capacity` table moved the beam/diffuse figures in their
  third or fourth decimal place (#825). Every page states, in "What this does not show", which
  `effective_capacity` table its figures rest on and when that table was built.
- **Never overwrite an output a merged page quotes; move it to `superseded/` first.** Every worktree
  writes to the main checkout's `data/studies/` (`sources.REPO_DATA_DIR`), so a re-run in a review
  worktree or a parallel session overwrites the same files.
- **Every page ends with the commands that reproduce it.**
- **Run a moved script end to end.** No linter evaluates a `sys.path` string: when
  `site_e_commissioning.py` moved up one directory, its `parents[3]` path hack pointed above the
  repository root, and nothing flagged the break until someone ran the script.

## Charts

**Put plenty of charts in every study page, because many technical readers look at the charts before
reading any text.** Each chart, with its title, subtitle, axis labels, and legend, tells its part of
the story without the prose around it. Load the `dataviz` skill before drawing any chart.

- **A headline chart opens every page**, directly under the opening paragraphs, showing the headline
  result with its 95% intervals. Every section whose claim rests on a number gets a chart too.
- **The title states the finding** ("CAMS describes past sunshine best, by a wide margin"), and
  matches the heading or bolded lead the chart sits under.
- **The subtitle names the quantity, the unit, the scope, and what a dot and a line mean**, and says
  which rows are exploratory or post hoc.
- **A labelled reference rule** at zero says what zero means ("same as ERA5"), and a side label says
  which direction is better.
- **Show paired differences with their intervals, not each arm's level with an interval.** The arms
  share their rows, so level intervals overlap even where the paired difference clearly excludes
  zero, and a reader takes the overlap to mean no difference. Put each arm's absolute error in its
  row label instead.
- **A chart of differences from one reference cannot show whether two other rows differ.** Put the
  named paired contrasts in a second panel beside it.
- **Take every number a chart shares with the page from the report**, so the chart cannot disagree
  with the page.
- **Take colours from `plotting.ocf_theme`, which encodes OCF's brand guidelines, and check every set
  with the `dataviz` skill's `validate_palette.js`.** On a published page use only the guidelines'
  main data colours: Brand Orange `#FF4901`, Data Blue `#306BFF`, Data Sky `#10C5F7`, Data Purple
  `#B701FF`, and Data Green `#17E58F`, with their light shades for a second condition of the same
  series. The guidelines mark the additional data colours, such as the dark teal `#009C75` and the
  amber `#FC9700`, for internal use only. `PALETTE`'s default order includes both of those, so set
  every colour explicitly. One colour per product fails the check (Data Purple and Data Blue sit 2.0
  ΔE apart under deuteranopia), so colour a group of products and put every product's name on the
  axis.
- **Write SVG, then optimise it** with `npx svgo@4 --multipass --precision=1 --final-newline`, as
  `CLAUDE.md` requires, and look at every chart rendered to PNG before committing it.

## Writing the page

The prose rules in `CLAUDE.md` apply, and the `long-form-prose` skill governs the page's order. The
two weather-product pages are the pattern:

1. **An opening paragraph** naming the consumers of the answer, then bolded leads stating each
   finding with its number and interval, then the evidence's size ("six metered solar farms inside
   one 25 km by 23 km box, and 79,384 generator-hours from December 2022 to September 2026"). The
   headline chart follows.
2. **What is being compared**, as a table: each product's lead, grid, coverage, history, and delay.
3. **How the comparison was made**: rows, model, folds, normalisation, intervals, and which contrasts
   were named before the run.
4. **One section per finding**, headed by the finding, each with its chart.
5. **Which option each consumer should use**, scoped to the evidence.
6. **What this does not show**: the region, the period, the per-generator recalibration, the
   capacity table, and every other scope a reader might over-read, such as equal leads.
7. **Reproducing the figures**.

**Every claim carries its scope and its interval.** The reviews of the weather-product pages caught
each of these overclaims, all described above:

- a gap that the way a value was built explains (UKV's end-of-hour snapshot);
- a win that leakage explains (the leave-one-site-out arm);
- a cause named without isolating it (ICON global's "coarser grid", which was mostly a sea-influenced
  grid cell and a pair of steps);
- a result from an arm that did not match the plan (the lost 10 m column);
- a comparison at unequal leads stated as a model comparison. Say which product had the shorter lead,
  and which way equal leads would move the gap.

**Where a result is unresolved, say so, and show the conflicting intervals.** "UKV against ERA5 is
unresolved" is a finding.

## Reviews

**Write each reviewer's brief to a file** under `.claude/worktrees/`, and give the Agent tool a
three-line prompt pointing at it. A long inline prompt can trip an API safeguard. Use Opus for every
review and every judgement, and Sonnet for mechanical work such as fetching documentation, checking
where a dataset is served, or mining transcripts.

**Each reviewer is fresh, and is told:**

- which files to read, and that the results are read-only;
- to do the review itself, never to dispatch sub-agents, and to keep scratch work under
  `.claude/worktrees/`, never `/tmp`;
- never to write a generator's identifier, name, or coordinates;
- what earlier reviews already settled, so the reviewer spends its effort elsewhere;
- to rank findings, give a verdict on each (must fix, should fix, fine) with the exact replacement
  or re-run, and end with an overall verdict.

**A scientific-validity review attacks:**

- whether every page number matches the report, sign and interval included;
- every overclaim;
- the row set, leakage, folds, normalisation, and intervals;
- served leads and timestamp conventions, checking at least one itself;
- confounds between arms: lead, height, grid cell, column count, and era;
- whether each recommendation follows from the evidence;
- what a critical outside reader would attack first.

The reviewer names which analyses must be re-run, and how.

**Triage every finding against the code and the data before acting on it.** Reviewers are often
wrong: one reviewer asserted that ICON-D2's lead was the precisely measured one and ICON-EU's the
approximate one, which the lineage files showed was backwards. A finding can also be right with the
wrong fix: a reviewer proposed deleting a test that asserted the wrong thing, where rewriting its
fixture kept the coverage. Record each rejected finding and its reason in the PR.

## Anonymisation

**Never publish a metered generator's time series with its name or identifier** (`CLAUDE.md`). In a
study:

- Relabel generators with `studies.anonymise.site_labels_for` before anything is written: A to F for
  solar, W1 to W3 for wind, each under its fixed permutation seed. The seed is not a secret; the
  roster being private is what protects the mapping.
- The labels belong to a fixed roster, and `site_labels_for` raises on a different count. A study
  with a different set of generators needs its own label tuple and seed, and never reuses A to F or
  W1 to W3 for other generators.
- Never write an identifier, name, or coordinate into a chart, a page, a report, a commit message, a
  PR or issue body, or a reviewer's brief.
- Never publish one generator's output in MW; normalise by its own capacity.
- A per-generator weather series can identify a generator too: a distinctive feature, such as a step
  in one product's served wind, can be matched against public archives at nearby grid cells. Plot
  such a series without its generator label.
- A script that has to send coordinates to a point service reads them at run time from the private
  roster, and writes only the label.

## GitHub hygiene

- **Tag every study issue and pull request as a spike.** Give each issue the org issue Type
  `Spike` and the `spike` label, and give each pull request the `spike` label, since a pull request
  cannot take an issue Type. The `github-graphql` skill has the mutation that sets the Type.
- **Keep closing keywords away from the parent issue's number.** A study usually lives under a parent
  issue, and "fixed" near "#809" in a PR body would have closed that parent on merge. The
  `github-issue-pr-workflow` skill has the check to run before merging.
- **Rename a PR titled "Plan: …" before merging**, because the title becomes the merge commit's
  subject.
