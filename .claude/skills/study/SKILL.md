---
name: study
description: >-
  How to run a one-off scientific study in this repository and publish it under `docs/studies/`,
  so the result is scientifically valid before anyone reads it: where a study's code, data, and page
  live; the order of work, from contrasts named before the run to two Opus scientific-validity
  reviews and a re-run whenever a reviewer asks; the design rules that keep a comparison fair
  (shared rows, equal column counts, month-block folds, month-resampled intervals, leakage through
  neighbouring generators); the data traps that fail silently (served lead, timestamp conventions,
  grid cells, steps in served data); how to chart a study so the charts tell the story alone; how to
  write the page without overclaiming; and anonymisation. Load before planning, running, re-running,
  charting, writing up, or reviewing any study under `studies/`, before changing
  `packages/studies/`, and before any experiment whose numbers will reach `docs/`.
---

# Running a study

A study answers one question with an experiment, such as "which weather product best describes past
sunshine?", and publishes the answer as a page under `docs/studies/`. The maintainer does not review
study code line by line, so **the adversarial reviews in this skill are the main defence of every
number a study publishes.** The priority, above speed and above tidiness, is a result that is
scientifically valid and says no more than its evidence supports.

**Work autonomously.** Plan, run, review, re-run, chart, and write up without stopping to ask, except
for decisions that belong to the maintainer: merging (unless the maintainer has said to merge once
the reviews are clean), spending money, or ordering data in someone's name. Post short progress
updates while working, and a full report at the end.

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
two studies need the same code, move that code into the package with tests rather than letting a
second script import a third script's private function.

**A study script is not unit-tested, so its check is its own output.** Every table the page quotes
is printed by the script into a `report.md`, never transcribed by hand, and every number on the page
is checked against that report.

## The order of work

1. **Plan the study** with the `plan-issue` skill, sized complex. The plan names the question, the
   arms, the row set, the folds, the metric, and the interval method.
2. **Name the deciding contrasts in the plan, before any result exists.** Two to five contrasts
   that answer the question. Every other number the study produces is exploratory and will be
   labelled so on the page. An analysis added after the first run is post hoc, and is labelled so
   too. A post-hoc result is still worth reporting, as long as it is labelled.
3. **Build the datasets, run, and write the report.** Check every arm has the columns the plan gave
   it before fitting (see "An arm can silently lose a column", below).
4. **First Opus scientific-validity review**, of the design and the first results. Triage it, fix,
   and **re-run whatever the reviewer asks to be re-run**.
5. **Second Opus scientific-validity review**, by a fresh reviewer, of the revised study and its
   written page, judged as an outside expert would judge it before publication. Triage, fix, re-run.
   Keep reviewing until a reviewer finds nothing that must be fixed.
6. **Chart the results and write the page** (sections below). Every result gets a chart.
7. **Diff review and prose review**, with the `implement-issue` and `prose-review` skills.
8. **Merge**, only as the maintainer has authorised, checking the PR's body and commit messages
   for closing keywords first (see "GitHub hygiene").

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
`colsample_bytree` below 1, an arm with more columns wins for no reason: on this repository's PV
data, 7 features padded with 2 duplicate columns scored 10% better at `colsample_bytree=0.8`, and
exactly equal at 1.0. Keep row subsampling (`subsample`), which is what makes seeds differ. Where
one arm genuinely needs more columns, give the other arm the same count, made of deterministic
functions of its own features: a separation model's output is exactly that, and doubles as a
negative control.

**An arm can silently lose a column.** In the wind study the planned "served 100 m" arm lost its 10 m
speed, and the page briefly claimed ICON-EU loses to ERA5 at 100 m. With the column restored ICON-EU
wins at both heights. Check each arm's column list against the plan in code (`_check_arms` in
`wind_products.py`) before fitting.

**Neighbouring generators share their weather, so the effective sample size is weather episodes, not
generator-hours.** The six solar farms in the trial area sit in one 25 km by 23 km box and in two
ERA5 grid cells. Three consequences:

- **Folds are contiguous blocks of whole months** (`studies.cross_validation.assign_folds`).
- **Intervals resample whole calendar months, paired across arms, and one fitting seed**
  (`studies.bootstrap.bootstrap_difference`, 2,000 resamples). A row-level bootstrap would be far
  too narrow.
- **A model that "generalises to an unseen generator" leaks unless the scored months are withheld at
  every generator.** A leave-one-site-out arm trained on the other five generators at the scored
  months learns each day's outcome from the neighbours. Before the fix, that arm appeared to beat
  the per-generator model; after it, transfer costs 0.06 to 0.18 points.

**Cut the folds on both sides of any change in an input's version, and tell the model which side each
hour falls on.** The Met Office upgraded UKV on 21 January 2026. With month-block folds, every
post-upgrade row landed in a fold whose model had trained on pre-upgrade UKV alone. A contrast
measured after the upgrade came out five and a half times its size before, and most of that jump was
the model meeting an input it had never trained on. The fix cuts the folds within each era
(`assign_folds(by=("site", "era"))`), adds an era feature, and drops the part-month that straddles
the change.

**Normalise each row's error by its own generator's capacity before any mean or difference.** A test
fixture where every row had the same capacity passed on the bug it existed to catch; give fixtures
unequal capacities.

**Report every arm's absolute error, not only the contrasts.** Which input is best can matter far
more than the contrast the issue asked about: in the beam/diffuse study, choosing the weather product
was worth about 4 points and choosing the beam/diffuse split about 0.1.

**Run a second hyperparameter setting, and report when a check fails.** It shows whether an ordering
belongs to the features or to the settings. Where a contrast changes sign or significance under the
second setting, the page says so.

**Build in controls.** A negative control (an arm known to carry no new information) shows the size
of difference the pipeline produces from nothing. A positive control (a synthetic target where the
effect must exist) shows the instrument can detect an effect at all. A null result means nothing
until the positive control has passed.

## Know what the archive actually serves

**A weather archive's value for an hour is a forecast made some hours earlier, and that served lead
is part of what a consumer gets.** Open-Meteo's historical-forecast archive keeps, for each hour, the
freshest run that covers it, so the lead follows each model's run cycle:

| Product | Served lead | How it was established |
|---|---|---|
| UKV | T+0, the analysis | `verify_ukv_lineage.py`: within 0.55 W m⁻² of the Met Office's files since August 2024 |
| ICON-EU | 1 to 3 h for radiation | `verify_icon_lineage.py --model icon-eu`: within 1 W m⁻² at 9 of 9 hours |
| ICON-D2 | 1 to 3 h for radiation | the same check matched only 7 of 9 hours, up to 44 W m⁻² off, so the 3-hour pattern in its own errors is the stronger evidence |
| ICON global | 1 to 6 h for radiation | inferred from its 6-hourly cycle; the check cannot read its grid |
| ERA5 | radiation from 1 to 12 h forecasts; wind an hourly analysis | ECMWF's documentation, and where the hour-to-hour jumps fall |

**Measure the lead; do not read it off documentation.** Compare the archive's value against the
source's own files for several hours and runs. Where that is impossible, use where the hour-to-hour
jumps fall in the served series, and say on the page which method each lead rests on.

**Radiation is a mean over the hour before its label; wind is an instantaneous value at its label.**
The ICON radiation lead follows `((h−1) % 3) + 1`, the instantaneous wind lead `h % 3` (`h % 6` for
ICON global). So solar power is aggregated to the hour ending at the label, and wind power to the
hour centred on it (shift the stamps back 30 minutes before `studies.power.hourly_from_half_hourly`).
Scan the offset: the solar convention applied to wind handicaps UKV by 0.27 points. UKV publishes
radiation as a snapshot, and Open-Meteo builds UKV's hourly value from the snapshot at the hour's
end; averaging the snapshots at both ends of the hour cut UKV's error by 0.60 points and erased a gap
the first draft reported as ICON-EU beating UKV.

**Read each product from a grid cell that represents the generator.** At one wind generator the
nearest ICON global cell is influenced by the sea; every product is read from its nearest land cell
(`studies.grid_sampling.sample_nearest_cell`).

**Look for steps in the served data before attributing a gap to a model.** ICON global's served
wind at one generator steps by about 12% in early June 2025 and back in early June 2026, at every
height, and no other product shows it. That pair of steps explains about half of ICON global's
deficit, which the first draft attributed to its coarser grid. Plot each product's series against a
sibling product's, per generator, before interpreting a per-generator gap.

**A feed's meaning can change mid-record.** From April 2026 two wind generators' feeds stop
publishing exact zeros, and their calm half-hours go missing instead. The wind study drops every hour
holding an exact zero, so the two periods match, and reports that keeping them moves no contrast by
more than 0.03 points.

**Check which variables a feed carries on a recent run.** Providers add fields over time, so a
listing from an old date gives a confident wrong answer.

**Know each source's access limits before planning around it:**

- **CDS ERA5:** one request at a time, about 2.5 minutes per month of three hourly fields, and a
  ceiling of 121,000 fields per request. Open-Meteo's ERA5 mirror is much faster, and
  `verify_era5_sources.py` shows the two agree.
- **Open-Meteo's Previous Runs API** serves only a `_previous_day1` column, from January 2024 for
  ICON and August 2024 for UKV.
- **UKV's hub-height wind** on Open-Meteo starts on 12 August 2024.
- **SARAH-3** is not on the Copernicus Climate Data Store. EUMETSAT's Data Store API cannot subset a
  region, so the data needs a manual order in the maintainer's name.

## Reproducibility

- **After a refactor, prove the outputs are unchanged, bit for bit.** Moving the fit loop into
  `packages/studies` was accepted only after 5,289,312 per-row losses and 504 intervals came out
  identical to `main`'s.
- **When a published number drifts because the data changed, file an issue; do not fold the change
  into unrelated work.** A rebuilt `effective_capacity` table moved the beam/diffuse figures in their
  third or fourth decimal place (#825), and each page's "What this does not show" section says which
  table its figures rest on.
- **Never overwrite an output a published page quotes.** Move it to `superseded/` first.
- **Every page ends with the commands that reproduce it.**
- **Run a moved or promoted script end to end.** No linter evaluates a `sys.path` string, so a script
  moved two directories deeper broke silently until someone ran it.

## Charts

**Put plenty of charts in every study page, because many technical readers look at the charts before
reading any text.** Each chart, with its title, subtitle, axis labels, and legend, tells its part of
the story without the prose around it. Load the `dataviz` skill before drawing any chart.

- **A headline chart opens every page**, directly under the opening paragraphs, showing the headline
  result with its 95% intervals. Every section whose claim rests on a number gets a chart too.
- **The title states the finding** ("CAMS describes past sunshine best, by a wide margin"), and
  matches the heading or bolded lead the chart sits under.
- **The subtitle names the quantity, the unit, the scope, and what a dot and a line mean**, and says
  "exploratory" for any contrast not named before the run.
- **A labelled reference rule** at zero says what zero means ("same as ERA5"), and a side label says
  which direction is better.
- **Show paired differences with their intervals, not each arm's level with an interval.** The arms
  share their rows, so level intervals overlap even where the paired difference clearly excludes
  zero, and a reader takes the overlap to mean no difference. Put each arm's absolute error in its
  row label instead.
- **Take every number a chart shares with the page from the report**, so the chart cannot disagree
  with the page.
- **Colours come from OCF's brand guidelines, through `plotting.ocf_theme`, and are checked with the
  `dataviz` skill's `validate_palette.js`.** One colour per product fails the check (Data Purple and
  Data Blue sit 2.0 ΔE apart under deuteranopia), so colour a group of products (satellite,
  reanalysis, weather model) and put every product's name on the axis. Use the guidelines' main data
  colours on published pages; the "additional" data colours are marked for internal use only.
- **Write SVG, then optimise it** with `npx svgo@4 --multipass --precision=1 --final-newline`, as
  `CLAUDE.md` requires, and look at every chart rendered to PNG before committing it.
- **No chart shows a generator's output in MW.** Use the anonymised labels, and normalise by each
  generator's own capacity.

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
6. **What this does not show**: region, period, recalibration, and anything else a reader might
   over-read.
7. **Reproducing the figures**.

**Every claim carries its scope and its interval.** The reviews on the weather-product pages caught
these overclaims, which are the shapes to look for:

- **A gap that a construction artefact explains.** "ICON-EU beats UKV" was Open-Meteo's end-of-hour
  UKV snapshot; rebuilt from both snapshots, the two cannot be told apart.
- **A win that leakage explains.** "A model trained on five generators predicts the sixth better"
  was the neighbouring generators sharing the scored months' weather.
- **A cause named without isolating it.** "ICON global's deficit is its coarser grid" was mostly a
  sea-influenced grid cell and a pair of steps in its served wind.
- **A result from an arm that did not match the plan.** "ICON-EU loses to ERA5 at 100 m" came from an
  arm that had lost a column.
- **A lead-unequal comparison stated as a model comparison.** Say which product had the shorter lead,
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

**A scientific-validity review attacks:** whether every page number matches the report, sign and
interval included; every overclaim; the row set, leakage, folds, normalisation, and intervals;
served leads and timestamp conventions, checking at least one itself; confounds between arms (lead,
height, grid cell, column count, era); whether each recommendation follows from the evidence; and
what a critical outside reader would attack first. The reviewer names which analyses must be re-run,
and how.

**Triage every finding against the code and the data before acting on it.** Reviewers are often
wrong: one asserted that ICON-D2's lead was the precisely measured one and ICON-EU's the approximate
one, which the lineage files showed was backwards. A finding can also be right with the wrong fix: a
reviewer proposed deleting a test that asserted the wrong thing, where rewriting its fixture kept the
coverage. Record each rejected finding and its reason in the PR.

**When a reviewer asks for a re-run, re-run.** Reasoning about what a re-run would show is not a
substitute.

## Anonymisation

**Never publish a metered generator's time series with its name or identifier** (`CLAUDE.md`). In a
study:

- Relabel generators with `studies.anonymise.site_labels_for` before anything is written: A to F for
  solar, W1 to W3 for wind, each under its fixed permutation seed. The seed is not a secret; the
  roster being private is what protects the mapping.
- Never write an identifier, name, or coordinate into a chart, a page, a report, a commit message, a
  PR or issue body, or a reviewer's brief.
- Never publish one generator's output in MW; normalise by its own capacity.
- A script that has to send coordinates to a point service reads them at run time from the private
  roster, and writes only the label.

## GitHub hygiene

- **Keep closing keywords away from the parent issue's number.** A study usually lives under an epic,
  and "fixed" near "#809" in a PR body would have closed the epic on merge. The
  `github-issue-pr-workflow` skill has the check to run before merging.
- **Rename a PR titled "Plan: …" before merging**, because the title becomes the merge commit's
  subject.
- **Report a design mistake outside the study's scope as its own issue**, and keep the study's PR to
  the study.
