# Promote the study machinery into a tested workspace package (#807) and stop the double stamp correction (#815)

**The problem.** The code under `scripts/experiments/beam_diffuse_split/` decides what production
gets built to do — which irradiance source feeds a PV forecast, which NWP we ingest — and has no
tests. One validation that was performed was written into a README as a markdown table instead of a
test file, and one defect (a filter that matched no arm and printed an empty table) survived an
adversarial reading and was only found by running the script. Separately, three of those scripts
apply NGED's half-hour stamp repair themselves, and [#790](https://github.com/openclimatefix/nged-substation-forecast/issues/790)
has since moved that repair into the ingest, so the power table has now been re-materialised already
repaired and two of the three scripts would apply the shift a second time.

**The planned solution.** Fix the double correction first, as its own commit, by deleting the
`--alignment` axis from every script that carries it and keeping the one arithmetic that is now
correct. Then move `scripts/experiments/` to a top-level `studies/`, promote eight pieces of
reusable machinery into a new `packages/studies/` workspace member with unit tests, and keep the
research dependencies out of the production image by listing the package in the root `dev`
dependency group — the mechanism `dashboard` already uses and that the Dockerfile's `--no-dev`
already excludes. The studies themselves stay as untested scripts.

## Verdict, size and departures

### Verdict: worth implementing, with four departures from the issue body

Both issues are worth doing. #815 is live today: PR #803 merged on 2026-09-22, the power table has
been re-materialised, and `build_dataset.py`'s default `--alignment piecewise` branch now shifts
readings that the ingest has already shifted. #807's argument stands on evidence from this
repository rather than on principle, and three of the six defects it lists are reproducible in the
code as it stands.

The user asked for both in one pull request. **Bundling a behaviour change into a rename is the
failure mode `plan-issue` step 7 names** — "is a refactor hiding a behaviour change inside it?" —
so the plan keeps them separable inside the one branch: #815 lands as the first commit, complete
with its test, before any file moves. A reviewer can read that commit alone.

### Departures from the issue body

- **The shared, tested fitting helper is deferred to its own issue.** The issue's closing section
  makes the strongest case in the whole issue for promoting the XGBoost fit loop, and the "What to
  build" list above it does not include the fit loop at all. Promoting it means rewriting
  `run_experiment.py`, `run_physics_experiment.py`, `run_hybrid_experiment.py`, `ens_horizons.py`
  and `shared_geometry.py`, and it changes the numbers those scripts produce, so it wants to land
  beside a re-run of the study rather than inside a rename. This plan proposes filing it as a new
  issue that blocks #809 and #810. **This is the biggest single call in the plan and is listed as
  an open question below.**
- **`stamp_alignment.py` does not double-correct.** #815 lists it alongside the other two, but the
  script applies no shift: `ALIGNMENT_FIXED_AT` there only labels each row `before` or `after`, and
  the only `offset_by` in the file is the deliberate lag sweep. What breaks is the script's stated
  expectation — its docstring says the `before` era will measure half an hour late, and against the
  repaired table both eras should now measure as aligned. That is a prose fix and a change of
  meaning, not an arithmetic one.
- **CLAUDE.md does not mention `scripts/experiments/`.** The issue's ship-time triage says the
  rename touches it; `grep` finds no reference. CLAUDE.md is still edited, but to add `studies` to
  the packages table rather than to fix a path.
- **Tidying `data/` is half a code change and half a manual step.** The code half — one
  `STUDIES_DATA_DIR` that every study path derives from, so ICON-D2, CAMS and ERA5 downloads land
  under `data/studies/` instead of beside `data/NWP/` — is in scope. Moving the existing files on
  the workstation is a `mv` the PR cannot perform, and #815 forces a rebuild of every built dataset
  anyway, so most of what would be moved is about to be regenerated. The PR body names the manual
  step.

### Size: complex

One line per trigger, as the sizing rule requires:

- **What gets stored** — no Patito model, Delta table or Dagster asset changes. #815 does change the
  content and the filenames of every built study dataset under `data/`, and the `data/studies/` move
  changes where they live, so what is stored on disk for the studies changes even though no contract
  does.
- **The production serving path** — no behaviour on it changes; nothing under `studies/` is imported
  by `src/` or `packages/` production code. The *install* surface does change: a new workspace
  member, a new `[tool.uv.sources]` entry, a new root `dev`-group entry, and a regenerated
  `uv.lock`. The Dockerfile is not edited, and `uv sync --frozen --no-dev` is what has to keep the
  image clean.
- **A degradation rule** — none. `docs/design-philosophy/inherent-stability.md` is untouched, no
  asset check is added or edited, and study code is R&D, where fail-fast is the rule.
- **More than one defensible design** — yes, several: where the package lives and what it is called
  when a directory already carries the name, whether the Fractions Skill Score belongs in `studies`
  or in `ml_core` given #805, which dependency group keeps the production install clean, whether the
  fit loop is promoted now, and whether the `--alignment` axis is deleted or kept.
- **Code whose callers you could not name without searching** — yes. Thirty-five scripts import each
  other by sibling module name, and finding every `--alignment` site, every `ALIGNMENT_FIXED_AT` and
  every consumer of the site-label mapping took a repository-wide `grep` each time.

Two triggers fire outright and a third fires partially, so the issue is **complex**: the plan gets
both plan reviews, and the diff gets both diff reviews in `implement-issue`.

## What changes, file by file

### Commit 1 — stop the double correction (#815)

- **`scripts/experiments/beam_diffuse_split/build_dataset.py`** — delete `ALIGNMENT_FIXED_AT`, the
  `AlignmentType` literal, the `--alignment` argument and the `piecewise` branch of `_hourly_power`.
  Keep the one arithmetic that is now correct: an hour ending at `T` is the mean of the half-hours
  stamped `T - 30 min` and `T`, which is `time.dt.offset_by("30m").dt.truncate("1h")` — what the
  deleted `as-labelled` branch did. Drop the `_{alignment}` segment from `output_path_for`.
- **`site_e_commissioning.py`** — delete `ALIGNMENT_FIXED_AT` and the `when/then` shift at lines
  146–149. Replace the hard-coded `SITE_IDS` map with the promoted label helper (commit 3).
- **`stamp_alignment.py`** — keep the era split; import the instant from
  `contracts.power_schemas.POWER_TIMESTAMPS_CORRECTED_BEFORE` rather than redeclaring it; rewrite the
  module docstring so the expected result is that **both** eras now measure as aligned, which is the
  evidence that the ingest repair worked.
- **`run_experiment.py`, `run_physics_experiment.py`, `run_hybrid_experiment.py`,
  `compare_sources.py`, `elevation_breakdown.py`, `sky_conditions.py`, `inverter_clipping.py`,
  `era_comparison.py`, `report_results.py`, `fractions_skill_score.py`, `shared_geometry.py`,
  `oracle_capacity.py`, `capacity_denominator.py`, `make_chart.py`** — delete the `--alignment`
  argparse argument and the `_{alignment}` segment from every path and results-directory name.
- **`docs/studies/beam-diffuse-split.md`** — rewrite the passage describing the alignment axis to say
  the ingest owns the repair, and the `stamp_alignment.py` mentions at lines 890 and 1040. The
  published numbers stand: they were computed under `piecewise`, which is arithmetically what the
  ingest now applies. Keep the `#the-power-timestamps-before-26-march-2026-are-half-an-hour-late`
  anchor intact — `packages/contracts/src/contracts/power_schemas.py:43` links to it.
- **`beam_diffuse_split/README.md`** — the same, for its own alignment section.

### Commit 2 — move `scripts/experiments/` to `studies/`

- `git mv scripts/experiments/beam_diffuse_split studies/beam_diffuse_split` and
  `git mv scripts/experiments/README.md studies/README.md`.
- Rewrite `studies/README.md`: the tier's promises table changes, because the machinery now *is*
  tested and the studies still are not. Rename "experiment" to "study" throughout.
- Rewrite the run command in all 35 module docstrings (commit 4, once the scripts import the
  package).

### Commit 3 — `packages/studies/`

New workspace member, `src/studies/`, one module per promoted piece. Each is the single
implementation the studies call, replacing the copies named beside it.

| Module | What moves into it | What it replaces |
|---|---|---|
| `paths.py` | `REPO_DATA_DIR`, `_find_project_root`, `_main_checkout`, and a new `STUDIES_DATA_DIR = REPO_DATA_DIR / "studies"` | `sources.py`'s copy |
| `anonymise.py` | `site_labels_for(roster)` — the seeded shuffle from `time_series_id` to `A`–`F` | three copies: `build_dataset._pv_sites`, `site_e_commissioning.SITE_IDS` (hard-coded), `anm_curtailment._site_labels` |
| `solar.py` | `zenith`, `cos_zenith_hour_mean`, the extraterrestrial-flux and clearness-index columns, `solar_geometry` | `fetch_open_meteo_point._solar_geometry` and friends, `build_dataset._add_solar_geometry` |
| `open_meteo.py` | `fetch_point_frame`, the retrying `_get_json`, the date-range helpers, `OpenMeteoModel` and `OPEN_METEO_MODELS`, `HISTORICAL_FORECAST_URL` | `fetch_open_meteo_point.py`'s fetcher half, the registry half of `sources.py` |
| `served_column_checks.py` | `check_hourly_value_is_a_backward_mean`, `check_direct_is_not_a_separation_model` | `fetch_open_meteo_point.py`'s two `_check_*` functions |
| `gridded.py` | `nearest_cell(sites, grid)` — projected-grid sampling at a set of coordinates | `build_dataset._nearest_era5_cell` |
| `power.py` | `hourly_from_half_hourly(frame)` — the pure stamp-to-hour aggregation, with the Delta read left behind in the study | the arithmetic half of `build_dataset._hourly_power` |
| `bootstrap.py` | `paired_monthly_block_interval(...)` — resample whole months, the same months for both arms, percentile interval | `run_experiment._bootstrap_difference`, `fractions_skill_score._bootstrap_fss_difference`, `ens_horizons._paired_month_bootstrap` |
| `fractions_skill_score.py` | `on_a_complete_hourly_grid`, `monthly_components`, `fss_from`, and **a raise when an arm filter selects no rows** | `fractions_skill_score.py`'s scoring half |

**The H3-cell mapping goes into `packages/geo`, not `packages/studies`.** `fetch_ens_point._cell_for_each_meter`
maps latitude and longitude to an H3 cell, which is `geo`'s stated purpose and where
`packages/geo/src/geo/h3.py` already lives. A new `cells_for_coordinates()` there serves the study
and anything else that needs it; a copy in `studies` would be the fault this issue exists to remove.

**Why `bootstrap.py` is one function and not three.** `run_experiment`, `fractions_skill_score` and
`ens_horizons` each resample whole months, each pair the two arms on the same drawn months, and each
take the same two percentiles. What differs is only the statistic re-formed from the resampled
sums, which the caller passes in.

### Commit 4 — dependency plumbing and the run commands

- `packages/studies/pyproject.toml` — depends on `contracts`, `geo`, `numpy`, `polars`, `patito`,
  `pvlib`. Deliberately **not** `cdsapi`, `xarray`, `netcdf4` or `xgboost`: the Copernicus
  downloaders and the GRIB reading stay in the study scripts, run with `uv run --with cdsapi`, and
  the fit loop is deferred.
- Root `pyproject.toml` — add `studies = { workspace = true }` under `[tool.uv.sources]` and
  `"studies"` to `[dependency-groups] dev`, with the same style of comment `dashboard` carries. It
  goes in `dev`, not in `[project] dependencies`, because `uv sync --frozen --no-dev` in the
  Dockerfile is what keeps `pvlib` out of the production image, and because `uv sync` with no flags
  has to install it or `uv run pytest` cannot collect the new tests.
- `uv.lock` — regenerated.
- All 35 module docstrings — the run command changes from `uv run --no-project --with polars …` to
  `uv run python studies/beam_diffuse_split/<script>.py`, plus `--with cdsapi` on the two Copernicus
  downloaders. A script importing a workspace package can no longer run under `--no-project`.
- Delete the `# ty: ignore[unresolved-import]` on every `import pvlib`, now that `pvlib` resolves.

### Commit 5 — docs

- **`packages/studies/README.md`** — new.
- **`docs/documentation-guide.md`** — line 19's tier row and line 110's "which place do I use?" row:
  the path changes and "no tests" becomes "the studies have no tests; the machinery they call does".
- **`docs/studies/index.md`** — the `scripts/experiments/` link.
- **`CLAUDE.md`** — add `studies` to the packages table.
- **`docs/architecture/testing.md`** — where the new tests live, and the dependency-group note.
- **`studies/beam_diffuse_split/README.md`** — delete the Fractions Skill Score validation table and
  point at the test file that now holds it. Deleting it is the point: a validation table in a README
  cannot fail.

## Design-philosophy check

**This is R&D, so fail-fast is correct and no degradation path is involved.** Nothing in
`studies/` or `packages/studies/` runs in production, enters the Dagster asset graph, or is reached
by `src/nged_substation_forecast/defs/`. No asset check is added or edited, so the
`WARN`/`blocking=False` rule does not apply. The one deliberate new raise — `fractions_skill_score`
rejecting an arm filter that selects no rows — is the R&D half of
[inherent stability](https://openclimatefix.github.io/nged-substation-forecast/design-philosophy/inherent-stability/)
applied correctly: a quietly-empty comparison poisons every number built on it, exactly as a quietly
degraded training run does.

**Principle 4, "an experiment must be cheap to try, and cheap to abandon", is the one being
traded.** A tested package raises the cost of the next study. What is bought is that the cost is
paid only on the machinery: the studies themselves stay untested scripts with no backwards
compatibility and no maintenance promise, which is the split the issue argues for and which
`studies/README.md` will state. Principle 2, "complexity belongs offline, not in the serving path",
is served rather than traded — the `--no-dev` boundary is what enforces it, and the plan adds a
verification command that proves it.

No hypothesis in `docs/design-philosophy/engineering-hypotheses.md` is claimed by this change.

## Tests

Each entry names the assertion that fails on `main` today.

1. **`test_hourly_power_places_a_reading_in_the_hour_ending_at_its_label`** — half-hours stamped
   09:30 and 10:00 on a date before 2026-03-26 form the hour ending 10:00. **Fails on `main`**:
   `build_dataset._hourly_power`'s default `piecewise` branch shifts both stamps 30 minutes earlier
   and puts them in the hour ending 09:30. This is the #815 test.
2. **`test_fractions_skill_score_matches_the_forecasts_whose_answer_is_known`** — the four cases the
   README table holds, as a parametrised test: identical to the observation scores 1.000 at every
   tolerance; one hour late scores 0.667 / 0.842 / 0.919 / 0.959; three hours late scores 0.000 /
   0.211 / 0.486 / 0.741; never predicting the event scores 0.000 at every tolerance. **Fails on
   `main`**: no test exists, and the last row is the control that catches a widened window inflating
   the score rather than crediting timing.
3. **`test_an_arm_filter_matching_no_rows_raises`** — selecting an arm name that no row carries
   raises rather than returning an empty frame. **Fails on `main`**: `fractions_skill_score.py`
   filtered on the gradient-boosted arm names, was run against the physical instrument, matched
   nothing and printed an empty table with no error.
4. **`test_a_window_spanning_a_gap_is_dropped`** — a site whose rows stop for a 12-hour night
   contributes no window across the gap, so the window count equals the complete windows only.
   **Fails on `main`**: no test, and a rolling window over the scored rows alone would join the
   hours either side of the night.
5. **`test_site_labels_are_derived_once`** — the derived mapping equals the mapping
   `site_e_commissioning.py` hard-codes, and adding a series that does not clear
   `MIN_YEARS_OF_READINGS` leaves every existing label unchanged. **Fails on `main`**: the two
   mappings are independent, and nothing checks that they agree.
6. **`test_backward_mean_check_rejects_a_half_hour_offset`** — a frame whose hourly column holds the
   instantaneous value makes `check_hourly_value_is_a_backward_mean` raise; a correctly-converted
   frame passes. **Fails on `main`**: no test, and this check is the only absolute guard on which
   hour a served label names.
7. **`test_direct_fraction_check_rejects_a_separation_model`** — a frame whose direct fraction is a
   pure function of clearness and zenith makes `check_direct_is_not_a_separation_model` raise; a
   frame with real within-bin spread passes. **Fails on `main`**: no test, and this check is what
   stands between arm C and being a copy of arm B.
8. **`test_paired_bootstrap_gives_a_zero_interval_for_an_arm_against_itself`**, and a companion
   asserting a non-zero interval for two genuinely different arms. **Fails on `main`**: no test, and
   the issue records a negative control that was arithmetically incapable of failing while its
   docstring asserted the opposite.
9. **`test_nearest_grid_cell_uses_the_descending_latitude_order`** — a coordinate between two ERA5
   rows picks the nearer one, with the archive's descending-latitude storage order. **Fails on
   `main`**: no test, and a reversed order is silently plausible.

## Docs to update

Listed by file in "Commit 5" above. This PR completes no roadmap item, so there is no ship-time
triage: no "Implementation details" section is deleted and no status banner moves. `plans/promote-study-machinery.md`
is deleted into the PR body at merge, per the standing rule.

## Verification commands

The green-before-push set from `implement-issue`, plus:

```bash
uv sync                                   # must install `studies` with no flags
uv lock --check
uv run ruff check . && uv run ruff format --check .
uv run ty check
uv run --isolated --no-project --with pydoclint==0.9.1 pydoclint --quiet --style=google .
uv run pytest
uv run pymarkdown scan -r docs README.md CLAUDE.md packages/*/README.md studies/README.md
uv run mkdocs build --strict
uv run python scripts/lint/check_docs_links.py
uv run pre-commit run --all-files
# The production install must not carry a research dependency:
uv export --no-dev --format requirements-txt | grep -ciE '^(pvlib|cdsapi)' | grep -qx 0
```

The last command is the one this change specifically needs, and it is the check that the `dev`-group
placement actually does what it is chosen for. It belongs in the PR body's evidence, and is worth
proposing as a CI step in a follow-up rather than adding to `ci.yml` inside this diff.

**Re-running any study is not part of the verification set.** Every built dataset under `data/` is
invalidated by #815 and by the `data/studies/` move, and a rebuild needs the workstation's data and
hours of compute. What the PR can assert is that the code is right; reproducing the published
figures is the user's step, on the workstation, and the PR body says so.

## Risks and open questions

1. **Should the shared fitting helper land here or in its own issue?** *Recommendation: its own
   issue, blocking #809 and #810.* It changes the numbers four scripts produce, so it wants to land
   beside a re-run rather than inside a rename, and this PR is already large. Against that: the next
   two studies are #809 and #810, and the issue's argument is that neither should be written on a
   fifth copy of the fit loop. Deferring it is only safe if the new issue lands before they start.
2. **Two things named `studies`.** The package `packages/studies/` (imported as `studies`) and the
   directory `studies/` holding the study scripts. Nothing shadows anything — `studies/` carries no
   `__init__.py` and a script run from `studies/beam_diffuse_split/` puts only its own directory on
   `sys.path` — but a reader meets the name twice meaning two things. *Recommendation: keep it*, as
   the issue proposes; the alternative is `packages/study_kit/` for the machinery, which reads
   clearly and costs nothing but a less obvious import name.
3. **Where the Fractions Skill Score lives, given #805.** #805 wants it on the production
   leaderboard beside MAE and the tail metrics, which would put it in `ml_core`.
   *Recommendation: `packages/studies/` now.* Moving it when #805 is planned is a rename in a young
   project with no external users, and placing it in `ml_core` today would be generalising for a
   caller that does not exist.
4. **Deleting the `--alignment` axis outright.** The three settings existed to test a convention the
   contract now settles, and two of the three are now arithmetically wrong.
   *Recommendation: delete all three*, which removes an argument from fourteen scripts and a segment
   from every output filename. The cost is that re-deriving the lateness needs the raw feed rather
   than a flag; `stamp_alignment.py` against the repaired table is the measurement that replaces it.
5. **`data/studies/` needs a manual `mv` on the workstation, and every study dataset rebuilt.** The
   rebuild is forced by #815 regardless. *Recommendation: proceed*, and name the `mv` in the PR body
   rather than trying to automate a move the PR cannot see.
6. **Every `uv sync` grows by `pvlib` and its dependencies.** That is the price of testing the solar
   geometry, and it is paid by every developer and by CI, not by the production image.
   *Recommendation: accept it.* `cdsapi`, `xarray` and `netcdf4` stay out, which is where the bulk
   would have been.
7. **Out of scope, reported rather than fixed:** `contracts.settings._find_project_root` resolves
   `data/` to a linked worktree's own root rather than the main checkout, which is why
   `sources.py` carries the `_main_checkout` walk at all. Fixing it in `contracts` would let
   `packages/studies/paths.py` drop the walk and would fix the same trap for every other caller.
   That is a change to a production package and belongs in its own issue.
