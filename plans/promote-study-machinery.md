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
correct. Then move `scripts/experiments/` to a top-level `studies/`, promote five pieces of
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
so the plan keeps them separable inside the one branch: #815 lands in commit 2, complete with its
test, before any file moves. It cannot be commit 1, because the test that pins it needs `pvlib`
installed to import the module under test, and commit 1 is what installs it. A reviewer can still
read the #815 commit alone, which is the point.

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

### Commit 1 — the dependency line, before anything that needs it

`pvlib` is the only dependency any planned test needs that `uv.lock` lacks, and
`build_dataset.py:42-43` imports `pvlib` and `xarray` at module scope, so **any** test that reaches
the hourly-power arithmetic needs `pvlib` installed before it can even be collected. The dependency
line therefore comes first, on its own: add `pvlib` to the root `dev` dependency group and
regenerate `uv.lock`. Under the package layout this commit also carries `packages/studies/pyproject.toml`
and the `[tool.uv.sources]` entry, because `[tool.uv.workspace] members = ["packages/*"]` makes a
new directory a member on sight and `uv lock --check` fails from the moment it appears until the
lock catches up.

### Commit 2 — stop the double correction (#815)

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
- **Nineteen scripts carry a `--alignment` argparse argument**, not the fourteen an earlier draft of
  this plan counted: the five it missed are `anm_curtailment.py:161`, `anm_setpoints.py:128`,
  `restart_basins.py:118`, `multi_nwp.py:213` and `ens_horizons.py:324`. Two more hold an
  `ALIGNMENT` constant instead (`make_chart.py:42`, `make_figures.py:49`), and
  `site_e_commissioning.py:51` holds its own `ALIGNMENT_FIXED_AT`, so the sweep is 22 files.
  Eighteen of the nineteen only pass the string into a path, almost all through three helpers —
  `run_experiment.dataset_path_for` and `results_dir_for` (`run_experiment.py:69-76`) and
  `run_physics_experiment.results_dir_for` (`:68-70`). Drop the keyword argument from those three
  first, then delete a three-line argparse block and one keyword argument per caller.
- **`docs/studies/beam-diffuse-split.md`** — a smaller edit than it looks. The page never names
  `as-labelled`, `shifted` or `piecewise`; the passage at `:861-868` says the numbers are "computed
  on the corrected reading" without saying who corrects, and stays true either way. What needs
  rewriting is the `stamp_alignment.py` material at `:890` and `:1040`. The alignment axis is
  described in `beam_diffuse_split/README.md:37`, `:51` and `:316` instead. **Do not touch the
  heading at `:861`** — three places link to its anchor (`power_schemas.py:43`,
  `docs/roadmap/data-cleaning.md:47`, `docs/roadmap/capacity-estimation.md:638`). The
  published numbers stand: they were computed under `piecewise`, which is arithmetically what the
  ingest now applies. Keep the `#the-power-timestamps-before-26-march-2026-are-half-an-hour-late`
  anchor intact — `packages/contracts/src/contracts/power_schemas.py:43` links to it.
- **`beam_diffuse_split/README.md`** — the same, for its own alignment section.

### Commit 3 — move `scripts/experiments/` to `studies/`

- `git mv scripts/experiments/beam_diffuse_split studies/beam_diffuse_split` and
  `git mv scripts/experiments/README.md studies/README.md`.
- Rewrite `studies/README.md`: the tier's promises table changes, because the machinery now *is*
  tested and the studies still are not. Rename "experiment" to "study" throughout.
- Rewrite the run command in all 35 module docstrings (commit 4, once the scripts import the
  package).

### Commit 4 — `packages/studies/`

New workspace member, `src/studies/`, one module per promoted piece. Each is the single
implementation the studies call, replacing the copies named beside it.

| Module | What moves into it | What it replaces |
|---|---|---|
| `anonymise.py` | `site_labels_for(eligible_ids)` — the seeded shuffle from `time_series_id` to `A`–`F` | three copies: `build_dataset._pv_sites`, `site_e_commissioning.SITE_IDS` (hard-coded), `anm_curtailment._site_labels` |
| `solar.py` | `zenith`, `cos_zenith_hour_mean`, the extraterrestrial-flux and clearness-index columns, `solar_geometry` | two copies of the same `pvlib` calls: `fetch_open_meteo_point._solar_geometry` and `build_dataset._add_solar_geometry` — keep the union of their columns, because the second adds the azimuth and elevation the built datasets carry |
| `served_column_checks.py` | `check_hourly_value_is_a_backward_mean`, `check_direct_is_not_a_separation_model` | `fetch_open_meteo_point.py`'s two `_check_*` functions |
| `power.py` | `hourly_from_half_hourly(frame)` — the pure stamp-to-hour aggregation, with the Delta read left behind in the study | the arithmetic half of `build_dataset._hourly_power` |
| `fractions_skill_score.py` | `on_a_complete_hourly_grid`, `monthly_components`, `fss_from` | `fractions_skill_score.py`'s scoring half |

**Five modules, not the nine an earlier draft of this plan carried.** The four that were cut, each
for its own reason:

- **`open_meteo.py`** (the fetcher and the model registry) — no test in this plan touches any of it,
  because exercising it needs the network, which is what the existing `--run-network` gate is for.
  Moving untested code into a package whose whole purpose is tested machinery inverts the issue.
- **`bootstrap.py`** — the three bootstraps are three different statistics, not three copies.
  `run_experiment._bootstrap_difference` resamples per-row differences within drawn months **and
  draws one of the seeds per resample**, which its docstring calls load-bearing;
  `ens_horizons._paired_month_bootstrap` takes per-month means with row-count weights, and says in
  its own docstring why it does not reuse `run_experiment`'s; `fractions_skill_score` re-forms a
  ratio from month sums. A shared function with a statistic callback expresses none of the seed axis
  or the weights, so unifying them would either change the intervals on the published page or grow a
  config surface to preserve three behaviours. The unification belongs with the deferred fit loop.
- **`gridded.py`** — cut, but the reason has to name the right function. #807's "projected-grid
  sampling at a set of coordinates" is `verify_ukv_lineage._sample_grid`
  (`verify_ukv_lineage.py:228-252`: a `pyproj` Lambert azimuthal equal-area transform, then an
  `xarray` `sel(..., method="nearest")`), not `build_dataset._nearest_era5_cell`. Both are cut,
  for different reasons. `_nearest_era5_cell` calls `np.sort` on its grid coordinates before the
  `argmin` (`build_dataset.py:571-572`), so the storage order it would have been tested against
  cannot reach the comparison: six lines of numpy, one caller, guarding a state that cannot occur.
  `_sample_grid` is genuinely testable — it needs no network, and `pyproj` and `xarray` are both
  already locked — and is deferred only because nothing else calls it yet. **Deferring it is a
  fifth departure from the issue body**, and the follow-up package issue should carry it.
- **`paths.py`** — `sources.py` already holds `_find_project_root`, `_main_checkout` and
  `REPO_DATA_DIR`, and is deliberately stdlib-only so the lean scripts can import it. The `data/`
  tidy the issue asks for is one line beside them: `STUDIES_DATA_DIR = REPO_DATA_DIR / "studies"`,
  with the study's path constants repointed at it. Moving the walk into the package would only
  duplicate the `contracts.settings` bug recorded in open question 7.

**The H3-cell mapping needs no new function anywhere.** `packages/nged_data/src/nged_data/read_nged_json.py:52`
already maps coordinates to cells with `polars_h3.latlng_to_cell`, and
`contracts.weather_schemas.ECMWF_ENS_H3_RESOLUTION` already names the resolution that
`fetch_ens_point.py:48` re-declares. `fetch_ens_point.py` runs in the project environment
already — its run command carries no `--no-project` — so it can import both today. Replace its
three-line list comprehension with the existing expression. An earlier draft of this plan proposed
adding `cells_for_coordinates()` to `packages/geo`, which would have been a second implementation of
a helper the repository already has: the exact fault the issue exists to remove.

### Commit 5 — the run commands

- `packages/studies/pyproject.toml` — depends on `contracts`, `numpy`, `polars`, `pvlib`.
  Deliberately **not** `cdsapi`, `xarray`, `netcdf4` or `xgboost`: the Copernicus downloaders and
  the GRIB reading stay in the study scripts, run with `uv run --with cdsapi`, and the fit loop is
  deferred. `pvlib` is the only one of these absent from `uv.lock` today, so it is the whole of the
  dependency change.
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

### Commit 6 — docs

- **`packages/studies/README.md`** — new.
- **`docs/documentation-guide.md`** — line 19's tier row and line 110's "which place do I use?" row:
  the path changes and "no tests" becomes "the studies have no tests; the machinery they call does".
- **`docs/studies/index.md`** — the `scripts/experiments/` link.
- **`docs/roadmap/data-sources.md:485`** — a fourth inbound reference to the old path, which an
  earlier draft of this plan missed. `grep -rn "scripts/experiments" docs/` finds all four; run it
  again at implementation time rather than trusting this list.
- **`CLAUDE.md`** — add `studies` to the packages table.
- **`docs/architecture/testing.md`** — where the new tests live, and the dependency-group note.
- **`studies/beam_diffuse_split/README.md`** — delete the Fractions Skill Score validation table and
  point at the test file that now holds it. Deleting it is the point: a validation table in a README
  cannot fail.

### What `data/studies/` must not swallow

The move is for what the studies download and build, not for what the pipeline owns. **These stay
where they are**: `POWER_DELTA_URI`, `METADATA_PATH` and `CAPACITY_DELTA_URI`
(`build_dataset.py:59-61`), and `NWP_ROOT` (`fetch_ens_point.py:42`) — all four are the production
tables the studies read. `data/NGED/anm/` (`export_cap.py:53`) is the genuinely ambiguous one: a
study writes it (`anm_setpoints.py`), a study reads it (`site_e_commissioning.py:118,130`), and the
docs page names it (`docs/studies/beam-diffuse-split.md:1036`). Decide it explicitly rather than by
whichever `sed` runs first. Note also that `make_chart.py:168` globs `REPO_DATA_DIR / "ERA5"`
directly, so a path constant alone will not catch it, and that three more docs lines name moved
paths: `beam_diffuse_split/README.md:44` and `:47`, and the docs page at `:1036`.

### Two behaviour changes hiding inside the moves

**The solar union changes the built dataset's schema and its cost, not its numbers.**
`fetch_open_meteo_point._solar_geometry` adds `cos_zenith_instant`, `cos_zenith_hour_mean` and
`clearness_index` that `build_dataset._add_solar_geometry` does not, and `cos_zenith_hour_mean` is
60 `pvlib` solar-position calls per stamp set (`fetch_open_meteo_point.py:282-310`). Every arm
selects its features by name (`run_experiment.py:85-109`), so no published number moves — but every
written parquet gains three columns and every build gets slower. Make the extra columns opt-in, or
say plainly that the cost is accepted.

**Two docstrings go stale the moment the run commands change.** `sources.py:135-139` states that
"every script here runs under `uv run --no-project` and so cannot import a workspace package", and
`scripts/experiments/README.md` says the same. That constraint is the stated reason for keeping
`sources.py` stdlib-only, and commit 5 removes it for the scripts that import the package. Rewrite
both rather than leaving a rule in place that the same commit falsifies.

## Design-philosophy check

**This is R&D, so fail-fast is correct and no degradation path is involved.** Nothing in
`studies/` or `packages/studies/` runs in production, enters the Dagster asset graph, or is reached
by `src/nged_substation_forecast/defs/`. No asset check is added or edited, so the
`WARN`/`blocking=False` rule does not apply, and the change adds no new raise. The one place the
R&D half of
[inherent stability](https://openclimatefix.github.io/nged-substation-forecast/design-philosophy/inherent-stability/)
is visible is a guard that already exists: `fractions_skill_score.py:263-270` raises when no
reported arm is present in the run, which is the fix for the silent empty table the issue narrates.
The plan pins that guard with a regression test rather than adding behaviour.

**Principle 4, "an experiment must be cheap to try, and cheap to abandon", is the one being
traded.** A tested package raises the cost of the next study. What is bought is that the cost is
paid only on the machinery: the studies themselves stay untested scripts with no backwards
compatibility and no maintenance promise, which is the split the issue argues for and which
`studies/README.md` will state. Principle 2, "complexity belongs offline, not in the serving path",
is served rather than traded — the `--no-dev` boundary is what enforces it, and the plan adds a
verification command that proves it.

No hypothesis in `docs/design-philosophy/engineering-hypotheses.md` is claimed by this change.

## Tests

Seven tests, not the nine an earlier draft carried.

**Only test 1 fails against behaviour this change alters.** An earlier draft claimed "fails on
`main`" for six of the seven, which was misleading in exactly the way the plan singled test 3 out
for: tests 2, 4, 5, 6 and 7 pin behaviour that already exists and works, and on `main` they would
fail only with `ImportError`, because the module they import does not exist yet. That is not a test
of this change. They are worth writing anyway — a move that silently altered any of them would
change numbers on a published page, and nothing currently stops it — but the honest description is
**"pins existing behaviour so the move cannot change it"**, and that is how each is labelled below.

1. **`test_hourly_power_places_a_reading_in_the_hour_ending_at_its_label`** — the #815 test, and
   the only one that fails against arithmetic rather than against an absent import. **The fixture
   needs four contiguous stamps, not two.** Measured in this worktree on stamps 09:00, 09:30, 10:00
   and 10:30, all before 2026-03-26: the surviving expression yields one complete hour, ending
   10:00, pooling the stamps 09:30 and 10:00 — the window `(09:00, 10:00]`, which is right. The
   `piecewise` branch yields two complete hours, and its hour ending 10:00 pools the stamps 10:00
   and 10:30 — the window `(09:30, 10:30]`, half an hour of the wrong weather. A two-stamp fixture
   would not show this: under `piecewise` the two stamps land in different hours and the
   `n_half_hours == 2` filter (`build_dataset.py:486`) drops both, so the frame comes back empty and
   the test passes for the wrong reason. **The promoted function must keep that filter**, because it
   is also what drops the single orphan half-hour at the correction boundary.
2. **`test_fractions_skill_score_matches_the_forecasts_whose_answer_is_known`** — the four cases the
   README table holds, as a parametrised test: identical to the observation scores 1.000 at every
   tolerance; one hour late scores 0.667 / 0.842 / 0.919 / 0.959; three hours late scores 0.000 /
   0.211 / 0.486 / 0.741; never predicting the event scores 0.000 at every tolerance. The last row is
   the control: widening the window must not rescue a forecast that never predicts the event.
   **Pins existing behaviour** — the arithmetic at `fractions_skill_score.py:102-183` is unchanged
   by this plan, and the test exists so the move cannot change it and so a validation currently
   living in a README can fail. Both fixtures need a `month` column, which `_monthly_components`
   drops nulls on (`:162`) and which `run_experiment.py:256` supplies upstream; the promoted
   function should derive it from `time` rather than make every caller carry it.
3. **`test_an_arm_filter_matching_no_rows_raises`** — a regression test on the guard at
   `fractions_skill_score.py:263-270`. **This one does not fail on `main`**, and is kept for that
   reason rather than in spite of it: the guard is what stands between a mistyped arm name and a
   silently empty comparison, and nothing currently holds it in place.
4. **`test_a_window_spanning_a_gap_is_dropped`** — a site whose rows stop for a 12-hour night
   contributes no window across the gap, so the window count equals the complete windows only.
   **Pins existing behaviour**: the complete-grid reindex and `min_samples=window_hours` already
   do this (`fractions_skill_score.py:102-137`, `:157`, `:160`). The test exists because a move
   that dropped either would be invisible in the numbers until a night fell inside a window.
5. **`test_site_labels_are_derived_once`** — `site_labels_for` maps a fixed roster of eligible
   `time_series_id`s to a fixed `A`–`F` mapping, and a roster of the wrong size raises. **Partly
   pins existing behaviour**: the wrong-size raise is already at `build_dataset.py:433-435`; what
   is new is that one implementation exists at all, where the mapping is derived three separate
   ways today and nothing checks that they agree. **The helper unifies the shuffle, not the
   roster** — `_pv_sites` (`build_dataset.py:404-438`) inner-joins the effective-capacity table
   before the row-count filter and `anm_curtailment._site_labels` (`:115-133`) does not, so the two
   rosters can differ; and `site_e_commissioning.py` imports nothing from `build_dataset`, so
   replacing its hard-coded `SITE_IDS` means importing the roster query rather than growing a third
   copy. The generator stays `np.random.default_rng(LABEL_PERMUTATION_SEED)`: any other random source
   relabels every site and orphans the `A`–`F` labels in the published write-up.
6. **`test_backward_mean_check_rejects_a_half_hour_offset`** — a frame whose hourly column holds the
   instantaneous value makes `check_hourly_value_is_a_backward_mean` raise; a correctly-converted
   frame passes. **Pins existing behaviour** (`fetch_open_meteo_point.py:335-384`). The fixture
   needs care: the ratio of the instantaneous to the hour-mean cosine is near 1.0 at midday, so
   hand-set geometry can land either side of the 5 W m⁻² threshold by accident. Build the geometry
   with `pvlib` at a generic, non-meter coordinate over a few days, so the failing case misses by
   tens of W m⁻² rather than by noise.
7. **`test_direct_fraction_check_rejects_a_separation_model`** — a frame whose direct fraction is a
   pure function of clearness and zenith makes `check_direct_is_not_a_separation_model` raise; a
   frame with real within-bin spread passes. **Pins existing behaviour**
   (`fetch_open_meteo_point.py:410-463`), and this check is what stands between arm C and being a
   copy of arm B. **Match on the message, not just the type**: the function raises `ValueError`
   twice, and the "no bin holds 30 rows" guard (`:444-449`) fires before the spread test
   (`:456-463`), so a thin fixture satisfies `pytest.raises(ValueError)` for the wrong reason.

**Two tests an earlier draft carried are cut.** A paired-bootstrap test is dropped with
`bootstrap.py`, because the three bootstraps stay where they are; the property it would have
asserted is worth writing against `_bootstrap_fss_difference` as it stands, and that is folded into
test 2's file. A nearest-grid-cell test is dropped with `gridded.py`, because
`build_dataset.py:571-572` sorts its own coordinates and the reversed order it would have guarded
against cannot occur.

**Test 5 is scoped to a pure function on purpose.** Asserting that the derived mapping equals the
one `site_e_commissioning.py` hard-codes would need `metadata.parquet`, the power table's row
counts and the effective-capacity table — private workstation data that CI does not have. What CI
can check is that one implementation exists and is deterministic; the three-way agreement is bought
by deleting two of the three copies, not by a test.

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
test "$(uv export --no-dev --format requirements-txt | grep -ciE '^(pvlib|cdsapi)')" -eq 0
```

The last command is the one this change specifically needs, and it is the check that the `dev`-group
placement actually does what it is chosen for. **Write it with `test "$(...)"`, not as a pipeline
into `grep -qx 0`.** Measured in this worktree: `grep -c` exits 1 when it counts zero matches, so
under `set -o pipefail` — which is what GitHub Actions gives every `run:` step — the pipeline form
exits 1 exactly when the check *passes*. The `test "$(...)"` form carries the count through stdout,
where the pipe's exit status cannot reach it. Do not reach for `! … | grep -q` either: `grep -q`
exits early, which can `SIGPIPE` `uv export` and make the negated pipeline pass on a match.

The check belongs in the PR body's evidence, and is worth proposing as a CI step in a
follow-up rather than adding to `ci.yml` inside this diff.

**Re-running any study is not part of the verification set.** Every built dataset under `data/` is
invalidated by #815 and by the `data/studies/` move, and a rebuild needs the workstation's data and
hours of compute. What the PR can assert is that the code is right; reproducing the published
figures is the user's step, on the workstation, and the PR body says so.

## Risks and open questions

**Approved on 2026-09-22: the package layout, and every recommendation below.** Question 1 is
settled in favour of `packages/studies/` rather than tests in place; questions 2 to 8 are settled as
each one recommends. The questions are kept as written, with their evidence, because the reasoning
is what a later reader needs — not just the answer. Nothing below is still open.


1. **Nothing mechanically forces the new package. Should it be deferred until #809 or #810 gives it
   a second caller?** The simplicity review argued this and its evidence checks out, verified in
   this worktree rather than reasoned about: a bare `uv run pytest` already collects a test placed
   at `scripts/experiments/beam_diffuse_split/tests/`, because `norecursedirs` does not exclude
   `scripts/`; adding that directory to `[tool.pytest.ini_options] pythonpath` makes the sibling
   imports resolve under `--import-mode=importlib` (probed both ways — it fails with
   `ModuleNotFoundError` without the entry and passes with it); and `pvlib` is the only dependency
   any planned test needs that `uv.lock` lacks, which is one `dev`-group line with or without a
   package. So the alternative is real: one `pyproject.toml` line, a `tests/` directory beside the
   scripts, and no package, no `[tool.uv.sources]` entry, no lockfile churn beyond `pvlib`, and no
   rewrite of 35 run commands.
   *Recommendation: keep the package.* The issue asks for one, and its argument is about the tier
   boundary rather than about mechanics — a package is what makes "the machinery is held to a
   standard, the studies are not" a fact about the repository rather than a convention about which
   files happen to have tests. But this is the human reviewer's call, not the plan's, and choosing
   the deferral would cut roughly two-thirds of the diff. Choosing it would also dissolve question 2
   below entirely.
2. **Two things named `studies`.** The package `packages/studies/` (imported as `studies`) and the
   directory `studies/` holding the study scripts. Nothing shadows anything — `studies/` carries no
   `__init__.py`, and a script run from `studies/beam_diffuse_split/` puts only its own directory on
   `sys.path` — but a reader meets the name twice meaning two things. *Recommendation: keep it*, as
   the issue proposes. Two alternatives: `packages/study_kit/` for the machinery, or keeping the
   scripts at `scripts/studies/` so only the word changes and not the level.
3. **Should the shared fitting helper land here or in its own issue?** *Recommendation: its own
   issue, blocking #809 and #810.* It changes the numbers four scripts produce, so it wants to land
   beside a re-run rather than inside a rename, and this PR is already large. Against that: the next
   two studies are #809 and #810, and the issue's argument is that neither should be written on a
   fifth copy of the fit loop. Deferring it is only safe if the new issue lands before they start.
   The bootstrap unification cut from commit 3 goes into the same follow-up.
4. **Where the Fractions Skill Score lives, given #805.** #805 wants it on the production
   leaderboard beside MAE and the tail metrics, which would put it in `ml_core`.
   *Recommendation: `packages/studies/` now.* Moving it when #805 is planned is a rename in a young
   project with no external users, and placing it in `ml_core` today would be generalising for a
   caller that does not exist.
5. **Deleting the `--alignment` axis outright**, across 22 files. Two of the three settings are now
   arithmetically wrong, and the third is what the ingest applies. *Recommendation: delete all
   three.* The only smaller change is a dishonest one — fix the two arithmetic sites and leave
   `_piecewise` in every output filename naming a setting no code has. The cost is that re-deriving
   the lateness needs the raw feed rather than a flag; `stamp_alignment.py` against the repaired
   table is the measurement that replaces it.
6. **`data/studies/` needs a manual `mv` on the workstation, and every study dataset rebuilt.** The
   rebuild is forced by #815 regardless. *Recommendation: proceed*, and name the `mv` in the PR body
   rather than trying to automate a move the PR cannot see.
7. **Every `uv sync` grows by `pvlib` and its dependencies.** That is the price of testing the solar
   geometry, and it is paid by every developer and by CI, not by the production image.
   *Recommendation: accept it.* `cdsapi`, `xarray` and `netcdf4` stay out, which is where the bulk
   would have been.
8. **Out of scope, reported rather than fixed:** `contracts.settings._find_project_root` resolves
   `data/` to a linked worktree's own root rather than the main checkout, which is why
   `sources.py` carries the `_main_checkout` walk at all. Fixing it in `contracts` would fix the
   same trap for every other caller. That is a change to a production package and belongs in its own
   issue.

## What the first review changed, and what was rejected

The first adversarial review hunted for a simpler way, saw none of the reasoning behind the plan,
and produced ten findings. Every claim it made about the code was checked against the code before
being applied.

**Accepted, and already folded into the plan above:** the package drops from nine modules to five
(`open_meteo.py`, `bootstrap.py`, `gridded.py` and `paths.py` cut, each for a reason recorded in
commit 3); the tests drop from nine to seven; the false "fails on `main`" claim on the empty-arm
test is corrected, because `fractions_skill_score.py:263-270` already carries that guard; the
`--alignment` sweep is 22 files rather than 14; `docs/roadmap/data-sources.md:485` is a fourth
inbound reference the plan had missed; the H3 mapping reuses the existing `polars_h3` expression
instead of a new `geo` function; and test 5 is scoped to a pure function because the comparison it
first proposed needs private workstation data CI does not have.

**Rejected, with the reason:**

- *"Ship #815 as its own pull request rather than as the first commit."* The user asked for one pull
  request, was told why bundling a behaviour change into a rename is risky, and confirmed. The
  first-commit split already buys the reviewability the finding is after.
- *"Defer the package until a second caller exists."* Not rejected on the merits — the evidence is
  sound and it is recorded as open question 1 above, for the human reviewer to decide. The plan
  proceeds with the package because the issue asks for one, per the rule that a proposal larger or
  smaller than the issue is the human's call rather than the plan's.
- *"Move the studies to `scripts/studies/` rather than a top-level `studies/`."* A preference call
  with no correctness content either way; recorded as an alternative under open question 2 rather
  than applied, because the issue names the top-level layout.

## What the second review changed, and what was rejected

The second adversarial review checked the plan for correctness and testability, saw neither the plan's
reasoning nor what the first review changed, and produced thirteen findings. Every claim was checked
against the code before being applied; four were re-measured in this worktree rather than reasoned
about.

**Accepted, and folded in above:** the commit order was wrong, because `build_dataset.py:42-43`
imports `pvlib` at module scope, so the #815 test cannot exist before the dependency line — the
dependency commit now comes first, and the package's `pyproject.toml` moves into it so
`uv lock --check` is never red between commits. Test 1's stated failure mechanism was wrong, and its
fixture needs four contiguous stamps rather than two: measured here, a two-stamp fixture returns an
*empty* frame under `piecewise` and so passes for the wrong reason. Five of the seven tests were
labelled "fails on `main`" when they pin existing behaviour and would fail only with `ImportError`,
which is the same fault the plan had singled test 3 out for. The proposed verification command
inverts under `pipefail` — measured: `grep -c` exits 1 on a zero count, so the pipeline form exits 1
exactly when the check passes, and GitHub Actions runs every step under `bash -eo pipefail`. The
`gridded.py` cut argued about the wrong function: #807's "projected-grid sampling" is
`verify_ukv_lineage._sample_grid`, which is a fifth departure and now recorded as one. And three
scoping points are now stated rather than left to the implementer: which `data/` paths must not
move, that the solar union adds three columns and 60 `pvlib` calls per stamp set, and that two
docstrings asserting the `--no-project` constraint are falsified by the commit that removes it.

**Rejected, with the reason:**

- *Nothing.* Every finding was either a real defect in the plan or a scoping point worth stating.
  Two findings about commit ordering (the dependency line, and `uv lock --check` between commits 3
  and 4) would partly dissolve under the tests-in-place alternative in open question 1, but the
  underlying constraint survives it: `pvlib` has to be installed before any test can import the
  module it exercises, whichever layout is chosen.

**Verified correct, and left alone:** every claim the plan makes about current behaviour; the #815
arithmetic, including the boundary — the repaired table has no row at 08:00 on 2026-03-26
(`power_schemas.py:153-155`), so the orphan half-hour is dropped by the `n_half_hours == 2` filter,
and piecewise-on-raw is row-for-row identical to the surviving expression on the repaired table,
which is why the published numbers stand; the dependency plumbing; and the claim that a top-level
`studies/` directory without `__init__.py` does not shadow the installed package.
