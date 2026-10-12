# Plan: remove the scorer checks that only stop deliberate cheating (#1129)

**Problem.** [PR #1056](https://github.com/openclimatefix/nged-substation-forecast/pull/1056) added
code to the leaderboard scorer that exists only to stop an AI researcher cheating on purpose: a
final-test date guard, a single-model-name check, an environment-scrubbing re-execution of
`score_study.py`, an `NGED_ALLOW_REPLACE` gate, and an `import-linter` contract. Phase 1 of agentic
research assumes an honest agent, and phase 2 hides validation data with file permissions, so these
checks cost reading and maintenance effort and protect against nothing the design now relies on.

**Solution.** A targeted removal of exactly those pieces, with their tests and the docs text that
describes them. The checks that catch honest mistakes stay: the reference row-key check, the
fold-window row check, the `study/` prefix and its promotion exclusion, the submission log, and the
`MetricsConfig` validator from PR #1125. This is not a `git revert`, because PR #1125 edited the
same code.

## Verdict, size and departures

**Verdict:** worth doing as described.

**Size: medium.** The five triggers:

- What gets stored: does not fire. `power_forecasts`, the metrics table, and the submission log keep
  their shapes. The removed `CvConfig.final_test_start` is configuration.
- Production serving path: does not fire. Only the R&D `metrics` asset, `score_study.py`, and the
  study helper `scan_power` change.
- Degradation rule: does not fire.
- More than one defensible design: does not fire. The issue lists every deletion.
- Callers not nameable without searching: borderline. `scan_power` has about 15 study callers (see
  Risks), but the names in this plan come from a repo-wide grep and every one is listed below.

**Review budget:** zero plan reviews, because the plan adds no abstraction, field, column, or file.
One diff review, for correctness (missed references, tests that still pass for the wrong reason). No
mutation pass, because the diff deletes production logic and adds none.

**Departures from the issue body:**

- `studies/production_feature_faults/audit_production_features.py` reads `final_test_start` from the
  CV YAML (line 661, as the `power_cutoff` stamp). The issue's list omits it, and the read would
  `KeyError` after the field goes. It gets the same local constant as the other studies.
- `docs/roadmap/auto-research.md` and `docs/roadmap/metrics-and-leaderboard.md` link to the heading
  "Fold hygiene: selection bias and a final-test window". That heading stays, so no anchor changes.
- `packages/ml_core/src/ml_core/metrics.py`'s module docstring, `CvConfig`'s docstring, and
  `studies/production_feature_faults/README.md` also describe removed code. The issue's docstring
  list names the first only in part.

## Solution: walk through

**Happy path.** A study calls `score_study.py predictions.parquet <study> <fold>`. `main` parses the
arguments and copies the file to a private directory. `_score_study` checks the study name and fold,
opens the predictions, calls `require_same_row_keys` against the reference experiment, writes the
rows, and materialises the `metrics` asset in leaderboard scope. The asset calls `_validate_group`,
which now runs `require_valid_times_within_window` and, for a `study/` experiment,
`require_same_row_keys`. The re-execution in a cleaned environment, the `NGED_ALLOW_REPLACE` gate,
`require_single_model_name`, and `require_window_within_guard` are gone from this route.

**Branches.** Ad hoc scope returns the window straight after `_resolve_eval_window`, as before, but
without the date guard in front. A `--replace` run now overwrites without an environment variable.
The live fold needs no special case any more, so `metrics.LIVE_FOLD_ID` goes. `checks.py` keeps its
own `_LIVE_FOLD_ID`.

**Error paths.** Unchanged for the kept checks: each still raises, because the asset is R&D code that
fails fast. `score_study.py` still logs a refused attempt to `study_submissions.jsonl` before
re-raising.

## What changes, file by file

**Code**

- `conf/cv/default.yaml`: delete `final_test_start` and its comment.
- `packages/contracts/src/contracts/config_schemas.py`: delete `CvConfig.final_test_start`; reduce
  `_check_guard_and_reference` to the reference-name check and rename it `_check_reference`; edit the
  class docstring.
- `packages/ml_core/src/ml_core/metrics.py`: delete `FinalTestWindowError`,
  `require_window_within_guard`, `LIVE_FOLD_ID`, `MultipleModelNamesError`, and
  `require_single_model_name`. Drop imports left unused (`date`, maybe `date_to_utc_datetime`). Edit
  the module docstring and the `require_valid_times_within_window` docstring.
- `src/nged_substation_forecast/defs/cv_assets.py`: delete `FINAL_TEST_ENV_VAR`,
  `_final_test_enabled`, the guard call, and the `require_single_model_name` call and imports. Edit
  the `_validate_group` and `metrics` docstrings ("Four checks" becomes two in every scope plus the
  row-key check for studies). Drop `os` if unused.
- `scripts/forecasting/score_study.py`: delete `ALLOWED_ENVIRONMENT`, `CLEAN_ENVIRONMENT_MARKER`,
  `ALLOW_REPLACE_VARIABLE`, `clean_environment`, the re-execution in `main`, the replace gate in
  `_score_study`, the `require_single_model_name` call, and the `Mapping`/`sys` imports if unused.
  Rewrite the module docstring without the environment, `sudo`, and autonomous-session passages.
- `packages/studies/src/studies/power.py`: `scan_power` returns every cleaned reading. Delete
  `CV_CONFIG_PATH`, the `load_cv_config` import, and the datetime imports that only the cutoff
  used. Rewrite the docstring. Check `CV_CONFIG_PATH` has no other importer
  (`audit_production_features.py` does; it gets its own constant, below).
- `studies/effective_capacity/build_capacity_inputs.py`, `studies/lag_features/build_lag_frame.py`:
  `final_test_cutoff()` returns a module constant for 2026-07-01 instead of reading the CV config.
  Keep the function name so `fit_lag_arms.py` and the other callers need no edit. Drop the
  `load_cv_config` and `CV_CONFIG_PATH` imports.
- `studies/production_feature_faults/audit_production_features.py`: stamp `power_cutoff` from a local
  constant `"2026-07-01"`, or drop the stamp if the audit no longer applies a cutoff (decide on
  reading the script; see Risks).
- `pyproject.toml`: delete `[tool.importlinter]`, its contract, and the `import-linter` dev
  dependency; run `uv lock` to regenerate `uv.lock`.
- `.pre-commit-config.yaml`: delete the `import-linter` hook and its comment.
- `.github/workflows/ci.yml`: the comment's "Six hooks" becomes "Five", and `import-linter` leaves
  the list.

**Tests**

- `tests/test_metrics.py`: delete the final-test guard tests (about lines 1096-1160), the
  `MultipleModelNamesError` tests (about 1032, 1377-1410), `test_clean_environment_*` (1195-), the
  `NGED_FINAL_TEST` and `NGED_ALLOW_REPLACE` environment tests (around 1528), and their imports. For
  each remaining test that sets `NGED_ALLOW_REPLACE=1` only to pass `--replace`, drop the
  `monkeypatch.setenv`.
- `tests/test_jobs.py`: drop `final_test_start=` from the `CvConfig` constructor.
- `packages/ml_core/tests/test_metrics.py`: delete `test_require_window_within_guard`,
  `test_require_single_model_name_*`, `_FINAL_TEST_START`, and the imports.
- `packages/contracts/tests/test_config_schemas.py`: drop `final_test_start=` from every constructor
  and `_FINAL_TEST_START`; delete the two final-test validator tests (237, 254).
- `packages/studies/tests/test_power.py`: replace `test_scan_power_stops_before_midnight_on_final_test_start`
  with one asserting `scan_power` returns a reading on and after 2026-07-01. That test fails on
  `main`, where the cutoff drops those rows. Remove the `load_cv_config` monkeypatch and the
  `SimpleNamespace` import if unused.
- A new test is not needed for the constants in the two studies: `final_test_cutoff()` is exercised
  by the studies' own scripts and has no unit test today.

**Docs**

- `docs/design-philosophy/design-principles.md`: delete the "Scope" paragraph of principle 3 only.
- `docs/design-philosophy/engineering-hypotheses.md`: edit the note under T2.1 (see Risks for which
  paragraph). T2.1's label and definition stay.
- `docs/ml_experimentation/cross-validation-folds.md`: rewrite "What the `metrics` asset refuses to
  score" as two refusals (rows inside the fold window; the reference row keys for a study). Remove
  the single-model-name wording and the `NGED_ALLOW_REPLACE` sentence in the submission-log bullet.
  Edit the line 237 sentence about `scan_power`. Check `dagster-workflow.md` and `index.md` for
  mentions (grep finds none by name, so these may need only a read).
- `docs/roadmap/metrics-and-leaderboard.md`: remove the "A narrow scoring guard" paragraph and the
  final-test enforcement sentence (line 447); leave the #960 design discussion.
- `docs/roadmap/auto-research.md`: re-read the two passages that link the fold-hygiene section; edit
  only if they mention the guard.
- `studies/README.md`, `studies/lag_features/README.md`, `studies/production_feature_faults/README.md`,
  `scripts/README.md`, `.claude/skills/study/SKILL.md`, `packages/studies/README.md`: remove the
  guard, the environment-scrub, and the `NGED_ALLOW_REPLACE` text. The lag-features README keeps its
  statement that the study's own cutoff is 2026-07-01, but attributes it to the build script and not
  to `scan_power`.
- Published study pages (`docs/studies/forecasts/production-feature-faults.md` line 23 and
  `lag-features.md` line 165) describe the data a past run read. They stay unless the sentence says
  `scan_power` enforces the date.
- Per the prose rules, no sentence anywhere may say what was removed. Rewrite for the present.

## Design-philosophy check

All changed code is R&D or tooling, so the fail-fast rule applies and no production path degrades or
raises differently. No asset check changes. The change deletes enforcement, so the trade is stated
openly: principle 3's "one scoring path" survives as convention, not as code that refuses a deviating
caller. That trade is the issue's decision. H-labels are unaffected.

## Tests that would fail on `main` today

- `test_scan_power_returns_readings_on_and_after_2026_07_01` (replacement above).
- The `CvConfig` tests construct the model without `final_test_start`; `main` rejects that with a
  missing-field error.
- A `score_study` test that runs `--replace` with no environment variable set; `main` raises the
  `NGED_ALLOW_REPLACE` `ValueError`.

## Verification commands

```bash
uv lock
uv run ruff check . && uv run ruff format --check .
uv run ty check
uv run pytest
uv run pytest --run-studies packages/studies
uv run pymarkdown scan -r docs README.md CLAUDE.md packages/*/README.md
uv run pre-commit run --all-files
uv run mkdocs build --strict && uv run python scripts/lint/check_docs_links.py
grep -rnE "final_test_start|NGED_FINAL_TEST|NGED_ALLOW_REPLACE|require_single_model_name|MultipleModelNamesError|importlinter|import-linter" . --exclude-dir=.git --exclude-dir=worktrees --exclude-dir=.venv
```

The final grep must return nothing except the two studies' local constants' comments and, if kept,
this plan.

## Risks and open questions

- **Removing the cutoff changes what about 15 other studies read when re-run.** `scan_power` callers
  include `quick_wins/*`, `past_weather/cerra_wind_levels.py`, `beam_diffuse_split/*`,
  `unmetered_battery_capacity`, `weather_downloads`, `production_feature_faults/*`, and the
  `studies` package modules `pv_dataset`, `wind_product_frames`, and `nged_battery_a`. A re-run would
  include readings from 2026-07-01 onward. The issue pins only the two named studies. Recommendation:
  follow the issue, leave the published pages as they stand, and let each study meet the new rows
  when someone next re-runs it. Say yes or no if you would rather pin more studies.
- **Which paragraph is "the note under T2.1"?** The only T2.1 text mentioning the scorer is the
  paragraph beginning "A count of T2.1 leaves out every MLflow experiment named with the prefix
  `study/`". That paragraph is about the kept `study/` prefix, so deleting it all would lose a kept
  fact. Recommendation: cut only its sentence about `score_study.py` going through no experiment
  registration if that sentence refers to the removed protections, otherwise leave the paragraph.
  Confirm on reading it again during implementation.
- **`audit_production_features.py` cutoff.** If the audit script only stamps `power_cutoff` and
  applies no filter itself, the stamp now reports a cutoff its `scan_power` call no longer applies.
  Recommendation: drop the stamp and say nothing, unless the audit's output is compared across runs.
