---
status: "reviewed"
version: "v1.1"
task_type: "standard"
requires_ml_review: true
requires_data_engineer: false
target_modules: ["packages/nged_data", "packages/xgboost_forecaster", "src/nged_substation_forecast", "packages/ml_core", "tests"]
---
# Implementation Plan: Rename `substation_number` to `time_series_id`

## Objective
Rename `substation_number` (and related terms like `substation_ids`, `substation_power_flows`) to `time_series_id` (and `time_series_ids`, `power_time_series`) throughout the codebase to align with the `TimeSeriesMetadata` and `PowerTimeSeries` data contracts.

**CRITICAL CONSTRAINT:** Do NOT remove or rename `substation_number` within the `TimeSeriesMetadata` schema or any mock dataframes that are validated against it, as it remains a required field in the contract.

## Conflicts
*   **Simplicity FLAW-002 (Incremental PRs) vs. Atomic Renaming:** Simplicity FLAW-002 recommends breaking the refactoring into smaller, incremental PRs. However, renaming variables that cross package boundaries (e.g., Dagster configs passed to ML core) can break the pipeline if done piecemeal.
*   *Resolution:* We will adopt a hybrid approach. We will split the work into logical phases (PRs) where safe (e.g., adding tests first), but the core cross-package rename (Dagster assets + ML core + tests) will be grouped into a single atomic PR to ensure the codebase remains functional at all commits.

## Step-by-Step Plan

### Phase 1: Pre-Refactor Safeguards & Baseline (PR 1)
1.  **Baseline Tests:** Run the entire test suite (`pytest -s`) with a timeout of at least 10 minutes (`600000` ms) to ensure a clean baseline before any changes.
2.  **Baseline Search:** Run a full-text search (e.g., `rg "substation_number|substation_ids|substation_power_flows"`) to establish a baseline count of instances to be changed.
3.  **Enforce Constraint (Tester FLAW-002):** Add specific tests in `tests/` to validate the `TimeSeriesMetadata` schema.
    *   Create a test that validates a sample DataFrame to ensure `substation_number` is present.
    *   Create an adversarial test that attempts to pass a DataFrame *without* `substation_number` to the `TimeSeriesMetadata` validator and asserts that it fails loudly.

### Phase 2: Automated Bulk Renaming (PR 2)
Instead of manual file-by-file edits, use automated tools (e.g., `sed`, `rg`, or IDE refactoring) to perform the bulk renaming across the codebase (Simplicity FLAW-001).

**Target Replacements:**
*   `substation_number` -> `time_series_id` (EXCEPT in `TimeSeriesMetadata` schema and related mock data).
*   `substation_ids` -> `time_series_ids`
*   `substation_power_flows` -> `power_time_series`
*   `allow_empty_substations` -> `allow_empty_time_series`
*   `healthy_substations` -> `healthy_time_series`
*   `_get_target_substations` -> `_get_target_time_series`

**Scope of Updates (Reviewer FLAW-002):**
Ensure the automated renaming covers:
*   Variable names and function signatures.
*   Configuration dictionary keys (e.g., in `XGBoostConfig`, `WeatherConfig`).
*   **Docstrings and Comments:** Update all explanations and references. Ensure any new or modified comments focus on the *why* (intent and rationale) rather than the *how*. Comments must "connect the dots" across the codebase, explaining how the `time_series_id` relates to the `TimeSeriesMetadata` contract.
*   **Type Hints:** Ensure any custom types or literals are updated.
*   **Logging Messages:** Update log outputs to use the new terminology.
*   **FORBIDDEN:** Do NOT include any `FLAW-XXX` IDs in the source code comments.

**Specific File Adjustments (from original plan):**
*   `packages/nged_data/src/nged_data/clean.py`: Update comment on line 60.
*   `packages/xgboost_forecaster/src/xgboost_forecaster/model.py`: In `_prepare_data_for_model`, remove `"substation_number"` from the `select` list when creating `metadata_lf`.
*   `packages/ml_core/src/ml_core/utils.py`: In `train_and_log_model` and `evaluate_and_save_model`, update temporal slicing logic to check for `"power_time_series"` instead of `"power_flows"`. Remove redundant `if "substation_power_flows" in sliced_data:` block.

### Phase 3: Verification & Cleanup (PR 3 or combined with PR 2)
1.  **Post-Refactor Search (Tester FLAW-001):** Run the same full-text search (`rg "substation_number|substation_ids|substation_power_flows"`) to ensure no instances remain, *except* where explicitly allowed (e.g., `TimeSeriesMetadata` schema definitions and its mock data in tests).
2.  **Run Full Test Suite (Reviewer FLAW-001 & Tester FLAW-003):** Execute `pytest -s` with a timeout of at least 10 minutes (`600000` ms) across all affected packages to ensure no regressions were introduced. Fix any failing tests immediately.
3.  **Integration Check:** Ensure the Dagster pipeline can materialize the updated assets without configuration errors.

## Review Responses & Rejections

*   **FLAW-001 [Tester]:** ACCEPTED. Added pre-refactor and post-refactor grep searches to ensure complete renaming, and mandated full test suite runs.
*   **FLAW-002 [Tester]:** ACCEPTED. Added explicit schema validation and adversarial tests in Phase 1 to lock in the `TimeSeriesMetadata` constraint.
*   **FLAW-003 [Tester]:** ACCEPTED. Added baseline and post-refactor test runs. Mutation testing is noted but deferred as out-of-scope for a simple rename, relying instead on strict schema tests.
*   **FLAW-001 [Reviewer]:** ACCEPTED. Added Phase 3: Verification, explicitly requiring `pytest` runs.
*   **FLAW-002 [Reviewer]:** ACCEPTED. Explicitly expanded the scope of the automated renaming to include docstrings, type hints, and logging messages.
*   **FLAW-001 [Simplicity]:** ACCEPTED. Replaced the manual file-by-file plan with an automated bulk renaming strategy using tools like `sed` or `rg`.
*   **FLAW-002 [Simplicity]:** ACCEPTED WITH MODIFICATIONS. The plan is broken into incremental phases (PRs). However, as noted in the `## Conflicts` section, the core renaming across interconnected packages must be atomic to prevent breaking the build between PRs.
