---
status: "reviewed"
version: "v1.1"
task_type: "standard"
requires_ml_review: true
requires_data_engineer: false
target_modules: ["packages/ml_core", "packages/xgboost_forecaster", "tests"]
---

# Implementation Plan: Rename `flows_30m` to `power_time_series`

## 1. Objective
Rename the variable, parameter, and dictionary key `flows_30m` to `power_time_series` throughout the entire codebase. This change improves domain language consistency, as the data represents a generic power time series rather than being strictly tied to a 30-minute resolution in its naming.

## 2. Strategy for Automated Refactoring
To ensure consistency, minimize human error, and avoid incomplete refactoring, we will use automated global search-and-replace tools rather than manual, file-by-file updates.

**Important Note on Code Comments:**
When updating docstrings, ensure they explicitly state that `power_time_series` is the source of truth for the historical power measurements. Comments must focus on the *why* (intent and rationale) rather than the *how*, connecting the dots across the codebase so a junior engineer can easily understand the architecture. Do not reference FLAW-XXX IDs in code comments.

## 3. Implementation Steps

### Step 1: Pre-check for Dynamic Access Patterns
Before performing the global rename, run a search for `flows_30m` *without* strict word boundaries (e.g., `rg "flows_30m"`) to identify any dynamic usage patterns (like `f"flows_{resolution}"` or `getattr(obj, "flows_30m")`). If any are found, they must be manually refactored to use the new name dynamically.

### Step 2: Global Search and Replace
Perform a global search-and-replace across the entire repository (including Python files, YAML/JSON configs, and documentation) using strict word boundaries.
*   Replace `\bflows_30m\b` with `power_time_series`.
*   Tools like `sed`, `rg --replace`, or IDE refactoring should be used to ensure all instances are caught in one go.

### Step 3: Docstring Updates
Review the updated docstrings in `packages/ml_core/src/ml_core/model.py`, `packages/ml_core/src/ml_core/experimental.py`, and `packages/xgboost_forecaster/src/xgboost_forecaster/features.py`. Ensure they explicitly state that `power_time_series` is the source of truth for historical power measurements.

### Step 4: Data Integrity Verification Test
Add a specific verification step (a unit test) to ensure that the *content* of the data being passed into `power_time_series` is identical to what was previously passed into `flows_30m`. This could be a simple check comparing the output of a data loading function before and after the rename, or an assertion in an existing test that verifies the schema and row count of the `power_time_series` input.

## 4. Verification Steps
After all replacements are made, the Builder must run the following verification steps:
1.  **Type Checking:** Run `uv run mypy .` or `uv run pyright` to ensure no interface mismatches occurred during the rename.
2.  **Unit Tests & Coverage:** Run `uv run pytest --cov` across the entire repository to ensure all tests pass and to identify any untested code paths that might have been affected.
3.  **Comprehensive Grep Check:** Run `rg "flows_30m"` (without strict word boundaries) across *all* files in the repository (including configs and docs) to verify that zero instances of the old variable name remain.

## 5. Risk Minimization Strategy
*   **Automated Tools:** Using global search-and-replace minimizes the risk of missing files or making typos.
*   **No Silent Failures:** Ensure that dictionary key lookups are updated symmetrically to prevent `KeyError` or silent fallback behaviors.
*   **Atomic Commits:** The Conductor should commit these changes as a single, atomic commit to ensure the codebase is never in a broken, intermediate state.

## Review Responses & Rejections

* **FLAW-001 [Reviewer]:** ACCEPTED (Partially). We will expand verification to all non-Python files and use automated global search-and-replace (`sed` or `rg`). REJECTED the pre-commit hook suggestion as it adds unnecessary complexity for a one-off rename; the expanded `rg` verification step is sufficient.
* **FLAW-001 [Simplicity]:** ACCEPTED. The manual step-by-step strategy has been replaced with a single automated global search-and-replace instruction.
* **FLAW-002 [Simplicity]:** ACCEPTED. The detailed breakdown of update instructions has been removed in favor of a single clear instruction.
* **Scientist Recommendations:** ACCEPTED. Added a requirement to include a data integrity test and to update docstrings to explicitly state `power_time_series` is the source of truth for historical power measurements.
* **Tester Recommendations:** ACCEPTED. Added a pre-check for dynamic access patterns without strict word boundaries, expanded the final grep check to all files, and added `--cov` to the pytest verification step.
