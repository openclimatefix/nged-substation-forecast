# Session Handoff: Substation to TimeSeriesID Refactor

## Objective
Rename `substation_number` (and related terms) to `time_series_id` (and related terms) throughout the codebase, while strictly preserving `substation_number` within the `TimeSeriesMetadata` schema and related mock data.

## Current State
- **Phase 0 (Planning & Review)::** Completed. The plan, which has been reviewed thoroughly by the
  reviewer agents, and merged, is available at `docs/temp/implementation_plan_v1_merged.md`.
- **Phase 1 (Baseline & Safeguards):** Completed.
  - Baseline tests and search performed.
  - Test failure in `packages/nged_data/tests/test_clean.py` fixed and committed.
  - Schema validation tests for `TimeSeriesMetadata` added and committed.
- **Phase 2 (Bulk Renaming):** In progress.
  - Automated bulk renaming was initiated but requires careful execution to avoid breaking the `TimeSeriesMetadata` constraint and to ensure all references (docstrings, type hints, etc.) are updated correctly.
- **Phase 3 (Verification):** Pending.

## Next Steps
1. **Complete Phase 2 (Bulk Renaming):**
   - Execute the automated renaming strategy carefully, ensuring the `TimeSeriesMetadata` constraint is respected.
   - Update all docstrings, type hints, and logging messages.
2. **Phase 3 (Verification):**
   - Run post-refactor search to ensure no unintended instances remain.
   - Run the full test suite (`pytest -s`) and fix any regressions.
   - Perform integration check to ensure the Dagster pipeline functions correctly.
