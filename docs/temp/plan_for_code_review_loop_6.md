# Conductor Session Plan: NGED Substation Forecast - Loop 6 (Final Scientist Audit & Horizon Logic)

## 1. Context & Current State
We have completed Loop 5, which addressed 16 flaws across Scientist, Tester, and Reviewer audits.
- **Wind Logic:** We shifted from circular interpolation of speed/direction to linear interpolation of Cartesian `u` and `v` components.
- **Horizon Logic:** We parameterized the lead-time filter to prevent lookahead bias.
- **Robustness:** We added anti-meridian wrap-around handling and strict coordinate validation.

## 2. Remaining Tasks (Loop 6: Final Scientist Audit)
This loop focuses on a deep-dive by the **Scientist** into the physical and ML implications of the recent changes.

### Station 1: Scientist Review (Phase 1: Verification & Fresh Audit)
- **Goal**: Verify the removal of redundant circular interpolation code, audit the new `u/v` logic, and ensure the model handles long-range (14-day) horizons correctly.
- **Specific Focus Areas**:
    1. **Wind Vector Interpolation:**
        - Confirm that the complex circular interpolation code from Loop 3 is fully removed.
        - Verify that `u` and `v` are interpolated linearly.
        - Ensure "why" comments explicitly mention the elimination of "phantom high wind" artifacts.
    2. **Horizon & Lead-Time Logic:**
        - Verify the code handles 14-day (336h) forecasts at 30-minute resolution.
        - **New Requirement:** Ensure the `lead_time_hours` (or `horizon`) is passed as a feature to the XGBoost model so it can learn the decay in NWP skill over time.
    3. **Tester's Changes Audit:**
        - Review the anti-meridian wrap-around logic in `processing.py` for physical correctness.
        - Review the weighted categorical mode implementation.
- **Files**:
    - `packages/geo/src/geo/h3.py`
    - `packages/geo/src/geo/assets.py`
    - `packages/dynamical_data/src/dynamical_data/processing.py`
    - `packages/xgboost_forecaster/src/xgboost_forecaster/data.py`
- **Output**: `docs/temp/scientist_code_review_6.md`

### Implementation Phase
- **Architect**: Update the plan based on the Scientist's findings in Loop 6.
- **Custom Build**:
    - Remove any lingering circular interpolation code.
    - Add `lead_time_hours` as a feature to the XGBoost training/inference set.
    - Fix any flaws identified in the Scientist's audit of the Tester's changes.

### Finalization (Track E)
- Update `README.md`, package docs, and ADRs.
- Final git commit and cleanup.

## 3. Technical Details
- **Target Horizon**: 14 days (336 hours).
- **Temporal Resolution**: 30 minutes.
- **Wind Components**: `wind_u`, `wind_v` (Float32).
- **Lead Time Feature**: Must be normalized or handled appropriately for XGBoost.
