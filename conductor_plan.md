# Conductor Session Plan: NGED Substation Forecast Refactoring

## 1. Context & Current State
We are in the middle of a major refactoring of the geospatial and weather data ingestion pipeline. The goal is to decouple generic geospatial logic from dataset-specific (ECMWF) logic and move towards a Dagster-centric orchestration model.

### Completed Milestones:
- **New Package `packages/geo`**: Created to house generic geospatial logic and assets.
- **Dagster Integration**: H3 grid weights are now a managed Dagster asset (`gb_h3_grid_weights`) instead of a static file in git.
- **Refactored `dynamical_data`**: Decoupled from static file paths; now uses injected Dagster assets.
- **ML Pipeline Updates**: `xgboost_forecaster` now handles circular variables (wind direction) via sine/cosine interpolation and categorical variables via forward-fill.
- **Code Review Loops 1 & 2**: Completed and all identified flaws (including critical ones like longitude wrap-around and data leakage) have been fixed and verified.

### Key Architectural Decisions:
- **ADR 010**: Separated generic geospatial logic into `packages/geo`.
- **ADR 011**: Allowed sub-packages to depend on Dagster and co-locate asset definitions (`assets.py`) for better DX and domain ownership.

## 2. Remaining Tasks (Loop 3: "Why" Comments)
We are currently in **Loop 3** of the Code Review track. The focus is on ensuring that "clever" or non-obvious logic is documented with "why" comments.

### Station 1: Scientist Review (Completed)
The Scientist identified 4 flaws related to missing physical/mathematical context in comments:
1. **Categorical Forward-fill**: Explain why linear interpolation is physically meaningless for categories.
2. **Circular Interpolation Assumption**: Document the assumption that 0-255 UInt8 maps perfectly to 0-360 degrees.
3. **Temporal Interpolation & Leakage**: Explain why interpolating over `valid_time` within a single `init_time` is not data leakage.
4. **Grid Snapping Formula**: Explain the mathematical intent behind the half-grid offset binning.

### Next Steps:
1. **Architect**: Update implementation plan to `v1.8_after_scientist_loop3.md` to include these comment fixes.
2. **Custom Build**: Implement the comments in the following files:
   - `packages/geo/src/geo/h3.py`
   - `packages/geo/src/geo/assets.py`
   - `packages/dynamical_data/src/dynamical_data/processing.py`
   - `packages/xgboost_forecaster/src/xgboost_forecaster/data.py`
3. **Tester**: Perform a fresh audit of "why" comments for robustness/testing logic.
4. **Reviewer**: Perform a fresh audit of "why" comments for style/maintainability.
5. **Finalization**: Update READMEs, docs, and ADRs.

## 3. Technical Details for Resumption
- **H3 API**: Using `h3.api.basic_int` (v4) for integer-based operations.
- **Coordinate System**: Using EPSG:27700 (British National Grid) for metric buffering of the UK boundary.
- **Data Contracts**: Using `Patito` models in `packages/contracts/src/contracts/data_schemas.py`.
- **Interpolation**: Circular variables use sine/cosine decomposition; categorical variables use forward-fill.
