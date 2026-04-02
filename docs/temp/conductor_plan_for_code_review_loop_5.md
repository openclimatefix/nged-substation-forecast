# Conductor Session Plan: NGED Substation Forecast - Loop 5 (Final Logical Audit)

## 1. Context & Current State
We have completed the refactoring of the geospatial and weather data ingestion pipeline.
- **Loop 1 & 2**: Fixed critical logical flaws (longitude wrap-around, data leakage, etc.).
- **Loop 3**: Added "why" comments for all non-obvious logic, heuristics, and constants.
- **Current Status**: The code is functionally complete and documented, but the user has requested one final rigorous audit for logical flaws before finalization.

## 2. Remaining Tasks (Loop 5: Final Logical Audit)
This loop returns to **Station 1** of the Code Review track for a final "sanity check" on logic, physics, and robustness.

### Station 1: Scientist Review (Phase 2: Fresh Audit)
- **Goal**: Perform a completely fresh, independent audit for math/ML rigor, physical realism, and data leakage.
- **Files**:
    - `packages/geo/src/geo/h3.py`
    - `packages/geo/src/geo/assets.py`
    - `packages/dynamical_data/src/dynamical_data/processing.py`
    - `packages/xgboost_forecaster/src/xgboost_forecaster/data.py`
- **Output**: `docs/temp/scientist_code_review_5.md`

### Station 2: Tester Review (Phase 2: Fresh Audit)
- **Goal**: Perform a fresh audit for robustness, edge cases, and testability.
- **Files**: Same as above.
- **Output**: `docs/temp/tester_code_review_5.md`

### Station 3: Reviewer Review (Phase 2: Fresh Audit)
- **Goal**: Perform a fresh audit for style, maintainability, and architectural alignment.
- **Files**: Same as above.
- **Output**: `docs/temp/reviewer_code_review_5.md`

### Finalization (Track E)
- Update `README.md` and package-specific docs.
- Update or create ADRs (ADR 010, 011, etc.).
- Final git commit and cleanup.

## 3. Technical Details
- **H3 API**: `h3.api.basic_int` (v4).
- **Coordinate System**: EPSG:27700.
- **Data Contracts**: `Patito` models in `packages/contracts`.
- **Interpolation**: Circular (sine/cosine) for wind, forward-fill for categories, linear for others (with radiation caveat).
