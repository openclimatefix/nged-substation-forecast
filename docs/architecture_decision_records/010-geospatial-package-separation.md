---
status: "Accepted"
date: "2026-04-01"
author: "Architect Agent (gemini-3.1-pro-preview)"
tags: ["architecture", "geospatial", "modularity", "h3"]
---

# ADR-010: Separate Generic Geospatial Logic into `packages/geo`

## 1. Context & Problem Statement
The `dynamical_data` package was originally responsible for both downloading/processing specific NWP datasets (like ECMWF) and handling generic geospatial operations, such as mapping latitude/longitude grids to H3 hexagons. As we prepare to add more geospatial datasets (e.g., GFS, satellite imagery, or static geographical features), keeping generic geospatial logic coupled with dataset-specific ingestion code leads to code duplication, circular dependencies, and a lack of clear boundaries. We needed a way to share geospatial utilities across the entire project without forcing other packages to depend on `dynamical_data`.

## 2. Options Considered

### Option A: Keep Geospatial Logic in `dynamical_data`
* **Description:** Continue adding generic geospatial functions (like H3 grid mapping) to the `dynamical_data` package.
* **Why it was rejected:** This violates the single responsibility principle. `dynamical_data` should focus on handling time-varying NWP data. If another package (e.g., a static features package) needs H3 mapping, it would have to depend on `dynamical_data`, which is conceptually incorrect and could lead to bloated dependencies.

### Option B: Move Geospatial Logic to `ml_core`
* **Description:** Place the H3 mapping and grid utilities into the `ml_core` package.
* **Why it was rejected:** `ml_core` is designed for machine learning model interfaces, training loops, and evaluation. Geospatial data processing is a distinct domain. Mixing data processing utilities with ML orchestration blurs the lines of responsibility.

### Option C: Create a Dedicated `packages/geo` Package
* **Description:** Extract all generic geospatial logic, including H3 grid weight computation and spatial mapping, into a new, lightweight `packages/geo` package. Define strict data contracts (e.g., `H3GridWeights`) in `packages/contracts` to ensure type safety when passing spatial data between packages.

## 3. Decision
We chose **Option C: Create a Dedicated `packages/geo` Package**. This approach decouples generic spatial operations from dataset-specific ingestion logic. It allows any package in the workspace to perform geospatial transformations by depending on a focused, lightweight library. We also parameterized the grid size and H3 resolution in the generic functions to ensure they can be reused for datasets with different spatial characteristics (e.g., 0.25-degree ECMWF vs. 1km satellite data).

## 4. Consequences
* **Positive:** Improved modularity and code reuse. New datasets can easily leverage the `geo` package for spatial mapping without duplicating code.
* **Positive:** Clearer dependency graph. Packages that only need spatial utilities don't need to depend on `dynamical_data`.
* **Positive:** The use of Patito contracts (`H3GridWeights`) ensures that the output of the generic spatial mapping functions is strictly validated before being used by downstream ingestion pipelines.
* **Negative/Trade-offs:** Introduces an additional package to manage in the workspace, slightly increasing the initial cognitive load for new developers navigating the repository structure.
