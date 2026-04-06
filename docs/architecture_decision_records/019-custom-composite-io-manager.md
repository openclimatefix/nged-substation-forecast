---
status: "Accepted"
date: "2026-04-06"
author: "AI Agent (gemini-3.1-flash-lite-preview)"
tags: ["dagster", "io-manager", "polars", "geospatial"]
---

# ADR-019: Custom Composite IO Manager for Geospatial Assets

## 1. Context & Problem Statement
The asset `gb_h3_grid_weights` returns a `pl.DataFrame`, while `uk_boundary` returns a `BaseGeometry` (a `MultiPolygon`). The default `FilesystemIOManager` (which uses `PickledObjectFilesystemIOManager`) was failing to correctly handle the `pl.DataFrame` in some environments, leading to `DagsterTypeCheckDidNotPass` errors.

## 2. Options Considered

### Option A: Use `PickledObjectFilesystemIOManager` for everything
* **Description:** Rely on the default Dagster IO manager.
* **Why it was rejected:** It was failing for `gb_h3_grid_weights` in the user's environment, returning a `pathlib.PosixPath` instead of the `pl.DataFrame`.

### Option B: Use `PolarsIOManager` for everything
* **Description:** Create a custom IO manager that only handles `pl.DataFrame` using Parquet.
* **Why it was rejected:** It failed for `uk_boundary` because `BaseGeometry` is not a `pl.DataFrame` and does not have a `write_parquet` method.

### Option C: Composite IO Manager
* **Description:** Create a custom `CompositeIOManager` that delegates to `PolarsIOManager` for `pl.DataFrame` and `PickledObjectFilesystemIOManager` (or a similar pickle-based approach) for other types.

## 3. Decision
We chose **Option C: Composite IO Manager**, because it provides the optimal balance of correctly handling both `pl.DataFrame` and other types, resolving the `DagsterTypeCheckDidNotPass` error.

## 4. Consequences
* **Positive:** Correctly handles both `pl.DataFrame` and other types, resolving the `DagsterTypeCheckDidNotPass` error.
* **Negative/Trade-offs:** Slightly more complex IO manager implementation.
