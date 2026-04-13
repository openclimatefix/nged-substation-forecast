---
status: "Accepted"
date: "2026-04-13"
author: "Software Architect (gemini-3.1-pro-preview)"
tags: ["orchestration", "dagster", "polars", "patito", "data-contracts"]
---

# ADR-021: Use `create_dagster_type_from_patito_model` for Dagster-Patito Integration

## 1. Context & Problem Statement
We need a robust, idiomatic way to integrate Patito data contracts (which validate Polars DataFrames) with Dagster's type system. Dagster needs to understand the types of assets being passed between ops/assets for type checking, UI rendering, and runtime validation. We must ensure that data flowing through the pipeline strictly adheres to our defined Patito schemas while maintaining a clean, maintainable Dagster implementation.

## 2. Options Considered

### Option A: Returning raw Polars DataFrames
* **Description:** Assets simply return `pl.DataFrame` or `pl.LazyFrame` without any explicit Dagster type wrapping. Validation would happen manually inside the asset body (e.g., calling `MyModel.validate(df)`).
* **Why it was rejected:** This approach loses the benefits of Dagster's type system. The Dagster UI won't show the specific schema expected, and runtime validation relies entirely on developers remembering to manually call validation methods inside every asset. It lacks visibility and enforcement at the orchestration layer.

### Option B: TypeAlias
* **Description:** Using Python's `TypeAlias` to alias `pl.DataFrame` to a specific Patito model name (e.g., `MyModelDataFrame: TypeAlias = pl.DataFrame`).
* **Why it was rejected:** While this provides some static typing hints for developers, Dagster does not natively enforce or validate `TypeAlias` at runtime. It is purely cosmetic and doesn't trigger Patito's validation logic when data is passed between assets. It fails to provide the strict runtime guarantees we require for data contracts.

### Option C: Side-effect assets
* **Description:** Creating separate Dagster assets whose sole purpose is to validate a DataFrame against a Patito model (e.g., an asset that takes a DataFrame, validates it, and acts as a side-effect or gatekeeper).
* **Why it was rejected:** This clutters the Dagster lineage graph with validation-only assets, making the pipeline harder to read and maintain. It separates the data generation from its validation, violating the principle of cohesion, and adds unnecessary overhead to the orchestration layer.

### Option D: `create_dagster_type_from_patito_model`
* **Description:** Using a dedicated factory function to generate a `DagsterType` directly from a Patito `Model`. This type can then be used in asset type hints.

## 3. Decision
We chose **Option D: `create_dagster_type_from_patito_model`**, because it seamlessly bridges Patito's schema definitions with Dagster's type system. It allows us to annotate asset return types and inputs with the generated `DagsterType`, enabling automatic runtime validation by Dagster and rich metadata display (like schema definitions) directly in the Dagster UI.

## 4. Consequences
* **Positive:** Automatic, framework-level runtime validation of Polars DataFrames against Patito schemas. Clear visibility of data contracts in the Dagster UI. Clean asset code without manual validation boilerplate.
* **Negative/Trade-offs:** Introduces a dependency on the specific integration function. Requires developers to understand how to generate and apply these custom Dagster types instead of using standard Python types.
