---
status: "Accepted"
date: "2026-04-01"
author: "Software Architect (gemini-3.1-pro-preview)"
tags: ["orchestration", "dagster", "architecture", "packages"]
---

# ADR-011: Co-locate Dagster Asset Definitions within Domain Sub-packages

## 1. Context & Problem Statement
As we modularize the codebase into distinct sub-packages (e.g., `packages/geo`, `packages/data`), we need to determine the boundary of responsibility for orchestration. Specifically, we must decide whether these sub-packages should remain entirely "Dagster-ignorant" (containing only pure Python logic), or if they should own their respective Dagster asset definitions (`assets.py`).

## 2. Options Considered

### Option A: Strict Separation (Pure Logic Packages, Assets in Main App)
* **Description:** Sub-packages contain only pure Python logic, functions, and classes. The main application (`src/nged_substation_forecast/`) imports this logic and wraps it in Dagster `@asset` decorators.
* **Why it was rejected:** This approach causes "wrapper fatigue" and splits the developer's mental model. A developer working on a geospatial feature would have to write the logic in `packages/geo`, but then switch contexts to `src/` to define how that logic is orchestrated and materialized. Consequently, the main app becomes a massive dumping ground for boilerplate wrapper code, reducing maintainability.

### Option B: Standalone Scripts / Git-committed Data
* **Description:** Sub-packages include standalone scripts to generate necessary data (e.g., static geospatial mappings), which are then either committed directly to git or stored externally without formal orchestration.
* **Why it was rejected:** This completely breaks data lineage and observability. It lacks data contract enforcement at the time of creation, pollutes the git history with data files, and makes it difficult to track when or how a specific dataset was generated.

### Option C: Domain-Driven Packages with Co-located Assets
* **Description:** Packages are fully self-contained domain modules. They own their pure logic (e.g., `h3.py`), their data schemas (`contracts.py`), and their Dagster wrappers (`assets.py`). The main application (`src/`) acts solely as a "Global Wiring" registry, importing the pre-defined assets from the sub-packages and assembling them into a global Dagster `Definitions` object.

## 3. Decision
We chose **Option C: Domain-Driven Packages with Co-located Assets**. This approach aligns with domain-driven design principles by keeping all related code—logic, schemas, and orchestration—highly cohesive and co-located. The main app is kept clean and focused strictly on global orchestration wiring.

## 4. Consequences
* **Positive:** Significantly better Developer Experience (DX). Developers can build, test, and reason about a domain (like geospatial processing) entirely within its own package. Centralized testing allows logic and its Dagster materialization to be tested together.
* **Negative/Trade-offs:** Sub-packages now have a hard dependency on `dagster`. This means the packages cannot be easily reused in non-Dagster projects without pulling in Dagster as a dependency. Given our commitment to Dagster as our orchestration engine, this is an acceptable trade-off.
