---
status: "Accepted"
date: "2026-04-02"
author: "Architect"
tags: ["testing", "dagster", "pytest", "mlops"]
---

# ADR-017: Full Pipeline Integration Testing with Dagster

## 1. Context & Problem Statement
To ensure our machine learning pipelines are robust, we need to test the entire lifecycle: data preparation, model training, MLflow logging, inference, and evaluation. Previously, we relied on manual Python scripts (e.g., `manual_xgboost_integration.py`) to test these components. However, these scripts often drift from the actual production Dagster assets, leading to a false sense of security. We need a testing strategy that executes the exact same Dagster assets used in production, but runs quickly and reliably in a CI/CD environment without requiring a heavy infrastructure setup.

## 2. Options Considered

### Option A: Manual Integration Scripts
* **Description:** Maintain standalone Python scripts that import core ML functions and run them sequentially.
* **Why it was rejected:** These scripts do not test the Dagster orchestration logic (e.g., asset dependencies, partitioning, resource injection). They easily become outdated as the Dagster pipeline evolves, violating the principle of "test what you run."

### Option B: Dagster `execute_in_process` with `mem_io_manager` in `pytest`
* **Description:** Use Dagster's native testing utilities (`execute_in_process`) within `pytest`. We construct a temporary Dagster job containing the exact production assets, but swap out the heavy production I/O managers (like Delta Lake) with Dagster's `mem_io_manager` (in-memory storage). We also use a local SQLite MLflow tracking URI for the test duration.
* **Why it was accepted:** It tests the exact production code paths and asset definitions. It is fast because it avoids disk I/O for intermediate datasets, and it integrates seamlessly with our standard `pytest` workflow.

### Option C: Mocking Dagster Assets
* **Description:** Extensively mock the inputs and outputs of individual Dagster assets using `unittest.mock`.
* **Why it was rejected:** High maintenance burden. Mocking complex Polars DataFrames and MLflow interactions is brittle and often hides integration bugs between assets.

## 3. Decision
We chose **Option B: Dagster `execute_in_process` with `mem_io_manager` in `pytest`**. We will use this pattern for all end-to-end integration tests of our ML pipelines.

## 4. Consequences
* **Positive:**
  * High confidence that the production Dagster pipeline works as expected.
  * Tests are fast and isolated (no shared state or database pollution).
  * Eliminates the need for separate, drift-prone manual integration scripts.
* **Negative/Trade-offs:**
  * Requires careful setup of Dagster resources and configuration within the `pytest` fixtures.
  * Developers need to understand Dagster's testing APIs.
