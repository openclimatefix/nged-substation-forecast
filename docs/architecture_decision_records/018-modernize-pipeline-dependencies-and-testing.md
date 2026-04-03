---
status: "Accepted"
date: "2026-04-03"
author: "Jack & Gemini 3.1 Pro Preview"
tags: ["dagster", "polars", "mlflow", "testing", "maintenance"]
---

# ADR-018: Modernize Pipeline Dependencies and Testing

## 1. Context & Problem Statement
We needed to fix several warnings and exceptions in the pipeline that were accumulating due to library updates and evolving best practices. Specifically, we were encountering Dagster deprecations (e.g., `build_op_context`), Polars sortedness warnings during joins and groupbys, Numpy timezone warnings related to datetime conversions, MLflow deprecations, and SQLite teardown issues in our test suite. Addressing these was necessary to ensure the pipeline remains robust, maintainable, and free of silent failures.

## 2. Options Considered

### Option A: Ignore warnings
* **Description:** Leave the codebase as is and ignore the deprecation and runtime warnings.
* **Why it was rejected:** This leads to technical debt. Deprecation warnings eventually become errors in future library versions, and runtime warnings (like Polars sortedness or Numpy timezone issues) can lead to silent failures, incorrect data processing, or degraded performance.

### Option B: Partial fixes
* **Description:** Only fix the most critical errors (like SQLite teardown issues) while leaving deprecation and performance warnings for later.
* **Why it was rejected:** This approach is inconsistent and incomplete. It leaves the codebase in a fragile state and requires multiple future interventions, increasing the overall maintenance burden.

### Option C: Full modernization
* **Description:** Systematically address all warnings and exceptions across the pipeline. This includes updating Dagster API usage, explicitly handling Polars sortedness flags, fixing Numpy datetime timezone awareness, updating MLflow calls, and ensuring proper SQLite connection teardown in tests.

## 3. Decision
We chose **Option C: Full modernization**, because it ensures long-term maintainability, correctness, and adherence to modern library standards. By proactively addressing these issues, we prevent future breakages and improve the overall health of the codebase.

## 4. Consequences
* **Positive:** Improved pipeline robustness, better test reliability, cleaner test output (free of warnings), and adherence to modern library standards. The codebase is now better prepared for future library upgrades.
* **Negative/Trade-offs:** Required a focused effort to track down and fix various warnings across different packages and test files, temporarily diverting resources from feature development.
