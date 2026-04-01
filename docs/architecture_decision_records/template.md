---
status: "{Proposed | Accepted | Rejected | Superseded}"
date: "YYYY-MM-DD"
author: "Your Name & AI Agent name and version"
tags: ["orchestration", "mlops", "polars", "dagster", etc.]
---

# ADR-[000]: [Short, imperative title, e.g., Use Static Dagster Partitions over Dynamic]

## 1. Context & Problem Statement
[Briefly describe the problem. What were we trying to solve today? E.g., "We need a way to ensure different machine learning models (XGBoost, GNNs) don't overwrite each other's outputs in the Dagster pipeline, while keeping the UI clean."]

## 2. Options Considered
[This is the most important section for the support agent. List the paths we explored.]

### Option A: [E.g., Dynamic Partitions based on MLflow Run ID]
* **Description:** [Briefly explain how it would work.]
* **Why it was rejected:** [CRITICAL: Be explicit here. E.g., "This causes the 'Small Files Problem' in Delta Lake, resulting in thousands of 50KB Parquet files that crash Polars on read."]

### Option B: [E.g., Partition by Git Commit Hash]
* **Description:** [...]
* **Why it was rejected:** ["High cardinality. It still results in too many tiny partitions because we commit code frequently."]

### Option C: [E.g., Static Partitions based on Hydra Model Name]
* **Description:** [...]

## 3. Decision
We chose **Option C: [Static Partitions based on Hydra Model Name]**, because it provides the optimal balance of a clean Dagster UI, robust Delta Lake file sizes (avoiding the small files problem), and requires zero boilerplate to maintain.

## 4. Consequences
* **Positive:** Delta Lake queries for the leaderboard will be blazingly fast due to large partition sizes and Z-ordering.
* **Negative/Trade-offs:** We must manually add a string to a Python list in `core/orchestration.py` whenever a completely new model architecture is introduced.
