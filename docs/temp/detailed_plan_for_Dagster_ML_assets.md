# Engineering Implementation Specification: NGED Probabilistic Forecasting Platform

This document outlines the architecture, data contracts, and progressive implementation steps for building the National Grid Electricity Distribution (NGED) probabilistic forecasting platform. It is formatted to provide an AI coding agent (e.g., Claude Code) with explicit design constraints, directory structures, and decoupled execution patterns.

---

## 1. Ultimate Aims of the Platform

* **Massive Experimentation Scale:** Enable research engineers to spin up hundreds of parallel Machine Learning experiments utilizing radically different frameworks (ARIMA, XGBoost, Temporal Fusion Transformer, Graph Neural Networks, and Differentiable Physics) without breaking production systems.
* **Production Robustness:** Support a continuous, production-grade inference engine running every **6 hours** to output 14-day ahead, half-hourly probabilistic forecasts across all 1,300 primary substations.
* **Scientific Rigor:** Maintain an immutable, scientifically fair leaderboard by validating models on strict, identical calendar-year blocks using expanding training windows.
* **Zero Training-Serving Skew:** Enforce a single `uv` monorepo configuration where experimentation code and production execution code use identical data-processing pipelines.
* **Compute Agnosticism:** Allow lightweight CPU workloads (data preparation, statistical modeling) to run on a central VM, while seamlessly bursting heavy deep-learning training tasks to serverless GPU compute (Modal).

---

## 2. Rejected Architectural Designs

Before finalizing this blueprint, several alternative structures were evaluated and explicitly rejected:

### Rejected: Using Dagster Partitions for Cross-Validation Folds

* *The Design:* Creating a `StaticPartitionsDefinition` (e.g., `["2023", "2024", "Live"]`) on Software-Defined Assets to handle both CV folds and production inference within the same asset.
* *Why Rejected:* Dagster assets represent the *current state of a data product*. If multiple researchers execute parallel experiments using the same asset definitions, they will overwrite each other's historical partitions, destroying the leaderboard. Furthermore, mixing categorical year blocks with an operational rolling time series breaks Dagster's built-in time-lineage and history tools.

### Rejected: Partitioning the Graph by Substation ID

* *The Design:* Creating a partition coordinate for each of the 1,300 primary substations to track local model failures and re-runs.
* *Why Rejected:* Triggering 1,300 separate asset executions every 6 hours generates massive orchestrator overhead (62,400 tasks per day), choking the scheduler. It also completely prevents multi-site models (like Graph Neural Networks) from accessing the global grid matrix natively during a forward pass.

---

## 3. Finalized System Design & Architecture

The system operates as a **two-layer funnel**: Dagster governs the time-series scheduling and data integrity layer, while MLflow acts as the abstract model shipping container via custom `pyfunc` wrappers.


### Architectural Principles For Code Generation:

1. **Strict Dependency Injection:** Sub-packages must never import modules from the Dagster root package in `src/`. They must interact with the system purely via flat input arguments (e.g., file paths, parameters) passed down by Dagster or Hydra.


## 4. Progressive Implementation Roadmap


### Phase 1: The Experimentation & Cross-Validation Engine

* **Task 1.1: Build Dynamic Map and Collect Jobs:** Build a dedicated Dagster Job for model research that uses `DynamicOut`. The generator Op maps over your strict, hardcoded calendar-year historical limits.
* **Task 1.2: Implement Cluster-Parallel Training:** Wire the mapped outputs to a training Op that triggers your training loop.

* **Later (not for today): Implement Modal Integration:** Use Dagster Pipes within the training Op to check if a model config requires a GPU. If true, pipe the execution context seamlessly to Modal to launch the serverless training container, streaming logs back to the main Dagster terminal.


### Phase 2: Production Inference & Champion Promotion

This is all for later. We won't implement this today.

* **Task 2.1: 6-Hourly Time-Window Partitioning:** Build the `canonical_grid_features` Dagster asset. Partition it on a 6-hourly cycle (`00:00`, `06:00`, `12:00`, `18:00`). This asset processes real-time telemetry and performs joins against the slower-moving Numerical Weather Prediction (NWP) dataset (which runs daily at midnight UTC via Dynamical.org) using a `TimeWindowPartitionMapping`.
* **Task 2.2: Implement the Production Bridge Asset:** Create a `champion_model` Dagster asset. This asset acts as a pass-through that hits the MLflow Model Registry API, requests the metadata for the active model tagged `"Production"`, and serializes it locally.
* **Task 2.3: Decoupled Multi-Model Serving:** Write the final `primary_substation_forecasts` inference asset. It loads the `champion_model`, passes the global 6-hourly data payload to it, and stores the long-form Polars output. Downstream business application databases are directed to read rows where `ensemble_member_id == 'aggregate'`, while monitoring tools extract individual traces.
* **Task 2.4: Financial Impact Assessment Asset:** Create a daily `production_model_settlement` asset. This asset must join historical forecasts against true telemetry as it rolls in. It will compute and log to MLflow real-world operational business metrics:
1. *Wasted Procurement Cost (£):* Excess capacity purchased due to over-conservative upper quantiles.
2. *Unmitigated Risk Hours (Count):* Severe load spikes missed by the upper tail bounds.
3. *Oracle Delta (£):* Total capital savings left on the table when compared to perfect future foresight.



### Phase 3: CI/CD & Automated Testing Layer

Not for today.

* **Task 3.1: Create Structural Contract Verifications:** Implement unit tests in `pytest` that check the MLflow `pyfunc` serialization logic against synthetic, mock Polars DataFrames to ensure schema conformance.
* **Task 3.2: Implement Fast-Forward Smoke Testing:** Configure your CI script (GitHub Actions) to intercept pull requests and execute the full Dagster training Job with a Hydra override setting `max_epochs=1` and `batch_limit=1`. This guarantees structural code execution validity across the entire orchestrator graph in under 3 minutes.
