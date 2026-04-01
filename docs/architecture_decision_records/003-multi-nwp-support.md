---
status: "Accepted"
date: "2026-03-27"
author: "Jack & opencode (gemini-3.1-pro-preview)"
tags: ["data", "nwp", "features", "hydra"]
---

# ADR-003: Multi-NWP Support via Enum Lists and Dict Passing

## 1. Context & Problem Statement
The forecasting models rely on Numerical Weather Prediction (NWP) data. Our requirements evolved from using a single NWP source to needing support for zero, one, or multiple different NWP sources (e.g., deterministic vs. ensemble, different providers) simultaneously. We needed a robust way to configure which NWPs to use and pass their data to the models without causing column name collisions (e.g., multiple NWPs providing a `temperature` column) or requiring complex, hardcoded merge logic in the feature engineering steps.

## 2. Options Considered

### Option A: Pre-merged "Mega-DataFrame"
* **Description:** Upstream data processing merges all requested NWP sources into a single massive Polars DataFrame before passing it to the ML models.
* **Why it was rejected:** Hard to manage column collisions dynamically. It forces outer/inner join decisions prematurely and creates a very wide, sparse dataframe if NWP timestamps don't perfectly align.

### Option B: Hardcoded NWP Parameters
* **Description:** Models explicitly expect `nwp_1` and `nwp_2` arguments.
* **Why it was rejected:** Extremely inflexible. Changing from 1 to 3 NWPs requires changing function signatures across the entire ML pipeline and interface.

### Option C: Enum List in Config & Dict of DataFrames
* **Description:** 
  1. Configuration defines a list of requested NWP sources using an `NwpModel` Enum (parsed via Pydantic).
  2. The orchestration layer loads the data and passes it to the `BaseForecaster` interface as a dictionary: `dict[NwpModel, pl.DataFrame]`.
  3. Inside the forecaster's feature engineering logic, columns are automatically prefixed with the Enum's name (e.g., `NwpModel.ECMWF.value + "_temperature"`) before joining or feature extraction.

## 3. Decision
We chose **Option C: Enum List in Config & Dict of DataFrames**. This provides the most flexible and robust architecture for handling variable NWP inputs.

## 4. Consequences
* **Positive:** Safe from column collisions due to dynamic prefixing based on the Enum name.
* **Positive:** Highly flexible. The system easily scales to N weather models simply by updating the Hydra YAML list and having the data available.
* **Negative/Trade-offs:** Model internals must loop over the dictionary to extract features and handle the merging of multiple dynamic NWP sources, slightly increasing the complexity of the feature engineering step.
