---
status: "Accepted"
date: "2026-04-10"
author: "Architect Agent"
tags: ["refactoring", "domain-language", "naming-conventions"]
---

# ADR-020: Rename `flows_30m` to `power_time_series`

## 1. Context & Problem Statement
The codebase previously used the variable name `flows_30m` to represent historical power measurements. This name was overly specific to a 30-minute resolution and didn't clearly convey that it was the source of truth for power time series data. As the project evolves, we need domain language consistency that represents generic power time series data, regardless of its temporal resolution. We needed to decide on a new, more robust naming convention.

## 2. Options Considered

### Option A: Keep `flows_30m`
* **Description:** Leave the variable name as is.
* **Why it was rejected:** It tightly couples the variable name to a specific temporal resolution (30 minutes). If we introduce 15-minute or 5-minute data, the name becomes misleading. It also doesn't clearly communicate that this is the primary power time series data.

### Option B: Rename to `power_flows`
* **Description:** Rename the variable to `power_flows`.
* **Why it was rejected:** While better than `flows_30m`, "flows" is slightly ambiguous in the context of energy forecasting (could be confused with network power flows rather than substation power measurements). It also lacks the explicit "time series" designation which is important for our data contracts.

### Option C: Rename to `power_time_series`
* **Description:** Rename the variable, parameter, and dictionary key to `power_time_series` throughout the codebase.

## 3. Decision
We chose **Option C: Rename to `power_time_series`**. This name is resolution-agnostic, clearly indicates the data type (time series), and aligns with our domain language for historical power measurements. It explicitly states that this data is the source of truth for historical power measurements.

## 4. Consequences
* **Positive:** Improved domain language consistency. The codebase is now more resilient to changes in data resolution. The intent of the variable is clearer to new engineers.
* **Negative/Trade-offs:** Required a large refactoring across multiple files, including tests, models, and feature engineering code, which introduced temporary code churn.
