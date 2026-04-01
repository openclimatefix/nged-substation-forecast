---
status: "Accepted"
date: "2026-03-27"
author: "Jack & opencode (gemini-3.1-pro-preview)"
tags: ["configuration", "pydantic", "hydra", "dagster"]
---

# ADR-001: Use Pydantic for Strictly Typed Hydra Configuration

## 1. Context & Problem Statement
The project uses Hydra for flexible YAML-based configuration. Historically, these configurations were passed into Dagster assets and ML models as raw dictionaries (`DictConfig` or `dict`). This lack of type safety at the system boundaries led to fragile code, where missing keys, typos in YAML files, or unexpected data types would only be discovered at runtime, often deep within the execution pipeline. We needed a robust way to validate configuration at the entry points of our Dagster assets and maintain strict typing throughout the ML pipeline.

## 2. Options Considered

### Option A: Raw Dictionaries / DictConfig
* **Description:** Continue passing Hydra's `DictConfig` or converting it to standard Python dictionaries and passing them to functions.
* **Why it was rejected:** No compile-time type checking, poor IDE auto-completion, and runtime errors occur deep in the call stack rather than failing fast at the system boundary.

### Option B: Python Dataclasses
* **Description:** Use standard library `@dataclass` to define configuration structures and instantiate them from the Hydra dictionaries.
* **Why it was rejected:** Dataclasses do not provide out-of-the-box deep validation, coercion, or parsing (e.g., converting a string list in YAML to an actual list of Enums) without writing custom `__post_init__` boilerplate.

### Option C: Pydantic Models at System Boundaries
* **Description:** Define configuration schemas using Pydantic `BaseModel`s. Parse and validate the Hydra dictionary immediately at the boundary of the Dagster assets before passing the strongly-typed Pydantic object to the underlying business logic.

## 3. Decision
We chose **Option C: Pydantic Models at System Boundaries**. We will use Pydantic models to parse and validate Hydra YAML configurations immediately inside Dagster assets. The resulting strongly-typed objects will be passed to down-stream ML components (`BaseForecaster`, `LocalForecasters`, etc.).

## 4. Consequences
* **Positive:** Fail-fast behavior for invalid configurations. Excellent IDE support and type hinting throughout the codebase. Reduces boilerplate validation code within the ML models themselves.
* **Positive:** Easy integration with our enum-based setups (like `NwpModel` lists).
* **Negative/Trade-offs:** Requires maintaining Pydantic models that mirror the structure of the Hydra YAML files, introducing a slight duplication of configuration structure definition.
