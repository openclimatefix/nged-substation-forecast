## ML Architecture & Pipeline Design: Energy Forecasting

From here: https://gemini.google.com/app/291f2b313ae1aa20

### 1. System Boundaries & Orchestration
* **Dagster's Role:** Acts as the high-level orchestrator and data router. It handles cross-fold validation and time-span filtering. It queries the heavily optimized Delta tables and passes a Polars `LazyFrame` directly into the model's interface.
* **Lazy Execution Preservation:** To prevent accidental materialization or pickling errors when passing `LazyFrame`s between Dagster steps, we will either use Dagster's `mem_io_manager` (for single-process runs) or a custom IO Manager that serializes/deserializes the Polars logical query plan (JSON).
* **Model Autonomy:** The ML models function as pure, self-contained units. Dagster does not engineer features; it hands raw, time-filtered data to the model, and the model handles its own preparation.

### 2. The Universal Model Interface (Template Method Pattern)
To ensure perfect symmetry between training and inference (preventing train-serve skew) and to protect junior developers from boilerplate errors, the ML pipeline relies on an Abstract Base Class (ABC) using the Template Method pattern.

* **Public Interface (Unbreakable):** The ABC defines standard `train(raw_data)` and `predict(raw_data)` methods. These internally trigger data validation, feature engineering, and `.collect()` materialization.
* **Protected Interface (Extensible):** Concrete implementations (e.g., `XGBoostEnergyModel`, `PyTorchEnergyModel`) only implement the pure math layer via `_fit_algo(X, y)` and `_predict_algo(X)`. 
* **Config Separation:** The `__init__` explicitly separates `selected_features: list[str]` (consumed by the base class for data prep) and `model_params: dict` (unpacked by the concrete subclass for algorithm config) to avoid kwarg pollution.

### 3. Feature Engineering Pattern
Feature engineering logic is decoupled from both Dagster and the concrete ML algorithms, relying on highly optimized Polars expressions.

* **The Kitchen (Polars Expressions):** Transformations are defined in a standalone `features.py` module using an Expression Registry for static features (e.g., `time_sin`) and Feature Factories for parameterized features (e.g., `build_lag_expr(window)`).
* **Execution:** The ABC's `_engineer_features()` method parses the requested features, gathers the corresponding `pl.Expr` objects from the registry/factories, and evaluates them simultaneously using a single `.with_columns()` call on the `LazyFrame`.

### 4. Data Contracts & Validation (Patito)
Patito is used strictly as a boundary validation layer ("Health Inspector"), not for complex ML transformations (`derive` rules are avoided for ML logic).

* **Input Contract:** Dagster passes data conforming to a `RawData` Patito model.
* **Output Contract:** Feature engineering returns an `AllFeatures` Patito model, where all possible engineered features are defined as `Optional` with strict bounds (e.g., ensuring sine/cosine bounds are exactly `[-1, 1]`).
* **Presence Validation:** Because `AllFeatures` uses `Optional` fields, the ABC performs a dynamic Polars schema check to ensure all features requested in the Hydra config were actually materialized, preventing silent failures.

### 5. Managing Duplication & Typos
To safely maintain the separation between the Hydra config (the request), the Polars Registry (the logic), and the Patito Schema (the validation), two safeguards are implemented:
* **Runtime Check:** The ABC `__init__` instantly checks the requested Hydra features against the Patito `AllFeatures` schema keys, crashing immediately on typos.
* **Enforcement Test:** A CI unit test asserts that every static feature defined in the Patito schema has a corresponding implementation in the Polars Expression Registry.

### 6. Inference State Management (Historical Padding)
To handle stateful time-series features (like lags and rolling windows) without building complex streaming architectures or breaking model purity:
* **Dagster Padding:** Dagster calculates the "max lookback" required by the Hydra config (e.g., 48 hours) and queries the Delta table for `[prediction_start - lookback, prediction_end]`.
* **Global Calculation:** The padded `LazyFrame` is passed to `predict()`. Polars calculates the lags and rolling features globally across the historical pad into the prediction horizon.
* **Trimming:** Right before passing the `X` matrix to `_predict_algo`, the ABC filters out the historical padding, retaining only the requested prediction horizon.
