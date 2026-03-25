# Architecture Implementation Plan

This document outlines the detailed, step-by-step plan for applying the "Write-Once" MLOps architecture described in `unified-ml-model-interface.md`, followed by a critical analysis of a few "gotchas" and missing pieces in the design doc that need to be addressed before writing code.

## 🛠️ Implementation Plan

### Phase 1: The Foundation (`packages/ml_core`)
We will create a new package to house the unified interface and shared ML utilities. This keeps the core logic decoupled from specific model implementations and ensures dependency isolation (see Phase 2).
1. **Create `packages/ml_core/src/ml_core/assets.py`**: Define the `FeatureAsset` Enum (the "Dagster Vocabulary").
2. **Create `packages/ml_core/src/ml_core/trainer.py`**:
   - Implement `BaseDataRequirements` inheriting from Pydantic's `BaseModel` (using `ConfigDict(arbitrary_types_allowed=True)` for Polars/Patito support).
   - Implement the `BaseTrainer` abstract class.
3. **Create `packages/ml_core/src/ml_core/model.py`**:
   - Implement the `BasePolarsModel` abstract class, inheriting from `mlflow.pyfunc.PythonModel`.

### Phase 2: Dependency Isolation & Data Contracts (`packages/contracts`)
We must strictly separate `contracts` from `ml_core`. `contracts` defines the shape of the data and should be extremely lightweight. `ml_core` contains heavy MLOps dependencies (like `mlflow-skinny`). If we merged them, any simple script needing a data schema would be forced to install MLflow.
1. Review the existing Patito models in `packages/contracts/src/contracts/data_schemas.py`.
2. Ensure these contracts align perfectly with the `FeatureAsset` Enum names.
3. **Update `PowerForecast` Contract**: Simplify and clarify the schema for deterministic ensemble forecasts:
   - Rename `timestamp` to `valid_time`.
   - Add `power_fcst_init_time` (datetime the power forecast was initialised).
   - Add `nwp_init_time` (datetime the underlying weather forecast was initialised).
   - Rename `model_name` to `ml_model_name`.
   - Rename `forecast_year_month` to `power_fcst_init_year_month` (for Delta Lake partitioning).
   - Keep `ensemble_member` (int) and `MW_or_MVA` (float).

### Phase 3: Configuration Management (Hydra)
1. Add `hydra-core` to the `pyproject.toml` dependencies.
2. Create a `conf/` directory at the project root.
3. Set up the Hydra directory structure:
   - `conf/config.yaml` (Main entry point)
   - `conf/model/xgboost_baseline.yaml` (Specific model config defining `trainer_class`, `data_split`, etc.)

### Phase 4: Refactoring XGBoost & Extracting Shared Logic (`packages/xgboost_forecaster`)
We will refactor the existing XGBoost code to fit the new interface, and extract generic ML logic into `ml_core` so future models (like GNNs) can reuse it.
1. **Extract to `ml_core`**: Move generic feature engineering (e.g., cyclical time features), data splitting logic, and scaling/normalization from `xgboost_forecaster` into `ml_core`.
2. **Define Requirements**: Create `XGBoostTrainData` and `XGBoostInferenceData` (inheriting from `BaseDataRequirements`).
3. **Implement Model**: Rewrite the existing `XGBoostPyFuncWrapper` to inherit from `BasePolarsModel`.
4. **Implement Trainer**: Create `XGBoostTrainer` (inheriting from `BaseTrainer`).

### Phase 5: The "Write-Once" Orchestrator (`src/nged_substation_forecast/defs/`)
1. **Replace Specific Assets**: Remove the hardcoded `train_xgboost_model` asset in `xgb_assets.py`.
2. **Implement Unified Asset**: Create a generic `model_assets.py` containing the dynamically partitioned `train_model` asset.
3. **Data Slicing**: Implement the Polars `.filter()` pushdown based on the `data_split` defined in the Hydra config before passing the `LazyFrames` to the Trainer.
4. **MLflow Logging**: Ensure the model is logged using `mlflow.pyfunc.log_model` with `signature=None` and the Hydra config is logged as an artifact.

### Phase 6: Evaluation & Storage (Delta Lake)
1. Add `deltalake` (or `polars[deltalake]`) to the `pyproject.toml` dependencies.
2. Create an `evaluate_model` asset that loads the `pyfunc` model from MLflow, collects the test set `LazyFrames`, and generates predictions.
3. **Storage**: Write the resulting forecasts to a single Delta Lake table (`evaluation_results.delta`), partitioned strictly by `ml_model_name` and `power_fcst_init_year_month`.

### Phase 7: Documentation & Educational Comments
1. **README Updates**: Update the root `README.md` to explain the sub-package architecture, specifically highlighting the dependency isolation between `contracts` and `ml_core`. Create/update READMEs in `packages/contracts` and `packages/ml_core`.
2. **Code Comments**: Add liberal comments throughout the new code explaining the *why* behind the architecture (e.g., why we bypass MLflow type hints, why we use Pydantic `arbitrary_types_allowed`), and cross-reference how the Dagster factory interacts with the Trainer classes.

---

## ⚠️ Critique & Missing Pieces (The "Gotchas")

While the design is conceptually brilliant, there are a few technical hurdles in the pseudo-code that will break in practice. Here is how we will fix them during implementation:

### 1. The `__orig_bases__` Type Hint Extraction is Fragile
**The Issue:** The design doc uses `__orig_bases__` to dynamically extract the Pydantic requirement class from the generic type hint. This is a known anti-pattern in Python typing; it is an implementation detail that can break across Python versions or complex inheritance chains.
**The Fix:** We will require subclasses to explicitly define their requirements class as an attribute. It is much safer and more Pythonic:
```python
class BaseTrainer(ABC, Generic[T_TrainReq]):
    requirements_class: Type[T_TrainReq] # Explicit is better than implicit

    @classmethod
    def data_requirements(cls) -> List[FeatureAsset]:
        return cls.requirements_class.get_required_assets()
```

### 2. Hydra + Dagster Execution Clash
**The Issue:** The pseudo-code `hydra_cfg = load_hydra_config(model_name)` glosses over a major architectural clash. Hydra expects to own the `main()` entrypoint (via `@hydra.main`), but Dagster owns the execution graph.
**The Fix:** Inside the Dagster asset, we must use Hydra's Compose API to programmatically load the config for the specific partition without taking over the thread:
```python
from hydra import compose, initialize

def load_hydra_config(model_name: str):
    with initialize(version_base=None, config_path="../../conf"):
        return compose(config_name="config", overrides=[f"model={model_name}"])
```

### 3. The "Write-Once" Asset Signature Trap
**The Issue:** The proposed Dagster asset hardcodes `weather_cerra`, `substation_scada`, and `grid_topology_edges` as arguments. If a future model (e.g., a solar-specific model) needs a new dataset like `solar_irradiance`, we would have to edit the "Write-Once" pipeline signature, breaking the philosophy. Furthermore, Dagster needs to know dependencies at definition time to build the DAG; we cannot dynamically load inputs mid-execution without breaking the UI.
**The Fix:** Instead of a single partitioned asset with hardcoded inputs, we should use a **Dagster Factory Pattern** to generate the `@asset` dynamically based on the available Hydra configurations. This way, Dagster knows exactly which model needs which data before execution begins, and the DAG remains clean:
```python
def create_model_asset(model_name: str, hydra_cfg: dict):
    TrainerClass = load_trainer_class(hydra_cfg)
    # Dynamically map only the required assets for this specific model
    required_assets = {req.value: AssetIn(req.value) for req in TrainerClass.data_requirements()}

    @asset(name=f"train_{model_name}", ins=required_assets)
    def _train_model(context, **kwargs):
        # kwargs now contains exactly the LazyFrames this model requested
        ...
    return _train_model
```

### 4. Delta Lake "Small Files" Nuance
**The Issue:** The doc correctly identifies that partitioning by `run_id` causes the small files problem. However, partitioning ONLY by `model_name` might still lead to massive, unwieldy partitions over time as years of forecasts accumulate.
**The Fix:** We should partition by `ml_model_name` AND a temporal dimension (`power_fcst_init_year_month`). This keeps partition sizes manageable while still avoiding the small files problem, allowing Delta Lake's Z-Ordering to work optimally.

### 5. Missing Dependencies
We will need to add `hydra-core` and `deltalake` to the `pyproject.toml` before we begin. We will also ensure we use `mlflow-skinny` instead of the full `mlflow` package where possible to keep dependencies light.
