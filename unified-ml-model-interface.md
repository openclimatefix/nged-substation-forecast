# System Design: The Unified Forecasting Architecture

## 1. The Philosophy: Why This Complexity?
At first glance, using Abstract Base Classes, Pydantic type reflection, and custom Polars MLflow wrappers seems like over-engineering for a forecasting project.

However, this complexity is an upfront investment to solve the most common cause of burnout in MLOps: **The Hamster Wheel of Boilerplate**. In traditional setups, adding a new model architecture requires writing a new training script, a new data ingestion pipeline, and new validation logic.

By building a highly defensive, strictly-typed foundation using **Dagster, Polars, and Hydra**, we achieve a "Write-Once" pipeline. The MLOps code is written exactly once. From that point on, contributors can experiment with radically different ML models (from simple XGBoost trees to complex Spatial GNNs) by writing purely mathematical Python code and tweaking YAML files, entirely shielded from the underlying data engineering.

---

## 2. Separation of Concerns
To make the "Write-Once" pipeline work, we strictly separate *Structure*, *Flavor*, and *Delivery*:
* **Python (Structure):** Defines what data tables the model physically requires to execute (e.g., "I need weather and SCADA data").
* **Hydra (Flavor/Configuration):** Defines the hyperparameters, temporal splits, and feature selection (e.g., "Use the ECMWF weather provider, learning rate 0.01").
* **Dagster (Delivery):** Reads the Python structure and the Hydra flavor, queries the Delta Lake, and delivers the requested data as memory-efficient `LazyFrames`.

---

## 3. Data Contracts & Standard Vocabulary

To prevent silent data drift and pipeline typos, we enforce strict contracts at the orchestration layer and the dataframe layer.

### A. The Dagster Vocabulary
A single Enum acts as the source of truth for all available upstream Dagster assets.
```python
from enum import Enum

class FeatureAsset(str, Enum):
    WEATHER_CERRA = "weather_cerra"
    SUBSTATION_POWER_FLOWS = "substation_power_flows"
    GRID_TOPOLOGY_EDGES = "grid_topology_edges"
```

### B. The Polars Schema
We use `patito` to define the physical columns and types expected in our Delta Lake tables.

(this is just a simple example... in practice, we must re-use the existing Patito data contracts in the code)

```python
import patito as pt
from datetime import datetime

class WeatherContract(pt.Model):
    timestamp: datetime
    substation_id: str
    temperature_2m: float

class ScadaContract(pt.Model):
    timestamp: datetime
    substation_id: str
    active_power_mw: float
```

---

## 4. The Unified ML Interface (Factory vs. Artifact)

We maintain a strict separation between the **Trainer** (a heavy script handling LazyFrames, targets, and GPU clusters) and the **Model** (a lightweight, deployable mathematical artifact handling eager DataFrames). Unifying these would break MLflow serialization (pickling massive training states) and violate the physical reality that inference data does not contain future power targets.

### A. The Base Classes (`core/trainer.py` & `core/model.py`)

**1. The Pydantic Payload Contract**
```python
from pydantic import BaseModel, ConfigDict
from abc import ABC, abstractmethod
from typing import TypeVar, Generic, get_origin, get_args, List, Type
import mlflow
import polars as pl

class BaseDataRequirements(BaseModel):
    """Dynamically maps Pydantic fields to Dagster FeatureAssets."""
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @classmethod
    def get_required_assets(cls) -> List[FeatureAsset]:
        return [FeatureAsset(f) for f in cls.model_fields.keys()]

T_TrainReq = TypeVar("T_TrainReq", bound=BaseDataRequirements)
T_InferReq = TypeVar("T_InferReq", bound=BaseDataRequirements)
```

**2. The Trainer Factory (Needs Targets & LazyFrames)**
```python
class BaseTrainer(ABC, Generic[T_TrainReq]):
    @classmethod
    def data_requirements(cls) -> List[FeatureAsset]:
        """Auto-resolves Dagster dependencies from the Pydantic type hint."""
        for base in getattr(cls, "__orig_bases__", []):
            if get_origin(base) is BaseTrainer:
                return get_args(base)[0].get_required_assets()
        raise TypeError("Must subclass with a BaseDataRequirements parameter.")

    @abstractmethod
    def train(self, data: T_TrainReq, config: dict) -> mlflow.pyfunc.PythonModel:
        """Executes heavy training logic and returns the lightweight artifact."""
        pass
```

**3. The Inference Artifact (Needs Eager Data & NO Targets)**
```python
class BasePolarsModel(mlflow.pyfunc.PythonModel, ABC, Generic[T_InferReq]):
    def get_inference_type(self) -> Type[T_InferReq]:
        for base in getattr(self.__class__, "__orig_bases__", []):
            if get_origin(base) is BasePolarsModel:
                return get_args(base)[0]
        raise TypeError("Must subclass with an inference BaseDataRequirements.")

    def predict(self, context, model_input: dict[str, pl.DataFrame]) -> pl.DataFrame:
        """
        Satisfies MLflow's rigid string-based signature, but immediately parses
        the raw dictionary into our strictly-typed, validated Pydantic payload.
        """
        InferenceClass = self.get_inference_type()
        typed_data = InferenceClass(**model_input) # Validates data presence instantly
        return self._run_inference(typed_data)

    @abstractmethod
    def _run_inference(self, data: T_InferReq) -> pl.DataFrame:
        """The developer writes their math here, enjoying full IDE autocomplete."""
        pass
```

### B. Example: The Developer Experience (`models/xgboost.py`)

A contributor writes zero boilerplate. They define their data shapes and write the math.

```python
# 1. Define Training Data (Lazy, includes SCADA targets)
class XGBoostTrainData(BaseDataRequirements):
    weather_cerra: pt.LazyFrame[WeatherContract]
    substation_scada: pt.LazyFrame[ScadaContract]

# 2. Define Inference Data (Eager, NO SCADA targets)
class XGBoostInferenceData(BaseDataRequirements):
    weather_cerra: pt.DataFrame[WeatherContract]

# 3. Implement the Model Artifact
class XGBoostPolarsWrapper(BasePolarsModel[XGBoostInferenceData]):
    def __init__(self, booster):
        self.booster = booster

    def _run_inference(self, data: XGBoostInferenceData) -> pl.DataFrame:
        # 🎉 PERFECT IDE TYPE HINTING 🎉
        data.weather_cerra.cast().validate() # Patito schema validation
        features = data.weather_cerra.to_pandas() # Or native Polars math
        return self.booster.predict(features)

# 4. Implement the Trainer Factory
class XGBoostTrainer(BaseTrainer[XGBoostTrainData]):
    def train(self, data: XGBoostTrainData, config: dict) -> BasePolarsModel:
        joined_df = data.substation_scada.join(
            data.weather_cerra, on=["timestamp", "substation_id"]
        ).collect() # DAG pushdown optimization triggers here

        model = xgboost.train(params=config["hyperparameters"], dtrain=...)
        return XGBoostPolarsWrapper(model)
```

---

## 5. Hydra Configuration & MLflow Tracking

Hydra injects the parameters into the pipeline.
**Example (`conf/model/xgboost.yaml`):**
```yaml
model_name: "xgboost_baseline"
trainer_class: "src.models.xgboost.XGBoostTrainer"
data_split:
  train_start: "2019-01-01"
  train_end: "2022-12-31"
  test_start: "2023-01-01"
  test_end: "2023-12-31"
features:
  nwp_provider: "ecmwf"
hyperparameters:
  learning_rate: 0.01
```

**Saving to MLflow:**
When Dagster runs, it logs the entire Hydra dictionary to MLflow, ensuring absolute reproducibility. It then registers the model under the `model_name` (e.g., "xgboost_baseline"). MLflow automatically handles the version increments.

---

## 6. Dagster: The "Write-Once" Orchestrator

We use exactly **one** Dagster asset for training, partitioned statically by the model architectures defined in our Hydra YAML files.

```python
from dagster import asset, StaticPartitionsDefinition
import importlib

model_partitions = StaticPartitionsDefinition(["xgboost_baseline", "st_gnn"])

@asset(partitions_def=model_partitions)
def train_model(context, weather_cerra, substation_scada, grid_topology_edges):
    model_name = context.partition_key
    hydra_cfg = load_hydra_config(model_name)

    # 1. Dynamically load the Trainer Class
    module_name, class_name = hydra_cfg["trainer_class"].rsplit(".", 1)
    TrainerClass = getattr(importlib.import_module(module_name), class_name)

    # 2. Map available LazyFrames
    available_assets = {
        FeatureAsset.WEATHER_CERRA: weather_cerra,
        FeatureAsset.SUBSTATION_SCADA: substation_scada,
        FeatureAsset.GRID_TOPOLOGY_EDGES: grid_topology_edges,
    }

    # 3. Filter and Build Payload based on Python Structure & Hydra Config
    payload_dict = {}
    for req in TrainerClass.data_requirements():
        lf = available_assets[req]
        # Temporal slicing using Polars pushdown (The model is ignorant of this!)
        lf = lf.filter(pl.col("timestamp").is_between(
            hydra_cfg["data_split"]["train_start"],
            hydra_cfg["data_split"]["train_end"]
        ))
        payload_dict[req.value] = lf

    # 4. Train and Log
    trainer = TrainerClass()
    pydantic_payload = trainer.__orig_bases__[0].__args__[0](**payload_dict)

    with mlflow.start_run() as run:
        mlflow.log_dict(hydra_cfg, "config.yaml")
        trained_model = trainer.train(pydantic_payload, hydra_cfg)

        # Log to registry, bypassing MLflow's Pandas-centric signature engine
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=trained_model,
            registered_model_name=model_name,
            signature=None
        )
```

### Addressing the PyTorch Validation Split Caveat
Because the Dagster pipeline slices the `LazyFrame` bounds *before* passing it to the Trainer, the model is ignorant of the outer split (e.g., separating the 2024 hold-out test set).

However, PyTorch requires an internal Validation set for early stopping during the training loop. To handle this cleanly:
1. Dagster passes the `LazyFrame` covering *both* the `train_start` and internal `val_end` dates.
2. The PyTorch `Trainer.train()` method reads the `hydra_cfg["data_split"]` directly.
3. The Trainer creates two separate PyTorch `DataLoaders`, applying a secondary Polars `.filter()` to split the LazyFrame internally before streaming batches to the GPU.

---

## 7. Output Storage & Partitioning (Delta Lake)

The `evaluate_model` asset loads the `pyfunc` model, passes the Test set `LazyFrames` (collected into eager DataFrames) into it, and generates predictions.

**The Storage Strategy:**
We store all forecasts from all models in a **single Delta Lake table** (`evaluation_results.delta`). We do not generate custom CSV files or nested directories.

**The Schema:**
* `model_name` (e.g., "xgboost_baseline") - **This is the ONLY Partition Column.**
* `model_version` (int)
* `mlflow_run_id` (string hash)
* `code_version` (Git commit hash - denormalized for fast plotting)
* `timestamp`, `substation_id`, `p10`, `p50`, `p90`

**Why partition ONLY by `model_name`?**
Partitioning by Git hash or MLflow Run ID causes the "Small Files Problem" (thousands of 50KB files that crash Polars). By partitioning broadly by `model_name`, Delta Lake builds massive, highly optimized Parquet files. When we want to query a specific `run_id` for the leaderboard, Delta Lake uses its internal JSON statistics log (Z-Ordering/Data Skipping) to read only the specific byte-ranges containing that run, executing the query in milliseconds.
