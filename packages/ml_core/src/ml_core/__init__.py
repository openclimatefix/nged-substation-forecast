"""The model-agnostic forecasting core, shared by every `BaseForecaster` implementation.

Holds the `BaseForecaster` interface itself, the feature-engineering pipeline that builds a
model's input, and the metrics that score its output. Also holds the cross-validation and MLflow
run helpers that give a training run an identity, the reproducibility provenance stamped onto
that run, and the production helpers the live service loads and serves a promoted model through.
"""

from ml_core.base_forecaster import BaseForecaster, BaseForecasterConfig

__all__ = ["BaseForecaster", "BaseForecasterConfig"]
