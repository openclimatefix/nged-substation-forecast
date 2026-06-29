"""Feature engineering package.

Public interface:

- ``FeatureEngineer`` — abstract base; implement to swap the full feature pipeline.
- ``TabularFeatureEngineer`` — default implementation: nearest-cell NWP spatial join followed
  by the declarative tabular pipeline.

Private sub-modules (importable by tests, not part of the public API):

- ``_parsed_features`` — typed feature descriptors and ``ParsedFeatures`` parser.
- ``_nwp`` — NWP upsampling, processing, and power/NWP join helpers.
- ``_lags`` — power-lag, weather-lag (dual-strategy), and leaky-lag nullification.
- ``tabular_feature_engineer`` — ``_engineer_features`` orchestrator and tabular pipeline helpers.
"""

from ml_core.features.feature_engineer import FeatureEngineer
from ml_core.features.tabular_feature_engineer import TabularFeatureEngineer

__all__ = ["FeatureEngineer", "TabularFeatureEngineer"]
