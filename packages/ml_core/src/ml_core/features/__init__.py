"""Feature engineering package.

Public interface:

- ``FeatureEngineer`` — abstract base; implement to swap the full feature pipeline.
- ``TabularFeatureEngineer`` — default implementation: nearest-cell NWP spatial join followed
  by the declarative tabular pipeline.
- ``NWP_PUBLICATION_DELAY_HOURS`` — hours after an NWP run's ``init_time`` before we treat that
  run as usable. Public because the constant is the default of three public signatures:
  ``FeatureEngineer.engineer``, ``TabularFeatureEngineer.engineer``, and
  ``ml_core.production_helpers.select_nwp_init_time``. A caller reasoning about any of those three
  signatures needs a name to reach the constant by.

Private sub-modules (importable by tests, not part of the public API):

- ``_parsed_features`` — typed feature descriptors and ``ParsedFeatures`` parser.
- ``_nwp`` — NWP upsampling, processing, and power/NWP join helpers.
- ``_lags`` — power-lag, weather-lag (dual-strategy), and leaky-lag nullification.
- ``tabular_feature_engineer`` — ``_engineer_features`` orchestrator and tabular pipeline helpers.
"""

from ml_core.features._nwp import NWP_PUBLICATION_DELAY_HOURS
from ml_core.features.feature_engineer import FeatureEngineer
from ml_core.features.tabular_feature_engineer import TabularFeatureEngineer

__all__ = ["NWP_PUBLICATION_DELAY_HOURS", "FeatureEngineer", "TabularFeatureEngineer"]
