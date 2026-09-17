"""Feature engineering package.

Public interface:

- ``FeatureEngineer`` — abstract base; implement to swap the full feature pipeline.
- ``TabularFeatureEngineer`` — default implementation: nearest-cell NWP spatial join followed by
  the declarative tabular pipeline.
- ``NWP_PUBLICATION_DELAY_HOURS`` — hours after an NWP run's ``init_time`` before we treat that
  run as usable. Public because the constant is the default of three public signatures:
  ``FeatureEngineer.engineer``, ``TabularFeatureEngineer.engineer``, and
  ``ml_core.production_helpers.select_nwp_init_time``. A caller reasoning about any of those
  three signatures needs a name to reach the constant by.

The two sub-modules holding that public interface:

- ``feature_engineer`` — the ``FeatureEngineer`` abstract base, plus ``DEFAULT_LOCAL_TIMEZONE``,
  the IANA zone the local-time features are computed in. The zone lives on the interface rather
  than on the default implementation so a ``FeatureEngineer`` forecasting another region can
  override the zone through the same ``engineer()`` call every production caller already makes.
- ``tabular_feature_engineer`` — ``TabularFeatureEngineer`` and the ``_engineer_features``
  orchestrator the class delegates the tabular pipeline to. The module name carries no leading
  underscore because the class is public; every function the module defines is private.

Private sub-modules (importable by tests, not part of the public API):

- ``_parsed_features`` — typed feature descriptors and the ``ParsedFeatures`` parser.
- ``_nwp`` — NWP upsampling, processing, and the two power/NWP join modes.
- ``_lags`` — power lags, weather lags (the dual-strategy join), and leaky-lag nullification.
"""

from ml_core.features._nwp import NWP_PUBLICATION_DELAY_HOURS
from ml_core.features.feature_engineer import FeatureEngineer
from ml_core.features.tabular_feature_engineer import TabularFeatureEngineer

__all__ = ["NWP_PUBLICATION_DELAY_HOURS", "FeatureEngineer", "TabularFeatureEngineer"]
