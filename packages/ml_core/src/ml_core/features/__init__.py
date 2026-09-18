"""Feature engineering package.

Four terms carry the rest of this docstring. *NWP* is numerical weather prediction: gridded
weather-model output, published as runs, each run initialised at an ``nwp_init_time`` and
predicting many future ``valid_time``s. The package runs in two *join modes* — bulk training over
every NWP run in the input, and single-run inference against one named run. A *lag feature* is a
value observed a fixed number of hours before the time being forecast, and a lag is *leaky* when
that observation would not yet exist at the moment the forecast is issued. The *dual-strategy
join* answers a weather lag from the row's own NWP run for a target time still in the future, and
from the freshest earlier run for a target time already past.

Public interface:

- ``FeatureEngineer`` — abstract base; implement to swap the full feature pipeline.
- ``TabularFeatureEngineer`` — default implementation: nearest-cell NWP spatial join followed by
  the declarative tabular pipeline.
- ``NWP_PUBLICATION_DELAY_HOURS`` — hours after an NWP run's ``init_time`` before we treat that
  run as usable. Public because the constant is the default of three public signatures:
  ``FeatureEngineer.engineer``, ``TabularFeatureEngineer.engineer``, and
  ``ml_core.production_helpers.select_nwp_init_time``. A caller reasoning about any of those
  three signatures needs a name to reach the constant by.

That public interface lives in the two public sub-modules below, plus the private ``_nwp``. This
package re-exports ``NWP_PUBLICATION_DELAY_HOURS`` from ``_nwp``. The two public sub-modules are:

- ``feature_engineer`` — the ``FeatureEngineer`` abstract base, plus ``DEFAULT_LOCAL_TIMEZONE``,
  the IANA zone the local-time features are computed in. The zone lives on the interface rather
  than on the default implementation. A ``FeatureEngineer`` forecasting another region can
  therefore override the zone through the same ``engineer()`` call every production caller
  already makes.
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
