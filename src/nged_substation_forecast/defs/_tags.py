"""The ``layer`` asset tag, which records whether the live service needs an asset.

See [Core
Components](https://openclimatefix.github.io/nged-substation-forecast/architecture/overview/#core-components).
"""

from typing import Final

LAYER_TAG_KEY: Final[str] = "layer"
"""Dagster asset-tag key naming which side of the system an asset belongs to.

See [Core
Components](https://openclimatefix.github.io/nged-substation-forecast/architecture/overview/#core-components)
for why a tag, not a ``group_name`` or a ``kind``.
"""

PRODUCTION_LAYER_TAGS: Final[dict[str, str]] = {LAYER_TAG_KEY: "production"}
"""Tags for an asset the deployed service runs to produce forecasts.

Applied as ``@asset(tags=PRODUCTION_LAYER_TAGS)``.
"""

RESEARCH_LAYER_TAGS: Final[dict[str, str]] = {LAYER_TAG_KEY: "research"}
"""Tags for an asset the deployed service never needs: cross-validation, and model promotion.

See [Core
Components](https://openclimatefix.github.io/nged-substation-forecast/architecture/overview/#core-components)
for what the tag does and does not say about where an asset runs.

``research`` rather than ``rnd``, which reads as an abbreviation of "random", and rather than
``R&D``, which Dagster rejects as a tag value.
"""
