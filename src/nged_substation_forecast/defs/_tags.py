"""The ``layer`` asset tag, which records whether the live service needs an asset.

Splitting the asset graph this way lets whoever operates the live service filter the experiment
assets out of the Dagster UI — see
<https://openclimatefix.github.io/nged-substation-forecast/architecture/overview/#core-components>
for which assets carry each value.
"""

from typing import Final

LAYER_TAG_KEY: Final[str] = "layer"
"""Dagster asset-tag key naming which side of the system an asset belongs to.

A tag rather than a ``group_name`` because a group is exclusive and is Dagster's primary structural
axis: spending it here would foreclose grouping the same assets by pipeline stage later, whereas a
tag composes with any grouping we choose. Not a ``kind`` either — Dagster reserves those
(``dagster/kind/*``) for naming the technology an asset uses.
"""

PRODUCTION_LAYER_TAGS: Final[dict[str, str]] = {LAYER_TAG_KEY: "production"}
"""Tags for an asset the deployed service runs to produce forecasts.

Applied as ``@asset(tags=PRODUCTION_LAYER_TAGS)``.
"""

RESEARCH_LAYER_TAGS: Final[dict[str, str]] = {LAYER_TAG_KEY: "research"}
"""Tags for an asset the deployed service never needs: cross-validation, and model promotion.

The tag says the live service does not need the asset, not where the asset runs (see the module
docstring's link).

``research`` rather than ``rnd``, which reads as an abbreviation of "random", and rather than
``R&D``, which Dagster rejects as a tag value.
"""
