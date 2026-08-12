"""The ``layer`` asset tag, which records where an asset runs.

Splitting the asset graph this way lets whoever operates the live service filter the experiment
assets out of the Dagster UI — see
<https://openclimatefix.github.io/nged-substation-forecast/architecture/overview/>.
"""

from typing import Final

LAYER_TAG_KEY: Final[str] = "layer"
"""Dagster asset-tag key naming the machine an asset runs on.

A tag rather than a ``group_name`` because a group is exclusive and is Dagster's primary structural
axis: spending it here would foreclose grouping the same assets by pipeline stage later, whereas a
tag composes with any grouping we choose. Not a ``kind`` either — Dagster reserves those
(``dagster/kind/*``) for naming the technology an asset uses.
"""

PRODUCTION_LAYER_TAGS: Final[dict[str, str]] = {LAYER_TAG_KEY: "production"}
"""Tags for an asset the AWS production box runs, as ``@asset(tags=PRODUCTION_LAYER_TAGS)``.

Safe to share across decorators: Dagster copies the mapping into the asset definition rather than
holding a reference to it.
"""

RESEARCH_LAYER_TAGS: Final[dict[str, str]] = {LAYER_TAG_KEY: "research"}
"""Tags for an asset that only ever runs on a researcher's laptop.

``research`` rather than ``rnd`` because ``rnd`` reads as an abbreviation of "random", and rather
than ``R&D`` because Dagster rejects ``&`` in a tag value.
"""
