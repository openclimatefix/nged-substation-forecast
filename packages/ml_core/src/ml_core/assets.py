"""Dagster asset vocabulary for the ML pipeline."""

from enum import Enum


class FeatureAsset(str, Enum):
    """Source of truth for all available upstream Dagster assets used as features.

    This Enum allows us to map Pydantic field names in our DataRequirements classes
    directly to Dagster asset names, enabling the 'Write-Once' pipeline.
    """

    WEATHER_CERRA = "weather_cerra"
    SUBSTATION_SCADA = "substation_scada"
    GRID_TOPOLOGY_EDGES = "grid_topology_edges"
    # Add more assets here as they become available in the Dagster graph
