"""Dagster asset vocabulary for the ML pipeline."""

from enum import Enum


class FeatureAsset(str, Enum):
    """Source of truth for all available upstream Dagster assets used as features.

    This Enum allows us to map Pydantic field names in our DataRequirements classes
    directly to Dagster asset names, enabling the 'Write-Once' pipeline.
    """

    WEATHER_ECMWF_ENS_0_25 = "ecmwf_ens_forecast"
    SUBSTATION_POWER_FLOWS = "live_primary_flows"
    SUBSTATION_METADATA = "substation_metadata"
    GRID_TOPOLOGY_EDGES = "grid_topology_edges"
