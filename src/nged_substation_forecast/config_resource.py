from dagster import ConfigurableResource
from contracts.config import Settings


class NgedConfig(ConfigurableResource, Settings):
    """Dagster resource for NGED substation forecast configuration."""

    pass
