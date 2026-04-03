import polars as pl
import patito as pt
from dagster import ConfigurableResource
from pydantic import Field
from contracts.data_schemas import H3GridWeights


class H3GridResource(ConfigurableResource):
    """Resource for loading and caching H3 grid weights."""

    h3_grid_weights_path: str = Field(description="Path to the H3 grid weights file (Parquet).")

    _cached_df: pt.DataFrame[H3GridWeights] | None = None

    def get_weights(self) -> pt.DataFrame[H3GridWeights]:
        """Load and return the H3 grid weights.

        Returns:
            pt.DataFrame[H3GridWeights]: The H3 grid weights.
        """
        if self._cached_df is None:
            # Load the parquet file
            df = pl.read_parquet(self.h3_grid_weights_path)
            # Validate against the contract
            self._cached_df = H3GridWeights.validate(df)
        return self._cached_df
