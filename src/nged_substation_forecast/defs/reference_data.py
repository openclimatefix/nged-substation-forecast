from geo.assets import gb_h3_grid_weights, uk_boundary

# Expose the assets to Dagster's load_from_defs_folder.
# These assets provide the foundational spatial reference data for the pipeline,
# including the UK boundary geometry and the H3 grid weights used for
# area-weighted averaging of weather data.
__all__ = ["uk_boundary", "gb_h3_grid_weights"]
