# Dynamical Data Assets

This directory contains static assets used by the `dynamical_data` package.

## Files

- `ecmwf_scaling_params.csv`: Defines the min/max bounds and ranges for scaling NWP variables to `uint8` for storage. These parameters are derived from historical ECMWF ENS data to ensure that the full range of physical values can be represented within the 0-255 range.
- `gb_h3_grid.parquet`: A precomputed mapping of Great Britain's landmass to H3 indices at resolution 5. This is used for spatial joins and weighting during NWP processing.
- `england_scotland_wales.geojson`: The source GeoJSON used to generate the H3 grid.
