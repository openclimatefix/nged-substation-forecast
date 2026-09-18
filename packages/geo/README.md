# Geo Package

Generic geospatial logic and data for the NGED substation forecast project: H3 spatial indexing and
the Great Britain boundary the numerical weather prediction (NWP) grid is clipped to. H3 is a grid
system that tiles the globe in hexagons at nested resolutions, each hexagon identified by an index.

## Map of Great Britain using H3 resolution 5 hexagons

![Map of Great Britain using H3 resolution 5
hexagons](assets/map-of-Great-Britain-H3-resolution-5.png)

## Purpose

The `geo` package decouples generic geospatial operations from dataset-specific ingestion logic,
such as the processing of European Centre for Medium-Range Weather Forecasts (ECMWF) data in
`dynamical_data`. Any package in the workspace can therefore perform a spatial transformation —
mapping a latitude/longitude grid to H3 hexagons, for example — without depending on heavy or
unrelated packages.

`compute_h3_grid_weights_for_boundary` accepts any boundary polygon, not only the Great Britain
shape this package ships. Accepting any boundary polygon adds no complexity here and means a new
region plugs into the same H3 gridding rather than forking the gridding code. Keeping the function
boundary-agnostic is [design principle
5](https://openclimatefix.github.io/nged-substation-forecast/design-philosophy/design-principles/#5-everything-around-the-model-is-general-purpose)
applied to this package.

Two neighbouring jobs are deliberately *not* here. The per-substation H3 index (`h3_res_5` on
`TimeSeriesMetadata`) is computed by `nged_data` straight from each substation's coordinates. The
spatial aggregation that consumes the grid weights computed here happens in `dynamical_data` at
ECMWF ingest. The weather grid is square and the H3 grid is hexagonal, so a grid weight is the
fraction of one hexagon that one square grid point covers, and each hexagon takes a weighted share
of the grid points overlapping it.

## Contents

- `h3` — `compute_h3_grid_weights_for_boundary()` and `compute_h3_grid_weights()`, which map the H3
  grid onto the regular lat/lon NWP grid. The sampling and snapping method is documented on the
  functions themselves.
- `great_britain.load` — `load_gb_boundary()`, which loads the Great Britain boundary polygon that
  the NWP grid is clipped to, from the packaged GeoJSON file.
