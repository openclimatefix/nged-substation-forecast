# Geo Package

Generic geospatial logic and data for the NGED substation forecast project: H3 spatial indexing
and the Great Britain boundary the NWP grid is clipped to.

## Map of Great Britain using H3 resolution 5 hexagons

![Map of Great Britain using H3 resolution 5 hexagons](assets/map-of-Great-Britain-H3-resolution-5.png)

## Purpose

The `geo` package decouples generic geospatial operations from dataset-specific ingestion logic
(e.g., ECMWF data processing in `dynamical_data`), so any package in the workspace can perform a
spatial transformation — such as mapping a latitude/longitude grid to H3 hexagons — without
depending on heavy or unrelated packages.

`compute_h3_grid_weights_for_boundary` accepts any boundary geometry, not only the Great Britain
shape this package ships. That generality is what lets a future region reuse the same H3 gridding
rather than fork it, and it is
[design principle 5](https://openclimatefix.github.io/nged-substation-forecast/design-philosophy/design-principles/#5-everything-around-the-model-is-general-purpose)
applied to this package.

Two neighbouring jobs are deliberately *not* here. The per-substation H3 index (`h3_res_5` on
`TimeSeriesMetadata`) is computed by `nged_data` straight from each substation's coordinates, and
the spatial aggregation that consumes the grid weights computed here happens in `dynamical_data`
at ECMWF ingest.

## Contents

- `h3` — `compute_h3_grid_weights_for_boundary()` and `compute_h3_grid_weights()`, which map the
  H3 grid onto the regular lat/lon NWP grid. The sampling and snapping method is documented on the
  functions themselves.
- `great_britain.load` — `load_gb_boundary()`, which loads the Great Britain boundary polygon that
  the NWP grid is clipped to, from the packaged GeoJSON file.
