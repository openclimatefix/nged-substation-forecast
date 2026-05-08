# Geo Package

This package contains generic geospatial logic and data for the NGED substation forecast project.

## Contents

- `geo.h3`: H3-related utilities, including grid weight computation.
- `assets/`: Generic geospatial assets like GeoJSON files.

![Map of Great Britain using H3 resolution 5 hexagons](assets/map-of-Great-Britain-H3-resolution-5.png)

## Purpose

The `geo` package is designed to decouple generic geospatial operations from dataset-specific ingestion logic (e.g., ECMWF data processing in `dynamical_data`). This ensures that any package in the workspace can perform spatial transformations, such as mapping latitude/longitude grids to H3 hexagons, without depending on heavy or unrelated packages.

