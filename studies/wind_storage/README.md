# Wind storage measurement

`measure_wind_storage.py` compares polar wind, Cartesian components, and components plus weighted
scalar-mean speed on the full runs in `manifest.json`. Its raw cache, Delta tables, column-byte
measurements, scalar-gap summaries, and provenance report live under `data/studies/wind_storage/`.

Run `fetch` once with explicit `--grid` and `--output` paths, starting with `--limit 1` for a pilot.
Run `write --representation polar`, `components`, and `scalar` in the corresponding experimental
schema worktrees against the same raw cache. Each variant calls the actual `write_nwp`. Run
`report` only after all arms finish; a pilot report must carry the same `--limit`.

The scalar arm adds grid-point magnitudes before H3 aggregation. The driver rejects different u/v
null footprints because those would require a separate reduction over paired valid points. It
reports compression on the sample, not archive-wide compression, and spatial speed gaps rather
than turbine power. No action writes to the serving archive.

The driver resolves missing dates using catalog availability before downloading any weather, and
records the requested and actual dates. Raw datasets, measured tables, and scalar-gap artifacts
carry content digests. Records also identify the imported schema, converter, writer, precision
code, lockfile, and driver. Reporting checks those identities against the requested manifest and
grid. A failed write is retried in a fresh study-owned table, keeping log overhead comparable.
