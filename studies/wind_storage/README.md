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

## Frozen measurement schemas

Create three worktrees from commit `e3bcb44c`, then cherry-pick `1bef88b2` and `ad2ea712`
into each. Leave the polar schema unchanged; apply `variants/components.patch` and
`variants/scalar.patch` to their corresponding worktrees and commit the changes. Do not apply
these patches to the current component schema. Use the root environment's Python with each
worktree's `packages/contracts/src`, `packages/dynamical_data/src`, and
`packages/delta_store/src` on `PYTHONPATH`.

```bash
python studies/wind_storage/measure_wind_storage.py fetch --grid GRID --output OUTPUT
python studies/wind_storage/measure_wind_storage.py write --grid GRID --output OUTPUT \
  --representation polar
python studies/wind_storage/measure_wind_storage.py report --grid GRID --output OUTPUT
python studies/wind_storage/summarize_wind_storage.py --grid GRID --output OUTPUT
```

Run `write` separately in all three matching schema worktrees, selecting the corresponding
representation each time, before running `report` in the frozen polar worktree. The summary
script runs from the current checkout and checks the completed measurement records against
`measurements.json`. Its `summary-provenance.json` hashes the records and generated summary.
Published tables are copied from `summary.md`; scalar-gap tables precede writer rounding.
Two independent scientific reviews checked the paired sample and raw reductions.

## Pending forecasting comparison

`leaderboard-reference.json` pins the polar code/table reference and recipe/CV hashes.
`register-interpolation-arm.yaml` uses the existing `register_experiment_job` in `register_only`
mode, with no model overrides. Registration was exercised in isolated local MLflow/Dagster
stores under `data/studies/wind_storage/experiments/`; no fit, score or serving cutover ran.
Re-register in the evaluation instance only with the matching component archive and pinned
recipe/CV files. Compare unchanged feature names on identical folds, seeds, scenarios and
evaluation keys, including 3–10 day horizon/type slices. Scoring remains pending the complete
component archive and matched polar baseline predictions.
