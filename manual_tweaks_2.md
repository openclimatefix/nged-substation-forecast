## In `packages/dynamical_data`

- Don't duplicate `def compute_h3_grid_weights` in `processing.py` and `precompute_h3_grid.py`! I
assume that function doesn't even need to exist in `processing.py`?
- I'd prefer `packages/dynamical_data/src/dynamical_data/scripts` to be moved to
`packages/dynamical_data/scripts/`.

### In `precompute_h3_grid.py`
- Can the function `compute_h3_grid_weights` use a Patito data contract
  for the input and/or output dataframe?
- Do we need the `if not geojson_path.exists()` check? Won't `open` raise a `FileNotFoundError` for
  us?
