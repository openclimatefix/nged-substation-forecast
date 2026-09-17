import marimo

__generated_with = "0.23.5"
app = marimo.App(width="full")

with app.setup:
    from geo.great_britain.load import load_gb_boundary
    from geo.h3 import compute_h3_grid_weights_for_boundary
    from lonboard import H3HexagonLayer, Map


@app.cell
def _():
    # The England/Scotland/Wales polygon bundled in `geo`, buffered outwards by 0.25 degrees so
    # coastal substations and nearby islands still fall inside it. The buffer takes ~30 seconds.
    boundary = load_gb_boundary()
    boundary
    return (boundary,)


@app.cell
def _(boundary):
    # Positionally: a 0.25-degree regular latitude/longitude grid (ECMWF ENS's spacing), H3
    # resolution 5 for the cells themselves, and resolution 7 children to sample each cell's
    # overlap with the grid. One row per (H3 cell, grid point) pair that overlaps the boundary.
    h3_grid_weights = compute_h3_grid_weights_for_boundary(boundary, 0.25, 5, 7)
    h3_grid_weights
    return (h3_grid_weights,)


@app.cell
def _(h3_grid_weights):
    # A cell appears once per grid point it overlaps, so dedupe before drawing the hexagons.
    unique_h3 = h3_grid_weights["h3_index"].unique().sort()
    return (unique_h3,)


@app.cell
def _(unique_h3):
    # Every H3 cell the NWP pipeline aggregates weather onto, drawn faintly over a base map.
    Map(H3HexagonLayer(unique_h3, get_hexagon=unique_h3, opacity=0.1))
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
