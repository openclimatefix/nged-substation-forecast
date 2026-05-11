import marimo

__generated_with = "0.23.5"
app = marimo.App(width="full")

with app.setup:
    import marimo as mo
    from lonboard import Map, H3HexagonLayer

    from geo.great_britain.load import load_gb_boundary
    from geo.h3 import compute_h3_grid_weights_for_boundary


@app.cell
def _():
    boundary = load_gb_boundary()
    boundary
    return (boundary,)


@app.cell
def _(boundary):
    h3_grid_weights = compute_h3_grid_weights_for_boundary(boundary, 0.25, 5, 7)
    h3_grid_weights
    return (h3_grid_weights,)


@app.cell
def _(h3_grid_weights):
    # Example: An Arrow table with H3 identifiers as a column
    layer = H3HexagonLayer(
        h3_grid_weights,
        get_hexagon=h3_grid_weights["h3_index"],
    )
    m = Map(layer)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
