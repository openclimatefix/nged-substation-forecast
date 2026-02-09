import marimo

__generated_with = "0.19.9"
app = marimo.App(width="full")

with app.setup:
    import marimo as mo
    from ipydeck import Deck, Layer, ViewState


@app.cell
def _():
    stops = Layer(
        type="ScatterplotLayer",
        data=[
            {"position": [-122.41, 37.784], "name": "Powell St", "ridership": 1200},
            {"position": [-122.419, 37.776], "name": "Civic Center", "ridership": 900},
            {"position": [-122.393, 37.776], "name": "Embarcadero", "ridership": 1500},
        ],
        get_position="@@=position",
        get_radius="@@=ridership",
        get_fill_color=[64, 170, 191],
        radius_scale=0.5,
        radius_units="meters",
        pickable=True,
        on_click=True,
    )

    d = Deck(
        layers=[stops],
        initial_view_state=ViewState(latitude=37.78, longitude=-122.41, zoom=10),
        tooltip={"text": "{name}"},
    )
    return (d,)


@app.cell
def _(d):
    mo.hstack([d, d.click], widths=[5, 1])
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
