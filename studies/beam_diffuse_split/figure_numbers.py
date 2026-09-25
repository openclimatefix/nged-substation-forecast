"""The figure numbers of the past-solar study page, in one place.

The page `weather-products-for-past-solar` holds 17 figures. `FIGURE_NUMBERS` gives each figure's
number by name, so renumbering a figure is one edit here. The redraw script reads the numbers from
this map when it draws each figure.
"""

from typing import Final, Literal

FigureKey = Literal[
    "leaderboard",
    "contrasts",
    "domains",
    "models_work_timeseries",
    "models_work_error",
    "cams_breakdown",
    "new_products",
    "era5_by_year",
    "icon_d2_leads",
    "weather_model_rivals",
    "ens_exploratory",
    "own_beam",
    "neighbours",
    "per_generator",
    "station_controls",
    "station_stations",
    "implied_capacity",
]
"""The name of each figure on the page, whichever chart script draws it."""

FIGURE_NUMBERS: Final[dict[FigureKey, int]] = {
    "leaderboard": 1,
    "contrasts": 2,
    "domains": 3,
    "models_work_timeseries": 4,
    "models_work_error": 5,
    "cams_breakdown": 6,
    "new_products": 7,
    "era5_by_year": 8,
    "icon_d2_leads": 9,
    "weather_model_rivals": 10,
    "neighbours": 11,
    "own_beam": 12,
    "ens_exploratory": 13,
    "station_controls": 14,
    "station_stations": 15,
    "per_generator": 16,
    "implied_capacity": 17,
}
"""Each figure's number on the page.

`leaderboard` and `contrasts` draw the main, extra, ENS and station row sets as stacked blocks, and
`contrasts` also draws each block's planned contrasts in a lower panel. `weather_model_rivals`
holds the ICON-EU rivals panels and the UKV-against-ERA5 panel. `per_generator` holds the ENS
per-generator panels and the station per-generator panels.
"""

SVG_FIGURES: Final[dict[str, FigureKey]] = {
    "sunshine_leaderboard": "leaderboard",
    "sunshine_contrasts": "contrasts",
    "weather_product_domains": "domains",
    "sunshine_models_work_timeseries": "models_work_timeseries",
    "sunshine_models_work_error": "models_work_error",
    "sunshine_cams_breakdown": "cams_breakdown",
    "sunshine_new_products": "new_products",
    "sunshine_era5_by_year": "era5_by_year",
    "sunshine_icon_d2_leads": "icon_d2_leads",
    "sunshine_weather_model_rivals": "weather_model_rivals",
    "sunshine_neighbours": "neighbours",
    "sunshine_own_beam": "own_beam",
    "ens_past_solar_exploratory_contrasts": "ens_exploratory",
    "station_past_solar_controls": "station_controls",
    "station_past_solar_stations": "station_stations",
    "station_past_solar_per_generator": "per_generator",
    "sunshine_implied_capacity": "implied_capacity",
}
"""The figure each SVG in `docs/studies/assets/` feeds, by file stem."""

SUPERSEDED_SVGS: Final[frozenset[str]] = frozenset(
    {
        "sunshine_all_leaderboard",
        "ens_past_solar_leaderboard",
        "station_past_solar_leaderboard",
        "sunshine_headline",
        "sunshine_all_contrasts",
        "ens_past_solar_planned_contrasts",
        "station_past_solar_planned_contrasts",
        "sunshine_icon_eu_rivals",
        "sunshine_ukv_against_era5",
        "station_past_solar_models_work",
    }
)
"""The SVGs no figure uses any more, still on disk because the past-solar page links them until
its prose is rewritten. No chart script draws them."""
