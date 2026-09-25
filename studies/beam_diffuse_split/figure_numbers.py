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
    "ens_exploratory": 11,
    "own_beam": 12,
    "neighbours": 13,
    "per_generator": 14,
    "station_controls": 15,
    "station_stations": 16,
    "implied_capacity": 17,
}
"""Each figure's number on the page.

`leaderboard` and `contrasts` each combine four charts, one per row set: the
stacked chart draws the main, extra, ENS and station row sets as blocks. `weather_model_rivals`
combines the ICON-EU rivals chart with the UKV-against-ERA5 chart. `per_generator` holds the
station per-generator chart and the ENS per-generator chart.
"""

SVG_FIGURES: Final[dict[str, FigureKey | None]] = {
    "sunshine_leaderboard": "leaderboard",
    "sunshine_all_leaderboard": "leaderboard",
    "ens_past_solar_leaderboard": "leaderboard",
    "station_past_solar_leaderboard": "leaderboard",
    "sunshine_headline": "contrasts",
    "sunshine_all_contrasts": "contrasts",
    "ens_past_solar_planned_contrasts": "contrasts",
    "station_past_solar_planned_contrasts": "contrasts",
    "weather_product_domains": "domains",
    "sunshine_models_work_timeseries": "models_work_timeseries",
    "sunshine_models_work_error": "models_work_error",
    "sunshine_cams_breakdown": "cams_breakdown",
    "sunshine_new_products": "new_products",
    "sunshine_era5_by_year": "era5_by_year",
    "sunshine_icon_d2_leads": "icon_d2_leads",
    "sunshine_icon_eu_rivals": "weather_model_rivals",
    "sunshine_ukv_against_era5": "weather_model_rivals",
    "ens_past_solar_exploratory_contrasts": "ens_exploratory",
    "sunshine_own_beam": "own_beam",
    "sunshine_neighbours": "neighbours",
    "station_past_solar_per_generator": "per_generator",
    "station_past_solar_controls": "station_controls",
    "station_past_solar_stations": "station_stations",
    "station_past_solar_models_work": None,
    "sunshine_implied_capacity": "implied_capacity",
}
"""The figure each SVG in `docs/studies/assets/` feeds, by file stem; `None` marks a dropped SVG."""
