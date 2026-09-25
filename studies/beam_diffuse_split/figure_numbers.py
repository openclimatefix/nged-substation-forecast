"""The figure numbers of the past-solar and past-wind study pages, in one place.

The page `past-weather/solar` holds 17 figures. `FIGURE_NUMBERS` gives each figure's
number by name, so renumbering a figure is one edit here. The redraw script reads the numbers from
this map when it draws each figure. The page `past-weather/wind` holds 15 figures,
numbered by `WIND_FIGURE_NUMBERS` in the same way.
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

WindFigureKey = Literal[
    "leaderboard",
    "contrasts",
    "domains",
    "models_work_timeseries",
    "models_work_error",
    "half_years",
    "per_generator",
    "icon_d2_leads",
    "icon_global_steps",
    "era5_by_year",
    "robustness",
    "reconciliation",
    "monthly_ratio",
    "time_of_day",
    "station_season",
]
"""The name of each figure on the past-wind page, whichever chart script draws it."""

WIND_FIGURE_NUMBERS: Final[dict[WindFigureKey, int]] = {
    "leaderboard": 1,
    "contrasts": 2,
    "domains": 3,
    "models_work_timeseries": 4,
    "models_work_error": 5,
    "half_years": 6,
    "per_generator": 7,
    "icon_d2_leads": 8,
    "icon_global_steps": 9,
    "era5_by_year": 10,
    "robustness": 11,
    "reconciliation": 12,
    "monthly_ratio": 13,
    "time_of_day": 14,
    "station_season": 15,
}
"""Each past-wind figure's number on the page.

`leaderboard` and `contrasts` draw the main, ICON-DREAM-EU, ECMWF, and station row sets as stacked
blocks, and `contrasts` also draws each block's planned contrasts in a lower panel. Each of
`models_work_timeseries`, `models_work_error`, and `per_generator` is drawn by more than one chart
script: the main row set's, the ECMWF row set's, and (for `per_generator`) the station row set's.
Each of those SVGs carries a letter, as `WIND_LETTERED_ROW_SETS` says.
"""

WIND_SVG_FIGURES: Final[dict[str, WindFigureKey]] = {
    "wind_leaderboard": "leaderboard",
    "wind_contrasts": "contrasts",
    "weather_product_domains": "domains",
    "wind_models_work_timeseries": "models_work_timeseries",
    "ens_hres_wind_models_work": "models_work_timeseries",
    "wind_models_work_error": "models_work_error",
    "ens_hres_wind_per_farm_error": "models_work_error",
    "wind_half_years": "half_years",
    "wind_per_generator": "per_generator",
    "ens_hres_wind_by_farm": "per_generator",
    "station_wind_by_farm": "per_generator",
    "wind_icon_d2_against_ukv": "icon_d2_leads",
    "wind_icon_global_steps": "icon_global_steps",
    "wind_era5_by_year": "era5_by_year",
    "ens_hres_wind_robustness": "robustness",
    "ens_hres_wind_reconciliation": "reconciliation",
    "ens_hres_wind_monthly_ratio": "monthly_ratio",
    "ens_hres_wind_split": "time_of_day",
    "station_wind_season": "station_season",
}
"""The figure each past-wind SVG in `docs/studies/assets/` feeds, by file stem."""

WindRowSetType = Literal["main", "ecmwf", "station"]
"""The row sets that draw one panel each of a lettered past-wind figure."""

WIND_LETTERED_ROW_SETS: Final[dict[WindFigureKey, tuple[WindRowSetType, ...]]] = {
    "models_work_timeseries": ("main", "ecmwf"),
    "models_work_error": ("main", "ecmwf"),
    "per_generator": ("main", "ecmwf", "station"),
}
"""The figures drawn by more than one SVG, each with the row set of each SVG in letter order.

The first row set is `a`, the second `b`, and the third `c`.
"""

WIND_ROW_SET_NAMES: Final[dict[WindRowSetType, str]] = {
    "main": "main rows",
    "ecmwf": "ECMWF rows",
    "station": "station rows",
}
"""What a lettered figure's title calls each row set."""


def wind_figure_number(*, key: WindFigureKey, row_set: WindRowSetType) -> str:
    """Return the number of one SVG of a lettered past-wind figure, such as `7b`.

    Args:
        key: The figure, one of `WIND_LETTERED_ROW_SETS`.
        row_set: The row set the SVG draws.

    Returns:
        The figure's number followed by the row set's letter.
    """
    letter = "abc"[WIND_LETTERED_ROW_SETS[key].index(row_set)]
    return f"{WIND_FIGURE_NUMBERS[key]}{letter}"


def wind_figure_title(*, row_set: WindRowSetType, title: str) -> str:
    """Return `title` led by the name of the row set its SVG draws, as `main rows - title`."""
    return f"{WIND_ROW_SET_NAMES[row_set]} - {title}"
