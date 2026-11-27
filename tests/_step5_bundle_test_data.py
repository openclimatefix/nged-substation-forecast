"""A synthetic step-5 bundle with random weights, for the forecaster's and the asset's tests.

Importable by bare name via the ``pythonpath = ["tests"]`` pytest setting, from the package's own
test suite and from the root integration tests alike.
"""

import json
from pathlib import Path
from typing import Any, Final, TypedDict

import numpy as np
from substation_demand_forecaster.features import B1_COLUMNS, STEP5_COLUMNS

N_PRIMARIES = 2
N_KNOTS = 4
LAG_FREE = 37
EMBEDDING = 8


class PrimaryEntry(TypedDict):
    """One Primary of a bundle's `primaries.json`."""

    primary_index: int
    time_series_id: int
    unit: str
    h3_res_5: int
    latitude: float
    longitude: float
    effective_capacity_mw: float
    capacity_mw: float


PRIMARIES: Final[list[PrimaryEntry]] = [
    {
        "primary_index": 0,
        "time_series_id": 101,
        "unit": "MVA",
        "h3_res_5": 599422966022799359,
        "latitude": 53.14,
        "longitude": 0.31,
        "effective_capacity_mw": 18.0,
        "capacity_mw": 18.0,
    },
    {
        "primary_index": 1,
        "time_series_id": 102,
        "unit": "MW",
        "h3_res_5": 599423194729807871,
        "latitude": 52.98,
        "longitude": -0.03,
        "effective_capacity_mw": 6.0,
        "capacity_mw": 6.0,
    },
]

SHARED = {
    "native_origin_hour_utc": 9,
    "publication_delay_hours": 9.0,
    "max_lead_hours_trained_and_stored": 339.0,
    "max_lead_hours_ens_summaries": 351.0,
    "grid_start_utc": "2019-09-01T00:00:00+00:00",
    "grid_step_minutes": 30,
    "calendar_timezone": "Europe/London",
    "holidays": {
        "calendar": "holidays.country_holidays('GB', subdiv='ENG')",
        "nearest_holiday_clip_days": 7,
        "christmas_clip_days": 14,
        "covid_first_day": "2020-03-23",
        "covid_last_day": "2021-07-19",
    },
}

STEP5 = {
    "input_clip_sd": 5.0,
    "lead_scale_hours": 336.0,
    "lags": {
        "cap_steps": 48,
        "week_steps": 336,
        "clip_sd": 5.0,
        "age_scale_hours": 336.0,
        "missing_age_code": {"day": 1.1428571428571428, "week": 1.5},
        "std_floor": 1e-06,
    },
    "column_slices": {
        "weather": [0, 7],
        "spread": [7, 14],
        "datetime": [14, 35],
        "lead": [35, 37],
        "day_lag": [37, 40],
        "week_lag": [40, 43],
        "b1": [46, 60],
    },
    "lag_grid_index_columns": [43, 44],
    "knot_position_column": 45,
    "n_b1": 14,
    "mixture_components": 3,
    "log_std_clamp": [-3.0, 3.0],
    "quantile_levels": [0.01, 0.02, 0.05, 0.1, 0.2, 0.35, 0.5, 0.65, 0.8, 0.9, 0.95, 0.98, 0.99],
    "bisection_range": 40.0,
    "bisection_steps": 40,
    "expected_weather": {"x37_positions": [*range(7), *range(14, 35)]},
    "b1_constants": {
        "cold_snap_threshold_c": 5.0,
        "cold_snap_cap_days": 10.0,
        "lookback_steps": 576,
        "min_clear_sky_w_m2": 20.0,
        "max_clear_sky_index": 1.5,
        "trailing_windows_steps": [6, 12, 48],
        "cams_columns_source": "arm drop",
    },
    "offset_knots": {"first_knot_month": "2019-09"},
}

_SHAPES = {
    "embedding.weight": (N_PRIMARIES, EMBEDDING),
    "weather.0.weight": (64, 29),
    "weather.0.bias": (64,),
    "weather.3.weight": (32, 64),
    "weather.3.bias": (32,),
    "datetime.0.weight": (64, 29),
    "datetime.0.bias": (64,),
    "datetime.3.weight": (32, 64),
    "datetime.3.bias": (32,),
    "lag.0.0.weight": (64, 83),
    "lag.0.0.bias": (64,),
    "lag.0.3.weight": (32, 64),
    "lag.0.3.bias": (32,),
    "mixer_ft.0.weight": (64, 145),
    "mixer_ft.0.bias": (64,),
    "mixer_ft.3.weight": (64, 64),
    "mixer_ft.3.bias": (64,),
    "mixer_ft.6.weight": (9, 64),
    "mixer_ft.6.bias": (9,),
    "lag_id.weight": (2, EMBEDDING),
    "knots_old": (N_PRIMARIES, N_KNOTS - 1, 1),
    "knot_new": (N_PRIMARIES, 1),
}


def _write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj))


def write_bundle(root: Path, *, manifest_overrides: dict | None = None) -> Path:
    """Write a synthetic two-Primary step-5 bundle with random weights under `root`."""
    step5 = root / "components" / "step5"
    weights = []
    rng = np.random.default_rng(0)
    for seed in (0, 1):
        path = step5 / "weights" / f"seed{seed}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        # `Any` values, because `ty` checks a `**` mapping of arrays against `allow_pickle: bool`.
        arrays: dict[str, Any] = {
            name: (rng.standard_normal(shape) * 0.1).astype(np.float32)
            for name, shape in _SHAPES.items()
        }
        np.savez(path, **arrays)
        weights.append(str(path.relative_to(root)))
    manifest = {
        "bundle_format_version": "2",
        "refit_month": "2019-12",
        "model_release": 0,
        "research_git": {"commit": "0" * 40},
        "components": {
            "step5": {
                "kind": "demand_network_step5",
                "seeds": [0, 1],
                "files": {
                    "columns": "components/step5/columns.json",
                    "standardisers": "components/step5/standardisers.json",
                    "offset_knots": "components/step5/offset_knots.json",
                    "weights": weights,
                },
            }
        },
        "shared_files": {"config": "config.json", "primaries": "primaries.json"},
    } | (manifest_overrides or {})
    _write_json(root / "manifest.json", manifest)
    _write_json(root / "config.json", {"shared": SHARED, "components": {"step5": STEP5}})
    _write_json(root / "primaries.json", {"primaries": PRIMARIES})
    _write_json(
        step5 / "columns.json", [{"index": i, "name": n} for i, n in enumerate(STEP5_COLUMNS)]
    )
    _write_json(step5 / "offset_knots.json", {"n_knots": N_KNOTS})
    _write_json(
        step5 / "standardisers.json",
        {
            "x37": {"mean": [0.0] * LAG_FREE, "std": [1.0] * LAG_FREE},
            "b1": {"mean": [0.0] * len(B1_COLUMNS), "std": [1.0] * len(B1_COLUMNS)},
            "lag": {"day": {"mean": 0.5, "std": 0.25}, "week": {"mean": 0.5, "std": 0.25}},
            "target": {
                "primary_index": list(range(N_PRIMARIES)),
                "mean": [0.6, 0.4],
                "std": [0.15, 0.1],
            },
        },
    )
    return root
