"""Export the fitted B1 core of each farm and its expected power over the whole ERA5 window.

Output: `work/b1_cores.json` (the fitted parameters per farm, labelled W1 to W3) and
`work/b1_expected_power.parquet` (`site`, `valid_time`, `expected_fraction`, `wind_speed_100m`),
one row per half-hour from 2019-09-01 to the end of the ERA5 file.

    uv run --no-sync python ml-experiments/00230-wind-physics-core-part-1/e00230_export_core.py
"""

import json
import os

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import e00230_common as common  # noqa: E402
import e00230_core as core_module  # noqa: E402
import e00230_step0 as step0  # noqa: E402
import polars as pl  # noqa: E402
import torch  # noqa: E402  # ty: ignore[unresolved-import]


def main() -> None:
    """Fit B1 per farm and write the parameters and the full-window expected power."""
    sites = common.site_table()
    era5 = common.era5_rows(sites=sites)
    train = common.training_table(sites=sites, era5=era5)
    parameters, pieces = {}, []
    for row in sites.iter_rows(named=True):
        series, farm = int(row["time_series_id"]), row["site"]
        fit_rows = train.filter(pl.col("time_series_id") == series)
        core = core_module.fit_core(
            weather=step0.weather_arrays(frame=fit_rows),
            target=fit_rows["target"].to_numpy(),
            with_idle_layer=farm != "W2",
            seed=0,
        )
        softplus = torch.nn.functional.softplus
        parameters[farm] = {
            "has_idle_layer": core.with_idle_layer,
            "hub_height_m": float(core.hub_height_m),
            "cut_in_m_s": float(core.cut_in_m_s),
            "rated_m_s": float(core.rated_m_s),
            "shape": float(core.raw_shape.sigmoid() * 4.0 + 1.0),
            "cut_out_m_s": float(core.raw_cut_out.sigmoid() * 15.0 + 20.0),
            "blur_m_s": float(softplus(core.raw_blur)),
            "scale_fraction_of_capacity": float(softplus(core.raw_scale)),
            "idle_floor_fraction": float(softplus(core.raw_idle_floor)),
            "dip_fraction": float(softplus(core.raw_dip)),
            "effective_capacity_mw": float(row["effective_capacity_mw"]),
        }
        weather_rows = era5.filter(pl.col("time_series_id") == series).sort("valid_time")
        expected = core_module.predict_expected_power(
            core=core, weather=step0.weather_arrays(frame=weather_rows)
        )
        pieces.append(
            weather_rows.select("valid_time", "wind_speed_100m").with_columns(
                site=pl.lit(farm), expected_fraction=pl.Series(expected)
            )
        )
    common.WORK_DIR.mkdir(parents=True, exist_ok=True)
    (common.WORK_DIR / "b1_cores.json").write_text(json.dumps(parameters, indent=1))
    pl.concat(pieces).select(
        "site", "valid_time", "expected_fraction", "wind_speed_100m"
    ).write_parquet(common.WORK_DIR / "b1_expected_power.parquet")
    print(json.dumps(parameters, indent=1))


if __name__ == "__main__":
    main()
