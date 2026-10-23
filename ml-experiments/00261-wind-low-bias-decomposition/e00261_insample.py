"""Share of rows below the ERA5-fed physics core's median, by quarter, in and out of sample.

The core and its residual quantiles are fitted on ERA5 before 2024-04-01 exactly as in
`e00261_decompose.py`. The residual quantiles are fitted in bins of the expected power, so the
in-sample share should sit near 0.5 by construction. Quarters from 2024-04 are out of sample.
The observed power ends at `TRAIN_END` (2025-06-17), so the scored year is covered by
`e00261_report.py` instead.

    uv run --no-sync python ml-experiments/00261-*/e00261_insample.py
"""

import json
import os
from datetime import UTC, datetime
from typing import Final

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import e00261_common as common  # noqa: E402
import e00261_core as core_module  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

BASE_END: Final = datetime(2024, 4, 1, tzinfo=UTC)


def main() -> None:
    """Write `insample.json`: per farm and quarter, the share below the median, kept rows only."""
    sites = common.site_table()
    table = common.training_table(sites=sites, era5=common.era5_rows(sites=sites))
    mask = common.idle_mask().with_columns(masked=pl.lit(value=True))
    table = table.join(mask, on=["time_series_id", "valid_time"], how="left").with_columns(
        masked=pl.col("masked").fill_null(value=False)
    )
    out: dict = {}
    for row in sites.iter_rows(named=True):
        farm_rows = table.filter(pl.col("time_series_id") == row["time_series_id"])
        train = farm_rows.filter(pl.col("valid_time") < BASE_END)

        def weather(frame: pl.DataFrame) -> core_module.WeatherArrays:
            return core_module.WeatherArrays(
                speed_10m=frame["wind_speed_10m"].to_numpy(),
                speed_100m=frame["wind_speed_100m"].to_numpy(),
                temperature_c=frame["temperature_c"].to_numpy(),
            )

        core = core_module.fit_core(
            weather=weather(train),
            target=train["target"].to_numpy(),
            with_idle_layer=row["site"] != "W2",
            seed=0,
        )
        residuals = core_module.ResidualQuantiles.fit(
            expected=core_module.predict_expected_power(core=core, weather=weather(train)),
            observed=train["target"].to_numpy(),
        )
        expected = core_module.predict_expected_power(core=core, weather=weather(farm_rows))
        median = residuals.quantiles(expected=expected)[:, common.MEDIAN_INDEX]
        scored = farm_rows.with_columns(
            median=pl.Series(median),
            below=pl.Series(farm_rows["target"].to_numpy() < median),
            quarter=pl.col("valid_time").dt.strftime("%Y-")
            + pl.col("valid_time").dt.quarter().cast(pl.String),
        )
        out[row["site"]] = {}
        for label, subset in (("kept", scored.filter(~pl.col("masked"))), ("all", scored)):
            out[row["site"]][label] = {
                r["quarter"]: float(r["share"])
                for r in subset.group_by("quarter")
                .agg(share=pl.col("below").mean())
                .sort("quarter")
                .iter_rows(named=True)
            }
            out[row["site"]][f"{label}_in_sample"] = float(
                np.mean(subset.filter(pl.col("valid_time") < BASE_END)["below"].to_numpy())
            )
            out[row["site"]][f"{label}_out_of_sample"] = float(
                np.mean(subset.filter(pl.col("valid_time") >= BASE_END)["below"].to_numpy())
            )
        print(
            row["site"],
            out[row["site"]]["kept_in_sample"],
            out[row["site"]]["kept_out_of_sample"],
            flush=True,
        )
    (common.WORK_DIR / "insample.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
