"""Step 1 of experiment 00356: the frozen Stage 3 plant on the daily runs after the hindcast.

Passes the 51 members of every daily run after the hindcast through the plant (arm y2 of
experiment 00347).

Reads experiment 00346's member rows (daily 00 UTC runs from 2025-06-17 to 2026-10-23) and stores,
per farm, `inputs/keys_<farm>.parquet` (the key columns, in case order) and
`inputs/summary_<farm>.npy` (cases, 6): the median, mean, standard deviation, 10th and 90th
percentile of the 51 member powers, and the saturation share (experiment 00347's `summarise`).

    PYTHONPATH=<see e00356_common> uv run --no-sync python e00356_inputs.py W1 [W2 W3]
"""

import os
import sys
import time

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import e00303_common as c303  # noqa: E402  # isort: skip  # ty: ignore[unresolved-import]
import e00347_common as c347  # noqa: E402  # isort: skip  # ty: ignore[unresolved-import]
import e00347_plant as plant  # noqa: E402  # isort: skip  # ty: ignore[unresolved-import]
import e00347_s4_inputs as s4  # noqa: E402  # isort: skip  # ty: ignore[unresolved-import]
import e00356_common as common  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402


def run_farm(*, farm: str) -> None:
    """Store the plant summaries of one farm for every daily run after the hindcast."""
    started = time.time()
    common.INPUTS_DIR.mkdir(parents=True, exist_ok=True)
    sites = c303.site_table()
    series = int(sites.filter(pl.col("site") == farm)["time_series_id"][0])
    parameters = plant.PlantParameters.from_dict(c347.read_json(name="plant_stage3.json")[farm])
    prior = c347.prior_scales()[farm]
    keys_all, summaries = [], []
    for path in sorted(common.E346_MEMBERS.glob("init=*.parquet")):
        rows = pl.read_parquet(path).filter(pl.col("time_series_id") == series)
        rows = rows.sort(*c303.KEY_COLUMNS, "ensemble_member")
        if rows.height % s4.MEMBERS:
            msg = f"{path.name}: {rows.height} rows is not a multiple of {s4.MEMBERS}"
            raise ValueError(msg)
        if rows.height == 0:
            continue
        keys = rows.gather_every(s4.MEMBERS).select(c303.KEY_COLUMNS)
        members = rows["ensemble_member"].to_numpy().reshape(-1, s4.MEMBERS)
        if not (members == np.arange(s4.MEMBERS)).all():
            msg = f"{path.name}: members are not 0 to {s4.MEMBERS - 1} in order for every case"
            raise ValueError(msg)
        weather = s4.member_weather(rows=rows)
        power = s4.expected_in_chunks(parameters=parameters, weather=weather, prior_scale=prior)
        summaries.append(
            s4.summarise(power=power.reshape(-1, s4.MEMBERS), flat_top=parameters.scale)
        )
        keys_all.append(keys)
    pl.concat(keys_all).write_parquet(common.INPUTS_DIR / f"keys_{farm}.parquet")
    np.save(common.INPUTS_DIR / f"summary_{farm}.npy", np.concatenate(summaries))
    print(f"{farm}: {sum(len(k) for k in keys_all):,} cases, {time.time() - started:.0f} s")


def main() -> None:
    """Run the farms named on the command line (all three by default)."""
    for farm in sys.argv[1:] or list(common.FARMS):
        run_farm(farm=farm)


if __name__ == "__main__":
    main()
