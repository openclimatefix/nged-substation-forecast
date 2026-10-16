"""Turn the extracted UKV shortwave into half-hour `u` for every ENS forecast origin.

For each daily ENS origin (the 00 UTC run plus 9 hours) of the training window and of experiment
00046's validation runs, the freshest UKV run available at the origin is chosen:

- `t54`: the 06 UTC run of the same day, assuming it is available 3 hours after initialisation
  (`UKV_LATENCY`), so it is just available at the 09:00 UTC origin; leads reach 51 h from the
  origin.
- `t54_late`: the 00 UTC run of the same day, a sensitivity check that assumes the 06 UTC run is
  not yet available; leads reach 45 h.
- `t120`: the 03 UTC run of the same day (the T120 files of the 03 and 15 UTC runs); leads reach
  117 h.

Two forecasts of `u` are built from each run:

- `u_ukv` uses the native steps (hourly to 48 h, 3-hourly after). The instantaneous flux is divided
  by the Haurwitz clear-sky flux at the same instant, the clear-sky index is interpolated linearly
  in time to the midpoint of each half-hour ending at the stamp, and multiplied by the clear-sky
  flux at that midpoint. The result is a half-hour mean ending at the stamp, the CAMS convention.
- `u_ukv3h` subsamples the run to the steps that are multiples of 3 h and passes the instantaneous
  values, labelled as if they were ENS's 3-hour window means, through the production resampler
  `_upsample_nwp_to_half_hourly` exactly as ENS is (which interpolates in clear-sky-index space
  and conserves each window's energy).
- `u_ukv3w` first turns the run into 3-hour window means ending at each multiple of 3 h (the
  trapezoid mean of the instantaneous values inside the window, which is hourly to 48 h and
  3-hourly after), the form in which ENS delivers its flux, and then passes them through the same
  resampler. It removes the half-window phase error that `u_ukv3h` carries.

Output: `work/ukv_u.parquet` with `variant`, `site`, `power_fcst_init_time`, `valid_time`, `u_ukv`,
`u_ukv3h`, `u_ukv3w` (null for `t54_late`).

    POLARS_MAX_THREADS=1 uv run python ml-experiments/00053-*/e00053_ukv_inputs.py
"""

import multiprocessing
import time
from datetime import UTC, datetime, timedelta
from typing import Final

import e00053_common as common
import numpy as np
import polars as pl
from ml_core.features._nwp import _upsample_nwp_to_half_hourly

SHORT_WAVE: Final[str] = "downward_short_wave_radiation_flux_surface"
VARIANT_RUN_OFFSET: Final[dict[str, timedelta]] = {
    "t54": timedelta(hours=3),
    "t54_late": timedelta(hours=9),
    "t120": timedelta(hours=6),
}
"""Hours from the chosen UKV run's initialisation to the forecast origin."""
PRODUCT_OF: Final[dict[str, str]] = {"t54": "t54", "t54_late": "t54", "t120": "t120"}
WORKERS: Final[int] = 4

_RUNS: dict[tuple[str, datetime, str], tuple[np.ndarray, np.ndarray]] = {}
_SERIES: dict[str, tuple[int, float, float]] = {}


def origins() -> list[datetime]:
    """Return every forecast origin of the training window and of the validation runs.

    Returns:
        Sorted UTC origins (run date at 09:00).
    """
    names = [path.stem for path in (common.E46_SOLAR / "ens_train").glob("init=*.parquet")]
    names += [path.stem for path in (common.E46_SOLAR / "ens_val").glob("init=*.parquet")]
    days = sorted({datetime.strptime(name[5:], "%Y-%m-%d").replace(tzinfo=UTC) for name in names})
    return [day + common.PUBLICATION_DELAY for day in days]


def window_means(*, steps: np.ndarray, ghi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the trapezoid mean over each 3-hour window ending at a multiple of 3 h.

    Args:
        steps: Sorted lead steps in hours with a value.
        ghi: Instantaneous irradiance at those steps.

    Returns:
        The window-end steps (3, 6, ...) whose start step also has a value, and the window means.
    """
    value = dict(zip(steps.tolist(), ghi.tolist(), strict=True))
    ends, means = [], []
    for end in range(3, int(steps.max()) + 1, 3):
        if end not in value or end - 3 not in value:
            continue
        inside = [step for step in range(end - 3, end + 1) if step in value]
        weights = np.array([0.5 if step in (end - 3, end) else 1.0 for step in inside])
        if len(inside) == 2:
            weights = np.array([0.5, 0.5])
        values = np.array([value[step] for step in inside])
        ends.append(end)
        means.append(float((weights * values).sum() / weights.sum()))
    return np.array(ends, dtype=np.int64), np.array(means, dtype=np.float32)


def process_origin(origin: datetime) -> pl.DataFrame:
    """Build every variant's half-hour `u` for one forecast origin and every site.

    Args:
        origin: The forecast origin.

    Returns:
        Columns `variant`, `site`, `power_fcst_init_time`, `valid_time`, `u_ukv`, `u_ukv3h`.
    """
    pieces = []
    for variant, offset in VARIANT_RUN_OFFSET.items():
        init = origin - offset
        radiation_rows = []
        for site, (time_series_id, latitude, longitude) in _SERIES.items():
            found = _RUNS.get((PRODUCT_OF[variant], init, site))
            if found is None:
                continue
            steps, ghi = found
            node_times = (
                np.datetime64(init.replace(tzinfo=None), "us") + steps.astype("timedelta64[h]")
            ).astype("datetime64[us]")
            last = node_times[-1].astype(datetime).replace(tzinfo=UTC)
            stamps = pl.datetime_range(
                origin + timedelta(minutes=30), last, interval="30m", time_zone="UTC", eager=True
            )
            half_hour = common.half_hour_ghi_from_instants(
                node_times=node_times,
                node_ghi=ghi.astype(np.float64),
                stamps=stamps,
                latitude=latitude,
                longitude=longitude,
            )
            u = common.u_from_half_hour_ghi(
                stamps=stamps, ghi=half_hour[None, :], latitude=latitude, longitude=longitude
            )[0]
            pieces.append(
                pl.DataFrame({"valid_time": stamps, "u_ukv": u.astype(np.float32)}).with_columns(
                    variant=pl.lit(variant),
                    site=pl.lit(site),
                    power_fcst_init_time=pl.lit(origin).cast(pl.Datetime("us", "UTC")),
                    u_ukv3h=pl.lit(None, dtype=pl.Float32),
                    u_ukv3w=pl.lit(None, dtype=pl.Float32),
                )
            )
            if variant != "t54_late":
                keep = (steps % 3 == 0) & (steps >= 3)
                window_end, window_mean = window_means(steps=steps, ghi=ghi)
                for name, end_steps, values in (
                    ("u_ukv3h", steps[keep], ghi[keep].astype(np.float32)),
                    ("u_ukv3w", window_end, window_mean),
                ):
                    radiation_rows.append(
                        pl.DataFrame(
                            {
                                "time_series_id": np.full(len(end_steps), time_series_id, np.int32),
                                "nwp_init_time": np.full(
                                    len(end_steps), np.datetime64(init.replace(tzinfo=None), "us")
                                ),
                                "valid_time": (
                                    np.datetime64(init.replace(tzinfo=None), "us")
                                    + end_steps.astype("timedelta64[h]")
                                ).astype("datetime64[us]"),
                                SHORT_WAVE: values,
                            }
                        ).with_columns(arm=pl.lit(name))
                    )
        for name in ("u_ukv3h", "u_ukv3w"):
            own = [frame for frame in radiation_rows if frame["arm"][0] == name]
            if own:
                pieces.append(
                    three_hourly(
                        rows=pl.concat(own).drop("arm"), origin=origin, variant=variant, name=name
                    )
                )
    return pl.concat(pieces, how="diagonal_relaxed") if pieces else pl.DataFrame()


def three_hourly(*, rows: pl.DataFrame, origin: datetime, variant: str, name: str) -> pl.DataFrame:
    """Pass 3-hourly UKV steps through the production ENS resampler and return `u`.

    Args:
        rows: Columns `time_series_id`, `nwp_init_time`, `valid_time`, and the short-wave column.
        origin: The forecast origin.
        variant: The variant name.
        name: The output column, `u_ukv3h` or `u_ukv3w`.

    Returns:
        Columns `variant`, `site`, `power_fcst_init_time`, `valid_time`, and the `name` column.
    """
    frame = rows.with_columns(
        pl.col("nwp_init_time").dt.replace_time_zone("UTC"),
        pl.col("valid_time").dt.replace_time_zone("UTC"),
    )
    coordinates = pl.DataFrame(
        {
            "time_series_id": [identifier for identifier, _, _ in _SERIES.values()],
            "latitude": [latitude for _, latitude, _ in _SERIES.values()],
            "longitude": [longitude for _, _, longitude in _SERIES.values()],
        }
    )
    half_hourly = _upsample_nwp_to_half_hourly(
        frame.lazy(), solar_coordinates=coordinates.lazy()
    ).collect()
    pieces = []
    for site, (time_series_id, latitude, longitude) in _SERIES.items():
        own = half_hourly.filter(
            (pl.col("time_series_id") == time_series_id) & (pl.col("valid_time") > origin)
        ).sort("valid_time")
        if own.is_empty():
            continue
        u = common.u_from_half_hour_ghi(
            stamps=own["valid_time"],
            ghi=own[SHORT_WAVE].to_numpy().astype(np.float64)[None, :],
            latitude=latitude,
            longitude=longitude,
        )[0]
        pieces.append(
            pl.DataFrame({"valid_time": own["valid_time"], name: u.astype(np.float32)})
            .with_columns(
                variant=pl.lit(variant),
                site=pl.lit(site),
                power_fcst_init_time=pl.lit(origin).cast(pl.Datetime("us", "UTC")),
            )
            .select("variant", "site", "power_fcst_init_time", "valid_time", name)
        )
    return pl.concat(pieces)


def load_runs() -> None:
    """Fill the module-level run and series tables (called once in each worker process)."""
    for row in common.series_table().iter_rows(named=True):
        _SERIES[row["site"]] = (row["time_series_id"], row["latitude"], row["longitude"])
    for product in ("t54", "t120"):
        table = pl.read_parquet(common.WORK_DIR / "ukv" / f"{product}.parquet")
        for key, group in table.sort("step").group_by(["init_time", "site"], maintain_order=True):
            _RUNS[(product, key[0], key[1])] = (
                group["step"].to_numpy().astype(np.int64),
                group["ghi"].to_numpy(),
            )


def main() -> None:
    """Build the half-hour `u` of every variant, origin, and site."""
    all_origins = origins()
    print(f"{len(all_origins)} origins", flush=True)
    started = time.monotonic()
    pieces = []
    context = multiprocessing.get_context("spawn")
    with context.Pool(WORKERS, initializer=load_runs) as pool:
        for count, frame in enumerate(pool.imap(process_origin, all_origins, chunksize=4), 1):
            if not frame.is_empty():
                pieces.append(frame)
            if count % 40 == 0:
                print(f"{count} origins in {time.monotonic() - started:.0f} s", flush=True)
    combined = pl.concat(pieces, how="diagonal_relaxed")
    combined = combined.group_by("variant", "site", "power_fcst_init_time", "valid_time").agg(
        pl.col("u_ukv").drop_nulls().first(),
        pl.col("u_ukv3h").drop_nulls().first(),
        pl.col("u_ukv3w").drop_nulls().first(),
    )
    common.write_atomically(frame=combined, target=common.WORK_DIR / "ukv_u.parquet")
    print(combined.group_by("variant").agg(pl.len(), pl.col("u_ukv").is_null().sum().alias("no_u")))


if __name__ == "__main__":
    main()
