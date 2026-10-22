# ruff: noqa: E501
"""Build the common rows and every arm's 11 quantiles from the stored forecasts, and run the gates.

Rows: experiment 169's arm A scored-year rows at the six substations that the network (experiment
201) also covers, with a reading, outside the hand-filtered switching events, at leads of at most
168 hours. Each arm is stored as an `(n_rows, 11)` array in the rows' order.

| Arm | Quantiles |
|---|---|
| `a` | experiment 169's arm A |
| `a_s` | experiment 202's R2 (the two-number solar summary, rolling replay) |
| `a_h` | A plus `delta`, the mean over experiment 201's three seeds of the arm 2 median correction |
| `a_hs` | R2 plus `delta` |
| `h_own` | the mean over the three seeds of arm 2's own 11 quantiles |
| `drop_widen` | experiment 160's table alone (169's `ref`), plus `delta`, plus R2 minus A |
| `drop_r` | A's offsets about experiment 155's arm B mean (no recipe R), plus `delta`, plus R2 minus A |
| `drop_offset` | A's offsets about arm C's mean (no winter night offset), plus the same two terms |
| `a_s_drop_r`, `a_s_drop_offset` | the same two rows without `delta` |

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00220-*/e00220_build.py
"""

import numpy as np
import polars as pl
from studies.dp_calendar import bank_holiday_names

import e00220_common as common  # isort: skip


def read_a_frame() -> pl.DataFrame:
    """Return experiment 169's arm A and `ref` rows for the scored year, six substations."""
    pieces = []
    for site in common.SCORED_SITES:
        for arm in ("a", "ref"):
            frame = pl.read_parquet(common.E169_DIR / f"{arm}_scored_{site}.parquet").with_columns(
                site=pl.lit(site)
            )
            pieces.append((arm, frame))
    base = pl.concat([f for a, f in pieces if a == "a"])
    ref = pl.concat([f for a, f in pieces if a == "ref"]).select(
        common.KEYS + [pl.col(q).alias(f"ref_{q}") for q in common.QUANTILES]
    )
    return base.join(ref, on=common.KEYS, how="inner", validate="1:1")


def read_network() -> pl.DataFrame:
    """Return, per row, the mean over the three seeds of `delta` and of the 11 quantiles."""
    frames = [
        pl.read_parquet(common.E201_DIR / f"arm2_seed{seed}.parquet")
        for seed in range(common.N_SEEDS)
    ]
    for frame in frames:
        if frame["era"].unique().to_list() != ["ens"]:
            msg = "experiment 201's scored frame holds rows other than the ensemble era"
            raise ValueError(msg)
    stacked = pl.concat(frames)
    columns = ["delta", *common.QUANTILES]
    return stacked.group_by(common.KEYS).agg(
        [pl.col(c).mean().alias(f"net_{c}") for c in columns]
        + [pl.col("actual").first().alias("actual_net"), pl.len().alias("n_seeds")]
    )


def read_r2() -> tuple[pl.DataFrame, np.ndarray]:
    """Return experiment 202's scored frame keys (with the daylight kind) and R2's quantiles."""
    frame = pl.read_parquet(common.E202_DIR / "scored_frame.parquet")
    r2 = np.load(common.E202_DIR / "scored_r2.npy").astype(np.float64)
    if frame.height != r2.shape[0]:
        msg = "experiment 202's frame and R2 array differ in length"
        raise ValueError(msg)
    return frame.select([*common.KEYS, "kind", "usable"]).with_columns(
        r2_index=pl.int_range(pl.len())
    ), r2


def read_155_mean(*, arm: str) -> pl.DataFrame:
    """Return experiment 155's arm mean forecast for the scored year, six substations."""
    frames = [
        pl.read_parquet(common.E155_DIR / f"{arm}_scored_{site}.parquet")
        .with_columns(site=pl.lit(site))
        .select([*common.KEYS, pl.col("mean").alias(f"mean_{arm}")])
        for site in common.SCORED_SITES
    ]
    return pl.concat(frames)


def annotate(*, rows: pl.DataFrame, cache) -> pl.DataFrame:  # noqa: ANN001
    """Add the cold-day flag, the valid month, and the bank-holiday and Christmas flags."""
    pieces = []
    for site in common.SCORED_SITES:
        part = rows.filter(pl.col("site") == site)
        features = cache.sites[site]
        target = np.searchsorted(
            cache.end_times, part["valid_time"].dt.replace_time_zone(None).to_numpy()
        )
        days = cache.end_times.astype("datetime64[D]")
        codes = np.unique(days, return_inverse=True)[1]
        temperature = features.temperature_c.astype(np.float64)
        finite = np.isfinite(temperature)
        total = np.bincount(codes, weights=np.where(finite, temperature, 0.0))
        count = np.bincount(codes, weights=finite.astype(np.float64))
        daily = (total / np.maximum(count, 1))[codes]
        pieces.append(
            part.with_columns(
                masked=pl.Series(features.switching_mask[target]),
                cold2=pl.Series(np.nan_to_num(daily[target], nan=99.0) < common.COLD_DAY_C),
            )
        )
    bank = list(bank_holiday_names())
    local_date = pl.col("valid_time").dt.convert_time_zone("Europe/London").dt.date()
    return pl.concat(pieces).with_columns(
        valid_month=pl.col("valid_time").dt.month(),
        bank_holiday=local_date.is_in(bank),
        christmas=(
            ((local_date.dt.month() == 12) & (local_date.dt.day() >= 24))
            | ((local_date.dt.month() == 1) & (local_date.dt.day() <= 1))
        ),
    )


def main() -> None:
    """Join the stored forecasts, build every arm, run the gates, and write them."""
    cache = common.load_cache()
    a = read_a_frame()
    net = read_network()
    r2_keys, r2 = read_r2()
    gates: dict = {}

    rows = a.join(net, on=common.KEYS, how="inner", validate="1:1")
    gates["rows_a"] = a.height
    gates["rows_after_network_join"] = rows.height
    gates["max_abs_actual_diff_network"] = float(
        np.abs(rows["actual"].to_numpy().astype(np.float64) - rows["actual_net"].to_numpy()).max()
    )
    gates["seeds_per_row_min"] = int(rows["n_seeds"].to_numpy().min())
    rows = rows.join(r2_keys, on=common.KEYS, how="inner", validate="1:1")
    for arm in ("b", "c", "d"):
        rows = rows.join(read_155_mean(arm=arm), on=common.KEYS, how="left", validate="1:1")
    gates["rows_after_all_joins"] = rows.height
    gates["max_abs_mean_d_minus_a"] = float(
        np.abs(rows["mean_d"].to_numpy() - rows["mean"].to_numpy()).max()
    )
    gates["null_mean_b"] = int(rows["mean_b"].is_null().sum())
    gates["null_mean_c"] = int(rows["mean_c"].is_null().sum())
    rows = annotate(rows=rows, cache=cache)
    rows = rows.filter(
        ~pl.col("masked")
        & pl.col("actual").is_finite()
        & (pl.col("lead_hours") <= common.HORIZON_HOURS)
        & pl.col("mean_b").is_not_null()
        & pl.col("mean_c").is_not_null()
    ).with_columns(
        month=pl.col("power_fcst_init_time").dt.strftime("%Y-%m"),
        daylight=pl.col("kind") == "daylight",
    )
    gates["rows_scored"] = rows.height

    qa = rows.select(list(common.QUANTILES)).to_numpy().astype(np.float64)
    qref = rows.select([f"ref_{q}" for q in common.QUANTILES]).to_numpy().astype(np.float64)
    q_net = rows.select([f"net_{q}" for q in common.QUANTILES]).to_numpy().astype(np.float64)
    delta = rows["net_delta"].to_numpy().astype(np.float64)[:, None]
    r2_rows = r2[rows["r2_index"].to_numpy()]
    gates["max_abs_a_minus_r2_nondaylight"] = float(
        np.abs(r2_rows - qa)[~rows["daylight"].to_numpy()].max()
    )
    s_change = r2_rows - qa
    mean_a = rows["mean"].to_numpy().astype(np.float64)[:, None]
    base_r = rows["mean_b"].to_numpy().astype(np.float64)[:, None] - mean_a
    base_off = rows["mean_c"].to_numpy().astype(np.float64)[:, None] - mean_a

    arms = {
        "a": qa,
        "a_s": r2_rows,
        "a_h": qa + delta,
        "a_hs": r2_rows + delta,
        "h_own": q_net,
        "drop_widen": qref + delta + s_change,
        "drop_r": qa + base_r + delta + s_change,
        "drop_offset": qa + base_off + delta + s_change,
        "a_s_drop_r": qa + base_r + s_change,
        "a_s_drop_offset": qa + base_off + s_change,
    }
    crossed = {name: int((np.diff(q, axis=1) < 0).any(axis=1).sum()) for name, q in arms.items()}
    gates["rows_with_crossing_quantiles_before_sort"] = crossed
    common.ARM_DIR.mkdir(parents=True, exist_ok=True)
    for name, q in arms.items():
        common.write_npy_atomic(
            path=common.ARM_DIR / f"{name}.npy", array=np.sort(q, axis=1).astype(np.float32)
        )
    drop = [c for c in rows.columns if c.startswith(("ref_", "net_"))]
    rows.drop(drop).write_parquet(common.ARM_DIR / "rows.parquet")
    common.write_json_atomic(path=common.REPORT_DIR / "gates_build.json", content=gates)
    print(gates)


if __name__ == "__main__":
    main()
