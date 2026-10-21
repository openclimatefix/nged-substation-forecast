"""Step 0 of experiment 152: how much reverse flow is there at each MVA-only Primary?

Reads experiment 110's stored L0 predictions (the convex model fitted through the MVA meter), its
fitted PV capacity, the CAMS irradiance in its feature cache, and NGED's monitoring file. Writes
`work/step0_table.parquet` and `work/step0_rows.parquet`.

The sign probability of a half-hour is the normal probability that the signed flow is below zero,
`Phi(-p_hat / sigma)`, where `sigma` is the standard deviation of the site's day-ahead residuals
`actual - mean` on sunny-midday rows in the same year. It is a screening number: the residual of
the reading `|S|` is not the error of the signed flow, which is unobserved.
"""

import csv
import json
import math
from datetime import timedelta

import numpy as np
import polars as pl
from e00152_common import (
    CLEAR_SKY_INDEX_MINIMUM,
    EXPERIMENT_110_DIR,
    MIDDAY_FIRST_HALF_HOUR_UTC,
    MIDDAY_LAST_HALF_HOUR_UTC,
    MVA_ONLY_PRIMARIES,
    SUNNY_MONTHS,
    WORK_DIR,
    load_cache,
)
from studies.baselines import haurwitz_w_m2
from studies.dp_features import default_input_paths
from studies.solar import zenith

NGED_FLAG_BY_SITE: dict[str, str] = {}


def _normal_cdf(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.vectorize(math.erf)(x / math.sqrt(2.0)))


def nged_reverse_flow_flags() -> dict[str, str]:
    """Return NGED's "Reverse Power Flow?" entry for each MVA-only Primary, by D-label."""
    from studies.sources import REPO_DATA_DIR

    series = pl.read_parquet(default_input_paths().series_table)
    label_of = dict(zip(series["time_series_id"].to_list(), series["site"].to_list(), strict=True))
    path = REPO_DATA_DIR / "NGED" / "substation_hierarchy_2026-10-15" / "Monitor Direction.csv"
    flags: dict[str, str] = {}
    with path.open() as handle:
        for row in csv.DictReader(handle):
            label = label_of.get(int(row["TimeSeriesID"]))
            if label in MVA_ONLY_PRIMARIES:
                flags[label] = row["Reverse Power Flow?"] or "blank"
    return flags


def main() -> None:
    """Build the Step 0 table and the per-row frame."""
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    cache = load_cache()
    details = json.loads((EXPERIMENT_110_DIR / "work" / "fit_details.json").read_text())
    flags = nged_reverse_flow_flags()
    row_parts = []
    for cut in ("replicate", "scored"):
        frame = pl.read_parquet(
            EXPERIMENT_110_DIR / "predictions" / f"{cut}_l0" / "predictions.parquet"
        ).filter(pl.col("filtered"))
        label_of = {cache.sites[s].table.time_series_id: s for s in MVA_ONLY_PRIMARIES}
        frame = frame.with_columns(
            pl.col("time_series_id").replace_strict(label_of, default=None).alias("site"),
            pl.lit(cut).alias("cut"),
        ).drop_nulls("site")
        row_parts.append(frame)
    rows = pl.concat(row_parts)
    pieces = []
    for site in MVA_ONLY_PRIMARIES:
        features = cache.sites[site]
        eff = features.table.effective_capacity_mw
        frame = rows.filter(pl.col("site") == site)
        valid = frame["valid_time"]
        start = valid.dt.offset_by("-30m")
        start_hour = start.dt.hour().cast(pl.Float64) + start.dt.minute().cast(pl.Float64) / 60.0
        # Position of each row on the cache grid, to read CAMS irradiance and the PV unit shape.
        stamps = valid.dt.replace_time_zone(None).dt.cast_time_unit("us").to_numpy()
        position = np.searchsorted(cache.end_times, stamps)
        ghi = features.ghi_w_m2[position]
        shape = features.pv_unit_shape[position]
        middle = valid.dt.offset_by("-15m")
        clear = haurwitz_w_m2(
            apparent_zenith_deg=zenith(
                stamps=middle, latitude=features.table.latitude, longitude=features.table.longitude
            )
        )
        index = np.where(clear > 20.0, ghi / np.maximum(clear, 1.0), np.nan)
        midday = (
            valid.dt.month().is_in(SUNNY_MONTHS)
            & (start_hour >= MIDDAY_FIRST_HALF_HOUR_UTC)
            & (start_hour <= MIDDAY_LAST_HALF_HOUR_UTC)
        ).to_numpy()
        frame = frame.with_columns(
            pl.Series("clear_sky_index", index),
            pl.Series("midday", midday),
            pl.Series(
                "sunny_midday", midday & (np.nan_to_num(index, nan=0.0) >= CLEAR_SKY_INDEX_MINIMUM)
            ),
            pl.Series("pv_unit_shape", shape),
            (pl.col("p_mw") * eff / 100.0).alias("flow_mw"),
            (pl.col("actual") * eff / 100.0).alias("reading_mva"),
        )
        # Residual spread of |S| on the site's sunny-midday rows, by cut.
        sigma = (
            frame.filter(pl.col("sunny_midday") & ~pl.col("switching_mask"))
            .group_by("cut")
            .agg((pl.col("actual") - pl.col("mean")).std().alias("sigma_points"))
        )
        frame = frame.join(sigma, on="cut", how="left").with_columns(pl.lit(site).alias("site"))
        z = (-frame["p_mw"] / frame["sigma_points"]).to_numpy()
        frame = frame.with_columns(pl.Series("p_reverse", _normal_cdf(z)))
        pieces.append(frame)
    all_rows = pl.concat(pieces, how="diagonal")
    all_rows.write_parquet(WORK_DIR / "step0_rows.parquet")

    table_rows = []
    for site in MVA_ONLY_PRIMARIES:
        eff = cache.sites[site].table.effective_capacity_mw
        for cut in ("replicate", "scored"):
            sub = all_rows.filter(
                (pl.col("site") == site)
                & (pl.col("cut") == cut)
                & pl.col("sunny_midday")
                & ~pl.col("switching_mask")
            )
            if sub.height == 0:
                continue
            cap_points = details[f"{cut}/l0/{site}"]["capacity"]
            cap_year = max(cap_points)
            capacity_points = cap_points[cap_year]
            reverse = sub.filter(pl.col("p_mw") < 0)
            lowest = sub.sort("reading_mva").row(0, named=True)
            table_rows.append(
                {
                    "site": site,
                    "cut": cut,
                    "effective_capacity_mw": eff,
                    "pv_capacity_points": capacity_points,
                    "pv_capacity_mw": capacity_points * eff / 100.0,
                    "n_sunny_midday_rows": sub.height,
                    "share_reverse": reverse.height / sub.height,
                    "mean_p_reverse": float(sub["p_reverse"].mean()),  # ty: ignore[invalid-argument-type]
                    "share_reverse_sure": float((sub["p_reverse"] > 0.9).mean()),  # ty: ignore[invalid-argument-type]
                    "largest_reverse_mw": float(-sub["flow_mw"].min()) if reverse.height else 0.0,  # ty: ignore[invalid-argument-type, unsupported-operator]
                    "p95_reverse_mw": float(-sub["flow_mw"].quantile(0.05))  # ty: ignore[unsupported-operator]
                    if reverse.height
                    else 0.0,
                    "lowest_reading_mva": lowest["reading_mva"],
                    "lowest_reading_share_of_pv": lowest["reading_mva"]
                    / max(capacity_points * eff / 100.0, 1e-9),
                    "sigma_points": float(sub["sigma_points"][0]),
                    "nged_flag": flags.get(site, "missing"),
                }
            )
    table = pl.DataFrame(table_rows)
    table.write_parquet(WORK_DIR / "step0_table.parquet")
    with pl.Config(tbl_rows=40, tbl_cols=20, fmt_str_lengths=20, tbl_width_chars=250):
        print(table)
    print(timedelta(0))


if __name__ == "__main__":
    main()
