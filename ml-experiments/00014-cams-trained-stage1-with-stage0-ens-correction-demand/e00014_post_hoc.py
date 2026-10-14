"""Post hoc variant of experiment 00014's stage 0: one quantile mapping per calendar month.

The pre-registered stage 0 (`e00014_fit.py`) fixes the validation-year mean of the daylight
shortwave but not its monthly pattern. This variant fits the same quantile mapping per calendar
month of the training pairs (valid time before 2025-06-17), per lead band, and per zenith bin, and
applies it to the CAMS-trained arm only. A bin with fewer than 1,000 pairs takes the mapping of the
same month and band in the nearest zenith bin that has enough, and a month and band with none takes
the all-year mapping. The pairs of April to June 2024 and 2025 give those months two years, and the
other months one.

Outputs, under `data/ml-experiments/00014-.../work/`: `stats_post_hoc/init=<date>.parquet` (arm
`cams-stage0-month`, input `stage0m`, same columns as `stats/`) and `shortwave_post_hoc/`.

    uv run python ml-experiments/00014-*/e00014_post_hoc.py
"""

import time
from pathlib import Path
from typing import Final

import e00014_common as common
import e00014_fit as fit
import numpy as np
import polars as pl
import xgboost as xgb
from studies.solar import zenith

STATS_DIR: Final[Path] = common.WORK_DIR / "stats_post_hoc"
SHORTWAVE_DIR: Final[Path] = common.WORK_DIR / "shortwave_post_hoc"
ARM: Final[str] = "cams-stage0-month"


def monthly_mapping(
    *, series: pl.DataFrame, data: dict[str, fit.SeriesData]
) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Fit the ENS-to-CAMS mapping for each calendar month, falling back to the all-year mapping."""
    ids = {row["time_series_id"]: row for row in series.iter_rows(named=True)}
    table = pl.read_parquet(
        common.ENS_TRAIN_PATH, columns=["time_series_id", "valid_time", "shortwave", "lead_hours"]
    ).filter(
        pl.col("time_series_id").is_in(list(ids))
        & (pl.col("valid_time") < common.FINAL_TRAIN_CUTOFF)
    )
    latest = table.select(pl.col("valid_time").max()).item()
    assert latest < common.FINAL_TRAIN_CUTOFF
    pieces = []
    for (series_id,), frame in table.group_by("time_series_id"):
        row = ids[series_id]
        index = np.asarray(common.grid_index(times=frame["valid_time"]))
        angle = np.asarray(
            zenith(
                stamps=frame["valid_time"].dt.offset_by("-15m"),
                latitude=row["latitude"],
                longitude=row["longitude"],
            )
        )
        pieces.append(
            pl.DataFrame(
                {
                    "ens": frame["shortwave"].to_numpy(),
                    "lead": frame["lead_hours"].to_numpy(),
                    "cos_zenith": np.cos(np.radians(angle)),
                    "cams": data[row["site"]].shortwave(target="cams")[index],
                    "month": frame["valid_time"].dt.month().to_numpy(),
                }
            )
        )
    pairs = pl.concat(pieces).filter(pl.all_horizontal(pl.all().is_finite()))
    all_year = np.load(fit.STAGE0_DIR / "cams.npz")
    xs_all, ys_all = all_year["xs"], all_year["ys"]
    out = {}
    for month in range(1, 13):
        part = pairs.filter(pl.col("month") == month)
        lead = part["lead"].to_numpy()
        cosine = part["cos_zenith"].to_numpy()
        band, zbin = fit._band(lead_hours=lead), fit._zenith_bin(cos_zenith=cosine)
        xs, ys = xs_all.copy(), ys_all.copy()
        for b in range(fit.N_BANDS):
            counts = np.array(
                [((band == b) & (zbin == z)).sum() for z in range(common.N_ZENITH_BINS)]
            )
            enough = np.flatnonzero(counts >= common.MIN_PAIRS_PER_BIN)
            if enough.size == 0:
                continue
            points = np.linspace(0.0, 1.0, common.N_QUANTILE_POINTS)
            for z in range(common.N_ZENITH_BINS):
                use = (
                    z
                    if counts[z] >= common.MIN_PAIRS_PER_BIN
                    else enough[np.abs(enough - z).argmin()]
                )
                keep = (band == b) & (zbin == use)
                xs[b, z] = np.quantile(part["ens"].to_numpy()[keep], points)
                ys[b, z] = np.quantile(part["cams"].to_numpy()[keep], points)
        out[month] = (xs, ys)
    print(f"monthly mappings fitted on {pairs.height} pairs", flush=True)
    return out


def apply_monthly(
    *, mappings: dict[int, tuple[np.ndarray, np.ndarray]], frame: pl.DataFrame
) -> np.ndarray:
    """Map each ENS row through the mapping of its valid time's calendar month."""
    ens = frame["shortwave"].to_numpy().astype(np.float32)
    lead = frame["lead_hours"].to_numpy()
    cosine = np.cos(np.radians(frame["solar_zenith"].to_numpy().astype(np.float64)))
    months = frame["valid_time"].dt.month().to_numpy()
    out = ens.copy()
    for month, mapping in mappings.items():
        keep = months == month
        if keep.any():
            out[keep] = fit.apply_mapping(
                mapping=mapping, ens=ens[keep], lead_hours=lead[keep], cos_zenith=cosine[keep]
            )
    return out


def main() -> None:
    """Fit the monthly mappings and apply the CAMS-trained models to every ENS run."""
    series = common.series_table()
    data = fit.load_series_data(series=series)
    mappings = monthly_mapping(series=series, data=data)
    models = fit.load_models(series=series)
    rows = {row["time_series_id"]: row for row in series.iter_rows(named=True)}
    column = common.feature_columns().index("shortwave")
    for path in sorted(common.VALIDATION_DIR.glob("init=*.parquet")):
        target = STATS_DIR / path.name
        if target.exists():
            continue
        started = time.monotonic()
        frame = pl.read_parquet(path).filter(pl.col("time_series_id").is_in(list(rows)))
        pieces, sums = [], []
        for series_id, row in rows.items():
            site = str(row["site"])
            ordered = common.with_derived(
                frame=frame.filter(pl.col("time_series_id") == series_id),
                latitude=float(row["latitude"]),
                longitude=float(row["longitude"]),
            ).sort("valid_time", "ensemble_member")
            members = ordered["ensemble_member"].n_unique()
            matrix = ordered.select(common.feature_columns()).to_numpy().astype(np.float32)
            mapped = apply_monthly(mappings=mappings, frame=ordered)
            matrix[:, column] = mapped
            keys = ordered.select(*fit.KEY_COLUMNS).unique(maintain_order=True)
            index = np.asarray(common.grid_index(times=keys["valid_time"]))
            actual = pl.Series("actual", data[site].power[index])
            for seed in common.SEEDS:
                forecast = (
                    models["cams", seed, site].predict(xgb.DMatrix(matrix)).reshape(-1, members)
                )
                quantiles = np.quantile(forecast, common.LEVELS, axis=1).T.astype(np.float32)
                pieces.append(
                    keys.hstack(
                        pl.DataFrame(quantiles, schema=list(common.LEVEL_COLUMNS), orient="row")
                    ).with_columns(actual, arm=pl.lit(ARM), seed=pl.lit(seed).cast(pl.Int8))
                )
            daylight = ordered["solar_zenith"].to_numpy() < common.MAX_ZENITH_DEGREES
            member_index = np.repeat(index, members)
            part = (
                ordered.select("time_series_id", "valid_time")
                .with_columns(
                    month=pl.col("valid_time").dt.strftime("%Y-%m"),
                    daylight=pl.Series(daylight),
                    to_cams=pl.Series(mapped),
                    cams=pl.Series(data[site].shortwave(target="cams")[member_index]),
                )
                .filter(pl.col("daylight"))
            )
            spread = part.group_by("time_series_id", "valid_time", "month").agg(
                pl.col("to_cams").std().alias("to_cams_spread")
            )
            sums.append(
                part.group_by("time_series_id", "month")
                .agg(
                    n=pl.len(),
                    to_cams=pl.col("to_cams").sum().cast(pl.Float64),
                    cams=pl.col("cams").sum().cast(pl.Float64),
                )
                .join(
                    spread.group_by("time_series_id", "month").agg(
                        to_cams_spread=pl.col("to_cams_spread").sum()
                    ),
                    on=["time_series_id", "month"],
                )
            )
        SHORTWAVE_DIR.mkdir(parents=True, exist_ok=True)
        pl.concat(sums).write_parquet(SHORTWAVE_DIR / path.name)
        target.parent.mkdir(parents=True, exist_ok=True)
        pl.concat(pieces).write_parquet(target.with_suffix(".tmp"))
        target.with_suffix(".tmp").rename(target)
        print(f"{path.name}: {time.monotonic() - started:.0f} s", flush=True)


if __name__ == "__main__":
    main()
