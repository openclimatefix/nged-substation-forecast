"""Fit experiment 00014's stage 0 mappings and apply experiment 00009's stage 1a models to ENS.

Stage 0 is an empirical quantile mapping from the ENS member's shortwave to a target irradiance
(CAMS at the series' own point, or ERA5 shortwave at its H3 cell), per lead band and per bin of the
cosine of the solar zenith, fitted on pairs with a valid time before 2025-06-17 and pooled over the
20 series. Night rows pass through. Each member is mapped on its own, so the members keep their
spread.

Outputs, under `data/ml-experiments/00014-.../work/`:

- `stage0/<target>.npz` and `stage0/pairs.parquet`: the mappings and the pair count of each bin.
- `stats/init=<date>.parquet`: for each of the 73 ENS runs of `fold_a_every5d_v1`, the 9 empirical
  quantiles of the 51 members' forecasts for each stage 1a arm (`era5`, `cams`), ENS input (`raw`,
  `stage0`), and seed, with the observed power over capacity.
- `shortwave/init=<date>.parquet`: daylight shortwave sums by series and month, raw and mapped,
  against CAMS and ERA5 at the same valid times, for the mismatch table.

Existing files are kept, so a crash costs one unit.

    uv run python ml-experiments/00014-*/e00014_fit.py
"""

import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import e00014_common as common
import numpy as np
import polars as pl
import xgboost as xgb
from studies.solar import zenith

STAGE0_DIR: Final[Path] = common.WORK_DIR / "stage0"
STATS_DIR: Final[Path] = common.WORK_DIR / "stats"
SHORTWAVE_DIR: Final[Path] = common.WORK_DIR / "shortwave"
KEY_COLUMNS: Final[tuple[str, ...]] = ("time_series_id", "power_fcst_init_time", "valid_time")
TARGETS: Final[tuple[str, ...]] = ("cams", "era5")
COS_ZENITH_DAYLIGHT_EDGE: Final[float] = float(np.cos(np.radians(common.MAX_ZENITH_DEGREES)))
COS_ZENITH_EDGES: Final[np.ndarray] = np.linspace(
    COS_ZENITH_DAYLIGHT_EDGE, 1.0, common.N_ZENITH_BINS + 1
)
N_BANDS: Final[int] = len(common.LEAD_BAND_EDGES) - 1


@dataclass(frozen=True)
class SeriesData:
    """One series' arrays on the dense half-hourly grid.

    Attributes:
        era5_features: Stage 1a's inputs with ERA5 shortwave, float32, one row per grid half-hour.
        cams_features: The same with CAMS irradiance in the shortwave column.
        power: Observed power over effective capacity, NaN where unobserved.
        valid_time: The grid's UTC timestamps.
    """

    era5_features: np.ndarray
    cams_features: np.ndarray
    power: np.ndarray
    valid_time: pl.Series

    def shortwave(self, *, target: str) -> np.ndarray:
        """Return the target's shortwave on the grid."""
        column = common.feature_columns().index("shortwave")
        return (self.cams_features if target == "cams" else self.era5_features)[:, column]


def load_series_data(*, series: pl.DataFrame) -> dict[str, SeriesData]:
    """Return each series' ERA5-arm and CAMS-arm features and observed power on the grid."""
    columns = common.feature_columns()
    weather = [c for c in columns if c not in common.DERIVED]
    grid = pl.DataFrame({"valid_time": common.grid_times()})
    era5 = pl.read_parquet(common.ERA5_PATH)
    observed = pl.read_parquet(
        common.OBSERVED_PATH, columns=["time_series_id", "valid_time", "power"]
    )
    out: dict[str, SeriesData] = {}
    for row in series.iter_rows(named=True):
        series_id = row["time_series_id"]
        joined = (
            grid.join(
                era5.filter(pl.col("time_series_id") == series_id).select("valid_time", *weather),
                on="valid_time",
                how="left",
                validate="1:1",
            )
            .join(
                observed.filter(pl.col("time_series_id") == series_id).select(
                    "valid_time", pl.col("power") / row["effective_capacity_mw"]
                ),
                on="valid_time",
                how="left",
                validate="1:1",
            )
            .join(
                common.cams_half_hourly(site=row["site"]),
                on="valid_time",
                how="left",
                validate="1:1",
            )
            .sort("valid_time")
        )
        derived = common.with_derived(
            frame=joined.select("valid_time"),
            latitude=row["latitude"],
            longitude=row["longitude"],
        )
        base = pl.concat(
            [joined.select(*weather, "cams"), derived.select(common.DERIVED)], how="horizontal"
        )
        out[row["site"]] = SeriesData(
            era5_features=base.select(columns).to_numpy().astype(np.float32),
            cams_features=base.with_columns(shortwave=pl.col("cams"))
            .select(columns)
            .to_numpy()
            .astype(np.float32),
            power=joined["power"].to_numpy().astype(np.float32),
            valid_time=joined["valid_time"],
        )
    return out


def _band(*, lead_hours: np.ndarray) -> np.ndarray:
    """Return each lead's band index, 0 for up to 1 day and 4 for 11 to 15 days."""
    return np.clip(
        np.searchsorted(common.LEAD_BAND_EDGES, lead_hours, side="right") - 1, 0, N_BANDS - 1
    )


def _zenith_bin(*, cos_zenith: np.ndarray) -> np.ndarray:
    """Return each cosine's bin index, or -1 where the sun is below the daylight limit."""
    index = np.searchsorted(COS_ZENITH_EDGES, cos_zenith, side="right") - 1
    index = np.minimum(index, common.N_ZENITH_BINS - 1)
    return np.where(cos_zenith < COS_ZENITH_DAYLIGHT_EDGE, -1, index)


def fit_mapping(
    *, ens: np.ndarray, target: np.ndarray, lead_hours: np.ndarray, cos_zenith: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit one quantile mapping per lead band and zenith bin.

    Args:
        ens: Paired ENS shortwave.
        target: Paired target irradiance.
        lead_hours: Lead of each pair.
        cos_zenith: Cosine of the solar zenith at each pair's valid time.

    Returns:
        Arrays of shape (bands, zenith bins, quantile points) holding the ENS quantiles and the
        target quantiles, and the pair count of each bin. A bin with too few pairs takes the
        mapping of the nearest zenith bin in the same band that has enough.

    Raises:
        ValueError: If a lead band has no bin with enough pairs.
    """
    points = np.linspace(0.0, 1.0, common.N_QUANTILE_POINTS)
    band, zbin = _band(lead_hours=lead_hours), _zenith_bin(cos_zenith=cos_zenith)
    xs = np.full((N_BANDS, common.N_ZENITH_BINS, points.size), np.nan)
    ys = np.full_like(xs, np.nan)
    counts = np.zeros((N_BANDS, common.N_ZENITH_BINS), dtype=np.int64)
    for b in range(N_BANDS):
        for z in range(common.N_ZENITH_BINS):
            keep = (band == b) & (zbin == z)
            counts[b, z] = keep.sum()
            if counts[b, z] >= common.MIN_PAIRS_PER_BIN:
                xs[b, z] = np.quantile(ens[keep], points)
                ys[b, z] = np.quantile(target[keep], points)
    for b in range(N_BANDS):
        enough = np.flatnonzero(counts[b] >= common.MIN_PAIRS_PER_BIN)
        if enough.size == 0:
            msg = f"lead band {b} has no zenith bin with {common.MIN_PAIRS_PER_BIN} pairs"
            raise ValueError(msg)
        for z in range(common.N_ZENITH_BINS):
            if counts[b, z] < common.MIN_PAIRS_PER_BIN:
                nearest = enough[np.abs(enough - z).argmin()]
                xs[b, z], ys[b, z] = xs[b, nearest], ys[b, nearest]
    return xs, ys, counts


def apply_mapping(
    *,
    mapping: tuple[np.ndarray, np.ndarray],
    ens: np.ndarray,
    lead_hours: np.ndarray,
    cos_zenith: np.ndarray,
) -> np.ndarray:
    """Map each ENS value through its bin's piecewise-linear quantile mapping.

    Night values pass through, and values beyond a bin's fitted range take the end point.
    """
    xs, ys = mapping
    band, zbin = _band(lead_hours=lead_hours), _zenith_bin(cos_zenith=cos_zenith)
    out = ens.astype(np.float64).copy()
    for b in range(N_BANDS):
        for z in range(common.N_ZENITH_BINS):
            keep = (band == b) & (zbin == z)
            if keep.any():
                out[keep] = np.interp(ens[keep], xs[b, z], ys[b, z])
    return out.astype(np.float32)


def fit_stage0(
    *, series: pl.DataFrame, data: dict[str, SeriesData]
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Fit (or load) the ENS-to-CAMS and ENS-to-ERA5 mappings on pairs before the cutoff."""
    STAGE0_DIR.mkdir(parents=True, exist_ok=True)
    if all((STAGE0_DIR / f"{t}.npz").exists() for t in TARGETS):
        loaded = {t: np.load(STAGE0_DIR / f"{t}.npz") for t in TARGETS}
        return {t: (loaded[t]["xs"], loaded[t]["ys"]) for t in TARGETS}
    ids = {row["time_series_id"]: row["site"] for row in series.iter_rows(named=True)}
    coordinates = {row["time_series_id"]: row for row in series.iter_rows(named=True)}
    table = pl.read_parquet(
        common.ENS_TRAIN_PATH,
        columns=["time_series_id", "valid_time", "shortwave", "lead_hours", "ensemble_member"],
    ).filter(
        pl.col("time_series_id").is_in(list(ids))
        & (pl.col("valid_time") < common.FINAL_TRAIN_CUTOFF)
    )
    latest = table.select(pl.col("valid_time").max()).item()
    assert latest < common.FINAL_TRAIN_CUTOFF
    print("training-table members:", table["ensemble_member"].value_counts().to_dicts(), flush=True)
    pieces = []
    for series_id, frame in table.group_by("time_series_id"):
        site = ids[series_id[0]]
        row = coordinates[series_id[0]]
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
                    "cams": data[site].shortwave(target="cams")[index],
                    "era5": data[site].shortwave(target="era5")[index],
                }
            )
        )
    pairs = pl.concat(pieces).filter(pl.all_horizontal(pl.all().is_finite()))
    mappings = {}
    count_rows = []
    for target in TARGETS:
        xs, ys, counts = fit_mapping(
            ens=pairs["ens"].to_numpy(),
            target=pairs[target].to_numpy(),
            lead_hours=pairs["lead"].to_numpy(),
            cos_zenith=pairs["cos_zenith"].to_numpy(),
        )
        np.savez(STAGE0_DIR / f"{target}.npz", xs=xs, ys=ys, counts=counts)
        mappings[target] = (xs, ys)
        count_rows += [
            {"target": target, "band": b, "zenith_bin": z, "pairs": int(counts[b, z])}
            for b in range(N_BANDS)
            for z in range(common.N_ZENITH_BINS)
        ]
    pl.DataFrame(count_rows).write_parquet(STAGE0_DIR / "pairs.parquet")
    print(f"stage 0 fitted on {pairs.height} pairs", flush=True)
    return mappings


def load_models(*, series: pl.DataFrame) -> dict[tuple[str, int, str], xgb.Booster]:
    """Load experiment 00009's boosters, keyed by arm, seed, and D label."""
    models = {}
    for arm in common.ARMS:
        for seed in common.SEEDS:
            for site in series["site"]:
                booster = xgb.Booster()
                booster.load_model(common.model_path(arm=arm, seed=seed, site=site))
                models[arm, seed, site] = booster
    return models


def run_statistics(
    *,
    run: pl.DataFrame,
    row: dict,
    arrays: SeriesData,
    models: dict[tuple[str, int, str], xgb.Booster],
    mappings: dict[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return one series' quantiles in one ENS run, and its daylight shortwave sums by month.

    Args:
        run: The run's rows for this series, every member.
        row: The series' row of `common.series_table()`.
        arrays: The series' `SeriesData`.
        models: Every booster.
        mappings: The stage 0 mappings by target.

    Returns:
        One row per case, arm, input, and seed with the 9 quantiles and `actual`, and the sums.
    """
    site = str(row["site"])
    ordered = common.with_derived(
        frame=run, latitude=float(row["latitude"]), longitude=float(row["longitude"])
    ).sort("valid_time", "ensemble_member")
    members = ordered["ensemble_member"].n_unique()
    matrix = ordered.select(common.feature_columns()).to_numpy().astype(np.float32)
    column = common.feature_columns().index("shortwave")
    keys = ordered.select(*KEY_COLUMNS).unique(maintain_order=True)
    index = np.asarray(common.grid_index(times=keys["valid_time"]))
    actual = pl.Series("actual", arrays.power[index])
    raw = matrix[:, column].copy()
    lead = ordered["lead_hours"].to_numpy()
    cosine = np.cos(np.radians(ordered["solar_zenith"].to_numpy().astype(np.float64)))
    mapped = {
        target: apply_mapping(mapping=mappings[target], ens=raw, lead_hours=lead, cos_zenith=cosine)
        for target in TARGETS
    }
    inputs = {
        "raw": {"era5": raw, "cams": raw},
        "stage0": {"era5": mapped["era5"], "cams": mapped["cams"]},
    }
    pieces = []
    for arm in common.ARMS:
        for input_name in common.INPUTS:
            matrix[:, column] = inputs[input_name][arm]
            for seed in common.SEEDS:
                forecast = models[arm, seed, site].predict(xgb.DMatrix(matrix)).reshape(-1, members)
                quantiles = np.quantile(forecast, common.LEVELS, axis=1).T.astype(np.float32)
                pieces.append(
                    keys.hstack(
                        pl.DataFrame(quantiles, schema=list(common.LEVEL_COLUMNS), orient="row")
                    ).with_columns(
                        actual,
                        arm=pl.lit(f"{arm}-{input_name}"),
                        seed=pl.lit(seed).cast(pl.Int8),
                    )
                )
    member_index = np.repeat(index, members)
    daylight = ordered["solar_zenith"].to_numpy() < common.MAX_ZENITH_DEGREES
    shortwave = (
        ordered.select("time_series_id", "valid_time")
        .with_columns(
            month=pl.col("valid_time").dt.strftime("%Y-%m"),
            daylight=pl.Series(daylight),
            raw=pl.Series(raw),
            to_cams=pl.Series(mapped["cams"]),
            to_era5=pl.Series(mapped["era5"]),
            cams=pl.Series(arrays.shortwave(target="cams")[member_index]),
            era5=pl.Series(arrays.shortwave(target="era5")[member_index]),
        )
        .filter(pl.col("daylight"))
    )
    spread = shortwave.group_by("time_series_id", "valid_time", "month").agg(
        pl.col("raw").std().alias("raw_spread"),
        pl.col("to_cams").std().alias("to_cams_spread"),
        pl.col("to_era5").std().alias("to_era5_spread"),
    )
    sums = (
        shortwave.group_by("time_series_id", "month")
        .agg(
            n=pl.len(),
            raw=pl.col("raw").sum().cast(pl.Float64),
            to_cams=pl.col("to_cams").sum().cast(pl.Float64),
            to_era5=pl.col("to_era5").sum().cast(pl.Float64),
            cams=pl.col("cams").sum().cast(pl.Float64),
            era5=pl.col("era5").sum().cast(pl.Float64),
        )
        .join(
            spread.group_by("time_series_id", "month").agg(
                n_times=pl.len(),
                raw_spread=pl.col("raw_spread").sum(),
                to_cams_spread=pl.col("to_cams_spread").sum(),
                to_era5_spread=pl.col("to_era5_spread").sum(),
            ),
            on=["time_series_id", "month"],
        )
    )
    return pl.concat(pieces), sums


def apply_to_ens(
    *,
    series: pl.DataFrame,
    data: dict[str, SeriesData],
    models: dict[tuple[str, int, str], xgb.Booster],
    mappings: dict[str, tuple[np.ndarray, np.ndarray]],
    limit_runs: int | None,
) -> None:
    """Write each ENS validation run's quantiles for every arm, input, series, and seed."""
    rows = {row["time_series_id"]: row for row in series.iter_rows(named=True)}
    paths = sorted(common.VALIDATION_DIR.glob("init=*.parquet"))[:limit_runs]
    for path in paths:
        target = STATS_DIR / path.name
        if target.exists():
            continue
        started = time.monotonic()
        frame = pl.read_parquet(path).filter(pl.col("time_series_id").is_in(list(rows)))
        results = [
            run_statistics(
                run=frame.filter(pl.col("time_series_id") == series_id),
                row=row,
                arrays=data[str(row["site"])],
                models=models,
                mappings=mappings,
            )
            for series_id, row in rows.items()
        ]
        SHORTWAVE_DIR.mkdir(parents=True, exist_ok=True)
        pl.concat([r[1] for r in results]).write_parquet(SHORTWAVE_DIR / path.name)
        target.parent.mkdir(parents=True, exist_ok=True)
        pl.concat([r[0] for r in results]).write_parquet(target.with_suffix(".tmp"))
        target.with_suffix(".tmp").rename(target)
        print(f"{path.name}: {time.monotonic() - started:.0f} s", flush=True)


def main() -> None:
    """Fit stage 0, then apply the stage 1a models to every ENS run."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--limit-runs", type=int, default=None, help="apply to this many runs only")
    args = parser.parse_args()
    series = common.series_table()
    data = load_series_data(series=series)
    mappings = fit_stage0(series=series, data=data)
    models = load_models(series=series)
    apply_to_ens(
        series=series, data=data, models=models, mappings=mappings, limit_runs=args.limit_runs
    )


if __name__ == "__main__":
    main()
