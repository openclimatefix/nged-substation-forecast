# ruff: noqa: E501
"""Score experiment 272: per-row losses (seed-averaged), farm-equal means, month-block intervals.

Rows are experiment 257's: experiment 224's scored-year, 12 to 36 h rows that every arm of that
experiment holds, which are also every row of this experiment's arms. The quantile score is twice
the mean pinball loss over the 11 levels, in points of capacity. A6 is the head arm, `a3_head` or
`a5_head`, with the lower mean validation loss in own history (validation months only).

    uv run --no-sync python ml-experiments/00272-*/e00272_score.py [phase1]
"""

import sys
from typing import Final

import e00272_common as common
import numpy as np
import polars as pl
from studies.tail_calibration import level_column

DRAWS: Final[int] = 20_000
QCOLS: Final[list[str]] = [level_column(level=lv) for lv in common.LEVELS]
CLOUDY_CSI: Final[float] = 0.5
FLOORS: Final[tuple[str, ...]] = ("floor_heuristic", "floor_climatology")
E243_ARMS: Final[tuple[str, ...]] = ("plant", "xgboost", "h5_quantile")
E257_ARMS: Final[tuple[str, ...]] = ("p2",)
BONFERRONI_TAILS: Final[tuple[float, float]] = (100 * 0.05 / 8 / 2, 100 - 100 * 0.05 / 8 / 2)
"""Tail percentiles of a 95% familywise level over eight planned contrasts."""
OBSERVED_WEATHER_FRACTION: Final[float] = 0.973
"""Experiment 218's fraction of sunny-half-hour output fitted with observed weather."""
CONTRASTS: Final[dict[str, tuple[str, str]]] = {
    "C1": ("a3_gam_full", "a1_identity"),
    "C2": ("a3_gam_full", "a2_gam_min"),
    "C3": ("a5_nn_full", "a3_gam_full"),
    "C4": ("a5_nn_full", "a4_nn_min"),
    "C5": ("a6_head", "a1_head"),
    "C6": ("a3_gam_full", "a3_placebo"),
    "C7": ("a5_nn_full", "a5_placebo"),
    "X1": ("a2_gam_min", "a1_identity"),
    "X2": ("a4_nn_min", "a1_identity"),
    "X3": ("blind", "a1_identity"),
    "X4": ("a5_nn_full", "a1_identity"),
}
"""Contrasts (arm, reference): C1 to C7 planned, X1 to X4 post hoc. C8 (the share of the gap closed) is derived from `fractions.csv`."""
REPORT_ARMS: Final[tuple[str, ...]] = (
    "a1_identity",
    "a2_gam_min",
    "a3_gam_full",
    "a4_nn_min",
    "a5_nn_full",
    "a6_head",
    "a1_head",
    "a3_head",
    "a5_head",
    "a3_shuffle",
    "a3_placebo",
    "a5_shuffle",
    "a5_placebo",
    "a3_alt",
    "a5_alt",
    "blind",
)


def reference_rows() -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return experiment 224's common-row keys and its two floors' quantiles at those rows."""
    frame = pl.read_parquet(common.E224_DIR / "predictions" / "scored_year.parquet").filter(
        pl.col("band") == 0
    )
    ok = frame.filter(pl.all_horizontal([pl.col(c).is_finite() for c in QCOLS]))
    keys = (
        ok.group_by("time_series_id", "valid_time")
        .agg(n=pl.col("arm").n_unique())
        .filter(pl.col("n") == frame["arm"].n_unique())
        .select("time_series_id", "valid_time")
    )
    floors = (
        frame.filter(pl.col("arm").is_in(FLOORS))
        .join(keys, on=["time_series_id", "valid_time"])
        .rename({"valid_time": "time"})
    )
    return keys.rename({"valid_time": "time"}), floors


def row_losses(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Return per-row losses (points of capacity) and the squared errors of the median."""
    q = frame.select(QCOLS).to_numpy()
    actual = frame["actual"].to_numpy()
    error = actual[:, None] - q
    pin = np.maximum(common.LEVELS * error, (common.LEVELS - 1.0) * error)
    mean_forecast = q.mean(axis=1)
    return frame.select("time_series_id", "time", "arm", "setting").with_columns(
        score=pl.Series(200.0 * pin.mean(axis=1)),
        ae_median=pl.Series(100.0 * np.abs(error[:, 5])),
        ae_mean=pl.Series(100.0 * np.abs(actual - mean_forecast)),
        median_minus_mean=pl.Series(100.0 * (q[:, 5] - mean_forecast)),
        pin01=pl.Series(100.0 * pin[:, 0]),
        pin05=pl.Series(100.0 * pin[:, 1]),
        pin95=pl.Series(100.0 * pin[:, 9]),
        pin99=pl.Series(100.0 * pin[:, 10]),
        cover01=pl.Series((actual <= q[:, 0]).astype(float)),
        cover99=pl.Series((actual <= q[:, 10]).astype(float)),
        sq_err=pl.Series(error[:, 5] ** 2),
        sq_actual=pl.Series(actual**2),
    )


def head_choice() -> str:
    """Return the head arm with the lower mean own-history validation loss (validation months only)."""
    loss = {}
    for arm in ("a3_head", "a5_head"):
        files = sorted((common.PRED_DIR / "own_history").glob(f"{arm}_s*_*.parquet"))
        loss[arm] = float(
            np.mean([pl.read_parquet(f, columns=["val_loss"])["val_loss"][0] for f in files])
        )
    print("head validation loss", loss)
    return min(loss, key=lambda k: loss[k])


def load_arms(*, phase1: bool) -> pl.DataFrame:
    """Return the seed-averaged per-row losses of every arm and setting on the common rows."""
    keys, floors = reference_rows()
    chosen = None if phase1 else head_choice()
    pieces = []
    for setting in common.SETTINGS:
        files = sorted((common.PRED_DIR / setting).glob("*.parquet"))
        files += [
            f
            for a in E243_ARMS
            for f in sorted((common.E243_DIR / "predictions" / setting).glob(f"{a}_s*_*.parquet"))
        ]
        files += [
            f
            for a in E257_ARMS
            for f in sorted((common.E257_DIR / "predictions" / setting).glob(f"{a}_s*_*.parquet"))
        ]
        columns = ["time", "time_series_id", "arm", "seed", "actual", "lead_hours", *QCOLS]
        frame = pl.concat(
            [
                pl.read_parquet(f, columns=columns).with_columns(
                    pl.col(QCOLS).cast(pl.Float64),
                    pl.col("actual").cast(pl.Float64),
                    pl.col("lead_hours").cast(pl.Float64),
                )
                for f in files
            ]
        )
        if chosen is not None:
            frame = pl.concat(
                [frame, frame.filter(pl.col("arm") == chosen).with_columns(arm=pl.lit("a6_head"))]
            )
        frame = frame.join(keys, on=["time_series_id", "time"], how="inner")
        pieces.append(
            row_losses(frame=frame.with_columns(setting=pl.lit(setting))).with_columns(
                seed=frame["seed"]
            )
        )
        pieces.append(
            row_losses(frame=floors.with_columns(setting=pl.lit(setting))).with_columns(
                seed=pl.lit(0)
            )
        )
    allrows = pl.concat(pieces)
    metrics = [
        c for c in allrows.columns if c not in ("setting", "arm", "time_series_id", "time", "seed")
    ]
    averaged = allrows.group_by("setting", "arm", "time_series_id", "time").agg(
        pl.col(metrics).mean()
    )
    sky = pl.concat(
        [
            common.load_farm(s).frame.select("time", "csi", "clear_sky", time_series_id=pl.lit(s))
            for s in common.FARMS
        ]
    )
    return averaged.join(sky, on=["time_series_id", "time"], how="left").with_columns(
        month=pl.col("time").dt.strftime("%Y-%m")
    )


def farm_means(
    *, rows: pl.DataFrame, value: str
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Return per-(arm, farm, month) sums and per-(farm, month) counts for a bootstrap."""
    arms = sorted(rows["arm"].unique().to_list())
    months = sorted(rows["month"].unique().to_list())
    sums = np.zeros((len(arms), len(common.FARMS), len(months)))
    counts = np.zeros((len(common.FARMS), len(months)))
    grouped = rows.group_by("arm", "time_series_id", "month").agg(
        total=pl.col(value).sum(), n=pl.len()
    )
    for r in grouped.iter_rows(named=True):
        a, f, m = (
            arms.index(r["arm"]),
            common.FARMS.index(r["time_series_id"]),
            months.index(r["month"]),
        )
        sums[a, f, m] = r["total"]
        counts[f, m] = r["n"]
    return sums, counts, arms, months


def boot(*, sums: np.ndarray, counts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the point estimate and bootstrap draws (arms, draws) of the farm-averaged mean."""
    rng = np.random.default_rng(272)
    n_months = counts.shape[1]
    farm_n = counts.sum(axis=1)
    point = np.nanmean(sums.sum(axis=2) / np.where(farm_n > 0, farm_n, np.nan)[None, :], axis=1)
    weights = rng.multinomial(n_months, np.ones(n_months) / n_months, size=DRAWS).astype(float)
    denominator = weights @ counts.T
    draws = np.empty((sums.shape[0], DRAWS))
    with np.errstate(invalid="ignore", divide="ignore"):
        for a in range(sums.shape[0]):
            draws[a] = np.nanmean(
                (weights @ sums[a].T) / np.where(denominator > 0, denominator, np.nan), axis=1
            )
    return point, draws


def fitted_fraction(*, rows: pl.DataFrame) -> pl.DataFrame:
    """Return per arm and farm 1 - SSE/SS of the median forecast on sunny half-hours."""
    sunny = rows.filter(pl.col("clear_sky") > common.SUNNY_CLEAR_SKY)
    return (
        sunny.group_by("setting", "arm", "time_series_id")
        .agg(fraction=1.0 - pl.col("sq_err").sum() / pl.col("sq_actual").sum(), n=pl.len())
        .sort("setting", "arm", "time_series_id")
    )


def main() -> None:
    """Print and write the score tables."""
    phase1 = "phase1" in sys.argv[1:]
    pl.Config.set_tbl_rows(300)
    pl.Config.set_tbl_cols(20)
    rows = load_arms(phase1=phase1)
    rows.write_parquet(common.DATA_DIR / "row_losses.parquet")
    out_scores, out_contrasts = [], []
    metrics = (
        "score",
        "ae_median",
        "ae_mean",
        "pin01",
        "pin05",
        "pin95",
        "pin99",
        "cover01",
        "cover99",
        "median_minus_mean",
    )
    for setting in common.SETTINGS:
        sub = rows.filter(pl.col("setting") == setting)
        for slice_name, slice_rows in (
            ("all", sub),
            ("cloudy", sub.filter(pl.col("csi") < CLOUDY_CSI)),
            ("clear", sub.filter(pl.col("csi") >= CLOUDY_CSI)),
        ):
            for metric in metrics:
                sums, counts, arms, _ = farm_means(rows=slice_rows, value=metric)
                point, draws = boot(sums=sums, counts=counts)
                idx = {a: i for i, a in enumerate(arms)}
                for a in arms:
                    lo, hi = np.nanpercentile(draws[idx[a]], [2.5, 97.5])
                    out_scores.append(
                        {
                            "setting": setting,
                            "slice": slice_name,
                            "metric": metric,
                            "arm": a,
                            "value": point[idx[a]],
                            "lo": lo,
                            "hi": hi,
                        }
                    )
                if metric not in ("score", "ae_median", "ae_mean", "pin99", "pin01"):
                    continue
                for name, (arm, ref) in CONTRASTS.items():
                    if arm not in idx or ref not in idx:
                        continue
                    change = 100.0 * (draws[idx[arm]] - draws[idx[ref]]) / draws[idx[ref]]
                    out_contrasts.append(
                        {
                            "contrast": name,
                            "setting": setting,
                            "slice": slice_name,
                            "metric": metric,
                            "arm": arm,
                            "reference": ref,
                            "percent_change": 100.0
                            * (point[idx[arm]] - point[idx[ref]])
                            / point[idx[ref]],
                            "lo": np.nanpercentile(change, 2.5),
                            "hi": np.nanpercentile(change, 97.5),
                            "lo_family": np.nanpercentile(change, BONFERRONI_TAILS[0]),
                            "hi_family": np.nanpercentile(change, BONFERRONI_TAILS[1]),
                        }
                    )
    scores, contrasts = pl.DataFrame(out_scores), pl.DataFrame(out_contrasts)
    per_farm = (
        rows.group_by("setting", "arm", "time_series_id")
        .agg(pl.col(list(metrics)).mean(), n=pl.len())
        .sort("setting", "arm", "time_series_id")
    )
    fractions = fitted_fraction(rows=rows)
    crossing = (
        pl.concat(
            [
                pl.read_parquet(f, columns=["arm", "setting", "crossed"])
                .group_by("arm", "setting")
                .agg(pl.col("crossed").mean())
                for f in sorted(common.PRED_DIR.glob("*/*.parquet"))
            ]
        )
        .group_by("arm", "setting")
        .agg(pl.col("crossed").mean())
        .sort("setting", "arm")
    )
    for name, table in (
        ("scores", scores),
        ("contrasts", contrasts),
        ("per_farm", per_farm),
        ("fractions", fractions),
        ("crossing", crossing),
    ):
        table.write_csv(common.DATA_DIR / f"{name}.csv")
    print(
        scores.filter((pl.col("slice") == "all") & (pl.col("metric") == "score")).sort(
            "setting", "value"
        )
    )
    print(
        contrasts.filter((pl.col("slice") == "all") & (pl.col("metric") == "score")).sort(
            "contrast"
        )
    )
    print(
        per_farm.filter(pl.col("setting") == "own_history")
        .select("arm", "time_series_id", "score")
        .pivot(on="time_series_id", index="arm", values="score")
    )
    print(
        fractions.filter(pl.col("setting") == "own_history").pivot(
            on="time_series_id", index="arm", values="fraction"
        )
    )
    print(crossing)


if __name__ == "__main__":
    main()
