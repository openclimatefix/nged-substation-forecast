# ruff: noqa: E501
"""Score experiment 287: per-row losses on sorted quantiles, farm-equal means, month-block intervals.

Rows are experiment 224's scored-year, 12 to 36 h rows that every arm holds, including its floors.
The quantile score is twice the mean pinball loss over the 11 levels, in points of capacity. Every
arm's quantiles are sorted per row before any loss. Intervals resample the 12 scored months with
farm-equal weights; the level is 98.75% (a Bonferroni correction across four headline contrasts).

    uv run --no-sync python ml-experiments/00287-*/e00287_score.py
"""

from typing import Final

import e00287_common as common
import numpy as np
import polars as pl
from studies.tail_calibration import level_column

DRAWS: Final[int] = 5_000
LO_PCT, HI_PCT = 0.625, 99.375
QCOLS: Final[list[str]] = [level_column(level=lv) for lv in common.LEVELS]
FLOORS: Final[tuple[str, ...]] = ("floor_heuristic", "floor_climatology")
HEADLINE: Final[tuple[tuple[str, str], ...]] = (
    ("nn", "gam"),
    ("mdn", "gam"),
    ("blend", "best_single"),
    ("gam", "p2"),
)
SINGLE_ARMS: Final[tuple[str, ...]] = ("gam", "nn", "xgb_x2")
TAIL_INDEX: Final[dict[str, int]] = {"01": 0, "05": 1, "95": 9, "99": 10}
CSI_BANDS: Final[tuple[float, ...]] = (0.3, 0.6, 0.9)
MEAN_WEIGHTS: Final[np.ndarray] = np.diff(
    np.concatenate([[0.0], (common.LEVELS[:-1] + common.LEVELS[1:]) / 2, [1.0]])
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
    """Return per-row losses on sorted quantiles (points of capacity), plus the raw crossing flag."""
    raw = frame.select(QCOLS).to_numpy()
    crossed = np.any(np.diff(raw, axis=1) < 0, axis=1)
    q = np.sort(raw, axis=1)
    actual = frame["actual"].to_numpy()
    error = actual[:, None] - q
    pin = np.maximum(common.LEVELS * error, (common.LEVELS - 1.0) * error)
    mean_forecast = q @ MEAN_WEIGHTS
    out = frame.select("time_series_id", "time", "arm", "setting").with_columns(
        score=pl.Series(200.0 * pin.mean(axis=1)),
        ae_median=pl.Series(100.0 * np.abs(error[:, 5])),
        ae_mean=pl.Series(100.0 * np.abs(actual - mean_forecast)),
        median_minus_mean=pl.Series(100.0 * (q[:, 5] - mean_forecast)),
        crossed=pl.Series(crossed.astype(float)),
    )
    for name, i in TAIL_INDEX.items():
        out = out.with_columns(
            pl.Series(f"pin{name}", 100.0 * pin[:, i]),
            pl.Series(f"cover{name}", (actual <= q[:, i]).astype(float)),
        )
    return out


def load_rows() -> pl.DataFrame:
    """Return the seed-averaged per-row losses of every arm and setting on the common rows."""
    keys, floors = reference_rows()
    pieces = []
    for setting in common.SETTINGS:
        files = sorted((common.PRED_DIR / setting).glob("*.parquet"))
        frame = pl.concat(
            [
                pl.read_parquet(f)
                .select(
                    "time",
                    "init_time",
                    "lead_hours",
                    "time_series_id",
                    "arm",
                    "seed",
                    "setting",
                    "actual",
                    *QCOLS,
                )
                .with_columns(
                    pl.col(QCOLS).cast(pl.Float64),
                    pl.col("actual").cast(pl.Float64),
                    pl.col("lead_hours").cast(pl.Float64),
                    pl.col("seed").cast(pl.Int64),
                )
                for f in files
            ]
        ).join(keys, on=["time_series_id", "time"], how="inner")
        pieces.append(row_losses(frame=frame))
        pieces.append(row_losses(frame=floors.with_columns(setting=pl.lit(setting))))
    metrics = [
        c for c in pieces[0].columns if c not in ("time_series_id", "time", "arm", "setting")
    ]
    averaged = (
        pl.concat(pieces)
        .group_by("setting", "arm", "time_series_id", "time")
        .agg([pl.col(m).mean() for m in metrics])
    )
    context = pl.concat(
        [
            common.load_farm(s).frame.select(
                "time",
                "csi",
                "lead_hours",
                "temperature_mean",
                "ghi_std",
                time_series_id=pl.lit(s),
            )
            for s in common.FARMS
        ]
    )
    return averaged.join(context, on=["time_series_id", "time"], how="left").with_columns(
        month=pl.col("time").dt.strftime("%Y-%m"),
        farm=pl.col("time_series_id"),
        csi_band=pl.when(pl.col("csi") < CSI_BANDS[0])
        .then(pl.lit("<0.3"))
        .when(pl.col("csi") < CSI_BANDS[1])
        .then(pl.lit("0.3-0.6"))
        .when(pl.col("csi") < CSI_BANDS[2])
        .then(pl.lit("0.6-0.9"))
        .otherwise(pl.lit(">=0.9")),
        lead_band=pl.when(pl.col("lead_hours") <= 24)
        .then(pl.lit("12-24"))
        .otherwise(pl.lit("24-36")),
        season=pl.col("time")
        .dt.month()
        .replace_strict(
            {
                12: "DJF",
                1: "DJF",
                2: "DJF",
                3: "MAM",
                4: "MAM",
                5: "MAM",
                6: "JJA",
                7: "JJA",
                8: "JJA",
                9: "SON",
                10: "SON",
                11: "SON",
            }
        ),
    )


def farm_sums(*, rows: pl.DataFrame, value: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return per-(arm, farm, month) sums and per-(arm, farm, month) counts."""
    arms = sorted(rows["arm"].unique().to_list())
    months = sorted(rows["month"].unique().to_list())
    sums = np.zeros((len(arms), len(common.FARMS), len(months)))
    counts = np.zeros_like(sums)
    grouped = rows.group_by("arm", "time_series_id", "month").agg(
        total=pl.col(value).sum(), n=pl.len()
    )
    for r in grouped.iter_rows(named=True):
        a, f, m = (
            arms.index(r["arm"]),
            common.FARMS.index(r["time_series_id"]),
            months.index(r["month"]),
        )
        sums[a, f, m], counts[a, f, m] = r["total"], r["n"]
    return sums, counts, arms


def boot(*, sums: np.ndarray, counts: np.ndarray, seed: int = 270) -> tuple[np.ndarray, np.ndarray]:
    """Return the point estimate and bootstrap draws (arms, draws) of the farm-averaged mean."""
    rng = np.random.default_rng(seed)
    n_months = sums.shape[2]
    weights = rng.multinomial(n_months, np.ones(n_months) / n_months, size=DRAWS).astype(float)
    point = np.nanmean(
        sums.sum(axis=2) / np.where(counts.sum(axis=2) > 0, counts.sum(axis=2), np.nan), axis=1
    )
    draws = np.empty((sums.shape[0], DRAWS))
    with np.errstate(invalid="ignore", divide="ignore"):
        for a in range(sums.shape[0]):
            den = weights @ counts[a].T
            draws[a] = np.nanmean((weights @ sums[a].T) / np.where(den > 0, den, np.nan), axis=1)
    return point, draws


SLICES: Final[tuple[str, ...]] = ("all", "csi_band", "season", "lead_band", "farm")
METRICS: Final[tuple[str, ...]] = (
    "score",
    "ae_median",
    "ae_mean",
    "pin01",
    "pin05",
    "pin95",
    "pin99",
)


def check_same_rows(*, rows: pl.DataFrame) -> None:
    """Raise when arms hold different numbers of rows per setting and farm."""
    counts = rows.group_by("setting", "time_series_id", "arm").agg(n=pl.len())
    spread = counts.group_by("setting", "time_series_id").agg(
        lo=pl.col("n").min(), hi=pl.col("n").max()
    )
    if spread.filter(pl.col("lo") != pl.col("hi")).height:
        msg = f"arms hold different row counts:\n{counts.sort('n')}"
        raise ValueError(msg)


def main() -> None:
    """Print and write the score tables."""
    best_by_setting: dict[str, str] = {}
    pl.Config.set_tbl_rows(200)
    pl.Config.set_tbl_cols(20)
    rows = load_rows()
    check_same_rows(rows=rows)
    rows.write_parquet(common.DATA_DIR / "row_losses.parquet")
    out_scores, out_contrasts = [], []
    for setting in common.SETTINGS:
        sub = rows.filter(pl.col("setting") == setting)
        for kind in SLICES:
            groups = (
                [("all", sub)]
                if kind == "all"
                else [
                    (str(v), sub.filter(pl.col(kind) == v))
                    for v in sorted(sub[kind].drop_nulls().unique().to_list())
                ]
            )
            for name, part in groups:
                for metric in METRICS if kind == "all" or name else ("score",):
                    if kind != "all" and metric not in ("score", "ae_median"):
                        continue
                    sums, counts, arms = farm_sums(rows=part, value=metric)
                    point, draws = boot(sums=sums, counts=counts)
                    idx = {a: i for i, a in enumerate(arms)}
                    for a in arms:
                        lo, hi = np.nanpercentile(draws[idx[a]], [LO_PCT, HI_PCT])
                        out_scores.append(
                            {
                                "setting": setting,
                                "slice_kind": kind,
                                "slice": name,
                                "metric": metric,
                                "arm": a,
                                "value": point[idx[a]],
                                "lo": lo,
                                "hi": hi,
                            }
                        )
                    if kind == "all" and metric == "score":
                        best_by_setting[setting] = min(
                            SINGLE_ARMS, key=lambda a: point[idx[a]] if a in idx else np.inf
                        )
                    best = best_by_setting[setting]
                    pairs = {
                        (a, "p2")
                        for a in arms
                        if a not in ("p2", "floor_heuristic", "floor_climatology")
                    }
                    pairs |= {
                        ("nn", "gam"),
                        ("mdn", "gam"),
                        ("blend", best),
                        ("blend", "gam"),
                        ("blend", "xgb_x2"),
                        ("gam", "xgb_x2"),
                        ("nn", "xgb_x2"),
                    }
                    pairs |= {("blend_mdn", r) for r in ("blend", "mdn", "xgb_x2", "gam")}
                    pairs |= {("nn_ens", "gam"), ("mdn_ens", "gam"), ("mdn_ens", "mdn")}
                    pairs |= {("nn", c) for c in ("ctrl_spread", "ctrl_lag", "ctrl_blind")}
                    for arm, ref in sorted(pairs):
                        if arm not in idx or ref not in idx:
                            continue
                        change = 100.0 * (draws[idx[arm]] - draws[idx[ref]]) / draws[idx[ref]]
                        lo, hi = np.nanpercentile(change, [LO_PCT, HI_PCT])
                        out_contrasts.append(
                            {
                                "setting": setting,
                                "slice_kind": kind,
                                "slice": name,
                                "metric": metric,
                                "arm": arm,
                                "reference": ref,
                                "is_headline": (arm, ref)
                                in {
                                    (h[0], best if h[1] == "best_single" else h[1])
                                    for h in HEADLINE
                                },
                                "percent_change": 100.0
                                * (point[idx[arm]] - point[idx[ref]])
                                / point[idx[ref]],
                                "lo": lo,
                                "hi": hi,
                            }
                        )
    scores, contrasts = pl.DataFrame(out_scores), pl.DataFrame(out_contrasts)
    scores.write_csv(common.DATA_DIR / "scores.csv")
    contrasts.write_csv(common.DATA_DIR / "contrasts.csv")
    allrows = pl.col("slice") == "all"
    for setting in common.SETTINGS:
        print(f"--- {setting}: quantile score")
        print(
            scores.filter(
                allrows & (pl.col("metric") == "score") & (pl.col("setting") == setting)
            ).sort("value")
        )
        print(f"--- {setting}: headline contrasts (quantile score, %)")
        for arm, ref in HEADLINE:
            ref = best_by_setting[setting] if ref == "best_single" else ref  # noqa: PLW2901
            print(
                contrasts.filter(
                    allrows
                    & (pl.col("metric") == "score")
                    & (pl.col("setting") == setting)
                    & (pl.col("arm") == arm)
                    & (pl.col("reference") == ref)
                )
            )
    farm = rows.group_by("setting", "arm", "time_series_id").agg(pl.col("score").mean())
    farm.write_csv(common.DATA_DIR / "per_farm.csv")
    print(best_by_setting)


if __name__ == "__main__":
    main()
