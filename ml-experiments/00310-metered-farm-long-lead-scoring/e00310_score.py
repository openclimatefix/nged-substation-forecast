# ruff: noqa: E501
"""Score experiment 310: per-row losses on sorted quantiles, farm-equal means, month-block intervals.

Per day band and history setting, the rows are the scored-year runs that every arm holds with
finite quantiles (including the two floors). The quantile score is twice the mean pinball loss over
the 11 levels, in points of capacity. Every arm's quantiles are sorted per row before any loss.
Intervals resample the 12 months of the run with farm-equal weights at the 98.75% level (a
Bonferroni correction across four planned contrasts per band).

    uv run --no-sync python ml-experiments/00310-*/e00310_score.py
"""

from typing import Final

import e00310_common as common
import numpy as np
import polars as pl
from studies.tail_calibration import level_column

DRAWS: Final[int] = 4_000
LO_PCT, HI_PCT = 0.625, 99.375
QCOLS: Final[list[str]] = [level_column(level=lv) for lv in common.LEVELS]
FLOORS: Final[tuple[str, ...]] = ("floor_heuristic", "floor_climatology")
SINGLES: Final[tuple[str, ...]] = ("gam", "mdn_ens", "xgb")
TAIL_INDEX: Final[dict[str, int]] = {"01": 0, "05": 1, "95": 9, "99": 10}
MEAN_WEIGHTS: Final[np.ndarray] = np.diff(
    np.concatenate([[0.0], (common.LEVELS[:-1] + common.LEVELS[1:]) / 2, [1.0]])
)
METRICS: Final[tuple[str, ...]] = (
    "score",
    "ae_median",
    "ae_mean",
    "pin01",
    "pin05",
    "pin95",
    "pin99",
    "cover01",
    "cover05",
    "cover95",
    "cover99",
    "median_minus_mean",
)
KEYS: Final[list[str]] = ["time_series_id", "time", "init_time"]


def row_losses(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Return per-row losses on sorted quantiles (points of capacity)."""
    raw = frame.select(QCOLS).to_numpy()
    q = np.sort(raw, axis=1)
    actual = frame["actual"].to_numpy()
    error = actual[:, None] - q
    pin = np.maximum(common.LEVELS * error, (common.LEVELS - 1.0) * error)
    mean_forecast = q @ MEAN_WEIGHTS
    out = frame.select(*KEYS, "arm", "setting", "band", "lead_hours").with_columns(
        score=pl.Series(200.0 * pin.mean(axis=1)),
        ae_median=pl.Series(100.0 * np.abs(error[:, 5])),
        ae_mean=pl.Series(100.0 * np.abs(actual - mean_forecast)),
        median_minus_mean=pl.Series(100.0 * (q[:, 5] - mean_forecast)),
    )
    for name, i in TAIL_INDEX.items():
        out = out.with_columns(
            pl.Series(f"pin{name}", 100.0 * pin[:, i]),
            pl.Series(f"cover{name}", (actual <= q[:, i]).astype(float)),
        )
    return out


def load_band(*, band: str) -> pl.DataFrame:
    """Return the per-row losses of every arm and setting on a band's common rows."""
    floors = pl.read_parquet(common.DATA_DIR / f"floors_{band}.parquet").filter(
        pl.all_horizontal([pl.col(c).is_finite() for c in QCOLS])
    )
    pieces = []
    for setting in common.SETTINGS:
        files = sorted((common.PRED_DIR / band / setting).glob("*.parquet"))
        frame = pl.concat(
            [
                pl.read_parquet(f)
                .select(*KEYS, "lead_hours", "arm", "seed", "setting", "band", "actual", *QCOLS)
                .with_columns(
                    pl.col(QCOLS).cast(pl.Float64),
                    pl.col("actual").cast(pl.Float64),
                    pl.col("lead_hours").cast(pl.Float64),
                    pl.col("seed").cast(pl.Int64),
                )
                for f in files
            ]
        )
        # Seed-average the neural arms' unaveraged copies; the blends and `mdn_ens` are already averaged.
        frame = frame.filter(~pl.col("arm").is_in(["mdn", "mdn_nolag"]))
        floor_rows = floors.with_columns(
            setting=pl.lit(setting),
            lead_hours=pl.col("lead_hours").cast(pl.Float64),
            seed=pl.lit(0).cast(pl.Int64),
        ).select(frame.columns)
        both = pl.concat([frame, floor_rows])
        # Keep rows every always-present arm holds with finite quantiles.
        finite = both.filter(pl.all_horizontal([pl.col(c).is_finite() for c in QCOLS]))
        required = [a for a in both["arm"].unique().to_list() if not a.endswith("_nolag")]
        required = [a for a in required if not a.endswith("nolag_ens")]
        keep = (
            finite.filter(pl.col("arm").is_in(required))
            .group_by(KEYS)
            .agg(n=pl.col("arm").n_unique())
            .filter(pl.col("n") == len(required))
            .select(KEYS)
        )
        pieces.append(row_losses(frame=finite.join(keep, on=KEYS, how="inner")))
    return pl.concat(pieces).with_columns(
        month=pl.col("init_time").dt.strftime("%Y-%m"),
        farm=pl.col("time_series_id"),
        day=(pl.col("lead_hours") / 24.0).ceil().cast(pl.Int64),
    )


def farm_sums(*, rows: pl.DataFrame, value: str) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return per-(arm, farm, month) sums and counts."""
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


def boot(*, sums: np.ndarray, counts: np.ndarray, seed: int = 310) -> tuple[np.ndarray, np.ndarray]:
    """Return the point estimate and bootstrap draws (arms, draws) of the farm-averaged mean."""
    rng = np.random.default_rng(seed)
    n_months = sums.shape[2]
    weights = rng.multinomial(n_months, np.ones(n_months) / n_months, size=DRAWS).astype(float)
    total = counts.sum(axis=2)
    point = np.nanmean(sums.sum(axis=2) / np.where(total > 0, total, np.nan), axis=1)
    draws = np.empty((sums.shape[0], DRAWS))
    with np.errstate(invalid="ignore", divide="ignore"):
        for a in range(sums.shape[0]):
            den = weights @ counts[a].T
            draws[a] = np.nanmean((weights @ sums[a].T) / np.where(den > 0, den, np.nan), axis=1)
    return point, draws


def contrast_pairs(*, arms: list[str], best_single: str) -> set[tuple[str, str]]:
    """Return the (arm, reference) pairs to report."""
    pairs: set[tuple[str, str]] = set()
    for arm in ("blend", "gam", "mdn_ens", "xgb", "p2"):
        pairs |= {(arm, "floor_climatology"), (arm, "floor_heuristic")}
    pairs |= {(arm, "xgb") for arm in ("blend", "gam", "mdn_ens")}
    pairs |= {("blend", "blend_nolag"), ("blend", best_single), ("blend", "p2")}
    pairs |= {("mdn_ens", "gam"), ("blend", "gam")}
    pairs |= {(f"{a}_nolag", a) for a in ("gam", "xgb")} | {("mdn_nolag_ens", "mdn_ens")}
    pairs |= {("floor_climatology", "floor_heuristic")}
    return {(a, r) for a, r in pairs if a in arms and r in arms}


def main() -> None:
    """Print and write the score tables."""
    pl.Config.set_tbl_rows(200)
    pl.Config.set_tbl_cols(20)
    out_scores, out_contrasts, out_best = [], [], []
    all_rows = []
    for band in common.BAND_ORDER:
        rows = load_band(band=band)
        all_rows.append(rows)
        for setting in common.SETTINGS:
            sub = rows.filter(pl.col("setting") == setting)
            counts = sub.group_by("arm").agg(n=pl.len())
            if counts["n"].min() != counts["n"].max() and not set(
                counts.filter(pl.col("n") != counts["n"].max())["arm"]
            ) <= {"blend_nolag", "gam_nolag", "xgb_nolag", "mdn_nolag_ens"}:
                msg = f"arms hold different row counts in {band} {setting}:\n{counts.sort('n')}"
                raise ValueError(msg)
            slices = [("all", "all", sub)]
            slices += [("farm", f, sub.filter(pl.col("farm") == f)) for f in common.FARMS]
            slices += [
                ("day", str(d), sub.filter(pl.col("day") == d))
                for d in sorted(sub["day"].unique().to_list())
            ]
            for kind, name, part in slices:
                for metric in (
                    METRICS if kind == "all" else ("score", "ae_median", "pin99", "cover99")
                ):
                    sums, counts_a, arms = farm_sums(rows=part, value=metric)
                    point, draws = boot(sums=sums, counts=counts_a)
                    idx = {a: i for i, a in enumerate(arms)}
                    for a in arms:
                        lo, hi = np.nanpercentile(draws[idx[a]], [LO_PCT, HI_PCT])
                        out_scores.append(
                            {
                                "band": band,
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
                    if metric != "score" and kind != "all":
                        continue
                    best_single = (
                        min((s for s in SINGLES if s in idx), key=lambda s: point[idx[s]])
                        if metric == "score"
                        else "xgb"
                    )
                    if kind == "all" and metric == "score":
                        out_best.append(
                            {"band": band, "setting": setting, "best_single": best_single}
                        )
                    for arm, ref in sorted(contrast_pairs(arms=arms, best_single=best_single)):
                        change = 100.0 * (draws[idx[arm]] - draws[idx[ref]]) / draws[idx[ref]]
                        lo, hi = np.nanpercentile(change, [LO_PCT, HI_PCT])
                        out_contrasts.append(
                            {
                                "band": band,
                                "setting": setting,
                                "slice_kind": kind,
                                "slice": name,
                                "metric": metric,
                                "arm": arm,
                                "reference": ref,
                                "percent_change": 100.0
                                * (point[idx[arm]] - point[idx[ref]])
                                / point[idx[ref]],
                                "lo": lo,
                                "hi": hi,
                            }
                        )
        print("scored", band, flush=True)
    pl.concat(all_rows).write_parquet(common.DATA_DIR / "row_losses.parquet")
    scores, contrasts = pl.DataFrame(out_scores), pl.DataFrame(out_contrasts)
    scores.write_csv(common.DATA_DIR / "scores.csv")
    contrasts.write_csv(common.DATA_DIR / "contrasts.csv")
    pl.DataFrame(out_best).write_csv(common.DATA_DIR / "best_single.csv")
    head = (pl.col("slice_kind") == "all") & (pl.col("metric") == "score")
    for setting in common.SETTINGS:
        print(f"--- {setting}: quantile score by band")
        print(
            scores.filter(head & (pl.col("setting") == setting))
            .pivot(on="arm", index="band", values="value")
            .sort("band")
        )


if __name__ == "__main__":
    main()
