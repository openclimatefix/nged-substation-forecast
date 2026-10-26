"""Score the capacity scales of experiment 352 and print the planned contrasts.

Rows, slices, and the quantile score are experiment 349's (`e00349_score`): the scored year, leads
up to 168 hours, rows outside the hand-filtered events, and the sunniest third of each Primary's
daylight half-hours. Intervals resample the 12 calendar months of the scored year with
replacement (a month-block bootstrap, paired across arms, 95% level).

    PYTHONPATH=FOLDER_349:FOLDER_352 \
    uv run --no-sync python ml-experiments/00352-*/e00352_score.py
"""

from typing import Final

import e00349_common as common349  # ty: ignore[unresolved-import]
import e00349_run_common as run_common  # ty: ignore[unresolved-import]
import e00349_score as score349  # ty: ignore[unresolved-import]
import e00352_common as common
import numpy as np
import polars as pl

N_RESAMPLES: Final[int] = 20000
SEED: Final[int] = 352
LEAD_RANGES: Final[dict[str, tuple[float, float]]] = score349.LEAD_RANGES
FLAT_BAND: Final[tuple[float, float]] = (0.8, 1.2)


def load_arm(*, scale: float, filter_name: str, sites: tuple[str, ...]) -> pl.DataFrame:
    """Return an arm's scorable rows for the scored year, with the unit shape and losses."""
    cache = common349.load_cache()
    pieces = []
    for site in sites:
        path = common.preds_path(scale=scale, filter_name=filter_name, period="scored", site=site)
        frame = pl.read_parquet(path).with_columns(label=pl.lit(site))
        frame = frame.filter(
            ~pl.col("switching_mask")
            & pl.col("actual").is_finite()
            & (pl.col("lead_hours") <= 168.0)
        )
        valid = frame["valid_time"].dt.replace_time_zone(None).to_numpy().astype("datetime64[us]")
        rows = np.searchsorted(cache.end_times, valid)
        pieces.append(
            frame.with_columns(unit=pl.Series(np.nan_to_num(cache.sites[site].pv_unit_shape[rows])))
        )
    return score349.with_losses(frame=pl.concat(pieces))


def month_sums(*, frame: pl.DataFrame, metric: str) -> tuple[np.ndarray, np.ndarray]:
    """Return per-calendar-month sums of a metric and row counts, 12 entries each."""
    grouped = (
        frame.with_columns(m=pl.col("valid_time").dt.month())
        .group_by("m")
        .agg(s=pl.col(metric).sum(), n=pl.len())
    )
    sums, counts = np.zeros(12), np.zeros(12)
    for m, s, n in grouped.iter_rows():
        sums[m - 1], counts[m - 1] = s, n
    return sums, counts


class Bootstrap:
    """Month-block draws shared by every statistic, so contrasts are paired across arms."""

    def __init__(self) -> None:
        """Draw the month multiplicities once."""
        generator = np.random.default_rng(SEED)
        picks = generator.integers(0, 12, size=(N_RESAMPLES, 12))
        self.weights = np.stack([np.bincount(row, minlength=12) for row in picks]).astype(float)

    def mean(self, *, sums: np.ndarray, counts: np.ndarray) -> np.ndarray:
        """Return the pooled mean in every draw."""
        return (self.weights @ sums) / np.maximum(self.weights @ counts, 1.0)

    @staticmethod
    def interval(*, draws: np.ndarray) -> tuple[float, float]:
        """Return the 95% percentile interval."""
        low, high = np.percentile(draws, [2.5, 97.5])
        return float(low), float(high)


def main() -> None:  # noqa: PLR0915
    """Score every arm and print the tables."""
    sites = run_common.SCORED_SITES
    frames = {
        (s, f): load_arm(scale=s, filter_name=f, sites=sites)
        for s in common.SCALES
        for f in common.FILTERS
    }
    reference = frames[(1.0, "kf")]
    edges = score349.tertile_edges(frame=reference)
    frames = {key: score349.add_slices(frame=f, edges=edges) for key, f in frames.items()}
    heights = {f.height for f in frames.values()}
    if len(heights) != 1:
        msg = f"arms cover different rows: {heights}"
        raise ValueError(msg)
    boot = Bootstrap()
    scopes: dict[str, pl.Expr] = {
        "sunny": pl.col("third") == "sunny",
        "all half-hours": pl.lit(True),
        "all daylight": pl.col("third") != "night",
        "night": pl.col("third") == "night",
        "winter nights": (pl.col("third") == "night") & pl.col("calendar_month").is_in([12, 1, 2]),
        "darkest third": pl.col("third") == "dark",
        **{
            f"sunny {site}": (pl.col("third") == "sunny") & (pl.col("label") == site)
            for site in sites
        },
        **{
            f"sunny leave out {site}": (pl.col("third") == "sunny") & (pl.col("label") != site)
            for site in sites
        },
    }
    rows = []
    for lead_name, (low, high) in LEAD_RANGES.items():
        in_range = (pl.col("lead_hours") > low) & (pl.col("lead_hours") <= high)
        for scope, expression in scopes.items():
            stats = {}
            for key, f in frames.items():
                part = f.filter(expression & in_range)
                for metric in ("qs", "abs_mean"):
                    stats[(key, metric)] = month_sums(frame=part, metric=metric)
            for metric in ("qs", "abs_mean"):
                means = {
                    key: boot.mean(sums=stats[(key, metric)][0], counts=stats[(key, metric)][1])
                    for key in frames
                }
                point = {
                    key: stats[(key, metric)][0].sum() / max(stats[(key, metric)][1].sum(), 1.0)
                    for key in frames
                }
                for (s, f), value in point.items():
                    low_i, high_i = boot.interval(draws=means[(s, f)])
                    rows.append({"lead": lead_name, "scope": scope, "metric": metric,
                                 "statistic": f"score s{s:.1f} {f}", "estimate": value,
                                 "lower": low_i, "upper": high_i})  # fmt: skip
                contrasts = {
                    "C1 kf 0.8-1.0": ((0.8, "kf"), (1.0, "kf")),
                    "C2 kf 1.2-1.0": ((1.2, "kf"), (1.0, "kf")),
                    "C3 nokf 0.8-1.0": ((0.8, "nokf"), (1.0, "nokf")),
                    "C4 nokf 1.2-1.0": ((1.2, "nokf"), (1.0, "nokf")),
                    "C5 nokf-kf at 1.0": ((1.0, "nokf"), (1.0, "kf")),
                    "kf 0.6-1.0": ((0.6, "kf"), (1.0, "kf")),
                    "kf 1.4-1.0": ((1.4, "kf"), (1.0, "kf")),
                    "nokf 0.6-1.0": ((0.6, "nokf"), (1.0, "nokf")),
                    "nokf 1.4-1.0": ((1.4, "nokf"), (1.0, "nokf")),
                }
                for name, (a, b) in contrasts.items():
                    draws = means[a] - means[b]
                    low_i, high_i = boot.interval(draws=draws)
                    rows.append({"lead": lead_name, "scope": scope, "metric": metric,
                                 "statistic": name, "estimate": point[a] - point[b],
                                 "lower": low_i, "upper": high_i})  # fmt: skip
                for f in common.FILTERS:
                    inside = [s for s in common.SCALES if FLAT_BAND[0] <= s <= FLAT_BAND[1]]
                    stack = np.stack([means[(s, f)] for s in inside])
                    draws = stack.max(axis=0) - stack.min(axis=0)
                    values = [point[(s, f)] for s in inside]
                    low_i, high_i = boot.interval(draws=draws)
                    rows.append({"lead": lead_name, "scope": scope, "metric": metric,
                                 "statistic": f"range 0.8-1.2 {f}",
                                 "estimate": max(values) - min(values),
                                 "lower": low_i, "upper": high_i})  # fmt: skip
    table = pl.DataFrame(rows)
    common.WORK_DIR.mkdir(parents=True, exist_ok=True)
    table.write_parquet(common.WORK_DIR / "scores.parquet")
    with pl.Config(float_precision=3, tbl_rows=1000, fmt_str_lengths=40, tbl_width_chars=140):
        for lead_name in LEAD_RANGES:
            print(f"== {lead_name}: quantile score, sunny, pooled and per Primary ==")
            print(
                table.filter(
                    pl.col("lead") == lead_name,
                    pl.col("metric") == "qs",
                    pl.col("statistic").str.starts_with("score"),
                    pl.col("scope").str.starts_with("sunny"),
                    ~pl.col("scope").str.contains("leave out"),
                ).sort("scope", "statistic")
            )
            print(f"== {lead_name}: contrasts and ranges, qs ==")
            print(
                table.filter(
                    pl.col("lead") == lead_name,
                    pl.col("metric") == "qs",
                    ~pl.col("statistic").str.starts_with("score"),
                    pl.col("scope").is_in(
                        [
                            "sunny",
                            "all half-hours",
                            "winter nights",
                            "all daylight",
                            "night",
                            "darkest third",
                        ]
                    ),
                ).sort("scope", "statistic")
            )
            print(f"== {lead_name}: error of the mean forecast, sunny ==")
            print(
                table.filter(
                    pl.col("lead") == lead_name,
                    pl.col("metric") == "abs_mean",
                    pl.col("scope") == "sunny",
                ).sort("statistic")
            )
    print("== tail calibration, all rows, 0.5 to 168 h ==")
    for (s, f), frame in frames.items():
        line = {}
        for level, column in ((0.01, "q01"), (0.05, "q05"), (0.95, "q95"), (0.99, "q99")):
            line[f"cover<{column}"] = float(np.mean((frame["actual"] < frame[column]).to_numpy()))
            err = (frame["actual"] - frame[column]).to_numpy()
            line[f"pin{level}"] = float(np.mean(np.maximum(level * err, (level - 1.0) * err)))
        print(f"s{s:.1f} {f}", {k: round(v, 4) for k, v in line.items()})
    print("== median minus mean forecast, per Primary (points) ==")
    gaps = pl.concat(
        [
            f.group_by("label")
            .agg(gap=(pl.col("q50") - pl.col("mean")).mean())
            .with_columns(arm=pl.lit(f"s{s:.1f} {fl}"))
            for (s, fl), f in frames.items()
        ]
    ).pivot(on="label", index="arm", values="gap")
    with pl.Config(float_precision=3, tbl_rows=20):
        print(gaps)


if __name__ == "__main__":
    main()
