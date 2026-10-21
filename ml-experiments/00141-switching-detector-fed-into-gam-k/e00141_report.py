"""Collate the injected-world scores of experiment 141 into the numbers the write-up quotes.

Reads the per-(site, world) score files, pairs every arm with K0 on the same rows, and prints the
paired differences in quantile score with 95% intervals that resample whole calendar months of the
target time. Positive means worse than K0. Also prints the mean-forecast error beside each
contrast, the q05 and q95 coverage, the leave-one-site-out check, and the tuning table.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Final, cast

import e00141_fed as fed
import numpy as np
import polars as pl
from e00141_common import WORK_DIR

SCORES: Final[Path] = WORK_DIR / "scores"
N_RESAMPLES: Final[int] = 20_000
ARMS: Final[tuple[str, ...]] = (
    "SKF",
    "SHE",
    "SORACLE",
    "PLACEBO",
    "VARINF",
    "SKF_END",
    "SORACLE_END",
)


def load() -> pl.DataFrame:
    """All scored rows with the target month and the K0 score on the same row."""
    frame = pl.concat([pl.read_parquet(p) for p in (SCORES / "score").glob("*.parquet")])
    end_times = fed.get_site("D5").cache.end_times
    starts = end_times - np.timedelta64(30, "m")
    month = (starts.astype("datetime64[M]").astype(np.int64) + 1970 * 12).astype(np.int64)
    hour = ((starts - starts.astype("datetime64[D]")) / np.timedelta64(1, "h")).astype(np.float64)
    calendar = pl.DataFrame(
        {"position": np.arange(len(end_times)), "month": month, "utc_hour": hour}
    )
    k0 = frame.filter(pl.col("arm") == "K0").select(
        "site",
        "world",
        "row",
        "period",
        k0_qs="qs",
        k0_err_mean=pl.col("err_mean").abs(),
    )
    return (
        frame.join(k0, on=["site", "world", "row", "period"])
        .join(calendar, on="position")
        .with_columns(
            diff=pl.col("qs") - pl.col("k0_qs"),
            err_mean_diff=pl.col("err_mean").abs() - pl.col("k0_err_mean"),
            winter_night=(pl.col("month") % 12 + 1).is_in([11, 12, 1, 2, 3])
            & (pl.col("utc_hour") < 6.0),
        )
    )


def pooled(*, frame: pl.DataFrame, column: str) -> tuple[np.ndarray, np.ndarray]:
    """Per (site, month): the sum and count of `column`, as arrays over months, for resampling."""
    grouped = frame.group_by("site", "month").agg(s=pl.col(column).sum(), n=pl.len())
    months = sorted(grouped["month"].unique().to_list())
    sites = sorted(grouped["site"].unique().to_list())
    sums = np.zeros((len(sites), len(months)))
    counts = np.zeros((len(sites), len(months)))
    for row in grouped.iter_rows(named=True):
        sums[sites.index(row["site"]), months.index(row["month"])] = row["s"]
        counts[sites.index(row["site"]), months.index(row["month"])] = row["n"]
    return sums, counts


def interval(*, frame: pl.DataFrame, column: str, seed: int = 141) -> dict[str, float]:
    """Equal-weight-over-sites mean of `column` with a 95% month-resampled interval."""
    sums, counts = pooled(frame=frame, column=column)
    rng = np.random.default_rng(seed)
    n_months = sums.shape[1]

    def estimate(index: np.ndarray) -> float:
        s, c = sums[:, index].sum(axis=1), counts[:, index].sum(axis=1)
        ok = c > 0
        return float(np.mean(s[ok] / c[ok]))

    point = estimate(np.arange(n_months))
    draws = np.array([estimate(rng.integers(0, n_months, n_months)) for _ in range(N_RESAMPLES)])
    return {
        "mean": point,
        "lo": float(np.quantile(draws, 0.025)),
        "hi": float(np.quantile(draws, 0.975)),
        "rows": int(counts.sum()),
        "months": n_months,
    }


def main() -> None:
    """Print and store the headline numbers."""
    frame = load()
    out: dict = {}
    for tag, mask in (
        ("alarm48", pl.col("in_alarm48")),
        ("transfer", pl.col("in_transfer")),
        ("tail", pl.col("in_event") & ~pl.col("in_transfer")),
        ("all_event_rows", pl.col("in_event") | pl.col("in_alarm48")),
    ):
        for arm in ARMS:
            sub = frame.filter((pl.col("arm") == arm) & mask)
            out[f"{tag}/{arm}"] = {
                "qs": interval(frame=sub, column="diff"),
                "mean_forecast_abs_error": interval(frame=sub, column="err_mean_diff"),
            }
    sub = frame.filter(pl.col("arm") == "K0")
    out["K0_absolute_qs_transfer"] = interval(
        frame=sub.filter(pl.col("in_transfer")), column="k0_qs"
    )
    out["K0_absolute_qs_alarm48"] = interval(frame=sub.filter(pl.col("in_alarm48")), column="k0_qs")
    out["K0_absolute_qs_all_event_rows"] = interval(
        frame=sub.filter(pl.col("in_event") | pl.col("in_alarm48")), column="k0_qs"
    )
    for arm in ("K0", "SKF", "SORACLE"):
        rows = frame.filter((pl.col("arm") == arm) & pl.col("in_alarm48"))
        night = rows.filter(pl.col("winter_night"))
        out[f"coverage/{arm}"] = {
            "q95_alarm48": float(cast("float", (rows["actual"] <= rows["q95"]).mean())),
            "q05_alarm48": float(cast("float", (rows["actual"] >= rows["q05"]).mean())),
            "q95_winter_night_proxy": float(cast("float", (night["actual"] <= night["q95"]).mean()))
            if night.height
            else None,
            "winter_night_rows": night.height,
        }
    leave = {}
    for site in sorted(frame["site"].unique().to_list()):
        sub = frame.filter(
            (pl.col("arm") == "SKF") & pl.col("in_alarm48") & (pl.col("site") != site)
        )
        leave[site] = interval(frame=sub, column="diff")["mean"]
    out["leave_one_site_out_SKF_alarm48"] = leave
    out["by_site_SKF_alarm48"] = {
        site: float(cast("float", g["diff"].mean()))
        for (site,), g in frame.filter((pl.col("arm") == "SKF") & pl.col("in_alarm48")).group_by(
            "site"
        )
    }
    out["by_period_SKF_alarm48"] = {
        period: float(cast("float", g["diff"].mean()))
        for (period,), g in frame.filter((pl.col("arm") == "SKF") & pl.col("in_alarm48")).group_by(
            "period"
        )
    }
    out["median_minus_mean_forecast_K0_alarm48"] = {
        site: float(cast("float", (g["err_median"] - g["err_mean"]).mean()))
        for (site,), g in frame.filter((pl.col("arm") == "K0") & pl.col("in_alarm48")).group_by(
            "site"
        )
    }
    out["tuning"] = json.loads((SCORES / "chosen.json").read_text())
    (WORK_DIR / "report.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
