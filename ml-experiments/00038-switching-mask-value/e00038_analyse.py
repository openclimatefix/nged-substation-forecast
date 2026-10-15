"""Tier A contrasts: month-resampled intervals, per-event gains, quiet rows, concentration.

Reads `tier_a_rows.parquet`. Writes `tier_a_contrasts.parquet`, `tier_a_events.parquet`, and
`tier_a_months.parquet`, and prints the numbers the write-up quotes.
"""

from typing import Final

import numpy as np
import polars as pl
from e00038_base import SOURCE_DIR, WORK_DIR
from scipy.stats import binomtest

N_RESAMPLES: Final[int] = 10_000
LONG_EVENT_HOURS: Final[float] = 24.0
AFTER_EVENT_STAMPS: Final[int] = 16 * 48
SPEC_TO_FIXED: Final[dict[str, str]] = {
    "d1_mask14": "d1fx_mask14",
    "d1_mask7": "d1fx_mask7",
    "random14": "randomfx14",
}
CONTRASTS: Final[tuple[tuple[str, str, str], ...]] = (
    # (label, arm, reference); the first is the primary contrast.
    ("primary: d1_mask7 vs plain7", "d1_mask7", "plain7"),
    ("d1_mask14 vs plain14", "d1_mask14", "plain14"),
    ("d0w48_mask14 vs plain14", "d0w48_mask14", "plain14"),
    ("oracle_conf14 vs plain14", "oracle_conf14", "plain14"),
    ("oracle_logged14 vs plain14", "oracle_logged14", "plain14"),
    ("oracle_conf7 vs plain7", "oracle_conf7", "plain7"),
    ("random14 vs plain14", "random14", "plain14"),
    ("random7 vs plain7", "random7", "plain7"),
    ("plain7 vs plain14", "plain7", "plain14"),
    ("post hoc: d1fx_mask7 vs plain7", "d1fx_mask7", "plain7"),
    ("post hoc: d1fx_mask14 vs plain14", "d1fx_mask14", "plain14"),
    ("post hoc: d0fx_mask14 vs plain14", "d0fx_mask14", "plain14"),
    ("post hoc: randomfx14 vs plain14", "randomfx14", "plain14"),
    ("post hoc: randomfx7 vs plain7", "randomfx7", "plain7"),
    ("d1_persist14 vs plain14 (all rows)", "d1_persist14", "plain14"),
    ("post hoc: d1fx_persist14 vs plain14 (all rows)", "d1fx_persist14", "plain14"),
)
ARMS: Final[tuple[str, ...]] = tuple(
    sorted({a for _, a, r in CONTRASTS} | {r for _, a, r in CONTRASTS})
)


def arms_in(*, rows: pl.DataFrame, metric: str) -> list[str]:
    """Return the arms that have a `metric` column in `rows`."""
    prefix = f"{metric}_"
    return [c[len(prefix) :] for c in rows.columns if c.startswith(prefix)]


def month_sums(*, rows: pl.DataFrame, metric: str) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Return the months, the row counts per month, and per-month sums of each arm's `metric`."""
    arms = arms_in(rows=rows, metric=metric)
    grouped = (
        rows.group_by("month")
        .agg(pl.len().alias("n"), *[pl.col(f"{metric}_{a}").sum().alias(a) for a in arms])
        .sort("month")
    )
    return (
        grouped["month"].to_list(),
        grouped["n"].to_numpy().astype(float),
        np.stack([grouped[a].to_numpy().astype(float) for a in arms], axis=1),
    )


def contrast_row(
    *, rows: pl.DataFrame, metric: str, label: str, arm: str, ref: str, view: str
) -> dict:
    """Point estimate, month-resampled interval, odd/even split, and largest-month-removed."""
    months, n, sums = month_sums(rows=rows, metric=metric)
    column = {a: i for i, a in enumerate(arms_in(rows=rows, metric=metric))}
    idx = np.random.default_rng(0).integers(0, len(months), size=(N_RESAMPLES, len(months)))
    arm_sum, ref_sum = sums[:, column[arm]], sums[:, column[ref]]
    draws = 100.0 * (arm_sum[idx].sum(axis=1) / ref_sum[idx].sum(axis=1) - 1.0)
    odd = np.array([int(m[5:]) % 2 == 1 for m in months])
    gain = ref_sum - arm_sum
    drop = int(np.argmax(np.abs(gain)))
    keep = np.arange(len(months)) != drop
    return {
        "view": view,
        "metric": metric,
        "contrast": label,
        "arm": arm,
        "reference": ref,
        "n_rows": int(n.sum()),
        "reference_mean": float(ref_sum.sum() / n.sum()),
        "arm_mean": float(arm_sum.sum() / n.sum()),
        "change_points": float((arm_sum.sum() - ref_sum.sum()) / n.sum()),
        "change_pct": float(100.0 * (arm_sum.sum() / ref_sum.sum() - 1.0)),
        "lo": float(np.percentile(draws, 2.5)),
        "hi": float(np.percentile(draws, 97.5)),
        "odd_pct": float(100.0 * (arm_sum[odd].sum() / ref_sum[odd].sum() - 1.0)),
        "even_pct": float(100.0 * (arm_sum[~odd].sum() / ref_sum[~odd].sum() - 1.0)),
        "largest_month": months[drop],
        "largest_month_share_of_gain": float(gain[drop] / gain.sum()) if gain.sum() else np.nan,
        "without_largest_pct": float(100.0 * (arm_sum[keep].sum() / ref_sum[keep].sum() - 1.0)),
    }


def long_events() -> pl.DataFrame:
    """Confirmed transfers of 24 hours or more, one row per side with telemetry."""
    oracle = pl.read_parquet(SOURCE_DIR / "oracle_real.parquet").filter(
        "primary", "confirmed", pl.col("hours") >= LONG_EVENT_HOURS
    )
    events = oracle.with_row_index("event")
    sides = [
        events.select(
            "event",
            "sheet",
            "first_stamp",
            "last_stamp_exclusive",
            "hours",
            "transfer_mw",
            "share",
            series=s,
        )
        for s in ("series_a", "series_b")
    ]
    return pl.concat(sides).drop_nulls("series")


def per_event_gains(*, rows: pl.DataFrame, arms: list[str], ref_of: dict[str, str]) -> pl.DataFrame:
    """Sum of threshold-weighted error per event window and arm, and the gain against reference."""
    out = []
    for event in long_events().iter_rows(named=True):
        window = rows.filter(
            pl.col("series") == event["series"],
            pl.col("band") == "A",
            pl.col("out_of_event"),
            pl.col("v") >= event["first_stamp"],
            pl.col("v") < event["last_stamp_exclusive"] + AFTER_EVENT_STAMPS,
        )
        if window.height == 0:
            continue
        record = {
            k: event[k] for k in ("event", "sheet", "series", "hours", "share", "transfer_mw")
        }
        record["n_rows"] = window.height
        for arm in arms:
            record[f"tw_{arm}"] = float(window[f"tw_{arm}"].sum())
            record[f"tw_{ref_of[arm]}"] = float(window[f"tw_{ref_of[arm]}"].sum())
        out.append(record)
    frame = pl.DataFrame(out)
    # One gain per event, pooling the two sides' rows.
    pooled = frame.group_by("event").agg(
        pl.col("sheet").first(),
        pl.col("hours").first(),
        pl.col("share").first(),
        pl.col("transfer_mw").first(),
        pl.col("n_rows").sum(),
        *[pl.col(c).sum() for c in frame.columns if c.startswith("tw_")],
    )
    return pooled.sort("share", descending=True)


def main() -> None:
    """Compute every contrast and print the summary."""
    pl.Config.set_tbl_rows(80)
    pl.Config.set_tbl_width_chars(220)
    pl.Config.set_tbl_cols(30)
    rows = pl.read_parquet(WORK_DIR / "tier_a_rows.parquet")
    table = []
    for band, band_name in (("A", "3.5 to 48 h"), ("B", "day 6 to 7")):
        in_band = rows.filter(pl.col("band") == band)
        for label, arm, ref in CONTRASTS:
            persist = "persist" in arm
            subsets = (
                {"all": in_band}
                if persist
                else {
                    "out_of_event": in_band.filter("out_of_event"),
                    "quiet": in_band.filter("quiet"),
                }
            )
            if not persist:
                subsets["all"] = in_band
            table.extend(
                contrast_row(rows=subset, metric=metric, label=label, arm=arm, ref=ref, view=view)
                | {"band": band_name}
                for view, subset in subsets.items()
                for metric in ("tw", "ae")
            )
    contrasts = pl.DataFrame(table)
    contrasts.write_parquet(WORK_DIR / "tier_a_contrasts.parquet")
    show = [
        "contrast",
        "n_rows",
        "reference_mean",
        "arm_mean",
        "change_pct",
        "lo",
        "hi",
        "odd_pct",
        "even_pct",
        "without_largest_pct",
        "largest_month",
        "largest_month_share_of_gain",
    ]
    for view in ("out_of_event", "quiet", "all"):
        print(f"=== band 3.5-48h, tw, view {view}")
        print(
            contrasts.filter(
                pl.col("band") == "3.5 to 48 h", pl.col("metric") == "tw", pl.col("view") == view
            ).select(show)
        )
    print("=== band 3.5-48h, ae (points), out_of_event and quiet")
    print(
        contrasts.filter(
            pl.col("band") == "3.5 to 48 h",
            pl.col("metric") == "ae",
            pl.col("view").is_in(["out_of_event", "quiet"]),
        ).select(["view", "contrast", "reference_mean", "arm_mean", "change_points", "change_pct"])
    )
    print("=== band day 6-7, tw, out_of_event")
    print(
        contrasts.filter(
            pl.col("band") == "day 6 to 7",
            pl.col("metric") == "tw",
            pl.col("view") == "out_of_event",
        ).select(show)
    )
    # Per event
    arms = [
        "d1_mask7",
        "d1_mask14",
        "d1fx_mask7",
        "d1fx_mask14",
        "d0fx_mask14",
        "oracle_conf14",
        "randomfx14",
        "random14",
    ]
    ref_of = {a: ("plain7" if a.endswith("7") else "plain14") for a in arms}
    events = per_event_gains(rows=rows, arms=arms, ref_of=ref_of)
    events.write_parquet(WORK_DIR / "tier_a_events.parquet")
    print(f"long confirmed events with a window: {events.height}")
    for arm in arms:
        ref = ref_of[arm]
        gain = events[f"tw_{ref}"] - events[f"tw_{arm}"]
        better = int((gain > 0).sum())
        test = binomtest(better, events.height, 0.5, alternative="greater")
        print(f"{arm} vs {ref}: better on {better}/{events.height}, binomial p={test.pvalue:.3f}")
    months, _, sums = month_sums(
        rows=rows.filter(pl.col("band") == "A", pl.col("out_of_event")), metric="tw"
    )
    column = {a: i for i, a in enumerate(ARMS)}
    pl.DataFrame({"month": months, **{a: sums[:, column[a]] for a in ARMS}}).write_parquet(
        WORK_DIR / "tier_a_months.parquet"
    )


if __name__ == "__main__":
    main()
