"""Score the real series: shares, year-to-year stability, interval widths, and contrast C3.

uv run python ml-experiments/00073-pvdis-6-orientation-mix-and-dc-ac-ratio/e00073_score_real.py
"""

import e00073_bootstrap as boot
import e00073_common as common
import numpy as np
import polars as pl


def stability(*, real: pl.DataFrame, arm: str) -> pl.DataFrame:
    """Return the total-variation distance between the shares of consecutive season-years."""
    rows = []
    shares = real.filter(pl.col("arm") == arm).sort("site", "year")
    for site in common.DIRECTIONAL_PRIMARIES:
        block = shares.filter(pl.col("site") == site).drop_nulls(["east", "south", "west"])
        values = block.select("east", "south", "west").to_numpy()
        years = block["year"].to_list()
        for k in range(1, len(years)):
            if years[k] == years[k - 1] + 1:
                change = 0.5 * np.abs(values[k] - values[k - 1]).sum()
                rows.append(
                    {"site": site, "year": years[k], "arm": arm, "tv": boot.as_float(change)}
                )
    return pl.DataFrame(rows)


def main() -> None:
    """Print and write the real-series report."""
    real = pl.read_parquet(common.REAL_PATH)
    heldout = pl.read_parquet(common.HELDOUT_PATH)
    lines = []
    for arm in ("O1", "O2"):
        s = stability(real=real, arm=arm)
        lines.append(
            f"Year-to-year change in shares, arm {arm}: median total-variation distance "
            f"{boot.as_float(s['tv'].median()):.3f} over {s.height} consecutive-year pairs "
            f"(the pooled arm holds its mix fixed across years, so this test does not apply to it)"
        )
    pooled = real.filter(pl.col("arm") == "O3_0.0")
    width = np.mean(
        [
            (pooled[f"{o}_high"] - pooled[f"{o}_low"]).to_numpy().mean()
            for o in ("east", "south", "west")
        ]
    )
    lines.append(f"Pooled-arm 95% intervals: mean width {width:.2f} of a share (range 0 to 1)")
    lines.append("Pooled shares by Primary (east, south, west):")
    for row in pooled.sort("site").iter_rows(named=True):
        name = common.PRIMARY_NAMES[row["site"]]
        lines.append(
            f"  {name:22s} total {row['total']:.2f} MW  "
            f"E {row['east']:.2f} ({row['east_low']:.2f}-{row['east_high']:.2f})  "
            f"S {row['south']:.2f} ({row['south_low']:.2f}-{row['south_high']:.2f})  "
            f"W {row['west']:.2f} ({row['west_low']:.2f}-{row['west_high']:.2f})"
        )
    per_primary = heldout.group_by("site").agg(pl.col("change").mean()).sort("site")
    values = per_primary["change"].to_numpy()
    mean, low, high = boot.cluster_interval(values=values, clusters=per_primary["site"].to_numpy())
    lines.append(
        f"\nC3, relative change in held-out RMSE, fitted pooled mix against south-only: "
        f"mean {mean:+.4f}, interval {low:+.4f} to {high:+.4f}; gain if upper < -0.005: "
        f"{high < -0.005}"
    )
    lines.append(
        "Per Primary: "
        + ", ".join(
            f"{common.PRIMARY_NAMES[s]} {v:+.4f}"
            for s, v in zip(per_primary["site"], values, strict=True)
        )
    )
    lines.append(f"Held-out year-folds scored: {heldout.height}")
    text = "\n".join(lines)
    (common.WORK_DIR / "report_real.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
