"""Read the gate's interval files: coverage, width ratios, bias, the planted failure, and power.

Coverage is the share of identified (node, parameter) rows whose 80% interval holds the truth,
averaged within each fleet and then across fleets; the standard error is the fleet-to-fleet one
(fleet-clustered). The width ratio compares an arm with its reference (H with G, NH with N) on the
same (node, parameter) rows of the same fleet.

    uv run python ml-experiments/00117-*/e00117_gate_report.py --tag power [--tag other ...]
"""

import argparse
import json
from typing import Any, Final

import e00117_common as common
import numpy as np
import polars as pl

REFERENCE: Final[dict[str, str]] = {
    "h_lo": "g",
    "h_hi": "g",
    "h_planted": "g",
    "nh_lo": "n",
    "nh_hi": "n",
    "n": "g",
}
"""The arm each arm's widths are read against."""
WIDTH_FAMILIES: Final[tuple[str, ...]] = ("slope", "scale", "capacity")
LEVEL: Final[float] = 0.983
"""98.3% matches three width families after a Bonferroni correction."""
N_RESAMPLES: Final[int] = 4000


def load(*, tags: list[str]) -> pl.DataFrame:
    """Read every interval file of the tags into one frame."""
    frames = [
        pl.read_parquet(path)
        for tag in tags
        for path in sorted((common.WORK_DIR / "gate" / tag).glob("*.parquet"))
    ]
    return pl.concat(frames)


def covered(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Add the columns `cover`, `width`, and `error` to the identified rows."""
    return frame.filter(pl.col("identified")).with_columns(
        cover=((pl.col("lo") <= pl.col("truth")) & (pl.col("truth") <= pl.col("hi"))).cast(
            pl.Float64
        ),
        width=pl.col("hi") - pl.col("lo"),
        error=pl.col("point") - pl.col("truth"),
    )


def pooled(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Pool the sets: every (set, fleet) pair becomes one fleet of set 0."""
    return frame.with_columns(
        fleet=pl.col("set_id") * 1000 + pl.col("fleet"), set_id=pl.lit(0, dtype=pl.Int64)
    )


def coverage_table(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Return coverage and its fleet-clustered standard error by set, cell, arm, and family."""
    keys = ["set_id", "cell", "arm", "family"]
    per_fleet = frame.group_by([*keys, "fleet"]).agg(
        pl.col("cover").mean().alias("coverage"),
        pl.col("width").median().alias("width"),
        pl.col("error").mean().alias("bias"),
        pl.col("width").mean().alias("mean_width"),
    )
    return (
        per_fleet.group_by(keys)
        .agg(
            pl.len().alias("fleets"),
            pl.col("coverage").mean(),
            (pl.col("coverage").std() / pl.len().sqrt()).alias("coverage_se"),
            pl.col("width").median().alias("median_width"),
            (pl.col("bias").abs() / pl.col("mean_width")).median().alias("bias_over_width"),
        )
        .sort(keys)
    )


def width_ratio_table(*, frame: pl.DataFrame, rng: np.random.Generator) -> pl.DataFrame:
    """Return the median width ratio of each arm against its reference, with a fleet interval."""
    rows = []
    wide = frame.select("set_id", "cell", "fleet", "node", "functional", "family", "arm", "width")
    for (set_id, cell), group in wide.group_by(["set_id", "cell"], maintain_order=True):
        for arm, reference in REFERENCE.items():
            if arm not in group["arm"].unique().to_list() or reference == arm:
                continue
            mine = group.filter(pl.col("arm") == arm)
            base = group.filter(pl.col("arm") == reference).select(
                "fleet", "node", "functional", pl.col("width").alias("base")
            )
            joined = mine.join(base, on=["fleet", "node", "functional"]).with_columns(
                ratio=pl.col("width") / pl.col("base")
            )
            for family, part in joined.group_by("family", maintain_order=True):
                per_fleet = part.group_by("fleet").agg(pl.col("ratio").median())["ratio"].to_numpy()
                draws = rng.choice(per_fleet, size=(N_RESAMPLES, len(per_fleet)))
                medians = np.median(draws, axis=1)
                low, high = np.quantile(medians, [(1 - LEVEL) / 2, 1 - (1 - LEVEL) / 2])
                rows.append(
                    {
                        "set_id": set_id,
                        "cell": cell,
                        "arm": arm,
                        "reference": reference,
                        "family": family[0],
                        "fleets": len(per_fleet),
                        "median_ratio": float(np.median(per_fleet)),
                        "low": float(low),
                        "high": float(high),
                    }
                )
    return pl.DataFrame(rows)


def power_check(*, coverage: pl.DataFrame) -> dict[str, Any]:
    """Return the planted failure's coverage gap to G and the standard errors, by family."""
    unlike = coverage.filter(pl.col("cell") == "unlike")
    out: dict[str, Any] = {}
    for family in unlike["family"].unique().sort().to_list():
        part = unlike.filter(pl.col("family") == family)
        g = part.filter(pl.col("arm") == "g")
        p = part.filter(pl.col("arm") == "h_planted")
        if g.is_empty() or p.is_empty():
            continue
        gap = common.as_float(g["coverage"].mean()) - common.as_float(p["coverage"].mean())
        se = float(
            np.hypot(
                common.as_float(g["coverage_se"].mean()), common.as_float(p["coverage_se"].mean())
            )
        )
        out[family] = {
            "g_coverage": common.as_float(g["coverage"].mean()),
            "planted_coverage": common.as_float(p["coverage"].mean()),
            "gap_points": 100 * gap,
            "se_points": 100 * se,
            "gap_in_se": gap / se if se > 0 else float("nan"),
        }
    return out


def main() -> None:
    """Print the gate's tables and write a JSON report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", nargs="+", required=True)
    arguments = parser.parse_args()
    frame = covered(frame=load(tags=arguments.tag))
    pool = pooled(frame=frame)
    coverage = coverage_table(frame=pool)
    ratios = width_ratio_table(frame=pool, rng=np.random.default_rng(0))
    by_set_coverage = coverage_table(frame=frame)
    by_set_ratios = width_ratio_table(frame=frame, rng=np.random.default_rng(0))
    power = power_check(coverage=coverage)
    with pl.Config(tbl_rows=200, tbl_cols=12, fmt_str_lengths=30):
        print(coverage)
        print(ratios)
    print(json.dumps(power, indent=2))
    out = common.WORK_DIR / "gate" / ("_".join(arguments.tag) + "_report.json")
    out.write_text(
        json.dumps(
            {
                "coverage": coverage.to_dicts(),
                "width_ratio": ratios.to_dicts(),
                "coverage_by_set": by_set_coverage.to_dicts(),
                "width_ratio_by_set": by_set_ratios.to_dicts(),
                "power": power,
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
