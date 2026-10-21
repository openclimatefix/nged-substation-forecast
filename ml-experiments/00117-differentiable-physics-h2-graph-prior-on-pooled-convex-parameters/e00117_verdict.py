"""Apply the pre-registered decision rule and the audit triggers to the finished results.

Reads the gate files (`e00117_gate_report.py`), the controls, the real-data report, the real
intervals, and the leave-one-out table, and prints each reading with its number so that the page
quotes what the rule computed.

    uv run python ml-experiments/00117-*/e00117_verdict.py --gate-tags gate gate_hi
"""

import argparse
import json
from typing import Any, Final

import e00117_common as common
import e00117_gate_report as gate_report
import numpy as np
import polars as pl

WIDTH_FAMILIES: Final[tuple[str, ...]] = ("slope", "scale", "capacity")
MARGIN: Final[float] = 0.03
SCORED_HALF_WIDTH_LIMIT: Final[float] = 0.025


def _read(path: str) -> Any:
    return json.loads((common.WORK_DIR / path).read_text())


def coverage_readings(*, coverage: pl.DataFrame) -> dict[str, Any]:
    """Return readings (a) and (c) and the refuted test for H and NH, per width family."""
    out: dict[str, Any] = {}
    for cell in ("smooth", "unlike"):
        for arm in ("h_lo", "nh_lo"):
            for family in (*WIDTH_FAMILIES, "holiday"):
                part = coverage.filter((pl.col("cell") == cell) & (pl.col("family") == family))
                mine = part.filter(pl.col("arm") == arm)["coverage"].mean()
                base = part.filter(pl.col("arm") == "g")["coverage"].mean()
                if mine is None or base is None:
                    continue
                out[f"{cell}/{arm}/{family}"] = {
                    "coverage": 100 * common.as_float(mine),
                    "g_coverage": 100 * common.as_float(base),
                    "gap": 100 * (common.as_float(mine) - common.as_float(base)),
                }
    return out


def reading_a(*, readings: dict[str, Any]) -> bool:
    """(a) Smooth-cell coverage of H and NH within 5 points of G's, in every width family."""
    return all(
        abs(readings[f"smooth/{arm}/{family}"]["gap"]) <= 5.0
        for arm in ("h_lo", "nh_lo")
        for family in WIDTH_FAMILIES
    )


def reading_c(*, readings: dict[str, Any]) -> bool:
    """(c) Unlike-neighbours coverage within 10 points of G's, in every width family."""
    return all(
        abs(readings[f"unlike/{arm}/{family}"]["gap"]) <= 10.0
        for arm in ("h_lo", "nh_lo")
        for family in WIDTH_FAMILIES
    )


def refuted_by_coverage(*, readings: dict[str, Any]) -> list[str]:
    """Return the smooth-cell family and arm pairs under 60% and over 10 points below G's."""
    return [
        key
        for key, value in readings.items()
        if key.startswith("smooth/")
        and key.split("/")[2] in WIDTH_FAMILIES
        and value["coverage"] < 60.0
        and value["gap"] < -10.0
    ]


def reading_b(*, ratios: pl.DataFrame) -> dict[str, Any]:
    """(b) Width ratio at most 0.85 in at least two of three families, interval wholly below 1."""
    out = {}
    for arm in ("h_lo", "nh_lo"):
        part = ratios.filter(
            (pl.col("cell") == "smooth")
            & (pl.col("arm") == arm)
            & pl.col("family").is_in(WIDTH_FAMILIES)
        )
        passed = part.filter((pl.col("median_ratio") <= 0.85) & (pl.col("high") < 1.0))
        out[arm] = {
            "families_passing": passed["family"].to_list(),
            "ratios": {r["family"]: r["median_ratio"] for r in part.to_dicts()},
            "passed": passed.height >= 2,
        }
    return out


def contrast_table(*, report: dict) -> dict[str, dict[str, Any]]:
    """Return the quantile-score contrasts of both years by name."""
    out: dict[str, dict[str, Any]] = {}
    for cut in ("scored", "replicate"):
        for row in report[cut]["contrasts"]:
            if row["metric"] == "qs":
                out.setdefault(row["name"], {})[cut] = row
    return out


def noninferior(*, row: dict[str, Any]) -> bool:
    """True if the interval's upper end is at most the margin."""
    return "upper" in row and row["upper"] <= MARGIN


def main() -> None:
    """Print the readings and the verdict row they imply."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--gate-tags", nargs="+", default=["gate"])
    arguments = parser.parse_args()
    frame = gate_report.pooled(
        frame=gate_report.covered(frame=gate_report.load(tags=arguments.gate_tags))
    )
    coverage = gate_report.coverage_table(frame=frame)
    ratios = gate_report.width_ratio_table(frame=frame, rng=np.random.default_rng(0))
    power = gate_report.power_check(coverage=coverage)
    readings = coverage_readings(coverage=coverage)
    controls = _read("controls.json")
    report = _read("report.json")
    contrasts = contrast_table(report=report)
    real_scored = _read("real_intervals_scored.json")
    print("controls passed:", {k: controls[k]["passed"] for k in ("p1", "p2", "p3", "p4", "p5")})
    print("planted failure (gap to G in points, in standard errors):")
    for family, row in power.items():
        print(f"  {family}: {row['gap_points']:.1f} points, {row['gap_in_se']:.1f} SE")
    planted_sees = any(row["gap_points"] >= 10.0 for row in power.values())
    print("planted failure sees over-shrinkage (>= 10 points in one family):", planted_sees)
    print(
        "(a) smooth coverage within 5 points of G, all width families:",
        reading_a(readings=readings),
    )
    print("(c) unlike coverage within 10 points of G:", reading_c(readings=readings))
    print("refuted by smooth-cell coverage:", refuted_by_coverage(readings=readings))
    print("(b) width:", json.dumps(reading_b(ratios=ratios), indent=1))
    for key, value in readings.items():
        print(
            f"  {key}: coverage {value['coverage']:.1f}%, G {value['g_coverage']:.1f}%, "
            f"gap {value['gap']:+.1f}"
        )
    for name in ("R1 H - G", "R2 NH - N"):
        for cut in ("scored", "replicate"):
            row = contrasts[name][cut]
            half = (row["upper"] - row["lower"]) / 2
            print(
                f"{name} {cut}: {row['difference']:+.3f} "
                f"[{row['lower']:+.3f}, {row['upper']:+.3f}] "
                f"half-width {half:.3f} non-inferior {noninferior(row=row)}"
            )
    print("H over H-perm (slope), scored:", real_scored["h_over_h_perm"]["slope"])
    for name in ("R1w H - G, winter", "R2w NH - N, winter"):
        row = contrasts[name]["scored"]
        print(f"{name} scored upper {row['upper']:+.3f}; hold back: {row['upper'] > MARGIN}")


if __name__ == "__main__":
    main()
