"""Summarise rung 6 with a wrong battery power: write `report_rung6_wrong_power.md`.

Reads `rung6_wrong_power_fits.parquet` and rung 6's `rung6_fits.parquet` (for `A8`, `A0b`, and
`A3b`, regional sky). Nothing is refitted. Intervals resample the 16 aggregates (4 batteries by 4
solar sets), as in `battery_rung4_summary`.

Run: `uv run python studies/battery_pv_separation/battery_rung6_wrong_power_summary.py`.
"""

from typing import Final

import numpy as np
import polars as pl
from battery_inputs import OUTPUT_DIR
from battery_rung4_summary import PERCENT, SOLAR_SHARES, interval
from battery_rung6 import ARM_LABELS
from battery_rung6_wrong_power import POWER_FACTORS, SKY
from battery_synthetic import AGGREGATE_P99_MW

ARMS: Final[tuple[str, ...]] = ("A0b", "A3b", "A8_p05", "A8", "A8_p2")
LABELS: Final[dict[str, str]] = {
    **ARM_LABELS,
    "A8_p05": "A8 with half the true power and energy capacity",
    "A8_p2": "A8 with double the true power and energy capacity",
}
CONTRASTS: Final[tuple[tuple[str, str], ...]] = (
    ("A8_p05", "A8"),
    ("A8_p2", "A8"),
    ("A8", "A0b"),
    ("A8_p05", "A0b"),
    ("A8_p2", "A0b"),
    ("A8", "A3b"),
    ("A8_p05", "A3b"),
    ("A8_p2", "A3b"),
)
FALSE_ALARM_LIMITS_MW: Final[tuple[float, float]] = (1.0, 2.0)
"""Rule 2's two thresholds: 1% and 2% of the aggregate's 99th percentile."""


def _fmt(*, values: np.ndarray, scale: float, digits: int) -> str:
    mean, low, high = interval(values=values * scale)
    return f"{mean:.{digits}f} [{low:.{digits}f}, {high:.{digits}f}]"


def _wide(*, fits: pl.DataFrame, metric: str) -> pl.DataFrame:
    """Return one row per aggregate with a share above 0 and one column per arm."""
    return (
        fits.filter(pl.col("share") > 0)
        .pivot(on="arm", index=["solar_set", "battery", "share"], values=metric)
        .sort("solar_set", "battery", "share")
    )


def _by_share(*, wide: pl.DataFrame) -> list[tuple[str, pl.DataFrame]]:
    """Return the aggregates of each share, and all shares pooled with one value per unit."""
    pooled = wide.group_by("solar_set", "battery", maintain_order=True).agg(
        pl.col(c).mean() for c in ARMS
    )
    return [(f"{share:.0%}", wide.filter(pl.col("share") == share)) for share in SOLAR_SHARES] + [
        ("Pooled", pooled)
    ]


def _level_table(*, fits: pl.DataFrame, metric: str, scale: float, digits: int) -> list[str]:
    wide = _wide(fits=fits, metric=metric)
    groups = _by_share(wide=wide)
    lines = ["| Arm | " + " | ".join(name for name, _ in groups) + " |"]
    lines.append("|---|" + "---|" * len(groups))
    for arm in ARMS:
        cells = [
            _fmt(values=g[arm].drop_nulls().to_numpy(), scale=scale, digits=digits)
            for _, g in groups
        ]
        lines.append(f"| {arm} | " + " | ".join(cells) + " |")
    return lines


def _contrast_table(*, fits: pl.DataFrame) -> list[str]:
    groups = _by_share(wide=_wide(fits=fits, metric="nmae_of_solar_p99"))
    lines = ["| Contrast | " + " | ".join(name for name, _ in groups) + " |"]
    lines.append("|---|" + "---|" * len(groups))
    for treatment, reference in CONTRASTS:
        cells = [
            _fmt(
                values=(g[treatment] - g[reference]).drop_nulls().to_numpy(),
                scale=PERCENT,
                digits=1,
            )
            for _, g in groups
        ]
        lines.append(f"| {treatment} minus {reference} | " + " | ".join(cells) + " |")
    return lines


def _false_alarm_table(*, fits: pl.DataFrame) -> list[str]:
    zero = fits.filter(pl.col("share") == 0)
    lines = [
        "| Arm | Mean fitted solar at 0% (MW) | Largest at 0% (MW) | "
        + " | ".join(f"False alarms above {limit:g} MW" for limit in FALSE_ALARM_LIMITS_MW)
        + " |",
        "|---|---|---|---|---|",
    ]
    for arm in ARMS:
        values = zero.filter(pl.col("arm") == arm)["fitted_ac_mw"].to_numpy()
        counts = [
            f"{int((values > limit).sum())} of {len(values)}" for limit in FALSE_ALARM_LIMITS_MW
        ]
        lines.append(
            f"| {arm} | {values.mean():.2f} | {values.max():.2f} | " + " | ".join(counts) + " |"
        )
    return lines


def main() -> None:
    """Write the report fragment."""
    rung6 = pl.read_parquet(OUTPUT_DIR / "rung6_fits.parquet").filter(
        (pl.col("sky") == SKY) & pl.col("arm").is_in(["A8", "A0b", "A3b"])
    )
    wrong = pl.read_parquet(OUTPUT_DIR / "rung6_wrong_power_fits.parquet")
    fits = pl.concat([rung6, wrong], how="diagonal")
    lines = [
        "## Rung 6: wrong battery power",
        "",
        (
            "Rung 6 gave the joint model the true battery's 99th-percentile power and rung 3's "
            "fitted energy capacity. Rungs 7 and 7b sweep the power instead, because the real "
            "BMUs' batteries are unknown. This run refits rung 6's `A8` (regional sky, all 16 "
            "aggregates and all shares) with the power and the energy capacity both multiplied "
            f"by {POWER_FACTORS['A8_p05']:g} (`A8_p05`) and by {POWER_FACTORS['A8_p2']:g} "
            "(`A8_p2`), which keeps the duration. `A8`, `A0b` (no battery), and `A3b` (true "
            "battery removed) are rung 6's. Intervals resample the 16 aggregates."
        ),
        "",
        "### Arms",
        "",
        *[f"- {arm}: {LABELS[arm]}" for arm in ARMS],
        "",
        "### Solar series error (% of solar p99)",
        "",
        *_level_table(fits=fits, metric="nmae_of_solar_p99", scale=PERCENT, digits=1),
        "",
        "### Energy ratio (recovered solar energy over true solar energy)",
        "",
        *_level_table(fits=fits, metric="energy_ratio", scale=1.0, digits=2),
        "",
        "### AC capacity ratio (fitted over the direct fit to the solar half)",
        "",
        *_level_table(fits=fits, metric="ac_ratio", scale=1.0, digits=2),
        "",
        "### Contrasts in solar error (points of solar p99; negative favours the first arm)",
        "",
        *_contrast_table(fits=fits),
        "",
        (
            "### False alarms among the 16 batteries-only aggregates (rule 2: fitted solar above "
            f"{FALSE_ALARM_LIMITS_MW[0] / AGGREGATE_P99_MW:.0%} and "
            f"{FALSE_ALARM_LIMITS_MW[1] / AGGREGATE_P99_MW:.0%} of the aggregate's 99th percentile)"
        ),
        "",
        *_false_alarm_table(fits=fits),
    ]
    (OUTPUT_DIR / "report_rung6_wrong_power.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
