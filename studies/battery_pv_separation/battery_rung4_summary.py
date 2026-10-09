"""Rung 4 summary: intervals, planned contrasts B3 and B4, and the report fragment.

Reads `rung4_fits.parquet` and writes `rung4_intervals.parquet` and `report_rung4.md`. Nothing is
refitted. A fit gives one number per aggregate, so intervals resample whole aggregates: the 16
aggregates (4 batteries by 4 solar sets) for a pooled interval, or the 4 solar sets for one
battery's interval. The aggregates share their battery series or their solar series, so the
resample is a rough guide to the spread, not a test that covers a different battery or solar site.

Run: `uv run python studies/battery_pv_separation/battery_rung4_summary.py`.
"""

from typing import Final

import numpy as np
import polars as pl
from battery_inputs import BATTERIES, NAMES, OUTPUT_DIR
from battery_rung4 import ARM_LABELS, ARMS, SKIES
from battery_synthetic import SOLAR_SET_LABELS, SOLAR_SETS, price_regressors

N_RESAMPLES: Final[int] = 2000
RESAMPLE_SEED: Final[int] = 4
PERCENT: Final[float] = 100.0
SOLAR_SHARES: Final[tuple[float, ...]] = (0.1, 0.25, 0.5)
METRICS: Final[tuple[str, ...]] = (
    "nmae_of_solar_p99",
    "energy_ratio",
    "ac_ratio",
    "correlation",
)
CONTRASTS: Final[tuple[tuple[str, str, str], ...]] = (
    ("A1", "A0", "planned B3"),
    ("A2", "A0", "exploratory"),
    ("A3", "A0", "exploratory"),
    ("A4", "A0", "exploratory"),
    ("A1", "A4", "exploratory"),
)
"""The treatment arm, the reference arm, and the kind of contrast."""


def interval(*, values: np.ndarray) -> tuple[float, float, float]:
    """Return the mean and the 95% interval from resampling the values with replacement.

    Args:
        values: One value per aggregate.

    Returns:
        The mean, and the 2.5th and 97.5th percentiles of the resampled means.
    """
    rng = np.random.default_rng(RESAMPLE_SEED)
    draws = rng.integers(0, len(values), size=(N_RESAMPLES, len(values)))
    low, high = np.percentile(values[draws].mean(axis=1), [2.5, 97.5])
    return float(values.mean()), float(low), float(high)


def _wide(*, fits: pl.DataFrame, sky: str, metric: str) -> pl.DataFrame:
    """Return one row per aggregate (solar set, battery, share) and one column per arm."""
    return (
        fits.filter((pl.col("sky") == sky) & (pl.col("share") > 0))
        .pivot(on="arm", index=["solar_set", "battery", "share"], values=metric)
        .sort("solar_set", "battery", "share")
    )


def _scale(*, metric: str) -> float:
    return PERCENT if metric == "nmae_of_solar_p99" else 1.0


def level_rows(*, fits: pl.DataFrame, arms: tuple[str, ...] = ARMS) -> list[dict]:
    """Return each arm's mean and interval, pooled and per battery, by share and over shares.

    Args:
        fits: `rung4_fits.parquet`.
        arms: The arms to summarise.

    Returns:
        Rows with the sky, group, share (0 means pooled over the three shares), metric, arm,
        mean, interval, and the number of resampled units.
    """
    rows = []
    for sky in SKIES:
        for metric in METRICS:
            wide = _wide(fits=fits, sky=sky, metric=metric)
            for group in ("pooled", *BATTERIES):
                part = wide if group == "pooled" else wide.filter(pl.col("battery") == group)
                for share in (None, *SOLAR_SHARES):
                    sub = part if share is None else part.filter(pl.col("share") == share)
                    # One value per battery-and-solar-set unit, averaging over the shares kept.
                    units = sub.group_by("solar_set", "battery", maintain_order=True).agg(
                        *[pl.col(arm).mean() for arm in arms]
                    )
                    for arm in arms:
                        values = units[arm].drop_nulls().to_numpy() * _scale(metric=metric)
                        mean, low, high = interval(values=values)
                        rows.append(
                            {
                                "sky": sky,
                                "group": group,
                                "share": 0.0 if share is None else share,
                                "metric": metric,
                                "quantity": arm,
                                "mean": mean,
                                "lower_95": low,
                                "upper_95": high,
                                "n": len(values),
                            }
                        )
    return rows


def contrast_rows(
    *,
    fits: pl.DataFrame,
    arms: tuple[str, ...] = ARMS,
    contrasts: tuple[tuple[str, str, str], ...] = CONTRASTS,
) -> list[dict]:
    """Return the paired differences in solar error between arms, with intervals.

    Args:
        fits: `rung4_fits.parquet`.
        arms: The arms the fits hold.
        contrasts: The treatment arm, reference arm, and kind of each contrast.

    Returns:
        Rows in the same shape as `level_rows`, with the metric `nmae_of_solar_p99` and the
        quantity `treatment minus reference`.
    """
    rows = []
    for sky in SKIES:
        wide = _wide(fits=fits, sky=sky, metric="nmae_of_solar_p99")
        for group in ("pooled", *BATTERIES):
            part = wide if group == "pooled" else wide.filter(pl.col("battery") == group)
            for share in (None, *SOLAR_SHARES):
                sub = part if share is None else part.filter(pl.col("share") == share)
                units = sub.group_by("solar_set", "battery", maintain_order=True).agg(
                    *[pl.col(arm).mean() for arm in arms]
                )
                for treatment, reference, _ in contrasts:
                    values = (units[treatment] - units[reference]).drop_nulls().to_numpy() * PERCENT
                    mean, low, high = interval(values=values)
                    rows.append(
                        {
                            "sky": sky,
                            "group": group,
                            "share": 0.0 if share is None else share,
                            "metric": "nmae_of_solar_p99",
                            "quantity": f"{treatment} minus {reference}",
                            "mean": mean,
                            "lower_95": low,
                            "upper_95": high,
                            "n": len(values),
                        }
                    )
    return rows


def _fmt(*, row: dict, digits: int = 1) -> str:
    return f"{row['mean']:.{digits}f} [{row['lower_95']:.{digits}f}, {row['upper_95']:.{digits}f}]"


def detection_lines(*, fits: pl.DataFrame) -> list[str]:
    """Return the report lines for detection against the no-solar threshold and planned B4.

    Args:
        fits: `rung4_fits.parquet`.

    Returns:
        Markdown lines.
    """
    lines = [
        "### Detection against the no-solar threshold",
        "",
        (
            "The statistic is the share of the variance of the four-hour changes, after the "
            "fitted regressors' contribution is removed, that the fitted plant explains. The "
            "cloud increment is that share minus the same share from a fit to CAMS's clear-sky "
            "irradiance. Each arm's own threshold is the largest value over the 16 no-solar "
            "aggregates (battery only, solar share 0%) of that arm and sky. An aggregate is "
            "detected when its value is strictly above its arm's threshold. Planned B4 instead "
            "applies A0's threshold to the no-solar aggregates of every arm: the count of "
            "no-solar aggregates above A0's threshold must not exceed A0's own count, which is "
            "0 by construction."
        ),
        "",
    ]
    for sky in SKIES:
        frame = fits.filter(pl.col("sky") == sky)
        lines += [f"#### Sky: {sky}", ""]
        lines.append(
            "| Arm | Max no-solar variance explained | Max no-solar cloud increment | "
            "No-solar above A0's threshold | Detected at 10% | 25% | 50% "
            "(variance) | Detected at 10% | 25% | 50% (increment) |"
        )
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        zero = frame.filter(pl.col("share") == 0)
        a0_threshold = float(zero.filter(pl.col("arm") == "A0")["variance_explained"].max())  # ty: ignore[invalid-argument-type]
        for arm in ARMS:
            arm_zero = zero.filter(pl.col("arm") == arm)
            thr = float(arm_zero["variance_explained"].max())  # ty: ignore[invalid-argument-type]
            inc = float(arm_zero["cloud_increment"].max())  # ty: ignore[invalid-argument-type]
            false_alarms = int((arm_zero["variance_explained"] > a0_threshold).sum())
            cells = []
            for column, limit in (("variance_explained", thr), ("cloud_increment", inc)):
                for share in SOLAR_SHARES:
                    sub = frame.filter((pl.col("arm") == arm) & (pl.col("share") == share))
                    hits = int((sub[column] > limit).sum())
                    cells.append(f"{hits} of {sub.height}")
            lines.append(
                f"| {arm} | {thr:.4f} | {inc:.4f} | {false_alarms} of {arm_zero.height} | "
                + " | ".join(cells[:3])
                + " | "
                + " | ".join(cells[3:])
                + " |"
            )
        lines.append("")
    return lines


def coefficient_lines(*, fits: pl.DataFrame) -> list[str]:
    """Return the report lines on the fitted regressor coefficients.

    Args:
        fits: `rung4_fits.parquet`.

    Returns:
        Markdown lines.
    """
    names = {
        "A1": ("day-ahead level (MW per standard deviation)", "within-day rank (MW per unit)"),
        "A2": (
            "day-ahead level (MW per standard deviation)",
            "within-day rank (MW per unit)",
            "imbalance spread (MW per standard deviation)",
        ),
        "A3": ("true battery (MW per MW; 1 is exact)",),
        "A4": ("level 7 days earlier (MW per standard deviation)", "rank 7 days earlier (MW)"),
    }
    lines = [
        "### Fitted regressor coefficients",
        "",
        (
            "Mean and range over the aggregates with solar (shares 10%, 25%, and 50%) and over "
            "the no-solar aggregates, regional sky."
        ),
        "",
        "| Arm | Regressor | Mean with solar | Range with solar | Mean no solar | Range no solar |",
        "|---|---|---|---|---|---|",
    ]
    regional = fits.filter(pl.col("sky") == "regional")
    for arm, columns in names.items():
        for index, name in enumerate(columns, start=1):
            column = f"coefficient_{index}"
            solar = regional.filter((pl.col("arm") == arm) & (pl.col("share") > 0))[column]
            none = regional.filter((pl.col("arm") == arm) & (pl.col("share") == 0))[column]
            lines.append(
                f"| {arm} | {name} | {solar.mean():.2f} | {solar.min():.2f} to {solar.max():.2f} "
                f"| {none.mean():.2f} | {none.min():.2f} to {none.max():.2f} |"
            )
    lines.append("")
    return lines


def _share_table(
    *,
    table: pl.DataFrame,
    sky: str,
    metric: str,
    rows: list[tuple[str, str]],
    digits: int,
) -> list[str]:
    """Return a markdown table with one row per quantity and one column per share.

    Args:
        table: `level_rows` or `contrast_rows` output.
        sky: The sky.
        metric: The metric.
        rows: For each table row, its label and the `quantity` it reads.
        digits: Decimal places.

    Returns:
        Markdown lines.
    """
    lines = ["| Row | 10% | 25% | 50% | All three shares |", "|---|---|---|---|---|"]
    for label, quantity in rows:
        cells = []
        for share in (*SOLAR_SHARES, 0.0):
            row = table.filter(
                (pl.col("sky") == sky)
                & (pl.col("group") == "pooled")
                & (pl.col("metric") == metric)
                & (pl.col("quantity") == quantity)
                & (pl.col("share") == share)
            ).row(0, named=True)
            cells.append(_fmt(row=row, digits=digits))
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    return [*lines, ""]


def _battery_table(*, table: pl.DataFrame, quantities: list[str], metric: str) -> list[str]:
    """Return the per-battery table, one column per quantity, regional sky, all three shares."""
    lines = [
        "| Battery | " + " | ".join(quantities) + " |",
        "|---|" + "---|" * len(quantities),
    ]
    for battery in BATTERIES:
        cells = []
        for quantity in quantities:
            row = table.filter(
                (pl.col("sky") == "regional")
                & (pl.col("group") == battery)
                & (pl.col("metric") == metric)
                & (pl.col("quantity") == quantity)
                & (pl.col("share") == 0.0)
            ).row(0, named=True)
            cells.append(_fmt(row=row))
        lines.append(f"| {NAMES[battery]} | " + " | ".join(cells) + " |")
    return [*lines, ""]


ARM_TABLE: Final[tuple[str, ...]] = (
    "| Arm | Meaning | Regressor columns |",
    "|---|---|---|",
    "| A0 | Solar only | none |",
    (
        "| A1 | Plus price level and rank | day-ahead price minus the day's mean (in standard "
        "deviations); within-day rank of the day-ahead price, scaled to [-1, 1] |"
    ),
    (
        "| A2 | Plus imbalance spread | the two of A1; system price minus day-ahead price (in "
        "standard deviations) |"
    ),
    "| A3 | Oracle | the true battery half (MW) |",
    "| A4 | Control | the two of A1, taken from 7 days earlier |",
)


def main() -> None:
    """Write the intervals and the report fragment."""
    fits = pl.read_parquet(OUTPUT_DIR / "rung4_fits.parquet")
    levels = pl.DataFrame(level_rows(fits=fits))
    contrasts = pl.DataFrame(contrast_rows(fits=fits))
    pl.concat([levels, contrasts]).write_parquet(OUTPUT_DIR / "rung4_intervals.parquet")
    _, missing = price_regressors()
    expected = len(SOLAR_SETS) * len(BATTERIES) * 4 * len(SKIES) * len(ARMS)
    arm_rows = [(f"{arm}: {ARM_LABELS[arm]}", arm) for arm in ARMS]
    contrast_labels = [
        (f"{treatment} minus {reference} ({kind})", f"{treatment} minus {reference}")
        for treatment, reference, kind in CONTRASTS
    ]
    lines = [
        "## Rung 4: does a price-based battery regressor help recover solar?",
        "",
        (
            "Synthetic aggregates: 4 solar halves (Burwell, Bishampton, Litchardon, and all three "
            "together) by 4 batteries by solar shares of 0%, 10%, 25%, and 50% of a 100 MW "
            "99th-percentile aggregate, 1 September 2025 to 31 August 2026. Every arm fits the "
            "physical plant (free tilt, azimuth, DC:AC ratio, and AC capacity) to four-hour "
            "changes with the same pairs, soft-L1 loss, and starting orientations, and differs "
            "only in its extra regressors. Errors are the mean absolute error of the recovered "
            "solar series over daylight half-hours, as a percentage of the true solar half's "
            "99th percentile. Intervals resample whole aggregates (see the script's docstring)."
        ),
        "",
        "### Arms and their regressor columns",
        "",
        *ARM_TABLE,
        "",
        f"Half-hours with no price (set to 0 in every price column): {missing}.",
        f"Fits expected: {expected}; fits present: {fits.height}.",
        "",
        "### Solar series error by arm and share (% of solar p99; pooled over 16 aggregates)",
        "",
    ]
    for sky in SKIES:
        lines += [f"#### Sky: {sky}", ""]
        lines += _share_table(
            table=levels, sky=sky, metric="nmae_of_solar_p99", rows=arm_rows, digits=1
        )
    lines += [
        "### Planned contrast B3 and the other differences (points of solar p99)",
        "",
        "Negative means the first arm recovers solar better. Only A1 minus A0 is planned (B3).",
        "",
    ]
    for sky in SKIES:
        lines += [f"#### Sky: {sky}, pooled over 16 aggregates", ""]
        lines += _share_table(
            table=contrasts, sky=sky, metric="nmae_of_solar_p99", rows=contrast_labels, digits=1
        )
    differences = [f"{t} minus {r}" for t, r, _ in CONTRASTS]
    lines += [
        "### Contrasts for each battery (regional sky, points of solar p99, all three shares)",
        "",
        "Intervals resample the 4 solar sets.",
        "",
        *_battery_table(table=contrasts, quantities=differences, metric="nmae_of_solar_p99"),
        "### Solar error for each battery (regional sky, % of solar p99, all three shares)",
        "",
        "Intervals resample the 4 solar sets.",
        "",
        *_battery_table(table=levels, quantities=list(ARMS), metric="nmae_of_solar_p99"),
        "### Energy ratio, AC capacity ratio, and correlation (pooled over 16 aggregates)",
        "",
        (
            "Energy ratio is the recovered solar energy over the true solar energy. The AC "
            "capacity ratio is the fitted AC capacity over the direct fit to the solar half, "
            "scaled to the aggregate's solar half. Both are 1 for a perfect recovery."
        ),
        "",
    ]
    for sky in SKIES:
        for metric in ("energy_ratio", "ac_ratio", "correlation"):
            lines += [f"#### Sky: {sky}, {metric}", ""]
            lines += _share_table(
                table=levels,
                sky=sky,
                metric=metric,
                rows=[(arm, arm) for arm in ARMS],
                digits=2,
            )
    lines += detection_lines(fits=fits)
    lines += coefficient_lines(fits=fits)
    lines += [
        "### Solar sets",
        "",
        *[
            f"- {SOLAR_SET_LABELS[name]} ({len(members)} BMUs)"
            for name, members in SOLAR_SETS.items()
        ],
        "",
    ]
    (OUTPUT_DIR / "report_rung4.md").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
