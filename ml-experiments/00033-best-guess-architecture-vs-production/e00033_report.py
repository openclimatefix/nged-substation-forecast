"""Print every number the experiment 00033 page quotes, as markdown tables, from the saved results.

Writes `work/report.md` and prints it. Reads `work/results/` (see `e00033_stats.py`) and
`work/metrics/` (see `e00033_metrics.py`).

    uv run python ml-experiments/00033-*/e00033_report.py
"""

from typing import Final

import e00033_common as common
import e00033_stats as stats
import polars as pl
from e00033_charts import HEADLINE

RESULTS: Final = common.WORK_DIR / "results"
METRICS: Final = common.WORK_DIR / "metrics"
BANDS: Final[list[str]] = [b[0] for b in common.LEAD_BANDS]


def fmt(value: float, digits: int = 2) -> str:
    """Format a number with a sign-safe fixed number of decimals."""
    return f"{value:.{digits}f}"


def absolute_table() -> list[str]:
    """Return the absolute scores of the headline arms, per group, over all leads."""
    absolute = pl.read_parquet(RESULTS / "absolute.parquet").filter(pl.col("scope") == "all leads")
    lines = []
    for group, arms in HEADLINE.items():
        lines += [
            f"### {group}: absolute scores over all leads",
            "",
            (
                "| Arm | Quantile score (95% interval) | MAE of the median | twCRPS at p99 | "
                "Brier at p99 | Above q90 | Above q95 | Inside q05-q95 |"
            ),
            "|---|---|---|---|---|---|---|---|",
        ]
        for arm, (label, _category) in arms.items():
            row = absolute.filter((pl.col("group") == group) & (pl.col("arm") == arm))
            if row.is_empty():
                continue
            r = row.row(0, named=True)
            lines.append(
                f"| {label} | {fmt(r['qs'])} ({fmt(r['qs_lower'])} to {fmt(r['qs_upper'])}) | "
                f"{fmt(r['mae'])} | {fmt(r['tw_p99'], 3)} | {fmt(r['brier_p99'], 4)} | "
                f"{fmt(r['ex90'], 3)} | {fmt(r['ex95'], 3)} | {fmt(r['cov90'], 3)} |"
            )
        lines.append("")
    return lines


def contrast_table() -> list[str]:
    """Return every planned contrast on the quantile score over all leads."""
    data = pl.read_parquet(RESULTS / "contrasts.parquet").filter(
        (pl.col("scope") == "all leads") & (pl.col("metric") == "qs")
    )
    lines = [
        "### Planned contrasts on the quantile score, over all leads",
        "",
        (
            "| Id | Family | Groups | Treatment minus reference | Difference | Planned interval "
            "(level) | 95% interval |"
        ),
        "|---|---|---|---|---|---|---|",
    ]
    for contrast_id in stats.CONTRASTS:
        rows = data.filter(pl.col("id") == contrast_id)
        planned = rows.filter(pl.col("level_label") == "planned").row(0, named=True)
        ninety_five = rows.filter(pl.col("level_label") == "95").row(0, named=True)
        lines.append(
            f"| {contrast_id} | {planned['family']} | {planned['groups']} | "
            f"`{planned['treatment']}` minus `{planned['reference']}` | "
            f"{fmt(planned['difference'], 3)} | {fmt(planned['lower'], 3)} to "
            f"{fmt(planned['upper'], 3)} ({fmt(planned['level'], 2)}%) | "
            f"{fmt(ninety_five['lower'], 3)} to {fmt(ninety_five['upper'], 3)} |"
        )
    return [*lines, ""]


def other_metric_contrasts() -> list[str]:
    """Return the decisive and demand contrasts on MAE, twCRPS at p99 and p95 (95% intervals)."""
    data = pl.read_parquet(RESULTS / "contrasts.parquet").filter(
        (pl.col("scope") == "all leads") & (pl.col("level_label") == "95")
    )
    lines = [
        "### Exploratory contrasts on the other metrics, over all leads (95% intervals)",
        "",
        "| Id | MAE of the median | twCRPS at p99 | twCRPS at p95 |",
        "|---|---|---|---|",
    ]
    for contrast_id in ("C1", "C2", "C3", "C4", "C5", "C7", "C13", "C17", "C20"):
        cells = []
        for metric in ("mae", "tw_p99", "tw_p95"):
            r = data.filter((pl.col("id") == contrast_id) & (pl.col("metric") == metric)).row(
                0, named=True
            )
            cells.append(
                f"{fmt(r['difference'], 3)} ({fmt(r['lower'], 3)} to {fmt(r['upper'], 3)})"
            )
        lines.append(f"| {contrast_id} | " + " | ".join(cells) + " |")
    return [*lines, ""]


def band_tables() -> list[str]:
    """Return the quantile score by lead band for the headline baselines."""
    absolute = pl.read_parquet(RESULTS / "absolute.parquet").filter(pl.col("scope").is_in(BANDS))
    wanted = {
        "Primary": ["n1_cs", "n1", "production", "manual_heuristic"],
        "solar": ["bg_base_spread", "n1", "production", "manual_heuristic"],
        "wind": ["bg_full", "n1", "production", "manual_heuristic"],
    }
    lines = ["### Quantile score by lead band", ""]
    for group, arms in wanted.items():
        lines += [f"**{group}**", "", "| Arm | " + " | ".join(BANDS) + " |", "|---|" + "---|" * 5]
        for arm in arms:
            cells = [
                fmt(
                    float(
                        absolute.filter(
                            (pl.col("group") == group)
                            & (pl.col("arm") == arm)
                            & (pl.col("scope") == band)
                        )["qs"][0]
                    )
                )
                for band in BANDS
            ]
            lines.append(f"| `{arm}` | " + " | ".join(cells) + " |")
        lines.append("")
    return lines


def ceiling_tables() -> list[str]:
    """Return the weather-ceiling contrasts by lead band, and the single-variable swaps."""
    data = pl.read_parquet(RESULTS / "contrasts.parquet").filter(
        (pl.col("level_label") == "95") & (pl.col("metric") == "qs")
    )
    lines = [
        "### Weather ceiling by lead band (95% intervals, quantile score)",
        "",
        "| Id | " + " | ".join(BANDS) + " |",
        "|---|" + "---|" * 5,
    ]
    for contrast_id in ("C27", "C28", "C28b", "C29", "C30", "C31", "C32"):
        cells = []
        for band in BANDS:
            r = data.filter((pl.col("id") == contrast_id) & (pl.col("scope") == band)).row(
                0, named=True
            )
            cells.append(f"{fmt(r['difference'])} ({fmt(r['lower'])} to {fmt(r['upper'])})")
        lines.append(f"| {contrast_id} | " + " | ".join(cells) + " |")
    absolute = pl.read_parquet(RESULTS / "absolute.parquet").filter(pl.col("scope") == "all leads")

    def score(group: str, arm: str) -> float:
        return float(absolute.filter((pl.col("group") == group) & (pl.col("arm") == arm))["qs"][0])

    lines += [
        "",
        "### Weather ceiling by variable (difference of mean quantile scores, no interval)",
        "",
        "| Swap | Group | Score of the swap | Score of its reference | Difference |",
        "|---|---|---|---|---|",
    ]
    swaps = [
        ("ERA5 shortwave", "Primary", "w22_era5_shortwave_c", "w22_ens_c"),
        ("ERA5 temperature", "Primary", "w22_era5_temperature_2m_c", "w22_ens_c"),
        ("ERA5 wind speed", "Primary", "w22_era5_wind_speed_10m_c", "w22_ens_c"),
        ("ERA5 all seven columns", "Primary", "w22_era5_all_c", "w22_ens_c"),
        ("ERA5 shortwave", "solar", "pw_sw_era5_spread", "bg_base_spread"),
        ("CAMS shortwave (CAMS-trained)", "solar", "pw_sw_cams_spread", "cams_spread"),
        ("ERA5 100 m wind speed", "wind", "pw_ws100_spread", "bg_full"),
    ]
    for label, group, arm, reference in swaps:
        a, b = score(group, arm), score(group, reference)
        lines.append(f"| {label} | {group} | {fmt(a)} | {fmt(b)} | {fmt(a - b)} |")
    return [*lines, ""]


def event_table() -> list[str]:
    """Return hits, false alarms, and events at the thresholds for two lead bands."""
    lines = [
        "### Threshold exceedance, 1 to 2 days and 2 to 6 days (counts over series, seed 0)",
        "",
        "| Group | Threshold | Lead | Arm | Events | Hits (p >= 0.5) | False alarms |",
        "|---|---|---|---|---|---|---|",
    ]
    wanted = {
        "Primary": ("n1_cs", "production", "manual_heuristic"),
        "solar": ("bg_base_spread", "production", "manual_heuristic"),
        "wind": ("bg_full", "production", "manual_heuristic"),
    }
    for group, arms in wanted.items():
        for threshold in ("p99", "p95"):
            for band, low, high in (("1-2 d", 24.0, 48.0), ("2-6 d", 48.0, 144.0)):
                for arm in arms:
                    d = pl.read_parquet(METRICS / group / f"{arm}.parquet").filter(
                        pl.col("shared")
                        & (pl.col("seed") == 0)
                        & (pl.col("lead_hours") >= low)
                        & (pl.col("lead_hours") < high)
                    )
                    lines.append(
                        f"| {group} | {threshold} | {band} | `{arm}` | "
                        f"{int(d[f'event_{threshold}'].sum())} | "
                        f"{int(d[f'hit_{threshold}'].sum())} | "
                        f"{int(d[f'false_alarm_{threshold}'].sum())} |"
                    )
    return [*lines, ""]


def count_table() -> list[str]:
    """Return the shared and dropped case counts per group."""
    counts = pl.read_parquet(METRICS / "shared_counts.parquet")
    lines = [
        "### Cases per group",
        "",
        "| Group | Shared cases | Largest arm | Smallest arm (cases) |",
        "|---|---|---|---|",
    ]
    for group in stats.GROUPS:
        part = counts.filter(pl.col("group") == group)
        smallest = part.sort("cases_before").row(0, named=True)
        lines.append(
            f"| {group} | {part['cases_shared'][0]:,} | {part['cases_before'].max():,} | "
            f"`{smallest['arm']}` ({smallest['cases_before']:,}) |"
        )
    thresholds = pl.read_parquet(common.WORK_DIR / "thresholds.parquet")
    lines += [
        "",
        (
            f"Thresholds: {thresholds.height} series; `historical_p99` ranges "
            f"{fmt(float(thresholds['p99'].to_numpy().min()), 3)} to "
            f"{fmt(float(thresholds['p99'].to_numpy().max()), 3)}, "
            f"`historical_p95` {fmt(float(thresholds['p95'].to_numpy().min()), 3)} to "
            f"{fmt(float(thresholds['p95'].to_numpy().max()), 3)} of capacity."
        ),
        "",
    ]
    return lines


def main() -> None:
    """Write and print the report."""
    lines = [
        *absolute_table(),
        *contrast_table(),
        *other_metric_contrasts(),
        *band_tables(),
        *ceiling_tables(),
        *event_table(),
        *count_table(),
    ]
    text = "\n".join(lines)
    (common.WORK_DIR / "report.md").write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
