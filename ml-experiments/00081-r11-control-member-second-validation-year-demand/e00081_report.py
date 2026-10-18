"""Print every number that experiment 00081's `result.md` quotes, from the saved results.

The numbers: each arm's input columns; the six planned contrasts of fold `b` at 99.17% with the
block-bootstrap interval and the t-interval over months, and the decision rule's verdict; the same
contrasts at 95% and on fold `c`; the descriptive contrasts and reads by lead band, series, and
month; C1 without three series; every arm's absolute score ("control-only fold") and 0.95 exceedance
rate; the Christmas window; the pooled Christmas-column read; the spread-factor check; the blend's
weights; and the descriptive 9 km check. The text goes to the console and to
`work/results/report.txt`.

    uv run python ml-experiments/00081-*/e00081_report.py [--smoke]
"""

import argparse
from pathlib import Path
from typing import Final

import e00081_common as common
import e00081_fit as fitting
import polars as pl

PRIMARY_ARMS: Final[tuple[str, ...]] = (
    "frozen_cs",
    "frozen_s",
    "monthly_s",
    "weekly_s",
    "roll12_s",
    "roll12_no_christmas_s",
    "roll12_hl2y_s",
    "blend",
    "manual_heuristic",
    "climatology",
)


def _table(*, frame: pl.DataFrame) -> str:
    with pl.Config(tbl_rows=300, tbl_cols=30, tbl_width_chars=250, fmt_str_lengths=60):
        return str(frame.with_columns(pl.col(pl.Float64).round(3)))


def columns_section() -> list[str]:
    """Return each configuration's input columns, so a reviewer can check the plan."""
    lines = ["ARM INPUT COLUMNS (the arms differ only where stated)"]
    for name, config in fitting.CONFIGS.items():
        columns = fitting.input_columns(drop=config.drop)
        lines.append(f"{name}: {len(columns)} columns: {', '.join(columns)}")
    lines.append("manual_heuristic: no input column (power lags of 1 to 6 and 49 to 55 weeks)")
    lines.append("climatology: no input column (local month, half-hour of day, day type)")
    return lines


def fold_section(*, fold: str, folder: Path) -> list[str]:
    """Return the report lines of one fold."""
    root = folder / fold
    lines = [f"===== FOLD {fold.upper()} ({common.FOLDS[fold].row_set_id}) ====="]  # ty: ignore[invalid-argument-type]
    if (root / "absolute.parquet").exists():
        absolute = pl.read_parquet(root / "absolute.parquet")
        lines.append(
            "ABSOLUTE, control-only fold, all leads (quantile score with 95% interval over 4-week "
            "blocks; MAE of the median; 0.95 exceedance rate, nominal 0.05; 90% coverage)"
        )
        lines.append(
            _table(
                frame=absolute.filter(pl.col("scope") == "all leads")
                .select("arm", "cases", "qs", "qs_lower", "qs_upper", "mae", "ex95", "cov90")
                .sort("qs")
            )
        )
        lines.append("ABSOLUTE, control-only fold, quantile score by lead band")
        lines.append(
            _table(frame=absolute.pivot(on="scope", index="arm", values="qs").sort("all leads"))
        )
        series = pl.read_parquet(root / "absolute_series.parquet")
        lines.append("QUANTILE SCORE BY SERIES, control-only fold")
        lines.append(
            _table(
                frame=series.filter(pl.col("arm").is_in(PRIMARY_ARMS))
                .pivot(on="arm", index="site", values="qs")
                .sort(pl.col("site").str.slice(1).cast(pl.Int32))
            )
        )
    path = root / "contrasts.parquet"
    if path.exists():
        contrasts = pl.read_parquet(path)
        planned = contrasts.filter(
            (pl.col("family") == "planned")
            & (pl.col("site") == "pooled")
            & (pl.col("scope") == "all leads")
            & (pl.col("metric") == "qs")
        )
        lines.append("PLANNED CONTRASTS, quantile score, all leads (points of capacity)")
        for row in planned.filter(pl.col("level_label") == "planned").iter_rows(named=True):
            narrow = planned.filter(
                (pl.col("id") == row["id"]) & (pl.col("level_label") == "95")
            ).row(0, named=True)
            lines.append(
                f"{row['id']}: {row['treatment']} minus {row['reference']}: "
                f"{row['difference']:+.3f}; 99.17% block interval [{row['lower']:+.3f}, "
                f"{row['upper']:+.3f}]; 95% block interval [{narrow['lower']:+.3f}, "
                f"{narrow['upper']:+.3f}]; 95% t-interval over {narrow['months']} months "
                f"[{narrow['t_lower']:+.3f}, {narrow['t_upper']:+.3f}] "
                f"({row['cases']:,} cases, {row['blocks']} blocks)"
            )
        lines.append("PLANNED CONTRASTS, other metric: MAE of the median, 95%")
        lines.append(
            _table(
                frame=contrasts.filter(
                    (pl.col("family") == "planned")
                    & (pl.col("site") == "pooled")
                    & (pl.col("metric") == "mae")
                ).select("id", "difference", "lower", "upper")
            )
        )
        lines.append("PLANNED CONTRASTS by lead band, 95% (descriptive)")
        lines.append(
            _table(
                frame=contrasts.filter(
                    (pl.col("family") == "planned")
                    & (pl.col("site") == "pooled")
                    & pl.col("scope").is_in([name for name, _, _ in common.LEAD_BANDS])
                ).select("id", "scope", "difference", "lower", "upper")
            )
        )
        lines.append("PLANNED CONTRASTS on each series alone, 95% (descriptive)")
        lines.append(
            _table(
                frame=contrasts.filter(
                    (pl.col("family") == "planned")
                    & (pl.col("site") != "pooled")
                    & (pl.col("scope") == "all leads")
                )
                .select("id", "site", "difference", "lower", "upper")
                .sort("id", pl.col("site").str.slice(1).cast(pl.Int32))
            )
        )
        lines.append(
            "PLANNED CONTRASTS by month of forecast start, 95% over weekly blocks (descriptive)"
        )
        lines.append(
            _table(
                frame=contrasts.filter(
                    (pl.col("family") == "planned") & pl.col("scope").str.starts_with("month ")
                ).select("id", "scope", "difference", "lower", "upper", "cases")
            )
        )
        lines.append("DESCRIPTIVE CONTRASTS, pooled, all leads, quantile score, 95%")
        lines.append(
            _table(
                frame=contrasts.filter(
                    (pl.col("family") == "descriptive")
                    & (pl.col("site") == "pooled")
                    & (pl.col("scope") == "all leads")
                    & (pl.col("metric") == "qs")
                    & (pl.col("level_label") == "95")
                ).select("id", "treatment", "reference", "difference", "lower", "upper", "cases")
            )
        )
    for name, title in (
        ("christmas_window", "CHRISTMAS WINDOW (a number, no interval)"),
        ("spread", "SPREAD FACTOR re-chosen on the first three months, effect on later months"),
        ("blend_weights", "BLEND WEIGHTS by month of forecast start"),
        ("nine_km", "9 KM CHECK: frozen model fitted to 2023-03-31 against manual_heuristic"),
    ):
        part = root / f"{name}.parquet"
        if part.exists():
            lines += [title, _table(frame=pl.read_parquet(part))]
    verdict = root / "verdict.json"
    if verdict.exists():
        lines += ["VERDICT (fold B decides; fold C cannot change it)", verdict.read_text()]
    return lines


def main() -> None:
    """Print the report and write it beside the results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    smoke = parser.parse_args().smoke
    folder = common.RESULTS_DIR.with_name("results_smoke") if smoke else common.RESULTS_DIR
    lines = columns_section()
    for fold in ("b", "c", "d"):
        if (folder / fold).exists():
            lines += fold_section(fold=fold, folder=folder)
    pooled = folder / "pooled" / "pooled_c4.parquet"
    if pooled.exists():
        lines += [
            "C4 POOLED over the winters (descriptive; precision-weighted, 95%)",
            _table(frame=pl.read_parquet(pooled)),
        ]
    text = "\n".join(lines) + "\n"
    folder.mkdir(parents=True, exist_ok=True)
    (folder / "report.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
