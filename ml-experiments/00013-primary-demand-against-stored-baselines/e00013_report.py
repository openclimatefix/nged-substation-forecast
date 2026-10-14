"""Print every number that experiment 00013's write-up quotes, from the saved result tables.

Reads `work/results/{absolute,contrasts,bias,debias}.parquet` and writes the Primary substations'
tables to `work/results/report.md`, rounded as the write-up quotes them.

    uv run python ml-experiments/00013-primary-demand-against-stored-baselines/e00013_report.py
"""

from typing import Final

import e00013_common as common
import polars as pl

RESULTS_DIR = common.WORK_DIR / "results"
SCOPE_ORDER: Final[tuple[str, ...]] = (
    "all leads",
    "up to 48 h",
    "3 d or more",
    "0-24 h",
    "1-2 d",
    "3-6 d",
    "7-10 d",
    "11-15 d",
)


def main() -> None:
    """Write and print the report."""
    with pl.Config(tbl_rows=200, tbl_cols=20, tbl_width_chars=220, fmt_str_lengths=80):
        text = _report()
    (RESULTS_DIR / "report.md").write_text(text + "\n")
    print(text)


def _report() -> str:
    """Return the report text."""
    absolute = pl.read_parquet(RESULTS_DIR / "absolute.parquet")
    contrasts = pl.read_parquet(RESULTS_DIR / "contrasts.parquet")
    parts = []
    for group in ("Primary", "BSP", "GSP"):
        for metric in ("median_error", "quantile_score"):
            table = (
                absolute.filter((pl.col("group") == group) & (pl.col("metric") == metric))
                .with_columns(pl.col("value").round(2))
                .pivot(on="scope", index="arm", values="value")
                .select("arm", *SCOPE_ORDER)
            )
            parts.append(f"## {group}: absolute {metric} (points of capacity)\n{table}")
    planned = (
        contrasts.filter(pl.col("planned"))
        .select(
            "label",
            "scope",
            "metric",
            "difference",
            "lower",
            "upper",
            "months_negative",
            "months",
            "series_negative",
            "series",
        )
        .with_columns(pl.col("difference", "lower", "upper").round(2))
    )
    parts.append(f"## Planned contrasts, Primary, 99%\n{planned}")
    exploratory = (
        contrasts.filter(~pl.col("planned") & (pl.col("group") == "Primary"))
        .select("label", "scope", "metric", "difference", "lower", "upper")
        .with_columns(pl.col("difference", "lower", "upper").round(2))
    )
    parts.append(f"## Exploratory contrasts, Primary, 95%\n{exploratory}")
    bias = (
        pl.read_parquet(RESULTS_DIR / "bias.parquet")
        .filter(pl.col("group") == "Primary")
        .with_columns(pl.col("bias").round(2))
        .pivot(on="scope", index="arm", values="bias")
    )
    parts.append(f"## Mean signed error of the median, Primary\n{bias}")
    debias = pl.read_parquet(RESULTS_DIR / "debias.parquet").with_columns(
        pl.col("mae", "mae_after_level_correction", "mean_offset").round(2)
    )
    parts.append(f"## Post hoc level correction, Primary\n{debias}")
    return "\n\n".join(parts)


if __name__ == "__main__":
    main()
