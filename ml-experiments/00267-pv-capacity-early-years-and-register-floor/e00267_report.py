"""Experiment 267: print every number the write-up quotes, and draw its figure.

The script reads `work/capacity.parquet` (from `e00267_run.py`), experiment 129's known-answer and
register tables, and writes `figure_01_capacity.svg` beside the write-up.

    uv run --no-sync python ml-experiments/00267-*/e00267_report.py
"""

from pathlib import Path
from typing import Final

import altair as alt
import e00267_common as common
import numpy as np
import plotting.ocf_theme as ocf
import polars as pl
from studies.charts import CONTENT_WIDTH_PX

EXPERIMENT_129_WORK: Final[Path] = (
    common.ML_EXPERIMENTS_DATA_DIR
    / "00129-differentiable-physics-pv-capacity-known-answer-test"
    / "work"
)
DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / f"{common.EXPERIMENT_ID:05d}-{common.SLUG}"
)
FIT_CUT_MW: Final[float] = 1.0
"""The register's 1 MW cut, the brief's floor."""


def with_floors(*, capacity: pl.DataFrame) -> pl.DataFrame:
    """Return the capacity table with each Primary's register floor, in AC and DC MW."""
    floors = (
        pl.read_parquet(EXPERIMENT_129_WORK / "registers.parquet")
        .filter(pl.col("arm") == "arm2", pl.col("cut_mw") == FIT_CUT_MW)
        .select("site", "fit_ac_mw", "fit_dc_mw")
    )
    return capacity.join(floors, on="site", how="left", validate="m:1")


def a1_table() -> pl.DataFrame:
    """Return the card's estimate beside experiment 241's divisor and a double division."""
    known = pl.read_parquet(EXPERIMENT_129_WORK / "kat_arm4@0.1.parquet").filter(
        pl.col("plant") == "matched",
        ~pl.col("cloud_load"),
        pl.col("source").is_in(common.FARMS),
    )
    divisor = known.group_by("site", "season_year").agg(a1_divisor=pl.col("recovery").median())
    card = pl.read_parquet(EXPERIMENT_129_WORK / "real_arm4@0.1_to_july.parquet")
    return card.join(divisor, on=["site", "season_year"], how="left", validate="1:1").select(
        "site",
        "season_year",
        a0_mw="estimate_mw",
        a0_divisor="divisor",
        a1_divisor="a1_divisor",
        a1_mw=pl.col("estimate_mw") * pl.col("divisor") / pl.col("a1_divisor"),
        double_mw=pl.col("estimate_mw") / pl.col("a1_divisor"),
        double_high_mw=pl.col("range_high_mw") / pl.col("a1_divisor"),
    )


def leave_one_farm_out_error() -> float:
    """Return the median absolute error of experiment 241's correction on card-fit recoveries."""
    known = pl.read_parquet(EXPERIMENT_129_WORK / "kat_arm4@0.1.parquet").filter(
        pl.col("plant") == "matched",
        ~pl.col("cloud_load"),
        pl.col("source").is_in(common.FARMS),
    )
    errors = []
    for _, group in known.group_by("site", "season_year"):
        for farm in common.FARMS:
            is_farm = (group["source"] == farm).to_numpy()
            recovery = group["recovery"].to_numpy()
            held = float(np.median(recovery[is_farm]))
            others = float(np.median(recovery[~is_farm]))
            errors.append(abs(held / others - 1.0))
    return float(np.median(errors))


def figure(*, table: pl.DataFrame) -> alt.VConcatChart:
    """Return one panel per Primary: both estimates with 80% ranges, and the register floor."""
    long = pl.concat(
        [
            table.select(
                "site",
                "season_year",
                arm=pl.lit("A0: satellite sunshine shape (the card)"),
                estimate="a0_mw",
                low="a0_low_mw",
                high="a0_high_mw",
            ),
            table.select(
                "site",
                "season_year",
                arm=pl.lit("A2: nearest metered farm as the shape"),
                estimate="a2_mw",
                low="a2_low_mw",
                high="a2_high_mw",
            ),
        ]
    ).with_columns((pl.col("season_year") + pl.lit(0)).cast(pl.Int32))
    domain = [
        "A0: satellite sunshine shape (the card)",
        "A2: nearest metered farm as the shape",
    ]
    colour = alt.Color(
        "arm:N",
        scale=alt.Scale(domain=domain, range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE]),
        legend=alt.Legend(
            title=None, orient="top", direction="vertical", labelLimit=CONTENT_WIDTH_PX
        ),
    )
    x = alt.X(
        "season_year:O",
        title="Season-year (July of the year shown to June of the next)",
        axis=alt.Axis(labelAngle=0),
    )
    panels = []
    for site in common.DECISION_SITES:
        data = long.filter(pl.col("site") == site)
        floor = table.filter(pl.col("site") == site)["fit_ac_mw"][0]
        base = alt.Chart(data)
        band = base.mark_rule(strokeWidth=4, opacity=0.5, aria=False).encode(  # ty: ignore[unresolved-attribute]
            x=x, y=alt.Y("low:Q", title="MW (AC)"), y2="high:Q", color=colour, xOffset="arm:N"
        )
        dots = base.mark_point(filled=True, size=50, aria=False).encode(  # ty: ignore[unresolved-attribute]
            x=x, y="estimate:Q", color=colour, xOffset="arm:N"
        )
        rule = (
            alt.Chart(pl.DataFrame({"floor": [floor]}))
            .mark_rule(strokeDash=[4, 3], color="black", aria=False)
            .encode(y="floor:Q")  # ty: ignore[unresolved-attribute]
        )
        panels.append(
            (band + dots + rule).properties(width=CONTENT_WIDTH_PX - 80, height=130, title=site)
        )
    return alt.vconcat(*panels).properties(
        title=alt.TitleParams(
            [
                "Capacity of unmetered solar: dot is the estimate, bar is the 80% range.",
                "Dashed line: the register's 1 MW-cut AC capacity, a lower bound on the capacity.",
            ],
            anchor="start",
            fontSize=12,
        )
    )


def main() -> None:
    """Print the quoted numbers and write the figure."""
    capacity = with_floors(capacity=pl.read_parquet(common.WORK_DIR / "capacity.parquet")).sort(
        "site", "season_year"
    )
    with pl.Config(tbl_rows=60, tbl_cols=30, float_precision=2):
        print("A1: the card's divisor against experiment 241's, and a double division")
        print(a1_table().sort("site", "season_year"))
        print("leave-one-farm-out median absolute error, card fit:", leave_one_farm_out_error())
        print("A3: months and training half-hours per season-year")
        print(capacity.select("site", "season_year", "months", "rows"))
        print("Estimates, upper range end, and floors (AC and DC, MW)")
        print(
            capacity.select(
                "site",
                "season_year",
                "a0_mw",
                "a0_high_mw",
                "a2_mw",
                "a2_high_mw",
                "fit_ac_mw",
                "fit_dc_mw",
                a2_over_a0=pl.col("a2_mw") / pl.col("a0_mw") - 1.0,
                a0_clears_ac=pl.col("a0_high_mw") >= pl.col("fit_ac_mw"),
                a0_clears_dc=pl.col("a0_high_mw") >= pl.col("fit_dc_mw"),
                a2_clears_ac=pl.col("a2_high_mw") >= pl.col("fit_ac_mw"),
                a2_clears_dc=pl.col("a2_high_mw") >= pl.col("fit_dc_mw"),
            )
        )
        print("A2 against A0 in 2021: ratio of raw capacity points and of divisors")
        print(
            capacity.filter(pl.col("season_year") == 2021).select(
                "site",
                raw_ratio=pl.col("ref_raw_points") / pl.col("raw_points"),
                divisor_ratio=pl.col("a2_divisor") / pl.col("a0_divisor"),
                estimate_ratio=pl.col("a2_mw") / pl.col("a0_mw"),
            )
        )
        print("reference farm distance and divisors")
        print(capacity.unique("site").select("site", "reference_km", "a2_divisor", "a0_divisor"))
        print("A2 known-answer error (leave one injected farm out), median absolute")
        print(
            pl.read_parquet(common.WORK_DIR / "known_answer_reference.parquet")
            .group_by("site")
            .agg(median_error=pl.col("corrected_error").abs().median(), farms=pl.len())
            .sort("site")
        )
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    figure(table=capacity).save(str(DOCS_DIR / "figure_01_capacity.svg"))


if __name__ == "__main__":
    main()
