"""Print every number of experiment 358's page and draw its two figures.

uv run --no-sync python ml-experiments/00358-*/e00358_report.py
"""

import json
from pathlib import Path
from typing import Any, Final

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

SLUG: Final[str] = "mva-fold-known-answer-at-folded-mw-primaries"
WORK: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"00358-{SLUG}" / "work" / "known_answer"
DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2] / "docs" / "ml-experiments" / f"00358-{SLUG}"
)
LEVEL_NAMES: Final[dict[float, str]] = {
    0.0: "natural (no injection)",
    1.0: "level 1.0",
    2.0: "level 2.0",
}
ARMS: Final[dict[str, str]] = {
    "s": "S: fitted to the signed series (ceiling)",
    "l0": "L0: fitted through the meter",
    "n": "N: reading taken as signed power",
}


def _chart(data: Any) -> Any:
    """Return an Altair chart typed loosely, because `ty` loses the chart type after `mark_*()`."""
    return alt.Chart(data)


def load() -> pl.DataFrame:
    """Return every job's result as one table."""
    rows = [json.loads(Path(path).read_text()) for path in sorted(WORK.glob("*.json"))]
    return pl.DataFrame(rows, infer_schema_length=None).with_columns(
        capacity_ratio=pl.col("capacity_l0") / pl.col("capacity_s"),
        capacity_ratio_n=pl.col("capacity_n") / pl.col("capacity_s"),
    )


def tables(*, frame: pl.DataFrame) -> None:
    """Print the numbers the page quotes."""
    pl.Config.set_tbl_rows(100)
    pl.Config.set_tbl_cols(30)
    pl.Config.set_tbl_width_chars(250)
    for cut in ("scored", "replicate"):
        for reactive in ("Q0", "Q1"):
            part = frame.filter(pl.col("cut") == cut, pl.col("reactive") == reactive).sort(
                "level", "site"
            )
            print(f"\n== {cut} cut, {reactive}: sunny sign accuracy and capacity ==")
            print(
                part.select(
                    "level",
                    "site",
                    "n_sunny",
                    pl.col("reverse_flow_share_sunny").round(2).alias("rev_share_sunny"),
                    *[pl.col(f"acc_{arm}_sunny").round(3) for arm in ("l0", "s", "n")],
                    *[pl.col(f"bal_acc_{arm}_sunny").round(3) for arm in ("l0", "s", "n")],
                    pl.col("bal_acc_l0_all").round(3),
                    pl.col("bal_acc_s_all").round(3),
                    pl.col("capacity_ratio").round(3),
                    pl.col("capacity_ratio_n").round(3),
                )
            )
            print(f"\n== {cut} cut, {reactive}: reading error (mean absolute, points) ==")
            print(
                part.select(
                    "level",
                    "site",
                    *[
                        pl.col(f"mae_reading_{arm}_{scope}").round(2)
                        for scope in ("all", "sunny")
                        for arm in ("l0", "n")
                    ],
                    *[pl.col(f"mae_signed_{arm}_negative").round(2) for arm in ("l0", "s", "n")],
                )
            )
    scored = frame.filter(pl.col("cut") == "scored", pl.col("reactive") == "Q0")
    for level in (1.0, 2.0):
        sub = scored.filter(pl.col("level") == level)
        print(
            f"\nlevel {level}, Q0, scored: sites with L0 sunny balanced accuracy >= 0.80:",
            int((sub["bal_acc_l0_sunny"] >= 0.80).sum()),
            "; plain accuracy >= 0.80:",
            int((sub["acc_l0_sunny"] >= 0.80).sum()),
            "; capacity ratio to S within 15%:",
            int(((sub["capacity_ratio"] - 1).abs() <= 0.15).sum()),
        )
    # Capacity rise against the injected capacity, level 1.0 minus natural, per site.
    for reactive in ("Q0", "Q1"):
        wide = frame.filter(pl.col("cut") == "scored", pl.col("reactive") == reactive).pivot(
            on="level", index="site", values=["capacity_l0", "capacity_s", "injected_points"]
        )
        print(f"\n== capacity rise from natural to level 1.0 and 2.0 (scored, {reactive}) ==")
        print(wide)


def figures(*, frame: pl.DataFrame) -> None:
    """Draw the sign-accuracy figure and the capacity-ratio figure."""
    scored = frame.filter(pl.col("cut") == "scored", pl.col("reactive") == "Q0")
    long = pl.concat(
        [
            scored.select(
                "site",
                level=pl.col("level").replace_strict(LEVEL_NAMES),
                arm=pl.lit(label),
                accuracy=pl.col(f"acc_{arm}_sunny"),
            )
            for arm, label in ARMS.items()
        ]
    ).to_pandas()
    chart = (
        _chart(long)
        .mark_bar()
        .encode(
            x=alt.X("site:N", title="Primary substation", axis=alt.Axis(labelAngle=0)),
            xOffset="arm:N",
            y=alt.Y(
                "accuracy:Q",
                title="Share of sunny half-hours with the flow direction right",
                scale=alt.Scale(domain=[0, 1]),
            ),
            color=alt.Color(
                "arm:N",
                scale=alt.Scale(
                    domain=list(ARMS.values()),
                    range=[ocf.DATA_BLUE_LIGHT, ocf.BRAND_ORANGE, ocf.DATA_BLUE],
                ),
                legend=alt.Legend(title=None, orient="top", direction="vertical"),
            ),
            column=alt.Column("level:N", title=None, sort=list(LEVEL_NAMES.values())),
        )
        .properties(width=170, height=220)
    )
    chart.save(DOCS_DIR / "figure_1_sign_accuracy_sunny_half_hours.svg")
    capacity = pl.concat(
        [
            frame.filter(pl.col("cut") == "scored", pl.col("reactive") == reactive).select(
                "site",
                level=pl.col("level").replace_strict(LEVEL_NAMES),
                reactive=pl.lit(
                    "Q = 0 in the data" if reactive == "Q0" else "power factor 0.97 in the data"
                ),
                ratio=pl.col("capacity_ratio"),
            )
            for reactive in ("Q0", "Q1")
        ]
    ).to_pandas()
    chart2 = (
        _chart(capacity)
        .mark_circle(size=90)
        .encode(
            x=alt.X("site:N", title="Primary substation", axis=alt.Axis(labelAngle=0)),
            y=alt.Y(
                "ratio:Q",
                title="Fitted capacity through the meter / signed fit",
                scale=alt.Scale(domain=[0.5, 1.4]),
            ),
            color=alt.Color(
                "level:N",
                scale=alt.Scale(
                    domain=list(LEVEL_NAMES.values()),
                    range=[ocf.DATA_BLUE_LIGHT, ocf.BRAND_ORANGE, ocf.DATA_BLUE],
                ),
                legend=alt.Legend(title=None, orient="top"),
            ),
            column=alt.Column("reactive:N", title=None),
        )
        .properties(width=250, height=220)
    )
    chart2.save(DOCS_DIR / "figure_2_capacity_through_the_meter.svg")


if __name__ == "__main__":
    data = load()
    tables(frame=data)
    figures(frame=data)
