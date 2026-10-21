"""Draw the figures of experiment 164's write-up from `work/report/report.json`.

uv run --no-sync python ml-experiments/00164-*/e00164_charts.py
"""

import json
import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import e00164_common as common
import plotting.ocf_theme as ocf
import polars as pl

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / f"{common.EXPERIMENT_ID:05d}-{common.SLUG}"
)
PNG_DIR: Final[Path] = common.WORK_DIR / "png"
REPORT: Final[Path] = common.WORK_DIR / "report" / "report.json"
LEAVE_OUT: Final[Path] = common.WORK_DIR / "report" / "leave_one_out.json"
WIDTH: Final[int] = 680
GREY: Final[str] = "#808080"
ARM_COLOURS: Final[dict[str, str]] = {
    "Reference": GREY,
    "T: temperature class": ocf.DATA_BLUE,
    "S: ensemble spread": ocf.DATA_GREEN,
    "O: one-sided": ocf.BRAND_ORANGE,
}
ARM_KEYS: Final[dict[str, str]] = {
    "ref": "Reference",
    "t": "T: temperature class",
    "s": "S: ensemble spread",
    "o": "O: one-sided",
}
COLD_KEY: Final[str] = "cold nights (day mean below 2 C)"
BIN_LABELS: Final[dict[str, str]] = {
    "(-inf, 0]": "0 or below",
    "(0, 2]": "0 to 2",
    "(2, 5]": "2 to 5",
    "(5, 8]": "5 to 8",
    "(8, 12]": "8 to 12",
    "(12, inf]": "above 12",
}
BIN_ORDER: Final[list[str]] = list(BIN_LABELS.values())


def _title(*, text: str, subtitle: str) -> alt.TitleParams:
    return alt.TitleParams(
        text=textwrap.wrap(text, 80),
        subtitle=textwrap.wrap(subtitle, 110),
        anchor="start",
        fontSize=14,
        subtitleFontSize=11,
    )


def _save(chart: alt.TopLevelMixin, *, name: str) -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    chart.save(DOCS_DIR / f"{name}.svg")
    chart.save(PNG_DIR / f"{name}.png", scale_factor=1.5)
    print("wrote", name)


def figure_coverage(*, data: dict) -> alt.TopLevelMixin:
    """Cold-night q95 coverage by arm, pooled, with the nominal and pass lines."""
    rows = []
    for period, label in (("replicate", "replicate months"), ("scored", "scored year")):
        for row in data[period]["tables"]["tail"][COLD_KEY]:
            if row["arm"] in ARM_KEYS:
                entry = {"arm": ARM_KEYS[row["arm"]], "period": label, "coverage": row["cov_q95"]}
                rows.append(entry)
    frame = pl.DataFrame(rows)
    points = (
        alt.Chart()  # ty: ignore[unresolved-attribute]
        .mark_point(filled=True, size=120)
        .encode(
            y=alt.Y("arm:N", sort=list(ARM_COLOURS), title=None, axis=alt.Axis(labelLimit=200)),
            x=alt.X(
                "coverage:Q",
                scale=alt.Scale(domain=[0.80, 0.96]),
                title="share of cold-night readings at or under the q95",
            ),
            color=alt.Color(
                "arm:N",
                scale=alt.Scale(domain=list(ARM_COLOURS), range=list(ARM_COLOURS.values())),
                legend=None,
            ),
        )
    )
    labels = points.mark_text(dy=-14, fontSize=11).encode(text=alt.Text("coverage:Q", format=".3f"))
    pass_line = alt.Chart().mark_rule(strokeDash=[4, 3], color="#444444").encode(x=alt.datum(0.90))  # ty: ignore[unresolved-attribute]
    nominal = alt.Chart().mark_rule(color="#444444").encode(x=alt.datum(0.95))  # ty: ignore[unresolved-attribute]
    return (
        alt.layer(points, labels, pass_line, nominal)
        .properties(width=WIDTH / 2 - 120, height=170)
        .facet(
            column=alt.Column("period:N", sort=["replicate months", "scored year"], title=None),
            data=frame,
        )
        .properties(
            title=_title(
                text="No fix reaches the 0.90 pass line on cold nights in the scored year",
                subtitle=(
                    "Pooled over D5, D10, D11, D14, and D16. "
                    "Dashed line: pass line 0.90. Solid line: nominal 0.95."
                ),
            )
        )
    )


def figure_diagnosis(*, data: dict) -> alt.TopLevelMixin:
    """The reference's residual mean and q95 coverage by ensemble temperature bin."""
    frame = pl.DataFrame(data["scored"]["diagnosis"]["by_temperature_bin"]).with_columns(
        residual_mean_share=pl.col("residual_mean") / pl.col("residual_std"),
        temperature_bin=pl.col("temperature_bin").replace(BIN_LABELS),
    )
    base = alt.Chart(frame).encode(
        x=alt.X(
            "temperature_bin:N",
            sort=BIN_ORDER,
            title="ensemble-mean 24-hour temperature (°C)",
            axis=alt.Axis(labelAngle=-30),
        )
    )
    cover = (
        base.mark_bar(color=ocf.DATA_BLUE, clip=True)  # ty: ignore[unresolved-attribute]
        .encode(y=alt.Y("cov_q95:Q", scale=alt.Scale(domain=[0.7, 1.0]), title="q95 coverage"))
        .properties(width=WIDTH / 2 - 60, height=220, title="Reference q95 coverage")
    )
    bias = (
        base.mark_bar(color=ocf.BRAND_ORANGE)  # ty: ignore[unresolved-attribute]
        .encode(y=alt.Y("residual_mean_share:Q", title="mean residual / residual std"))
        .properties(
            width=WIDTH / 2 - 60,
            height=220,
            title="Reference mean residual (actual minus mean forecast)",
        )
    )
    return alt.hconcat(cover, bias).properties(
        title=_title(
            text="The coldest forecasts are under-forecast on average and under-covered",
            subtitle=(
                "Scored year, five decision substations, all hours. "
                "The mean residual is a share of the residual standard deviation."
            ),
        )
    )


def figure_contrasts(*, data: dict) -> alt.TopLevelMixin:
    """All-row and cold-night quantile-score contrasts with their intervals."""
    names = {
        "C1": "C1 T minus reference",
        "C2": "C2 S minus reference",
        "C3": "C3 O minus reference",
    }
    rows = []
    for period, label in (("replicate", "replicate months"), ("scored", "scored year")):
        for key, name in names.items():
            for scope, scope_label in (("all", "all rows"), (COLD_KEY, "cold nights")):
                c = data[period]["contrasts"][key][scope]["qs"]
                rows.append(
                    {
                        "contrast": name,
                        "period": label,
                        "scope": scope_label,
                        "estimate": c["estimate"],
                        "low": c["lower"],
                        "high": c["upper"],
                    }
                )
    base = alt.Chart().encode(
        y=alt.Y("contrast:N", sort=list(names.values()), title=None, axis=alt.Axis(labelLimit=250)),
        yOffset=alt.YOffset("period:N", sort=["replicate months", "scored year"]),
        color=alt.Color(
            "period:N",
            scale=alt.Scale(
                domain=["replicate months", "scored year"], range=[ocf.DATA_SKY, ocf.BRAND_ORANGE]
            ),
            legend=alt.Legend(title=None, orient="top"),
        ),
    )
    layer = alt.layer(
        base.mark_rule().encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("low:Q", title="quantile-score change (points; negative is better)"),
            x2="high:Q",
        ),
        base.mark_point(filled=True, size=70).encode(x="estimate:Q"),  # ty: ignore[unresolved-attribute]
    ).properties(width=WIDTH / 2 - 80, height=140)
    return (
        layer.facet(
            column=alt.Column("scope:N", sort=["all rows", "cold nights"], title=None),
            data=pl.DataFrame(rows),
        )
        .resolve_scale(x="independent")
        .properties(
            title=_title(
                text="Fix O is no worse over all rows and better on cold nights",
                subtitle=(
                    "Paired month-resampled 98.3% intervals (Bonferroni across three contrasts)."
                ),
            )
        )
    )


def main() -> None:
    """Draw every figure."""
    data = json.loads(REPORT.read_text())
    _save(figure_coverage(data=data), name="figure_1_cold_night_coverage")
    _save(figure_diagnosis(data=data), name="figure_2_diagnosis_by_temperature")
    _save(figure_contrasts(data=data), name="figure_3_contrasts")


if __name__ == "__main__":
    main()
