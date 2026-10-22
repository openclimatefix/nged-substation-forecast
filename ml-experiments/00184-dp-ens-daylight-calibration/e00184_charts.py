# ruff: noqa: E501
"""Draw the figures of experiment 184's write-up from `work/report/report.json` and `extra.json`.

uv run --no-sync python ml-experiments/00184-*/e00184_charts.py
"""

import json
import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import e00184_common as common
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
EXTRA: Final[Path] = common.WORK_DIR / "report" / "extra.json"
GATES: Final[Path] = common.WORK_DIR / "gates.json"
WIDTH: Final[int] = 680
ARM_LABELS: Final[dict[str, str]] = {
    "m": "M: members through the solar term",
    "f": "F: members through every input",
    "s": "S: spread statistics set the band width",
    "mr": "Placebo MR: rescale the bands, no members",
    "mp": "Placebo MP: members from another day",
    "sp": "Placebo SP: statistics from another day",
}
ARM_COLOURS: Final[dict[str, str]] = {
    ARM_LABELS["m"]: ocf.BRAND_ORANGE,
    ARM_LABELS["f"]: ocf.DATA_SKY,
    ARM_LABELS["s"]: ocf.DATA_BLUE,
    ARM_LABELS["mr"]: ocf.DATA_PURPLE,
    ARM_LABELS["mp"]: ocf.DATA_GREEN,
    ARM_LABELS["sp"]: "#808080",
}


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


def _change_rows(*, report: dict, subsets: list[tuple[str, str]], arms: list[str]) -> list[dict]:
    rows = []
    for period, period_label in (("scored", "scored year"), ("replicate", "replicate months")):
        for subset, subset_label in subsets:
            entry = report[period]["subsets"].get(subset)
            if entry is None:
                continue
            for arm in arms:
                c = entry[arm]["change_pct"]
                rows.append(
                    {
                        "period": period_label,
                        "subset": subset_label,
                        "arm": ARM_LABELS[arm],
                        "estimate": c["point"],
                        "low": c["lower"],
                        "high": c["upper"],
                    }
                )
    return rows


def _interval_chart(
    *, rows: list[dict], row_field: str, order: list[str], title: str, subtitle: str, height: int
) -> alt.TopLevelMixin:
    base = alt.Chart().encode(
        y=alt.Y(f"{row_field}:N", sort=order, title=None, axis=alt.Axis(labelLimit=260)),
        yOffset=alt.YOffset("arm:N", sort=list(ARM_COLOURS)),
        color=alt.Color(
            "arm:N",
            scale=alt.Scale(domain=list(ARM_COLOURS), range=list(ARM_COLOURS.values())),
            legend=alt.Legend(title=None, orient="top", columns=1, labelLimit=400),
        ),
    )
    layer = alt.layer(
        base.mark_rule().encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("low:Q", title="change against arm A (%; lower is better)"),
            x2="high:Q",
        ),
        base.mark_point(filled=True, size=60).encode(x="estimate:Q"),  # ty: ignore[unresolved-attribute]
        alt.Chart().mark_rule(color="#444444").encode(x=alt.datum(0)),  # ty: ignore[unresolved-attribute]
    ).properties(width=WIDTH / 2 - 110, height=height)
    return layer.facet(
        column=alt.Column("period:N", sort=["scored year", "replicate months"], title=None),
        data=pl.DataFrame(rows),
    ).properties(title=_title(text=title, subtitle=subtitle))


def figure_headline(*, report: dict) -> alt.TopLevelMixin:
    """Daylight change by arm, scored year and replicate months."""
    arms = ["m", "f", "s", "mr", "mp", "sp"]
    rows = _change_rows(report=report, subsets=[("daylight", "all daylight half-hours")], arms=arms)
    return _interval_chart(
        rows=rows,
        row_field="subset",
        order=["all daylight half-hours"],
        title="Passing the 51 members through the solar term lowers the daylight score; the placebos do not",
        subtitle=(
            "Dot: change in the quantile score (twice the mean pinball loss) against arm A, daylight "
            "half-hours (sun 5 degrees or higher), leads 0.5 to 168 hours, five decision substations. "
            "Line: 97.5% interval for M and S, 95% for the rest, resampling whole months. Zero: same as A."
        ),
        height=170,
    )


def figure_cloud(*, report: dict) -> alt.TopLevelMixin:
    """Daylight change by clear-sky-index band."""
    arms = ["m", "s", "mr", "mp"]
    bands = ["overcast (under 0.4)", "partly cloudy (0.4 to 0.7)", "clear (0.7 and over)"]
    rows = _change_rows(report=report, subsets=[(f"daylight, {b}", b) for b in bands], arms=arms)
    return _interval_chart(
        rows=rows,
        row_field="subset",
        order=bands,
        title="The member pass helps most when the ensemble mean is partly cloudy",
        subtitle=(
            "Change in the quantile score against arm A by the clear-sky index of the ensemble-mean "
            "irradiance (negative is better). Line: 95% interval from resampling whole months."
        ),
        height=230,
    )


def figure_lead(*, report: dict) -> alt.TopLevelMixin:
    """Daylight change by lead band."""
    arms = ["m", "s", "mr"]
    bands = ["under 24 h", "24 to 72 h", "72 to 168 h"]
    rows = _change_rows(report=report, subsets=[(f"daylight, {b}", b) for b in bands], arms=arms)
    return _interval_chart(
        rows=rows,
        row_field="subset",
        order=bands,
        title="The gain does not fade with lead; it is larger beyond 72 hours",
        subtitle=(
            "Change in the daylight quantile score against arm A by lead band (negative is better). "
            "Line: 95% interval from resampling whole months."
        ),
        height=200,
    )


def figure_coverage(*, report: dict) -> alt.TopLevelMixin:
    """q05-to-q95 coverage of each arm by clear-sky band, scored year."""
    arms = {
        "a": "A: reference",
        "m": ARM_LABELS["m"],
        "s": ARM_LABELS["s"],
        "mr": ARM_LABELS["mr"],
        "mp": ARM_LABELS["mp"],
    }
    colours = {"A: reference": "#808080", **{k: ARM_COLOURS[k] for k in list(arms.values())[1:]}}
    bands = ["overcast (under 0.4)", "partly cloudy (0.4 to 0.7)", "clear (0.7 and over)"]
    rows = []
    for band in [*bands, None]:
        subset = "daylight" if band is None else f"daylight, {band}"
        entry = report["scored"]["subsets"][subset]
        for key, label in arms.items():
            rows.append(
                {"band": band or "all daylight", "arm": label, "coverage": entry[key]["inside90"]}
            )
    chart = (
        alt.Chart(pl.DataFrame(rows))  # ty: ignore[unresolved-attribute]
        .mark_point(filled=True, size=70)
        .encode(
            y=alt.Y(
                "band:N", sort=[*bands, "all daylight"], title=None, axis=alt.Axis(labelLimit=260)
            ),
            yOffset=alt.YOffset("arm:N", sort=list(colours)),
            x=alt.X(
                "coverage:Q",
                scale=alt.Scale(domain=[0.80, 0.94]),
                title="share of readings between the q05 and the q95",
            ),
            color=alt.Color(
                "arm:N",
                scale=alt.Scale(domain=list(colours), range=list(colours.values())),
                legend=alt.Legend(title=None, orient="top", columns=1, labelLimit=400),
            ),
        )
    )
    nominal = alt.Chart().mark_rule(color="#444444").encode(x=alt.datum(0.90))  # ty: ignore[unresolved-attribute]
    return alt.layer(chart, nominal, data=pl.DataFrame(rows)).properties(
        width=WIDTH - 280,
        height=230,
        title=_title(
            text="Arm A's daylight bands cover 85% of readings; the member pass brings that to 89%",
            subtitle="Scored year, five decision substations, leads 0.5 to 168 hours. Line: nominal 0.90.",
        ),
    )


def figure_bg2(*, extra: dict) -> alt.TopLevelMixin:
    """Gap to BG2 on the sunniest third of daylight half-hours."""
    scopes = [
        "five substations, all leads",
        "five substations, under 72 h",
        "five substations without D5",
        "D5 only",
    ]
    arms = {"a": "A: reference", "m": ARM_LABELS["m"], "mp": ARM_LABELS["mp"]}
    colours = {
        "A: reference": "#808080",
        ARM_LABELS["m"]: ocf.BRAND_ORANGE,
        ARM_LABELS["mp"]: ocf.DATA_GREEN,
    }
    rows = []
    for scope in scopes:
        for key, label in arms.items():
            g = extra["bg2_gap"][scope][key]["gap_abs_mw"]
            rows.append(
                {
                    "scope": scope,
                    "arm": label,
                    "estimate": g["mean"],
                    "low": g["lower"],
                    "high": g["upper"],
                }
            )
    base = alt.Chart().encode(
        y=alt.Y("scope:N", sort=scopes, title=None, axis=alt.Axis(labelLimit=260)),
        yOffset=alt.YOffset("arm:N", sort=list(colours)),
        color=alt.Color(
            "arm:N",
            scale=alt.Scale(domain=list(colours), range=list(colours.values())),
            legend=alt.Legend(title=None, orient="top", columns=1, labelLimit=400),
        ),
    )
    layer = alt.layer(
        base.mark_rule().encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("low:Q", title="median error minus BG2's (MW; lower is better)"), x2="high:Q"
        ),
        base.mark_point(filled=True, size=60).encode(x="estimate:Q"),  # ty: ignore[unresolved-attribute]
        alt.Chart().mark_rule(color="#444444").encode(x=alt.datum(0)),  # ty: ignore[unresolved-attribute]
    ).properties(width=WIDTH - 280, height=230)
    return alt.layer(layer, data=pl.DataFrame(rows)).properties(
        title=_title(
            text="The member pass narrows the sunny-daylight gap to BG2, but the interval still includes zero",
            subtitle=(
                "Scored year, sunniest third of daylight half-hours (experiment 178's rule). Zero: same "
                "error as BG2. Line: 95% interval from resampling whole months."
            ),
        )
    )


def figure_spread(*, gates: dict) -> alt.TopLevelMixin:
    """Raw against calibrated member spread by clear-sky band."""
    bands = ["overcast (under 0.4)", "partly cloudy (0.4 to 0.7)", "clear (0.7 and over)"]
    rows = []
    for band in bands:
        g = gates["g4_spread_by_clear_sky_band"][band]
        rows.append({"band": band, "version": "raw ensemble", "std": g["raw_std"]})
        rows.append(
            {
                "band": band,
                "version": "after calibration and solar shrink",
                "std": g["calibrated_std"],
            }
        )
    colours = {
        "raw ensemble": ocf.DATA_BLUE,
        "after calibration and solar shrink": ocf.BRAND_ORANGE,
    }
    return (
        alt.Chart(pl.DataFrame(rows))  # ty: ignore[unresolved-attribute]
        .mark_bar()
        .encode(
            y=alt.Y("band:N", sort=bands, title=None, axis=alt.Axis(labelLimit=260)),
            yOffset=alt.YOffset("version:N", sort=list(colours)),
            x=alt.X(
                "std:Q", title="median standard deviation across members of the clear-sky index"
            ),
            color=alt.Color(
                "version:N",
                scale=alt.Scale(domain=list(colours), range=list(colours.values())),
                legend=alt.Legend(title=None, orient="top"),
            ),
        )
        .properties(
            width=WIDTH - 280,
            height=170,
            title=_title(
                text="Calibration shrinks the members' cloud spread by about 8%, so the spread survives it",
                subtitle="Scored year, daylight half-hours, five decision substations and D9.",
            ),
        )
    )


def main() -> None:
    """Draw every figure."""
    report = json.loads(REPORT.read_text())
    extra = json.loads(EXTRA.read_text())
    gates = json.loads(GATES.read_text())
    _save(figure_headline(report=report), name="figure_1_headline")
    _save(figure_cloud(report=report), name="figure_2_cloud")
    _save(figure_coverage(report=report), name="figure_3_coverage")
    _save(figure_lead(report=report), name="figure_4_lead")
    _save(figure_bg2(extra=extra), name="figure_5_bg2_gap")
    _save(figure_spread(gates=gates), name="figure_6_spread")


if __name__ == "__main__":
    main()
