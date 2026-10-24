# ruff: noqa: E501
# ruff: noqa: E501
"""Draw the figures of experiment 332 from the stored contrast tables.

uv run --no-sync python e00332_charts.py
"""

import json
import textwrap
from pathlib import Path
from typing import Any, Final

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

DATA: Final = ML_EXPERIMENTS_DATA_DIR / "00332-lsc-curtailment-cost-of-label-free-mask"
DOCS: Final = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00332-lsc-curtailment-cost-of-label-free-mask"
)
WIDTH: Final = 560
GROUP_LABELS: Final = {
    "DA_during_after": "During and after pooled",
    "G3_during": "During a shift",
    "G4_after": "After (48 hours)",
    "G2_before_alarm": "Before the alarm",
    "G1_ordinary": "Ordinary half-hours",
}


def _chart(data: pl.DataFrame) -> Any:
    """Return an Altair chart typed loosely, because `ty` loses the chart type after `mark_*()`."""
    return alt.Chart(data)


def _save(chart: alt.TopLevelMixin, *, name: str) -> None:
    DOCS.mkdir(parents=True, exist_ok=True)
    chart.save(DOCS / f"{name}.svg")
    print("wrote", name)


def _title(*, title: str, subtitle: str) -> alt.TitleParams:
    return alt.TitleParams(
        text=textwrap.wrap(title, 80),
        subtitle=textwrap.wrap(subtitle, 105),
        anchor="start",
        fontSize=14,
        subtitleFontSize=11,
    )


def _forest(
    *,
    rows: list[dict],
    order: list[str],
    colours: dict[str, str],
    x_title: str,
    title: str,
    subtitle: str,
    name: str,
    height: int,
) -> None:
    base = _chart(pl.DataFrame(rows)).encode(
        y=alt.Y("label:N", sort=order, title=None),
        yOffset="series:N",
        color=alt.Color(
            "series:N",
            scale=alt.Scale(domain=list(colours), range=list(colours.values())),
            legend=alt.Legend(title=None, orient="top"),
        ),
    )
    zero = (
        _chart(pl.DataFrame({"x": [0.0]}))
        .mark_rule(strokeDash=[4, 3], color="#444444")
        .encode(x="x:Q")
    )
    chart = (
        zero
        + base.mark_rule(strokeWidth=2).encode(x=alt.X("lo:Q", title=x_title), x2="hi:Q")
        + base.mark_point(size=70, filled=True).encode(x="est:Q")
    ).properties(width=WIDTH, height=height, title=_title(title=title, subtitle=subtitle))
    _save(chart, name=name)


def main() -> None:
    """Draw all four figures."""
    inj = json.loads((DATA / "inj_contrasts.json").read_text())
    real = json.loads((DATA / "real_cost.json").read_text())
    # 1. A minus C by row group, observed power and true series
    rows = []
    for target, series in (
        ("obs", "Observed power (money target)"),
        ("nra", "True normal running arrangement series"),
    ):
        for g, label in GROUP_LABELS.items():
            r = inj[f"all|{target}|f0.5|{g}|A-C"]
            rows.append(
                {"label": label, "series": series, "est": r["est"], "lo": r["lo"], "hi": r["hi"]}
            )
    _forest(
        rows=rows,
        order=list(GROUP_LABELS.values()),
        colours={
            "Observed power (money target)": ocf.BRAND_ORANGE,
            "True normal running arrangement series": ocf.DATA_BLUE,
        },
        x_title="Arm A minus arm C, cost per half-hour (GBP; below zero means the mask lowers cost)",
        title="Masking lowers cost against the true series but not against observed power",
        subtitle="Injected load shifts at six Primaries, 44 worlds, limit 0.5, 12 to 36 hour leads, GBP 2,000 per MWh-equivalent. Primaries have no published limit, so this ranks arms and states no money saved. Line: 95% interval from resampling whole worlds.",
        name="figure_1_a_minus_c_by_group",
        height=250,
    )
    # 2. contrasts during and after, observed power
    pairs = [
        ("A-C", "A: label-free mask minus none"),
        ("B-C", "B: 24 hour residual mask minus none"),
        ("D-C", "D: true spans masked minus none"),
        ("A_plus-C", "A with moved MW added back minus none"),
        ("D_plus-C", "D with moved MW added back minus none"),
    ]
    rows = []
    for g, series in (
        ("DA_during_after", "During and after pooled"),
        ("G3_during", "During a shift"),
    ):
        for key, label in pairs:
            r = inj[f"all|obs|f0.5|{g}|{key}"]
            rows.append(
                {"label": label, "series": series, "est": r["est"], "lo": r["lo"], "hi": r["hi"]}
            )
    _forest(
        rows=rows,
        order=[p[1] for p in pairs],
        colours={"During and after pooled": ocf.BRAND_ORANGE, "During a shift": ocf.DATA_PURPLE},
        x_title="Cost per half-hour minus arm C (GBP; below zero is better)",
        title="On observed power no mask lowers cost with certainty, not even the true spans",
        subtitle="Injected worlds, observed power, limit 0.5. Adding the moved megawatts back uses the true megawatts, as in experiment 312, so it is an upper bound.",
        name="figure_2_contrasts_observed",
        height=330,
    )
    # 3. per Primary
    sites = ["D10", "D11", "D14", "D16", "D5", "D9"]
    rows = [
        {
            "label": s,
            "series": "A minus C, during and after",
            "est": inj[f"site|{s}"]["est"],
            "lo": inj[f"site|{s}"]["lo"],
            "hi": inj[f"site|{s}"]["hi"],
        }
        for s in sites
    ]
    _forest(
        rows=rows,
        order=sites,
        colours={"A minus C, during and after": ocf.BRAND_ORANGE},
        x_title="Arm A minus arm C, cost per half-hour (GBP; below zero is better)",
        title="Two Primaries gain and none loses with certainty",
        subtitle="Observed power, limit 0.5, during and after shifts pooled; each line resamples whole worlds at that Primary.",
        name="figure_3_per_primary",
        height=200,
    )
    # 4. real record
    rec = real["f0.5|G1_ordinary"]
    names = {
        "A-C": "A minus C",
        "B-C": "B minus C",
        "D-C": "D (hand labels) minus C",
        "E-C": "E (two passes) minus C",
        "AF-C": "Filter-only mask minus C",
    }
    rows = [
        {
            "label": names[k],
            "series": "Cost, GBP",
            "est": rec["contrasts"][k]["est"] / 1e3,
            "lo": rec["contrasts"][k]["lo"] / 1e3,
            "hi": rec["contrasts"][k]["hi"] / 1e3,
        }
        for k in names
    ]
    _forest(
        rows=rows,
        order=list(names.values()),
        colours={"Cost, GBP": ocf.DATA_BLUE},
        x_title="Cost minus arm C (GBP thousand over six months; below zero is better)",
        title="On the real record the mask changes cost by less than the metric can resolve",
        subtitle="Bulk supply point D13 with its metered solar farms, ordinary half-hours of the six even months with farm forecasts, limit 0.5. Line: 95% interval from resampling whole months.",
        name="figure_4_real_record",
        height=200,
    )


if __name__ == "__main__":
    main()
