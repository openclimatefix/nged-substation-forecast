# ruff: noqa: E501
"""Draw the figures of experiment 277 from `work/report_*.json`.

uv run --no-sync python ml-experiments/00277-*/e00277_charts.py
"""

import json
import textwrap
from pathlib import Path
from typing import Any, Final

import altair as alt
import e00277_common as common
import plotting.ocf_theme as ocf
import polars as pl

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / f"{common.EXPERIMENT_ID:05d}-{common.SLUG}"
)
PNG_DIR: Final[Path] = common.OUT_DIR / "png"
WIDTH: Final[int] = 560
COLOURS: Final[dict[str, str]] = {
    "Climatology": "#7A7A7A",
    "Manual heuristic": "#222222",
    "XGBoost BG2": ocf.BRAND_ORANGE,
    "GAM stack (arm 1)": ocf.DATA_PURPLE,
    "Weak XGBoost (reference)": "#B5B5B5",
    "Top-down GAM": ocf.DATA_BLUE,
    "Bottom-up GAM": ocf.DATA_SKY,
    "GAM-K stack, unmasked (K0)": ocf.DATA_PURPLE,
}
PAIR_LABELS: Final[dict[str, str]] = {
    "climatology-arm1": "Climatology minus GAM stack",
    "manual_heuristic-arm1": "Manual heuristic minus GAM stack",
    "bg2-arm1": "BG2 minus GAM stack",
    "climatology-bg2": "Climatology minus BG2",
    "manual_heuristic-bg2": "Manual heuristic minus BG2",
    "climatology-manual_heuristic": "Climatology minus manual heuristic",
}


def _chart(data: pl.DataFrame) -> Any:
    """Return an Altair chart typed loosely, because `ty` loses the chart type after `mark_*()`."""
    return alt.Chart(data)


def _save(chart: alt.TopLevelMixin, *, name: str) -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    chart.save(DOCS_DIR / f"{name}.svg")
    chart.save(PNG_DIR / f"{name}.png", scale_factor=1.5)
    print("wrote", name)


def _title(*, title: str, subtitle: str) -> alt.TitleParams:
    return alt.TitleParams(
        text=textwrap.wrap(title, 80),
        subtitle=textwrap.wrap(subtitle, 105),
        anchor="start",
        fontSize=14,
        subtitleFontSize=11,
    )


def _scale(labels: list[str]) -> alt.Scale:
    return alt.Scale(domain=labels, range=[COLOURS[label] for label in labels])


def cost_by_arm(*, a: dict) -> None:
    """Avoidable cost of each forecast at the 0.5 limit, with its interval."""
    arms = a["A_cost"]["0.5"]["all"]["arms"]
    rows = [
        {
            "arm": common.ARM_LABELS[k],
            "est": v["avoidable"]["est"] / 1e6,
            "lo": v["avoidable"]["lo"] / 1e6,
            "hi": v["avoidable"]["hi"] / 1e6,
        }
        for k, v in arms.items()
    ]
    order = [
        common.ARM_LABELS[k] for k in ("climatology", "manual_heuristic", "bg2", "arm1", "xgboost")
    ]
    base = _chart(pl.DataFrame(rows)).encode(
        y=alt.Y("arm:N", sort=order, title=None),
        color=alt.Color("arm:N", scale=_scale(order), legend=None),
    )
    chart = (
        base.mark_rule(strokeWidth=2).encode(
            x=alt.X(
                "lo:Q",
                title="Avoidable curtailment cost (GBP million, six Primaries, one year; smaller is better)",
            ),
            x2="hi:Q",
        )
        + base.mark_point(size=90, filled=True).encode(x="est:Q")
    ).properties(
        width=WIDTH,
        height=190,
        title=_title(
            title="At the 0.5 limit the GAM stack has the lowest curtailment cost of the four forecasts",
            subtitle="Cost above a perfect forecast; 12 to 36 hour leads; limit 0.5 times each Primary's largest earlier demand; GBP 2,000 per MWh-equivalent. Primaries have no published limit, so this ranks forecasts and states no money saved. Dot: estimate. Line: 95% interval from resampling whole months.",
        ),
    )
    _save(chart, name="figure_1_cost_by_arm")


def contrasts(*, a: dict) -> None:
    """Cost differences between pairs of forecasts at 0.5 and 0.75 of the largest earlier demand."""
    rows = []
    for fraction in ("0.5", "0.75"):
        for pair, v in a["A_cost"][fraction]["all"]["contrasts"].items():
            rows.append(
                {
                    "pair": PAIR_LABELS[pair],
                    "limit": f"limit {fraction}",
                    "est": v["est"] / 1e6,
                    "lo": v["lo"] / 1e6,
                    "hi": v["hi"] / 1e6,
                }
            )
    order = [PAIR_LABELS[p] for p in PAIR_LABELS]
    base = _chart(pl.DataFrame(rows)).encode(
        y=alt.Y("pair:N", sort=order, title=None),
        color=alt.Color(
            "limit:N",
            scale=alt.Scale(
                domain=["limit 0.5", "limit 0.75"], range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE]
            ),
            legend=alt.Legend(title=None, orient="top"),
        ),
        yOffset="limit:N",
    )
    zero = (
        _chart(pl.DataFrame({"x": [0.0]}))
        .mark_rule(strokeDash=[4, 3], color="#444444")
        .encode(x="x:Q")
    )
    chart = (
        base.mark_rule(strokeWidth=2).encode(
            x=alt.X(
                "lo:Q",
                title="Cost of the first forecast minus the second (GBP million; positive means the second is cheaper)",
            ),
            x2="hi:Q",
        )
        + base.mark_point(size=70, filled=True).encode(x="est:Q")
        + zero
    ).properties(
        width=WIDTH,
        height=230,
        title=_title(
            title="The GAM stack beats BG2 at the 0.5 limit, and the intervals at the 0.75 limit are borderline",
            subtitle="Dashed line: no difference. Dot: estimate. Line: 95% interval from resampling whole months. Six Primaries, 12 to 36 hour leads, GBP 2,000 per MWh-equivalent.",
        ),
    )
    _save(chart, name="figure_2_cost_contrasts")


def detectability(*, a: dict) -> None:
    """Observed difference against the smallest difference the design resolves, in percent of the worse forecast."""
    rows = []
    for pair, v in a["A_detectability"]["cost"].items():
        rows.append(
            {
                "pair": PAIR_LABELS[pair],
                "metric": "Avoidable cost",
                "kind": "Observed difference",
                "percent": 100.0 * v["observed"] / (v["mde"] / (v["mde_relative_percent"] / 100.0)),
            }
        )
        rows.append(
            {
                "pair": PAIR_LABELS[pair],
                "metric": "Avoidable cost",
                "kind": "Smallest resolvable",
                "percent": v["mde_relative_percent"],
            }
        )
    for pair, v in a["A_detectability"]["score"].items():
        for metric, label in (("qs", "CRPS"), ("tw95", "twCRPS at 95th percentile")):
            x = v[metric]
            rows.append(
                {
                    "pair": PAIR_LABELS[pair],
                    "metric": label,
                    "kind": "Observed difference",
                    "percent": 100.0
                    * x["observed"]
                    / (x["mde"] / (x["mde_relative_percent"] / 100.0)),
                }
            )
            rows.append(
                {
                    "pair": PAIR_LABELS[pair],
                    "metric": label,
                    "kind": "Smallest resolvable",
                    "percent": x["mde_relative_percent"],
                }
            )
    order = [PAIR_LABELS[p] for p in PAIR_LABELS]
    chart = (
        _chart(pl.DataFrame(rows))
        .mark_point(size=80, filled=True)
        .encode(
            y=alt.Y("pair:N", sort=order, title=None),
            x=alt.X(
                "percent:Q", title="Difference as a percentage of the first forecast's own score"
            ),
            color=alt.Color(
                "kind:N",
                scale=alt.Scale(
                    domain=["Observed difference", "Smallest resolvable"],
                    range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE],
                ),
                legend=alt.Legend(title=None, orient="top"),
            ),
            row=alt.Row("metric:N", title=None),
        )
        .properties(
            width=WIDTH - 120,
            height=130,
            title=_title(
                title="One scored year resolves cost differences of about 13% to 30%, but CRPS differences of only about 6% to 13%",
                subtitle="Smallest resolvable: 2.8 standard errors of the paired difference, from resampling whole months (80% power at the 5% level). An observed dot to the right of the blue dot is a difference the design is expected to detect.",
            ),
        )
    )
    _save(chart, name="figure_3_detectability")


def groups(*, c: dict) -> None:
    """CRPS by row group for the four forecasts on experiment 163's rows."""
    names = {
        "K0": "GAM-K stack, unmasked (K0)",
        "bg2": "XGBoost BG2",
        "manual_heuristic": "Manual heuristic",
        "climatology": "Climatology",
    }
    order_groups = ["G1 ordinary rows", "G3 during", "G4 after the end"]
    rows = []
    for g in order_groups:
        for arm, label in names.items():
            v = c["metrics"][g][arm]["qs"]
            rows.append(
                {
                    "group": f"{g} ({c['n_cards'][g] or 12} {'months' if g.startswith('G1') else 'transfers'})",
                    "arm": label,
                    "est": v["est"],
                    "lo": v["lo"],
                    "hi": v["hi"],
                }
            )
    labels = list(names.values())
    order = list(dict.fromkeys(r["group"] for r in rows))
    base = _chart(pl.DataFrame(rows)).encode(
        y=alt.Y("arm:N", sort=labels, title=None, axis=alt.Axis(labels=True)),
        color=alt.Color("arm:N", scale=_scale(labels), legend=None),
    )
    chart = (
        (
            base.mark_rule(strokeWidth=2).encode(
                x=alt.X("lo:Q", title="CRPS (points of effective capacity; smaller is better)"),
                x2="hi:Q",
            )
            + base.mark_point(size=80, filled=True).encode(x="est:Q")
        )
        .properties(width=WIDTH - 60, height=120)
        .facet(row=alt.Row("group:N", sort=order, title=None))
        .properties(
            title=_title(
                title="After a logged transfer ends the unmasked GAM stack has the highest CRPS of the four forecasts, but the intervals overlap",
                subtitle="Six Primaries, scored year, experiment 163's rows. Dot: estimate. Line: 95% interval from resampling whole months (ordinary rows) or whole logged transfers (during, after). The during and after groups rest on 3 and 7 transfers, so their lines understate the uncertainty.",
            )
        )
    )
    _save(chart, name="figure_4_row_groups")


def bsp(*, b: dict) -> None:
    """Avoidable cost at the two bulk supply points, as demand-only nodes."""
    rows = []
    for fraction in ("0.5", "0.75"):
        for arm, v in b["cost"][fraction]["arms"].items():
            rows.append(
                {
                    "arm": common.ARM_LABELS[arm],
                    "limit": f"limit {fraction}",
                    "est": v["avoidable"]["est"] / 1e6,
                    "lo": v["avoidable"]["lo"] / 1e6,
                    "hi": v["avoidable"]["hi"] / 1e6,
                }
            )
    order = [common.ARM_LABELS[k] for k in ("climatology", "manual_heuristic", "td", "bu")]
    base = _chart(pl.DataFrame(rows)).encode(
        y=alt.Y("arm:N", sort=order, title=None),
        color=alt.Color("arm:N", scale=_scale(order), legend=None),
    )
    chart = (
        (
            base.mark_rule(strokeWidth=2).encode(
                x=alt.X(
                    "lo:Q",
                    title="Avoidable curtailment cost (GBP million, D13 and D17, one year; smaller is better)",
                ),
                x2="hi:Q",
            )
            + base.mark_point(size=80, filled=True).encode(x="est:Q")
        )
        .properties(width=WIDTH - 60, height=120)
        .facet(row=alt.Row("limit:N", title=None))
        .properties(
            title=_title(
                title="At the two bulk supply points the two GAM forecasts cost less than the baselines, but two nodes cannot separate them",
                subtitle="Demand-only nodes (no solar farms attached), 12 to 36 hour leads, GBP 2,000 per MWh-equivalent. Dot: estimate. Line: 95% interval from resampling whole months.",
            )
        )
    )
    _save(chart, name="figure_5_bulk_supply_points")


def main() -> None:
    """Draw every figure."""
    a = json.loads((common.OUT_DIR / "report_a.json").read_text())
    b = json.loads((common.OUT_DIR / "report_b.json").read_text())
    c = json.loads((common.OUT_DIR / "report_c.json").read_text())
    cost_by_arm(a=a)
    contrasts(a=a)
    detectability(a=a)
    groups(c=c)
    bsp(b=b)


if __name__ == "__main__":
    main()
