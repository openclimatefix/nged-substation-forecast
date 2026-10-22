# ruff: noqa: E501
"""Draw the figures of experiment 228 from `work/report/report_scored.json`; write dp_preds files.

The `dp_preds` files follow the layout of experiment 169: `<arm>_scored_<site>.parquet`, quantiles
in points of effective capacity, so that a later assembly can reuse the stored forecasts.

    uv run --no-sync python ml-experiments/00228-*/e00228_charts.py
"""

import json
import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl

import e00228_common as common  # isort: skip

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs" / "ml-experiments" / f"{common.EXPERIMENT_ID:05d}-{common.SLUG}"
)  # fmt: skip
PNG_DIR: Final[Path] = common.WORK_DIR / "png"
WIDTH: Final[int] = 640
ARMS: Final[dict[str, tuple[str, str]]] = {
    "d_minus_bg2": ("D: arm D with calibrated bands", ocf.DATA_SKY),
    "dh_minus_bg2": ("DH: D plus network head", ocf.DATA_BLUE),
    "dhs_minus_bg2": ("DH+S: DH plus solar summary", ocf.BRAND_ORANGE),
}


def _save(chart: alt.TopLevelMixin, *, name: str) -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    chart.save(DOCS_DIR / f"{name}.svg")
    chart.save(PNG_DIR / f"{name}.png", scale_factor=1.5)
    print("wrote", name)


def forest(
    *, report: dict, key: str, order: list[str], title: str, subtitle: str, name: str
) -> None:
    """Draw each arm's score minus BG2's for each group of rows, with its interval."""
    rows = [
        {"group": g, "arm": label, "estimate": v["est"], "low": v["lo"], "high": v["hi"]}
        for contrast, (label, _) in ARMS.items()
        for g, v in report["contrasts"][contrast][key].items()
    ]
    domain = [v[0] for v in ARMS.values()]
    base = alt.Chart(pl.DataFrame(rows)).encode(
        y=alt.Y("group:N", sort=order, title=None),
        yOffset=alt.YOffset("arm:N", sort=domain),
        color=alt.Color(
            "arm:N",
            scale=alt.Scale(domain=domain, range=[v[1] for v in ARMS.values()]),
            legend=alt.Legend(title=None, orient="top", direction="vertical", labelLimit=500),
        ),
    )
    zero = alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(color="#333333").encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    chart = (
        zero
        + base.mark_rule(strokeWidth=2).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X("low:Q", title="Quantile score minus BG2 (points of capacity; below zero is better)"),
            x2="high:Q",
        )
        + base.mark_point(filled=True, size=40).encode(x="estimate:Q")  # ty: ignore[unresolved-attribute]
    ).properties(
        width=WIDTH,
        height=max(120, 26 * len(order) * 2),
        title=alt.TitleParams(
            text=textwrap.wrap(title, 85), subtitle=textwrap.wrap(subtitle, 110),
            anchor="start", fontSize=14, subtitleFontSize=11,
        ),
    )  # fmt: skip
    _save(chart, name=name)


def write_dp_preds() -> None:
    """Write each arm's quantiles for the scored year as `<arm>_scored_<site>.parquet`."""
    import numpy as np

    base = common.ARM_DIR / "scored"
    rows = pl.read_parquet(base / "rows.parquet").select(
        "site", *common.KEYS, "run", "lead_hours", "seed", "actual", "mean"
    )
    out = common.WORK_DIR / "dp_preds"
    out.mkdir(parents=True, exist_ok=True)
    for arm in ("d", "dh", "dhs", "bg2", "climatology", "manual_heuristic"):
        q = np.load(base / f"{arm}.npy")
        frame = rows.with_columns(
            [pl.Series(name, q[:, i]) for i, name in enumerate(common.QUANTILES)]
        )
        for site in common.SCORED_SITES:
            frame.filter(pl.col("site") == site).write_parquet(out / f"{arm}_scored_{site}.parquet")


def main() -> None:
    """Draw the four forest figures and store the forecasts."""
    report = json.loads((common.REPORT_DIR / "report_scored.json").read_text())
    c = report["contrasts"]["dh_minus_bg2"]
    months = list(c["by_month"])
    forest(report=report, key="by_month", order=months,
           title="The stack beats BG2 from November to March and is level or worse from April to October",
           subtitle="Quantile score minus BG2 by month of forecast run, six substations. Intervals resample initialisation days within the month.",
           name="figure_1_by_month")  # fmt: skip
    forest(report=report, key="by_site", order=list(common.SCORED_SITES),
           title="The network head beats BG2 at four of six substations",
           subtitle="Quantile score minus BG2 by substation, scored year. Intervals resample months.",
           name="figure_2_by_site")  # fmt: skip
    forest(report=report, key="by_season", order=list(c["by_season"]),
           title="The gain over BG2 is a winter gain",
           subtitle="Quantile score minus BG2 by season of the valid time. Intervals resample months.",
           name="figure_3_by_season")  # fmt: skip
    forest(report=report, key="by_lead", order=list(c["by_lead"]),
           title="The gain over BG2 shrinks as lead time grows",
           subtitle="Quantile score minus BG2 by lead band, six substations. Intervals resample months.",
           name="figure_4_by_lead")  # fmt: skip
    write_dp_preds()


if __name__ == "__main__":
    main()
