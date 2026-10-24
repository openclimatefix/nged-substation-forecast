# ruff: noqa: E501
"""Draw the figures of experiment 335 from `work/report.json` and `work/oracle.json`.

uv run --no-sync python ml-experiments/00335-*/e00335_charts.py
"""

import json
from pathlib import Path
from typing import Any, Final

import altair as alt
import e00335_common as common
import plotting.ocf_theme as ocf
import polars as pl

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / f"{common.EXPERIMENT_ID:05d}-{common.SLUG}"
)
NAMES: Final[dict[str, str]] = {
    "d3": "Default (Primary demand research default 3)",
    "c1": "C1 per-band, per-group widening",
    "c2": "C2 Student-t tail",
    "c3": "C3 adaptive conformal",
    "oracle": "C1 fitted with hindsight (post hoc)",
}
COLOURS: Final[dict[str, str]] = {
    NAMES["d3"]: ocf.DATA_BLUE, NAMES["c1"]: ocf.DATA_GREEN, NAMES["c2"]: ocf.DATA_PURPLE,
    NAMES["c3"]: ocf.BRAND_ORANGE, NAMES["oracle"]: "#7A7A7A",
}  # fmt: skip


def _chart(data: Any) -> Any:
    """Return an Altair chart typed loosely, because `ty` loses the chart type after `mark_*()`."""
    return alt.Chart(data)


def _line(
    *,
    data: pl.DataFrame,
    y: str,
    title: str,
    subtitle: str,
    ytitle: str,
    rule: float | None,
    name: str,
) -> None:
    base = _chart(data)
    layer = base.mark_line(point=True).encode(
        x=alt.X("lead:Q", title="Lead (hours; centre of each 24-hour band)"),
        y=alt.Y(f"{y}:Q", title=ytitle, scale=alt.Scale(zero=False)),
        color=alt.Color(
            "arm:N",
            scale=alt.Scale(domain=list(COLOURS), range=list(COLOURS.values())),
            title=None,
            legend=alt.Legend(orient="bottom", columns=2, labelLimit=500),
        ),
    )
    layers = [layer]
    if rule is not None:
        layers.append(
            _chart(pl.DataFrame({"r": [rule]}))
            .mark_rule(strokeDash=[4, 4], color="black")
            .encode(y="r:Q")
        )
    chart = alt.layer(*layers).properties(
        width=620, height=280, title=alt.TitleParams(text=title, subtitle=subtitle, anchor="start")
    )
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    chart.save(DOCS_DIR / f"{name}.svg")
    chart.save(common.WORK_DIR / f"{name}.png", scale_factor=1.5)


def main() -> None:
    """Draw the tail-rate, coverage, and tail-pinball figures."""
    rep = json.loads((common.WORK_DIR / "report.json").read_text())
    oracle = json.loads((common.WORK_DIR / "oracle.json").read_text())
    rows = []
    for i, label in enumerate(common.BAND_LABELS):
        for a in ("d3", "c1", "c2", "c3"):
            t = rep["tails_by_band"][label][a]
            rows.append(
                {
                    "lead": i * 24 + 12,
                    "arm": NAMES[a],
                    **{k: 100 * t[k] for k in ("below_q01", "above_q99", "in_5_95")},
                    "q01_pinball": t["q01_pinball"],
                    "q99_pinball": t["q99_pinball"],
                }
            )
        o = oracle["by_band"][label]
        rows.append(
            {
                "lead": i * 24 + 12,
                "arm": NAMES["oracle"],
                **{k: 100 * o[k] for k in ("below_q01", "above_q99", "in_5_95")},
                "q01_pinball": None,
                "q99_pinball": None,
            }
        )
    data = pl.DataFrame(rows)
    sub = "Six Primaries, scored year 2025-07-02 to 2026-06-30, origin 09:00 UTC. Calibrations are fitted as-of on earlier origins."
    _line(
        data=data,
        y="below_q01",
        title="Share of observations below q01 (nominal 1%)",
        subtitle=sub,
        ytitle="Share below q01 (%)",
        rule=1.0,
        name="tail_below_q01",
    )
    _line(
        data=data,
        y="above_q99",
        title="Share of observations above q99 (nominal 1%)",
        subtitle=sub,
        ytitle="Share above q99 (%)",
        rule=1.0,
        name="tail_above_q99",
    )
    _line(
        data=data,
        y="in_5_95",
        title="Coverage of the 5% to 95% band (nominal 90%)",
        subtitle=sub,
        ytitle="Coverage (%)",
        rule=90.0,
        name="coverage_5_95",
    )
    diff = data.filter(pl.col("q01_pinball").is_not_null()).with_columns(
        d99=pl.col("q99_pinball")
        - pl.col("q99_pinball").filter(pl.col("arm") == NAMES["d3"]).first().over("lead")
    )
    _line(
        data=diff,
        y="d99",
        title="q99 pinball loss against the default (above zero is worse)",
        subtitle=sub,
        ytitle="Pinball loss difference (points)",
        rule=0.0,
        name="q99_pinball_difference",
    )


if __name__ == "__main__":
    main()
