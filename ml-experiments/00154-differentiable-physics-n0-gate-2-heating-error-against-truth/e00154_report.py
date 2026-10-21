"""Print the numbers and draw the figures of experiment 154 from the stored gate runs.

    uv run --no-sync python ml-experiments/00154-*/e00154_report.py

Reads `work/gate/e154_seed*.json`, writes `report.txt` and the figures beside the experiment's
`result.md`.
"""

import json
import textwrap
from pathlib import Path
from typing import Any, Final

import altair as alt
import e00154_common as common
import e00154_gate as gate
import numpy as np
import plotting.ocf_theme as ocf
import polars as pl
from studies.charts import CONTENT_WIDTH_PX

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / (f"{common.EXPERIMENT_ID:05d}-{common.SLUG}")
)
WRAP: Final[int] = 110
ESTIMATORS: Final[dict[str, str]] = {
    "pooled": "Pooled fit (tuned penalty)",
    "oracle": "Fair oracle",
    "complete_pooling": "Complete pooling",
    "no_pooling": "No pooling",
}
COLOURS: Final[tuple[str, ...]] = (ocf.DATA_BLUE, ocf.BRAND_ORANGE, ocf.DATA_GREEN, ocf.DATA_PURPLE)


def _title(*, text: str, subtitle: str) -> alt.TitleParams:
    return alt.TitleParams(
        text=text,
        subtitle=textwrap.wrap(subtitle, WRAP),
        anchor="start",
        fontSize=14,
        subtitleFontSize=11,
    )


def _save(chart: alt.TopLevelMixin, *, name: str) -> None:
    path = DOCS_DIR / f"{name}.svg"
    chart.save(path)
    with path.open("a", encoding="utf-8") as handle:
        handle.write("\n")
    print("wrote", name)


def _colour() -> alt.Color:
    return alt.Color(
        "estimator:N",
        scale=alt.Scale(domain=list(ESTIMATORS.values()), range=list(COLOURS)),
        legend=alt.Legend(orient="top", title=None, columns=2),
    )


def error_figure(*, runs: list[dict[str, Any]]) -> alt.TopLevelMixin:
    """Return the heating error per seed and estimator (smaller is better)."""
    rows = [
        {"seed": f"seed {s['seed'] - 20261100}", "estimator": label, "error": s["errors"][key]}
        for s in (gate.score_seed(run=run) for run in runs)
        for key, label in ESTIMATORS.items()
    ]
    return (
        alt.Chart(pl.DataFrame(rows))  # ty: ignore[unresolved-attribute]
        .mark_point(filled=True, size=80, opacity=0.9)
        .encode(
            x=alt.X(
                "error:Q", title="Heating error (points of effective capacity; smaller is better)"
            ),
            y=alt.Y("seed:N", title=None),
            color=_colour(),
            shape=alt.Shape("estimator:N", legend=None),
            yOffset=alt.YOffset("estimator:N", sort=list(ESTIMATORS.values())),
        )
        .properties(
            width=CONTENT_WIDTH_PX - 90,
            height=190,
            title=_title(
                text="The pooled fit has a smaller heating error than both controls on all 5 seeds",
                subtitle="Dot: root mean square, over the 8 substations that took a turn as the "
                "4-week substation, of the fitted minus injected heating contribution over the "
                "winter rows. Substations are synthetic; 4 weeks held before 2025-02-01.",
            ),
        )
    )


def offset_figure(*, runs: list[dict[str, Any]]) -> alt.TopLevelMixin:
    """Return each substation's fitted offset against its injected offset, for three estimators."""
    rows = []
    for run in runs:
        for turn in run["turns"]:
            rows.extend(
                {
                    "seed": f"seed {run['seed'] - 20261100}",
                    "estimator": ESTIMATORS[key],
                    "true": turn["true_offset"],
                    "fitted": turn[key]["offset"],
                }
                for key in ("pooled", "oracle")
            )
    data = pl.DataFrame(rows)
    limit = float(data.select(pl.col("true").abs().max())[0, 0]) * 1.6
    points = (
        alt.Chart(data)  # ty: ignore[unresolved-attribute]
        .mark_point(filled=True, size=40, opacity=0.7, clip=True)
        .encode(
            x=alt.X(
                "true:Q",
                scale=alt.Scale(domain=[-limit, limit]),
                title="Injected offset (points of effective capacity)",
            ),
            y=alt.Y(
                "fitted:Q",
                scale=alt.Scale(domain=[-limit, limit]),
                title="Fitted offset (points)",
            ),
            color=alt.Color(
                "estimator:N",
                scale=alt.Scale(domain=list(ESTIMATORS.values()), range=list(COLOURS)),
                legend=None,
            ),
        )
    )
    diagonal = (
        alt.Chart(data)  # ty: ignore[unresolved-attribute]
        .mark_line(color="#888888", strokeDash=[4, 3])
        .encode(x="true:Q", y="true:Q", order="true:Q")
    )
    return (
        (points + diagonal)
        .properties(width=(CONTENT_WIDTH_PX - 90) // 2, height=(CONTENT_WIDTH_PX - 90) // 3)
        .facet(column=alt.Column("estimator:N", title=None, sort=list(ESTIMATORS.values())))
        .properties(
            title=_title(
                text="From 4 weeks, a substation's own heating offset is found on only some seeds",
                subtitle="Dot: one synthetic substation on one of 5 seeds. Dashed line: fitted "
                "equals injected. The offset is the mean heating contribution over winter rows "
                "minus the fleet curve's. No pooling is left out: its offsets mostly fall "
                "outside these axes.",
            )
        )
    )


def curve_figure(*, runs: list[dict[str, Any]]) -> alt.TopLevelMixin:
    """Return the fleet curve's error against the average of per-site fits' error, per seed."""
    rows = []
    for run in runs:
        score = gate.score_seed(run=run)
        seed = f"seed {run['seed'] - 20261100}"
        rows.append(
            {"seed": seed, "estimator": "Pooled fleet curve", "error": score["fleet_curve_rmse"]}
        )
        rows.append(
            {
                "seed": seed,
                "estimator": "Average of per-site fits",
                "error": score["per_site_mean_curve_rmse"],
            }
        )
    return (
        alt.Chart(pl.DataFrame(rows))  # ty: ignore[unresolved-attribute]
        .mark_point(filled=True, size=80)
        .encode(
            x=alt.X(
                "error:Q",
                title="Fleet-curve error (points of effective capacity; smaller is better)",
            ),
            y=alt.Y("seed:N", title=None),
            color=alt.Color(
                "estimator:N",
                scale=alt.Scale(
                    domain=["Pooled fleet curve", "Average of per-site fits"],
                    range=[ocf.DATA_BLUE, ocf.DATA_PURPLE],
                ),
                legend=alt.Legend(orient="top", title=None),
            ),
            yOffset=alt.YOffset("estimator:N"),
        )
        .properties(
            width=CONTENT_WIDTH_PX - 90,
            height=150,
            title=_title(
                text="The pooled fleet curve is as close to the truth as per-site fits",
                subtitle="Dot: root mean square error of the shared heating curve against the "
                "injected fleet mean, over the winter rows of the 8 substations, from the joint "
                "fit of each turn.",
            ),
        )
    )


def main() -> None:
    """Write `report.txt` and the figures."""
    runs = [json.loads(p.read_text()) for p in sorted(gate.GATE_DIR.glob("e154_seed*.json"))]
    lines = ["== Decision rule", json.dumps(gate.summarise(), indent=2)]
    lines.append("== Heating error per seed (points of effective capacity)")
    for run in runs:
        s = gate.score_seed(run=run)
        errors = {k: round(v, 2) for k, v in s["errors"].items()}
        lines.append(f"seed {run['seed']}: penalty {run['lambda']:g}; {errors}")
    lines.append("== Coefficient-space error (root mean square over the 8 turns)")
    for run in runs:
        parts = {
            key: round(
                float(np.sqrt(np.mean([t[key]["coefficient_rmse"] ** 2 for t in run["turns"]]))), 2
            )
            for key in ESTIMATORS
        }
        lines.append(f"seed {run['seed']}: {parts}")
    lines.append(
        "== Offset sign: substations of 8 with the injected sign, per estimator (post hoc: all)"
    )
    for run in runs:
        counts = {
            key: int(
                sum(np.sign(t[key]["offset"]) == np.sign(t["true_offset"]) for t in run["turns"])
            )
            for key in ("pooled", "oracle", "no_pooling")
        }
        slope = {
            key: round(
                float(
                    sum(t[key]["offset"] * t["true_offset"] for t in run["turns"])
                    / sum(t["true_offset"] ** 2 for t in run["turns"])
                ),
                2,
            )
            for key in ("pooled", "oracle", "no_pooling")
        }
        size = [round(abs(t["true_offset"]), 2) for t in run["turns"]]
        lines.append(f"seed {run['seed']}: sign {counts}; slope {slope}; |injected offsets| {size}")
    lines.append("== Fair oracle capacity ratio (fitted over true) per turn")
    lines.extend(
        f"seed {run['seed']}: "
        + " ".join(f"{t['oracle']['capacity_ratio']:.2f}" for t in run["turns"])
        for run in runs
    )
    (DOCS_DIR / "report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    _save(error_figure(runs=runs), name="figure_01_heating_error")
    _save(curve_figure(runs=runs), name="figure_02_fleet_curve")
    _save(offset_figure(runs=runs), name="figure_03_offsets")


if __name__ == "__main__":
    main()
