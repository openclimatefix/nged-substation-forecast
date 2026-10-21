"""Print the numbers and draw the figures of experiment 150 from the stored gate runs.

    uv run --no-sync python ml-experiments/00150-*/e00150_report.py

Reads `work/gate/gate_seed*.json` (the pre-registered runs, 4 June weeks held) and
`work/gate/winter_seed*.json` (the post hoc January runs), writes `report.txt` and the figures
beside the experiment's `result.md`.
"""

import json
import textwrap
from pathlib import Path
from typing import Any, Final

import altair as alt
import e00150_common as common
import e00150_gate as gate
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
ESTIMATORS: Final[tuple[str, ...]] = (
    "Pooled fit (tuned penalty)",
    "Fair oracle",
    "Complete pooling",
    "No pooling",
)
COLOURS: Final[tuple[str, ...]] = (ocf.DATA_BLUE, ocf.BRAND_ORANGE, ocf.DATA_GREEN, ocf.DATA_PURPLE)


def _load(*, pattern: str) -> list[dict[str, Any]]:
    return [json.loads(p.read_text()) for p in sorted(gate.GATE_DIR.glob(pattern))]


def _rows(*, runs: list[dict[str, Any]], period: str, june: bool) -> list[dict[str, Any]]:
    rows = []
    for run in runs:
        b = run["checks"]["b_shrinkage_on_4_weeks"] if june else run
        pooled, controls = b["pooled"], b["controls"]
        seed = str(run["seed"] - 20261100)
        for name, source in (
            (ESTIMATORS[0], pooled),
            (
                ESTIMATORS[1],
                {
                    "slope_on_heating_contribution": pooled["oracle_slope_on_heating_contribution"],
                    "rmse": pooled["oracle_rmse"],
                },
            ),
            (ESTIMATORS[2], controls["complete_pooling"]),
            (ESTIMATORS[3], controls["no_pooling"]),
        ):
            rows.append(
                {
                    "period": period,
                    "seed": f"seed {seed}",
                    "estimator": name,
                    "slope": source["slope_on_heating_contribution"],
                    "rmse": source["rmse"],
                }
            )
    return rows


def _title(*, text: str, subtitle: str) -> alt.TitleParams:
    return alt.TitleParams(
        text=text,
        subtitle=textwrap.wrap(subtitle, WRAP),
        anchor="start",
        fontSize=14,
        subtitleFontSize=11,
    )


def _save(chart: alt.TopLevelMixin, *, name: str) -> None:
    chart.save(DOCS_DIR / f"{name}.svg")
    with (DOCS_DIR / f"{name}.svg").open("a", encoding="utf-8") as handle:
        handle.write("\n")
    chart.save(f"/tmp/png150/{name}.png", scale_factor=1.5)
    print("wrote", name)


def dot_figure(
    *, data: pl.DataFrame, column: str, x_title: str, title: str, subtitle: str
) -> alt.TopLevelMixin:
    """Return dots per seed and estimator, one panel per held period."""
    colour = alt.Color(
        "estimator:N",
        scale=alt.Scale(domain=list(ESTIMATORS), range=list(COLOURS)),
        legend=alt.Legend(orient="top", title=None, columns=2),
    )
    return (
        alt.Chart(data)  # ty: ignore[unresolved-attribute]
        .mark_point(filled=True, size=70, opacity=0.9)
        .encode(
            x=alt.X(f"{column}:Q", title=x_title),
            y=alt.Y("seed:N", title=None),
            color=colour,
            shape=alt.Shape("estimator:N", legend=None),
            yOffset=alt.YOffset("estimator:N", sort=list(ESTIMATORS)),
        )
        .properties(width=CONTENT_WIDTH_PX - 90, height=130)
        .facet(row=alt.Row("period:N", title=None, sort=["4 June weeks", "4 January weeks"]))
        .properties(title=_title(text=title, subtitle=subtitle))
    )


def tuning_figure(*, runs: list[dict[str, Any]]) -> alt.TopLevelMixin:
    """Return the tuned penalty beside the best penalty on the noise-free truth, per seed."""
    rows = []
    for run in runs:
        d = run["checks"]["d_tuning_within_one_step"]
        seed = f"seed {run['seed'] - 20261100}"
        rows.append({"seed": seed, "which": "Tuning rule", "penalty": max(d["lambda_star"], 30.0)})
        rows.append(
            {"seed": seed, "which": "Best on truth", "penalty": max(d["lambda_truth_best"], 30.0)}
        )
    return (
        alt.Chart(pl.DataFrame(rows))  # ty: ignore[unresolved-attribute]
        .mark_point(filled=True, size=90)
        .encode(
            x=alt.X(
                "penalty:Q",
                scale=alt.Scale(type="log", domain=[30, 3000]),
                axis=alt.Axis(values=[100, 300, 1000, 3000]),
                title="Deviation penalty (rows of evidence; grid values 0, 100, 300, 1,000, 3,000)",
            ),
            y=alt.Y("seed:N", title=None),
            color=alt.Color(
                "which:N",
                scale=alt.Scale(
                    domain=["Tuning rule", "Best on truth"], range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE]
                ),
                legend=alt.Legend(orient="top", title=None),
            ),
            yOffset=alt.YOffset("which:N"),
        )
        .properties(
            width=CONTENT_WIDTH_PX - 90,
            height=150,
            title=_title(
                text="The tuning rule is within one grid step of the best penalty on 5 of 5 seeds",
                subtitle="Each dot is a penalty on the grid; 0 is drawn at 30. Best on truth: "
                "the penalty that minimises the error of the mean forecast against the "
                "noise-free injected truth, by the same rule.",
            ),
        )
    )


def main() -> None:
    """Write `report.txt` and the three figures."""
    june, winter = _load(pattern="gate_seed*.json"), _load(pattern="winter_seed*.json")
    rows = _rows(runs=june, period="4 June weeks", june=True)
    rows += _rows(runs=winter, period="4 January weeks", june=False)
    data = pl.DataFrame(rows)
    lines = ["== Line (b) per seed: slope on the heating contribution and heating RMSE"]
    lines.append(str(data.pivot(on="estimator", index=["period", "seed"], values="rmse")))
    lines.append(str(data.pivot(on="estimator", index=["period", "seed"], values="slope")))
    lines.append("== Decision rule")
    lines.append(json.dumps(gate.summarise(), indent=2))
    lines.append("== Lines (a), (c), (d), (e) per seed")
    for run in june:
        c = run["checks"]
        d = c["d_tuning_within_one_step"]
        lines.append(
            f"seed {run['seed']}: (a) profile rms {c['a_fleet_mean']['profile_rms_points']:.3f}, "
            f"max band error {max(c['a_fleet_mean']['band_relative_error']):.3f}; "
            f"(c) slope {c['c_full_history_keeps_own']['slope_on_heating_contribution']:.3f}; "
            f"(d) lambda* {d['lambda_star']:g}, truth-best {d['lambda_truth_best']:g}, "
            f"mae-tuned {d['lambda_star_mae']:g}; "
            f"(e) within 25%: {c['e_capacity_from_a_short_window']['within_25_percent_of_8']}"
        )
    lines.append("== Fair oracle capacity ratios (fitted over true), June, per seed")
    for run in june:
        ratios = run["checks"]["b_shrinkage_on_4_weeks"]["pooled"]["oracle_capacity_ratios"]
        lines.append(f"seed {run['seed']}: " + " ".join(f"{r:.2f}" for r in ratios))
    lines.append("== Winter (post hoc) line (b)")
    for run in winter:
        p = run["pooled"]
        lines.append(
            f"seed {run['seed']}: pooled slope {p['slope_on_heating_contribution']:.2f} rmse "
            f"{p['rmse']:.2f}; oracle slope {p['oracle_slope_on_heating_contribution']:.2f} rmse "
            f"{p['oracle_rmse']:.2f}; passes line (b): complete pooling "
            f"{run['controls']['complete_pooling']['passes_line_b']}, no pooling "
            f"{run['controls']['no_pooling']['passes_line_b']}"
        )
    (DOCS_DIR / "report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))
    _save(
        dot_figure(
            data=data,
            column="rmse",
            x_title="Heating-slope error (root mean square, coefficient units; smaller is better)",
            title="The pooled fit has a smaller heating error than the fair oracle on all 5 seeds",
            subtitle="Eight synthetic substations take turns holding only 4 weeks. Complete "
            "pooling is as accurate in June, when the weather carries little heating signal.",
        ),
        name="figure_01_heating_error",
    )
    _save(
        dot_figure(
            data=data,
            column="slope",
            x_title="Slope of fitted on true heating deviation (1 keeps all, 0 keeps none)",
            title="In January the pooled fit shrinks heating deviations harder than the oracle",
            subtitle="Slope of the fitted heating contribution on the true one. In June the "
            "oracle's slope is already near 0 on 2 seeds, so complete pooling passes line (b).",
        ),
        name="figure_02_shrinkage_slope",
    )
    _save(tuning_figure(runs=june), name="figure_03_tuning")


if __name__ == "__main__":
    main()
