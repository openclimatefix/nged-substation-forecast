# ruff: noqa: E501, PERF401
"""Draw experiment 163's figures as SVG from `event_rows.parquet`, `joined.parquet`, and `report.json`.

Afterwards optimise each with `npx svgo@4 --multipass --precision=1 --final-newline`.
The series are Primary substations, not NGED generators.

    OMP_NUM_THREADS=1 uv run --no-sync python ml-experiments/00163-*/e00163_charts.py
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl
from e00163_common import WORK_DIR
from e00163_report import boot

HERE: Final[Path] = Path(__file__).resolve().parent
OUT_DIR: Final[Path] = (
    HERE.parents[1]
    / "docs"
    / "ml-experiments"
    / "00163-switching-real-event-comparison-gam-stack-against-bg2"
)
WIDTH: Final[int] = 520
LABELS: Final[dict[str, str]] = {
    "K0": "Stack, no mask",
    "GAM_ONLY": "Stack, weather model only",
    "MASK_NEW": "Stack, mask from pair alarm",
    "MASK_UNION": "Stack, mask from either alarm",
    "MASK_RONC": "Stack, mask from RO-NC alarm",
    "ORACLE": "Stack, mask from logged interval (ceiling)",
    "BG2": "Best guess 2 (XGBoost)",
    "ARM_C": "Experiment 155 arm C (stack with recipe R, no mask)",
    "MANUAL": "Manual heuristic",
    "CLIM": "Climatology",
}
COLOURS: Final[dict[str, str]] = {
    LABELS["K0"]: ocf.BRAND_ORANGE,
    LABELS["GAM_ONLY"]: ocf.DATA_PURPLE,
    LABELS["MASK_NEW"]: ocf.DATA_GREEN,
    LABELS["MASK_UNION"]: ocf.DATA_BLUE,
    LABELS["MASK_RONC"]: ocf.DATA_SKY,
    LABELS["ORACLE"]: "#444444",
    LABELS["BG2"]: "#8A8A8A",
    LABELS["ARM_C"]: "#F4A582",
    LABELS["MANUAL"]: "#BBBBBB",
    LABELS["CLIM"]: "#DDDDDD",
}


def _title(*, text: str, line: str) -> alt.TitleParams:
    return alt.TitleParams(
        text=textwrap.wrap(text, 70),
        subtitle=textwrap.wrap(line, 85),
        anchor="start",
        fontSize=14,
        subtitleFontSize=11,
    )


def _colour() -> alt.Color:
    return alt.Color(
        "label:N", legend=None, scale=alt.Scale(domain=list(COLOURS), range=list(COLOURS.values()))
    )


def _scale(arms: tuple[str, ...]) -> alt.Scale:
    names = [LABELS[a] for a in arms]
    return alt.Scale(domain=names, range=[COLOURS[n] for n in names])


def _save(*, chart: alt.TopLevelMixin, name: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    chart.save(OUT_DIR / f"{name}.svg")


def _dots(*, data: pl.DataFrame, title: str, rule: float | None, height: int) -> alt.TopLevelMixin:
    enc_y = alt.Y("label:N", title=None, sort=None, axis=alt.Axis(labelLimit=330))
    layers = [
        alt.Chart(data)
        .mark_rule(strokeWidth=2)
        .encode(x=alt.X("low:Q", title=title), x2="high:Q", y=enc_y, color=_colour()),  # ty: ignore[unresolved-attribute]
        alt.Chart(data)
        .mark_point(filled=True, size=70)
        .encode(x="mean:Q", y=enc_y, color=_colour()),  # ty: ignore[unresolved-attribute]
    ]
    if rule is not None:
        layers.append(
            alt.Chart(pl.DataFrame({"v": [rule]}))
            .mark_rule(strokeDash=[4, 3], color="#888")
            .encode(x="v:Q")  # ty: ignore[unresolved-attribute]
        )
    return alt.layer(*layers).properties(width=WIDTH, height=height)


def main() -> None:
    """Draw the figures."""
    ev = pl.read_parquet(WORK_DIR / "event_rows.parquet")
    report = json.loads((WORK_DIR / "report.json").read_text())
    # figure 1: AFTER, shared rows, mean score by arm with a 95% interval over logged transfers
    after = ev.filter((pl.col("set") == "AFTER") & pl.col("shared"))
    arms = [
        "K0",
        "GAM_ONLY",
        "MASK_NEW",
        "MASK_UNION",
        "MASK_RONC",
        "ORACLE",
        "BG2",
        "ARM_C",
        "MANUAL",
        "CLIM",
    ]
    rows = []
    for arm in arms:
        sub = after.filter(pl.col(arm).is_not_null())
        m, lo, hi = boot(num=sub[arm].to_numpy(), den=None, groups=sub["card"].to_numpy())
        rows.append({"label": LABELS[arm], "mean": m, "low": lo, "high": hi})
    cards = after["card"].n_unique()
    _save(
        chart=_dots(
            data=pl.DataFrame(rows),
            title="Mean quantile score, points of effective capacity (smaller is better)",
            rule=None,
            height=250,
        ).properties(
            title=_title(
                text="After a logged transfer ends, masking scores about half as high as the unmasked stack, but 7 transfers leave wide intervals",
                line=f"Dot: mean score of every row in the 72 hours after the logged end, forecasts issued after the transfer began. Line: 95% interval from resampling the {cards} logged transfers. Winter dominates the rows. Six Primary substations, July 2025 to June 2026.",
            )
        ),
        name="figure_1_headline",
    )
    # figure 2: per event, AFTER
    per = (
        after.group_by("card", "site")
        .agg(
            pl.len().alias("rows"),
            *[pl.col(a).mean() for a in ("K0", "MASK_NEW", "GAM_ONLY", "BG2", "ORACLE")],
            pl.col("winter").first(),
        )
        .sort("K0", descending=True)
        .with_row_index("n", offset=1)
    )
    long = []
    for r in per.iter_rows(named=True):
        name = (
            f"Transfer {r['n']} ({'winter' if r['winter'] else 'other months'}, {r['rows']} rows)"
        )
        for arm in ("K0", "MASK_NEW", "GAM_ONLY", "BG2", "ORACLE"):
            long.append({"event": name, "label": LABELS[arm], "score": r[arm]})
    long_frame = pl.DataFrame(long)
    chart2 = (
        alt.Chart(long_frame)  # ty: ignore[unresolved-attribute]
        .mark_point(filled=True, size=70)
        .encode(
            x=alt.X(
                "score:Q",
                title="Mean quantile score, points of effective capacity (smaller is better)",
            ),
            y=alt.Y("event:N", title=None, sort=None, axis=alt.Axis(labelLimit=330)),
            color=alt.Color(
                "label:N",
                legend=alt.Legend(orient="top", columns=2, title=None, labelLimit=400),
                scale=_scale(("K0", "MASK_NEW", "GAM_ONLY", "BG2", "ORACLE")),
            ),
        )
        .properties(
            width=WIDTH,
            height=26 * per.height + 30,
            title=_title(
                text="The gain comes from a few long transfers; short transfers change little",
                line="Each row is one logged transfer at one substation, scored on the 72 hours after it ends. Transfers are numbered from the largest unmasked score; substations are not named.",
            ),
        )
    )
    _save(chart=chart2, name="figure_2_per_transfer")
    # figure 3: three row sets
    long3 = []
    for name in ("INSIDE", "AFTER"):
        sub = ev.filter((pl.col("set") == name) & pl.col("shared"))
        for arm in ("K0", "GAM_ONLY", "MASK_NEW", "BG2"):
            long3.append(
                {
                    "set": f"{name} ({sub['card'].n_unique()} transfers, {sub.height} rows)",
                    "label": LABELS[arm],
                    "score": float(sub[arm].mean()),  # ty: ignore[invalid-argument-type]
                }
            )
    chart3 = (
        alt.Chart(pl.DataFrame(long3))  # ty: ignore[unresolved-attribute]
        .mark_point(filled=True, size=80)
        .encode(
            x=alt.X(
                "score:Q",
                title="Mean quantile score against observed power, points (smaller is better)",
            ),
            y=alt.Y("set:N", title=None, sort=None, axis=alt.Axis(labelLimit=330)),
            color=alt.Color(
                "label:N",
                legend=alt.Legend(orient="top", columns=2, title=None, labelLimit=400),
                scale=_scale(("K0", "GAM_ONLY", "MASK_NEW", "BG2")),
            ),
        )
        .properties(
            width=WIDTH,
            height=100,
            title=_title(
                text="While a transfer runs, the observed power contains the transfer, so a mask scores worse",
                line="INSIDE: targets from the first alarm until the logged end. AFTER: the 72 hours after the end, when observed power is the normal running arrangement. Only AFTER is a fair test of a mask.",
            ),
        )
    )
    _save(chart=chart3, name="figure_3_row_sets")
    # figure 4: no harm
    rows4 = []
    for season in ("winter", "other"):
        for arm, label in (
            ("MASK_NEW", "MASK_NEW minus K0"),
            ("MASK_UNION", "MASK_UNION minus K0"),
            ("MASK_RONC", "MASK_RONC minus K0"),
            ("GAM_ONLY", "GAM_ONLY minus K0"),
        ):
            m, lo, hi = report[f"quiet|{season}|{label}"]
            rows4.append({"label": LABELS[arm], "season": season, "mean": m, "low": lo, "high": hi})
        for arm, label in (("K0", "K0 minus BG2"), ("MASK_NEW", "MASK_NEW minus BG2")):
            m, lo, hi = report[f"quiet|{season}|{label}"]
            rows4.append(
                {
                    "label": LABELS[arm] + " minus BG2",
                    "season": season,
                    "mean": m,
                    "low": lo,
                    "high": hi,
                }
            )
    d4 = pl.DataFrame(rows4)
    d4 = d4.with_columns(row=pl.col("label") + " (" + pl.col("season") + ")")
    chart4 = alt.layer(
        alt.Chart(d4)  # ty: ignore[unresolved-attribute]
        .mark_rule(strokeWidth=2)
        .encode(
            x=alt.X(
                "low:Q",
                title="Score minus the unmasked stack (or minus best guess 2 where stated), points; negative is better",
            ),
            x2="high:Q",
            y=alt.Y("row:N", title=None, sort=None, axis=alt.Axis(labelLimit=330)),
            color=alt.Color(
                "season:N",
                legend=alt.Legend(orient="top", title=None),
                scale=alt.Scale(
                    domain=["winter", "other"], range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE]
                ),
            ),
        ),
        alt.Chart(d4)  # ty: ignore[unresolved-attribute]
        .mark_point(filled=True, size=70)
        .encode(
            x="mean:Q",
            y=alt.Y("row:N", sort=None),
            color=alt.Color(
                "season:N",
                legend=None,
                scale=alt.Scale(
                    domain=["winter", "other"], range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE]
                ),
            ),
        ),
        alt.Chart(pl.DataFrame({"v": [0.0]}))
        .mark_rule(strokeDash=[4, 3], color="#888")
        .encode(x="v:Q"),  # ty: ignore[unresolved-attribute]
    ).properties(
        width=WIDTH,
        height=380,
        title=_title(
            text="Outside transfers the pair-alarm mask costs about 0.03 points; masks from RO-NC cost more",
            line="Real rows outside logged transfers and their 72 hours after, July 2025 to June 2026, six Primary substations. Dot: mean difference. Line: 95% interval from resampling calendar months. Dashed line: no difference.",
        ),
    )
    _save(chart=chart4, name="figure_4_no_harm")


if __name__ == "__main__":
    main()
