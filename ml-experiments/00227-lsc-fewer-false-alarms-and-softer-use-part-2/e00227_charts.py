# ruff: noqa: E501
"""Draw the figures of experiment 227 as SVG beside the write-up.

OMP_NUM_THREADS=1 uv run --no-sync python e00227_charts.py
"""

from __future__ import annotations

from pathlib import Path
from typing import Final

import altair as alt
import plotting.ocf_theme as ocf
import polars as pl
from e00227_common import WORK_DIR
from studies.charts import CONTENT_WIDTH_PX, wrapped

OUT: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00227-lsc-fewer-false-alarms-and-softer-use-part-2"
)
W: Final[int] = CONTENT_WIDTH_PX - 150
FAMILY: Final[dict[str, str]] = {
    "LR": ocf.DATA_BLUE,
    "UR": ocf.DATA_SKY,
    "RO": ocf.BRAND_ORANGE,
    "PG": ocf.DATA_PURPLE,
}
FAMILY_NAME: Final[dict[str, str]] = {
    "LR": "learned score (LR)",
    "UR": "learned or pair alarm (UR)",
    "RO": "RO-NC (RO)",
    "PG": "pair alarm gate (PG)",
}


def save(chart: alt.TopLevelMixin, name: str) -> None:
    """Write a chart as SVG."""
    chart.configure_view(stroke=None).save(OUT / f"{name}.svg")


def fig_precision_recall() -> None:
    """Matched load shifts against alarms per matched load shift, by source."""
    t = (
        pl.read_parquet(WORK_DIR / "part_a.parquet")
        .filter(
            ~pl.col("source").str.starts_with("LR148")
            & pl.col("matched").is_not_null()
            & (pl.col("matched") > 0)
        )
        .with_columns(
            family=pl.col("source").str.split("_").list.get(0),
            inside=pl.col("inside"),
            label=pl.col("source"),
        )
        .with_columns(
            family_name=pl.col("family").replace(FAMILY_NAME),
            harm_text=pl.when(pl.col("inside"))
            .then(pl.lit("inside ±0.05"))
            .otherwise(pl.lit("outside ±0.05")),
        )
    )
    colour = alt.Color(
        "family_name:N",
        scale=alt.Scale(domain=list(FAMILY_NAME.values()), range=[FAMILY[k] for k in FAMILY_NAME]),
        legend=alt.Legend(title=None, orient="top", columns=2),
    )
    base = alt.Chart(t.to_pandas() if False else t).encode(
        x=alt.X(
            "alarms_per_match:Q",
            scale=alt.Scale(type="log", domain=[2, 120]),
            axis=alt.Axis(
                values=[2, 5, 10, 20, 50, 100],
                title="Alarms per matched load shift (log scale; fewer is better)",
            ),
        ),
        y=alt.Y(
            "matched:Q",
            scale=alt.Scale(domain=[0, 28]),
            axis=alt.Axis(
                values=[0, 5, 10, 15, 20, 25], title="Load shifts matched, of 35 (more is better)"
            ),
        ),
        color=colour,
    )
    lines = base.mark_line(strokeWidth=1.5, opacity=0.6).encode(detail="family:N")  # ty: ignore[unresolved-attribute]
    pts = base.mark_point(size=90, strokeWidth=2).encode(  # ty: ignore[unresolved-attribute]
        fill=alt.Fill(
            "harm_text:N",
            scale=alt.Scale(domain=["inside ±0.05", "outside ±0.05"], range=["#FF4901", "white"]),
            legend=None,
        ),
        shape=alt.Shape(
            "harm_text:N",
            legend=alt.Legend(
                title="Ordinary-day harm 95% interval (filled: inside ±0.05 points)", orient="top"
            ),
        ),
        opacity=alt.value(1),
    )
    pts = base.mark_point(size=90, filled=True).encode(  # ty: ignore[unresolved-attribute]
        shape=alt.Shape(
            "harm_text:N",
            scale=alt.Scale(domain=["inside ±0.05", "outside ±0.05"], range=["circle", "cross"]),
            legend=alt.Legend(title="Ordinary-day harm interval", orient="top"),
        )
    )
    rule_x = (
        alt.Chart(pl.DataFrame({"x": [5.0]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(strokeDash=[4, 4], color="grey")
        .encode(x="x:Q")
    )
    rule_y = (
        alt.Chart(pl.DataFrame({"y": [12.0]}))  # ty: ignore[unresolved-attribute]
        .mark_rule(strokeDash=[4, 4], color="grey")
        .encode(y="y:Q")
    )
    title = alt.TitleParams(
        text=wrapped(
            text="Several learned-score settings match 20 or more of 35 load shifts with 5 or fewer alarms each",
            width=70,
        ),
        subtitle=wrapped(
            text="Each point is one alarm budget (or balance gate). Circles: the 95% interval of ordinary-day harm (resampled by whole months) lies inside ±0.05 points of score. Crosses: it does not. Dashed lines: the pre-registered 5 alarms and 12 matches. LR_b20 is the carried setting.",
            width=95,
        ),
        anchor="start",
    )
    save(
        (lines + pts + rule_x + rule_y).properties(width=W, height=300, title=title),
        "precision_recall",
    )


def interval_chart(
    *,
    t: pl.DataFrame,
    x: str,
    lo: str,
    hi: str,
    xtitle: str,
    title: str,
    subtitle: str,
    zero: float | None,
    limit: bool,
    domain: list[float] | None = None,
) -> alt.VConcatChart:
    """A dot-and-interval chart with one row per use and one panel per source."""
    uses = ["MASK", "SOFT_2", "SOFT_4", "SOFT_16", "HALF", "SUB", "PAIRSUM"]
    sources = t["source"].unique(maintain_order=True).to_list()
    panels = []
    for source in sources:
        d = t.filter((pl.col("source") == source) & pl.col("use").is_in(uses))
        scale = alt.Scale(domain=domain) if domain else alt.Scale()
        base = alt.Chart(d).encode(
            y=alt.Y("use:N", sort=uses, title=None),
            x=alt.X(f"{x}:Q", title=xtitle if source == sources[-1] else None, scale=scale),
        )
        layers = [
            base.mark_rule(strokeWidth=2, color=ocf.DATA_BLUE).encode(x=f"{lo}:Q", x2=f"{hi}:Q"),  # ty: ignore[unresolved-attribute]
            base.mark_point(filled=True, size=70, color=ocf.DATA_BLUE),
        ]
        if zero is not None:
            layers.insert(
                0,
                alt.Chart(pl.DataFrame({"z": [zero]})).mark_rule(color="grey").encode(x="z:Q"),  # ty: ignore[unresolved-attribute]
            )
        if limit:
            layers.insert(
                0,
                alt.Chart(pl.DataFrame({"z": [-0.05, 0.05]}))  # ty: ignore[unresolved-attribute]
                .mark_rule(strokeDash=[4, 4], color=ocf.BRAND_ORANGE)
                .encode(x="z:Q"),
            )
        panels.append(
            alt.layer(*layers).properties(
                width=W, height=130, title=alt.TitleParams(source, anchor="start", fontSize=12)
            )
        )
    return alt.vconcat(*panels).properties(
        title=alt.TitleParams(
            text=wrapped(text=title, width=70),
            subtitle=wrapped(text=subtitle, width=95),
            anchor="start",
        )
    )


def fig_real() -> None:
    """Real-record harm and AFTER share by use."""
    t = pl.read_parquet(WORK_DIR / "part_b.parquet").filter(pl.col("use") != "NONE")
    harm = interval_chart(
        t=t,
        x="harm",
        lo="harm_lo",
        hi="harm_hi",
        xtitle="Ordinary-day score minus K0 (points of capacity; below zero is better)",
        title="Averaging the masked and unmasked forecast removes most of the mask's ordinary-day harm",
        subtitle="Real record, 1,579,711 ordinary rows at six substations. Dot: estimate. Line: 95% interval from resampling whole months. Dashed orange: ±0.05 limit. Soft down-weighting at factors 2, 4, and 16 leaves the forecast unchanged.",
        zero=0.0,
        limit=True,
        domain=[-0.1, 1.1],
    )
    save(harm, "real_harm_by_use")
    share = interval_chart(
        t=t,
        x="share",
        lo="share_lo",
        hi="share_hi",
        xtitle="Share of the logged-transfer AFTER gain recovered (1 = all of it)",
        title="Masking and the pair-sum correction recover most of the gain in the 72 hours after a load shift",
        subtitle="Real record, 5,993 rows in the 72 hours after the 11 logged load shifts that fall in the scored months. Dot: estimate. Line: 95% interval from resampling whole windows. The intervals are wide because the windows are few.",
        zero=0.0,
        limit=False,
        domain=[-1.3, 1.5],
    )
    save(share, "real_share_by_use")


def fig_injected() -> None:
    """Injected-world gain by phase and harm per false alarm."""
    t = pl.read_parquet(WORK_DIR / "part_b_injected.parquet")
    g = t.filter(
        (pl.col("kind") == "gain")
        & (pl.col("source") == "DET")
        & pl.col("stratum").str.starts_with("headline")
    )
    designs = ["MASK", "SOFT_2", "SOFT_4", "SOFT_16", "HALF", "SUB"]
    panels = []
    for phase in ("BEFORE_ALARM", "DURING", "AFTER"):
        d = g.filter(pl.col("phase") == phase)
        ref = t.filter(
            (pl.col("kind") == "reference_gain")
            & (pl.col("phase") == phase)
            & pl.col("stratum").str.starts_with("headline")
        )["est"][0]
        base = alt.Chart(d).encode(
            y=alt.Y("design:N", sort=designs, title=None),
            x=alt.X(
                "est:Q",
                title="Gain over K0 (points of capacity; more is better)"
                if phase == "AFTER"
                else None,
                scale=alt.Scale(domain=[-0.8, 2.4]),
            ),
        )
        layers = [
            alt.Chart(pl.DataFrame({"z": [0.0]})).mark_rule(color="grey").encode(x="z:Q"),  # ty: ignore[unresolved-attribute]
            alt.Chart(pl.DataFrame({"z": [ref]}))  # ty: ignore[unresolved-attribute]
            .mark_rule(strokeDash=[4, 4], color=ocf.BRAND_ORANGE)
            .encode(x="z:Q"),
            base.mark_rule(strokeWidth=2, color=ocf.DATA_BLUE).encode(x="lo:Q", x2="hi:Q"),  # ty: ignore[unresolved-attribute]
            base.mark_point(filled=True, size=70, color=ocf.DATA_BLUE),
        ]
        panels.append(
            alt.layer(*layers).properties(  # ty: ignore[invalid-argument-type]
                width=W,
                height=120,
                title=alt.TitleParams(phase.replace("_", " ").lower(), anchor="start", fontSize=12),
            )
        )
    save(
        alt.vconcat(*panels).properties(
            title=alt.TitleParams(
                text=wrapped(
                    text="On injected load shifts, the average of masked and unmasked forecasts keeps 92% of the mask's gain after a load shift",
                    width=70,
                ),
                subtitle=wrapped(
                    text="Injected load shifts of 25% of load or more lasting over 1 day, found by RO-NC, 44 worlds of experiment 141. Dot: estimate. Line: 95% interval from resampling whole transfers. Dashed orange: the gain from perfect knowledge of the mask. Grey: no gain.",
                    width=95,
                ),
                anchor="start",
            )
        ),
        "injected_gain_by_phase",
    )
    h = t.filter(pl.col("kind") == "harm_per_false_alarm")
    base = alt.Chart(h).encode(
        y=alt.Y("design:N", sort=designs, title=None),
        x=alt.X(
            "est:Q",
            title="Score points summed over ordinary rows, per false alarm (lower is better)",
        ),
    )
    save(
        (
            alt.Chart(pl.DataFrame({"z": [0.0]})).mark_rule(color="grey").encode(x="z:Q")  # ty: ignore[unresolved-attribute]
            + base.mark_rule(strokeWidth=2, color=ocf.DATA_BLUE).encode(x="lo:Q", x2="hi:Q")  # ty: ignore[unresolved-attribute]
            + base.mark_point(filled=True, size=70, color=ocf.DATA_BLUE)
        ).properties(
            width=W,
            height=160,
            title=alt.TitleParams(
                text=wrapped(
                    text="A false alarm costs the mask about 1,000 score points; the average of masked and unmasked costs little",
                    width=70,
                ),
                subtitle=wrapped(
                    text="Ordinary rows within 6 days of an RO-NC false alarm in the injected worlds. Dot: estimate. Line: 95% interval from resampling worlds and substations.",
                    width=95,
                ),
                anchor="start",
            ),
        ),
        "injected_harm_per_false_alarm",
    )


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    fig_precision_recall()
    fig_real()
    fig_injected()
