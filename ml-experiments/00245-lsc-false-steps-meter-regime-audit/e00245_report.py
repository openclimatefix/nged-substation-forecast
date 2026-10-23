"""Print the numbers of experiment 245 and draw its figures as SVG into the docs folder.

Only Primaries (D-labels) appear. Afterwards optimise each SVG with
`npx svgo@4 --multipass --precision=1 --final-newline`.

    uv run --no-sync python ml-experiments/00245-lsc-false-steps-meter-regime-audit/e00245_report.py
"""

import textwrap
from pathlib import Path
from typing import Final

import altair as alt
import e00245_audit as audit
import numpy as np
import plotting.ocf_theme as ocf
import polars as pl

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00245-lsc-false-steps-meter-regime-audit"
)
PNG_DIR: Final[Path] = Path("/tmp/e00245_png")
WIDTH: Final[int] = 640
WRAP: Final[int] = 100
FAILING: Final[tuple[str, ...]] = ("D10", "D11")
CATEGORIES: Final[dict[str, str]] = {
    "balanced": "Balanced opposite neighbour step",
    "alarm_RO-NC": "RO-NC alarm",
    "alarm_Pair alarm": "Pair alarm",
    "alarm_Learned score": "Learned score alarm",
    "any_alarm": "Any alarm",
    "explained": "Any explanation",
}
N_BOOT: Final[int] = 5000


def poisson_binomial_tail(*, probs: np.ndarray, k: int) -> float:
    """Return P(X >= k) for a sum of independent Bernoulli variables with the given rates."""
    dist = np.array([1.0])
    for p in probs:
        dist = np.convolve(dist, [1 - p, p])
    return float(dist[k:].sum())


def shares(*, events: pl.DataFrame, controls: pl.DataFrame, column: str) -> dict[str, float]:
    """Return the observed share, the matched base rate, and the ratio for one explanation."""
    rate = controls.group_by("site", "year", "month").agg(
        rate=pl.col(column).cast(pl.Float64).mean()
    )
    joined = events.join(rate, on=["site", "year", "month"], how="left")
    obs = joined[column].cast(pl.Float64).to_numpy()
    base = joined["rate"].to_numpy()
    ok = np.isfinite(base)
    k = int(obs[ok].sum())
    rng = np.random.default_rng(245)
    idx = rng.integers(0, ok.sum(), size=(N_BOOT, ok.sum()))
    o, b = obs[ok][idx].mean(axis=1), base[ok][idx].mean(axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        ratio = o / b
    return {
        "n": int(ok.sum()),
        "k": k,
        "observed": float(obs[ok].mean()),
        "base": float(base[ok].mean()),
        "ratio": float(obs[ok].mean() / base[ok].mean()) if base[ok].mean() else float("nan"),
        "ratio_low": float(np.nanpercentile(ratio, 2.5)),
        "ratio_high": float(np.nanpercentile(ratio, 97.5)),
        "p_tail": poisson_binomial_tail(probs=base[ok], k=k),
    }


def print_numbers(*, events: pl.DataFrame, controls: pl.DataFrame) -> pl.DataFrame:
    """Print every number the page quotes."""
    pl.Config.set_tbl_rows(100)
    pl.Config.set_tbl_cols(40)
    pl.Config.set_tbl_width_chars(250)
    print(
        events.select(
            "site",
            "year",
            "month",
            "kind",
            "n_row_sets",
            "change_mw",
            "own_z",
            "own_mw",
            "visible",
            "best_opp_z",
            "balanced",
            "alarm_RO-NC",
            "alarm_Pair alarm",
            "alarm_Learned score",
            "any_wide",
            "flagged_or_zero",
            "bank_holiday",
            "explained",
            "n_controls",
        )
    )
    local = events.filter(pl.col("kind") == "summer").with_columns(
        balanced=pl.col("balanced").fill_null(False)
    )
    cols = {c: c for c in CATEGORIES}
    out = []
    for group, sites in (
        ("D10+D11", FAILING),
        ("D10", ("D10",)),
        ("D11", ("D11",)),
        ("D14+D16", ("D14", "D16")),
        ("all four", ("D10", "D11", "D14", "D16")),
    ):
        ev = local.filter(pl.col("site").is_in(sites))
        ct = controls.filter(pl.col("site").is_in(sites)).with_columns(
            explained=pl.col("explained")
        )
        for col in cols:
            if col not in ev.columns:
                continue
            out.append(
                {"group": group, "explanation": col, **shares(events=ev, controls=ct, column=col)}
            )
    table = pl.DataFrame(out)
    print(table)
    wide = local.filter(pl.col("site").is_in(FAILING))
    print("wide-window any alarm:", int(wide["any_wide"].sum()), "of", wide.height)
    print("controls per event:", local.select("site", "year", "month", "n_controls").to_dicts())
    print(
        "controls: share visible own step",
        controls["visible"].mean(),
        "bank holiday",
        controls["bank_holiday"].mean(),
        "flagged/zero",
        controls["flagged_or_zero"].mean(),
        "wide",
        controls["any_wide"].mean(),
    )
    print(
        "events: bank holiday",
        local["bank_holiday"].sum(),
        "flagged or zero",
        local["flagged_or_zero"].sum(),
    )
    # sign agreement: false step sign vs own residual step sign
    print(
        local.select("site", "year", "month", "change_mw", "own_mw").with_columns(
            same=(pl.col("change_mw").sign() == pl.col("own_mw").sign())
        )
    )
    # gate recount
    rows = []
    for site in ("D5", "D10", "D11", "D14", "D16"):
        ev = events.filter(pl.col("site") == site)
        total = int(ev["n_row_sets"].sum())
        explained = int(ev.filter(pl.col("explained").fill_null(False))["n_row_sets"].sum())
        denom = 5 * len(audit.PLACEBO_YEARS)
        winter = int(ev.filter(pl.col("kind") == "winter")["n_row_sets"].sum())
        rows.append(
            {
                "site": site,
                "before": total / denom,
                "before_without_winter": (total - winter) / denom,
                "after": (total - explained) / denom,
                "after_without_winter": (total - explained - winter) / denom,
            }
        )
    gate = pl.DataFrame(rows).with_columns(
        pass_before=pl.col("before") < 1.0, pass_after=pl.col("after") < 1.0
    )
    print(gate)
    return table


def save_chart(chart: alt.TopLevelMixin, *, name: str) -> None:
    """Write the chart as an SVG into the docs folder and as a PNG for checking."""
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    chart.save(DOCS_DIR / f"{name}.svg")
    chart.save(PNG_DIR / f"{name}.png", scale_factor=1.5)


def chart_title(*, text: str, subtitle: str) -> alt.TitleParams:
    """Return a left-aligned title with the subtitle wrapped to the plot width."""
    return alt.TitleParams(
        text=text, subtitle=textwrap.wrap(subtitle, WRAP), anchor="start", subtitleColor="#444444"
    )


def figure_shares(*, table: pl.DataFrame) -> alt.TopLevelMixin:
    """Return the observed share against the matched base rate for each explanation."""
    data = table.filter(
        pl.col("group") == "D10+D11", pl.col("explanation").is_in(list(CATEGORIES))
    ).with_columns(label=pl.col("explanation").replace(CATEGORIES))
    long = pl.concat(
        [
            data.select(
                "label",
                pl.col("observed").alias("share"),
                pl.lit("False steps at D10 and D11").alias("series"),
            ),
            data.select(
                "label",
                pl.col("base").alias("share"),
                pl.lit("Matched random windows").alias("series"),
            ),
        ]
    )
    order = list(CATEGORIES.values())
    chart = (
        alt.Chart(long)
        .mark_point(filled=True, size=90)
        .encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y("label:N", sort=order, title=None, axis=alt.Axis(labelLimit=300)),
            x=alt.X(
                "share:Q",
                title="Share of windows with the explanation",
                scale=alt.Scale(domain=[0, 1]),
            ),
            color=alt.Color(
                "series:N",
                scale=alt.Scale(
                    domain=["False steps at D10 and D11", "Matched random windows"],
                    range=[ocf.BRAND_ORANGE, ocf.DATA_BLUE],
                ),
                legend=alt.Legend(title=None, orient="top", columns=1),
            ),
        )
        .properties(width=WIDTH - 260, height=200)
    )
    return chart.properties(
        title=chart_title(
            text="The false steps are no more often explained than random windows",
            subtitle=(
                "Share of 30-day windows with each explanation. Orange: the 7 false steps that"
                " localise to a month at D10 and D11. Blue: mean over matched windows at the same"
                " Primary, same calendar month, with a residual step within 30% of the same size."
            ),
        )
    )


def figure_example(
    *, panel: audit.Panel, site: str, year: int, month: int, name_title: str, subtitle: str
) -> alt.TopLevelMixin:
    """Return the Primary's residual step and its neighbours' steps around one false step."""
    series = panel.site_series[site]
    centre = audit.stamp_of(year=year, month=month)
    lo, hi = centre - audit.WINDOW_HALF, centre + audit.WINDOW_HALF
    tags = panel.window_tags(series=series, centre=centre)
    sites_of = {s: d for d, s in panel.site_series.items()}
    pieces = []
    for s in (series, *(n for n in audit.ADJACENCY[series] if n in panel.shift)):
        z = panel.shift[s][lo:hi] / panel.scale[s]
        step = 6
        pieces.append(
            pl.DataFrame(
                {
                    "day": (np.arange(lo, hi) - centre)[::step] / audit.DAY,
                    "z": np.nan_to_num(z[::step]),
                    "who": ("This Primary: " if s == series else "Neighbour: ") + sites_of[s],
                    "own": s == series,
                }
            )
        )
    data = pl.concat(pieces)
    t0 = (tags["t0"] - centre) / audit.DAY
    own = data.filter(pl.col("own"))
    others = data.filter(~pl.col("own"))
    neighbours = (
        alt.Chart(others)
        .mark_line(strokeWidth=1, opacity=0.8)
        .encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "day:Q",
                title="Days from the first day of the month",
                scale=alt.Scale(domain=[-15, 15]),
                axis=alt.Axis(values=[-15, -10, -5, 0, 5, 10, 15]),
            ),
            y=alt.Y("z:Q", title="One-day level shift (noise units)"),
            color=alt.Color(
                "who:N",
                scale=alt.Scale(
                    range=[
                        ocf.DATA_BLUE,
                        ocf.DATA_SKY,
                        ocf.DATA_PURPLE,
                        ocf.DATA_GREEN,
                        ocf.DATA_BLUE_LIGHT,
                    ]
                ),
                legend=alt.Legend(title=None, orient="top", columns=3),
            ),
        )
    )
    line = (
        alt.Chart(own).mark_line(strokeWidth=2.5, color=ocf.BRAND_ORANGE).encode(x="day:Q", y="z:Q")  # ty: ignore[unresolved-attribute]
    )
    rule = (
        alt.Chart(pl.DataFrame({"day": [t0]}))
        .mark_rule(strokeDash=[4, 3], color="#444444")
        .encode(x="day:Q")  # ty: ignore[unresolved-attribute]
    )
    return alt.layer(neighbours, line, rule).properties(
        width=WIDTH - 90,
        height=220,
        title=chart_title(text=name_title, subtitle=subtitle),
    )


def main() -> None:
    """Print the numbers and write the figures."""
    events = pl.read_parquet(audit.OUT_DIR / "events.parquet")
    controls = pl.read_parquet(audit.OUT_DIR / "controls.parquet")
    table = print_numbers(events=events, controls=controls)
    save_chart(figure_shares(table=table), name="figure_1")
    panel = audit.Panel()
    save_chart(
        figure_example(
            panel=panel,
            site="D10",
            year=2022,
            month=8,
            name_title="D10, August 2022: alarms, but no balanced neighbour step",
            subtitle=(
                "Orange: D10's one-day level shift of the arm B residual, in noise units."
                " Other lines: its metered neighbours. Dashed line: D10's largest step in the"
                " window."
            ),
        ),
        name="figure_2",
    )
    save_chart(
        figure_example(
            panel=panel,
            site="D11",
            year=2023,
            month=5,
            name_title="D11, May 2023: no alarm and no balanced neighbour step",
            subtitle="Same layout as the figure above, for a false step with no explanation.",
        ),
        name="figure_3",
    )


if __name__ == "__main__":
    main()
