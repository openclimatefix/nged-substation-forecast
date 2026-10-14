"""Draw experiment 00009's figures and print every number the write-up quotes.

Reads `work/results/*.parquet` and `work/arms/*.parquet` from `e00009_score.py`. Substations appear
by their anonymous labels only. Writes `figure_1.svg` to `figure_3.svg` (and 2x PNGs for checking)
to `docs/ml-experiments/00009-.../`.

    uv run python ml-experiments/00009-cams-trained-stage1a-vs-era5-trained-demand/e00009_charts.py
"""

from pathlib import Path
from typing import Final

import altair as alt
import e00009_common as common
import e00009_score as score
import plotting.ocf_theme as ocf
import polars as pl
from studies.charts import CONTENT_WIDTH_PX

from studies import bootstrap, experiment_store

FIGURES_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / f"{common.EXPERIMENT_ID:05d}-{common.SLUG}"
)
"""This checkout's page folder, which is not the main checkout's `ML_EXPERIMENTS_DOCS_DIR`."""
PLANNED_THRESHOLD: Final[float] = -0.2
ROW_HEIGHT_PX: Final[int] = 26
LABEL_PX: Final[int] = 230


def _dot_and_interval(
    *,
    frame: pl.DataFrame,
    title: list[str],
    x_title: str,
    order: list[str],
    zero_label: str,
    domain: tuple[float, float],
    show_key: bool = True,
) -> alt.LayerChart:
    """Return rows of a dot with its interval against a zero rule, coloured by `kind`."""
    colours = {
        "Planned (97.5% interval)": ocf.BRAND_ORANGE,
        "Exploratory (95% interval)": ocf.DATA_BLUE,
    }
    base = alt.Chart(frame).encode(
        y=alt.Y("label:N", sort=order, title=None, axis=alt.Axis(labelLimit=LABEL_PX)),
        color=alt.Color(
            "kind:N",
            scale=alt.Scale(domain=list(colours), range=list(colours.values())),
            legend=alt.Legend(title=None, orient="top", columns=2) if show_key else None,
        ),
    )
    scale = alt.Scale(domain=list(domain))
    x = alt.X("difference:Q", title=x_title, scale=scale)
    zero = alt.Chart(pl.DataFrame({"x": [0.0], "text": [zero_label]}))
    rules = zero.mark_rule(color="#444444", strokeDash=[4, 3]).encode(x=alt.X("x:Q", scale=scale))
    words = zero.mark_text(align="left", dx=4, dy=-6, baseline="bottom", fontSize=11).encode(
        x=alt.X("x:Q", scale=scale), y=alt.value(0), text="text:N"
    )
    return (
        rules
        + words
        + base.mark_rule(strokeWidth=2).encode(
            x=alt.X("lower:Q", title=x_title, scale=scale), x2="upper:Q"
        )
        + base.mark_circle(size=70, opacity=1.0).encode(x=x)
    ).properties(
        width=CONTENT_WIDTH_PX - LABEL_PX - 20,
        height=ROW_HEIGHT_PX * len(order),
        title=alt.TitleParams(title, anchor="start", fontSize=12),
    )


def figure_1(*, planned: pl.DataFrame, exploratory: pl.DataFrame) -> None:
    """Draw the contrast of the CAMS-trained arm against the ERA5-trained arm by scope."""
    wanted = {
        "C1 by lead band: 0-24 h": "0 to 24 h ahead",
        "C1 by lead band: 1-2 d": "1 to 2 days ahead",
        "C1 by lead band: 3-6 d": "3 to 6 days ahead",
        "C1 by lead band: 7-10 d": "7 to 10 days ahead",
        "C1 by lead band: 11-15 d": "11 to 15 days ahead",
        "C1, daylight valid times": "Daylight valid times, all leads",
        "C1, night valid times": "Night valid times, all leads",
    }
    rows = [
        {
            "label": "C1: all leads, 16 Primary series"
            if row["label"].startswith("C1")
            else "C2: all leads, without series P2",
            "kind": "Planned (97.5% interval)",
            **{k: row[k] for k in ("difference", "lower", "upper")},
        }
        for row in planned.iter_rows(named=True)
    ]
    rows += [
        {
            "label": wanted[row["label"]],
            "kind": "Exploratory (95% interval)",
            **{k: row[k] for k in ("difference", "lower", "upper")},
        }
        for row in exploratory.iter_rows(named=True)
        if row["label"] in wanted
    ]
    frame = pl.DataFrame(rows)
    chart = _dot_and_interval(
        frame=frame,
        title=[
            "The CAMS-trained arm scores 0.035 points better than the ERA5-trained arm,",
            "a sixth of the 0.2-point gain the plan required",
        ],
        x_title=(
            "CAMS-trained minus ERA5-trained quantile score, Primary (points of capacity; "
            "more negative is better)"
        ),
        order=frame["label"].to_list(),
        zero_label="same as ERA5-trained",
        domain=(-0.25, 0.06),
    )
    planned_rule = alt.Chart(
        pl.DataFrame({"x": [PLANNED_THRESHOLD], "text": ["planned gain: 0.2 points"]})
    )
    scale = alt.Scale(domain=[-0.25, 0.06])
    layers = planned_rule.mark_rule(color=ocf.BRAND_ORANGE, strokeDash=[2, 2]).encode(
        x=alt.X("x:Q", scale=scale)
    ) + planned_rule.mark_text(
        align="left", dx=4, dy=-6, baseline="bottom", fontSize=11, color=ocf.BRAND_ORANGE
    ).encode(x=alt.X("x:Q", scale=scale), y=alt.value(0), text="text:N")
    _save(chart=chart + layers, name="figure_1")


def figure_2(*, exploratory: pl.DataFrame) -> None:
    """Draw the per-series contrast of the 16 Primary series."""
    rows = exploratory.filter(pl.col("label").str.ends_with(" only")).with_columns(
        label=pl.col("label").str.replace("C1, ", "").str.replace(" only", ""),
        kind=pl.lit("Exploratory (95% interval)"),
    )
    order = rows.sort("difference")["label"].to_list()
    chart = _dot_and_interval(
        frame=rows,
        title=[
            "The CAMS-trained arm scores better at 12 of the 16 Primary series, and worst at P2,",
            "the series with the largest residual in experiment 7",
        ],
        x_title=(
            "CAMS-trained minus ERA5-trained quantile score (points of capacity; "
            "more negative is better)"
        ),
        order=order,
        zero_label="same as ERA5-trained",
        domain=(-0.6, 0.75),
        show_key=False,
    )
    _save(chart=chart, name="figure_2")


def ens_median_error_contrast(*, band: str, arms: dict[str, pl.DataFrame]) -> dict[str, float]:
    """Return the daylight, Primary median-error contrast of the two arms at ENS, 95% interval."""
    losses = experiment_store.losses_for_bootstrap(
        arm_losses={
            name: score._scope(frame=arms[name], band=band, daylight=True)
            for name in (score.TREATMENT, score.REFERENCE)
        }
    )
    lower, upper = bootstrap.bootstrap_difference_at_level(
        losses=losses,
        treatment=score.TREATMENT,
        reference=score.REFERENCE,
        metric="median_error",
        level=95.0,
        n_resamples=score.DEFAULT_RESAMPLES,
    )
    paired = (
        losses.filter(pl.col("arm") == score.TREATMENT)
        .select("site", "time", "seed", treated=pl.col("median_error"))
        .join(
            losses.filter(pl.col("arm") == score.REFERENCE).select(
                "site", "time", "seed", base=pl.col("median_error")
            ),
            on=["site", "time", "seed"],
        )
    )
    return {
        "difference": paired.select((pl.col("treated") - pl.col("base")).mean()).item(),
        "lower": lower,
        "upper": upper,
    }


def figure_3(*, observed: pl.DataFrame, arms: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Draw the daylight median-error gain at the observed weather against the gain at ENS."""
    primary = observed.filter(pl.col("group") == "Primary").row(0, named=True)
    rows = [
        {
            "label": "Observed weather (each arm reads its own input)",
            "difference": primary["difference"],
            "lower": primary["lower_95"],
            "upper": primary["upper_95"],
            "kind": "Exploratory (95% interval)",
        }
    ]
    for band, label in (
        ("0-24 h", "ENS forecasts, 0 to 24 h ahead"),
        ("all leads", "ENS forecasts, all leads"),
    ):
        rows.append(
            {
                "label": label,
                **ens_median_error_contrast(band=band, arms=arms),
                "kind": "Exploratory (95% interval)",
            }
        )
    frame = pl.DataFrame(rows)
    chart = _dot_and_interval(
        frame=frame,
        title=[
            "Training on CAMS helps when CAMS is also the input, and the gain is gone when",
            "the input is an ENS forecast",
        ],
        x_title=(
            "CAMS-trained minus ERA5-trained error of the median, daylight, Primary "
            "(points of capacity; more negative is better)"
        ),
        order=frame["label"].to_list(),
        zero_label="same as ERA5-trained",
        domain=(-1.1, 0.2),
        show_key=False,
    )
    _save(chart=chart, name="figure_3")
    return frame


def _save(*, chart: alt.TopLevelMixin, name: str) -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    chart.save(FIGURES_DIR / f"{name}.svg")
    chart.save(common.WORK_DIR / "results" / f"{name}.png", scale_factor=2)


def main() -> None:
    """Print the quoted numbers and draw the figures."""
    results = common.WORK_DIR / "results"
    planned = pl.read_parquet(results / "planned.parquet")
    exploratory = pl.read_parquet(results / "exploratory.parquet")
    observed = pl.read_parquet(results / "observed_weather.parquet")
    absolute = pl.read_parquet(results / "absolute.parquet")
    arms = {
        name: pl.read_parquet(score.ARMS_DIR / f"{name}.parquet")
        for name in (score.REFERENCE, score.TREATMENT)
    }
    pl.Config.set_tbl_rows(100).set_tbl_cols(20).set_fmt_str_lengths(60)
    print(planned)
    print(exploratory.filter(~pl.col("label").str.contains("only|without")))
    print(observed)
    print(
        absolute.filter((pl.col("scope") == "all leads") & (pl.col("group") == "Primary")).select(
            "arm", "metric", "value", "lower_95", "upper_95"
        )
    )
    figure_1(planned=planned, exploratory=exploratory)
    figure_2(exploratory=exploratory)
    print(figure_3(observed=observed, arms=arms))
    shortwave = pl.read_parquet(results / "shortwave_by_month.parquet")
    print(
        shortwave.group_by("period", "source").agg(pl.col("value").mean().round(1)).sort("period")
    )


if __name__ == "__main__":
    main()
