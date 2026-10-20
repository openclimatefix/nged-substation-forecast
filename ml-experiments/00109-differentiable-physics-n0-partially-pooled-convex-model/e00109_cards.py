"""Draw the substation cards of experiment 109 (provisional) from the pooled fit at the scored cut.

The cards show, for each decision site under its D-label, the coefficients of the pooled model N
beside the fleet mean and the per-site model G, and how far the deviation penalty pulled each
family towards the fleet mean. A last figure compares the capacity three ways: the real-time
vintage, the capacity fitted from 4 weeks and from 3 months by N, and the same by G. The numbers go
to `cards.json` beside the figures.

    uv run python ml-experiments/00109-*/e00109_cards.py
"""

import json
from pathlib import Path
from typing import Final

import altair as alt
import e00109_arms as arms
import e00109_common as common
import numpy as np
import plotting.ocf_theme as ocf
import polars as pl
import torch
from studies.dp_calendar import HALF_HOURS_PER_DAY, HEATING_THRESHOLD_C, SMOOTHING_HOURS
from studies.dp_gam import N_BANDS, GamFit, GamLayout
from studies.dp_gam_pooled import (
    POOLED_FAMILIES,
    PooledFit,
    PoolingStrength,
    centring_matrix,
    fit_pooled,
)

DOCS_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / f"{common.EXPERIMENT_ID:05d}-{common.SLUG}"
)
PNG_DIR: Final[Path] = Path("/tmp/e00109_png")
WIDTH: Final[int] = 560
SITES: Final[tuple[str, ...]] = common.DECISION_SITES
SITE_COLOURS: Final[list[str]] = [
    ocf.BRAND_ORANGE,
    ocf.DATA_BLUE,
    ocf.DATA_SKY,
    ocf.DATA_PURPLE,
    ocf.DATA_GREEN,
]
BAND_NAMES: Final[tuple[str, ...]] = ("night", "morning", "midday", "evening")
HEATING_TEMPERATURES: Final[np.ndarray] = np.linspace(-5.0, HEATING_THRESHOLD_C, 22)
"""Steady temperatures, in degrees Celsius, at which the heating response is drawn."""


def _save(chart: alt.TopLevelMixin, *, name: str) -> None:
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    PNG_DIR.mkdir(parents=True, exist_ok=True)
    chart.save(DOCS_DIR / f"{name}.svg")
    with (DOCS_DIR / f"{name}.svg").open("a", encoding="utf-8") as handle:
        handle.write("\n")
    chart.save(PNG_DIR / f"{name}.png", scale_factor=1.5)
    print("wrote", name)


def heating_response(*, theta: np.ndarray, layout: GamLayout) -> np.ndarray:
    """Return the heating contribution of each hour band at steady temperatures, in points.

    Args:
        theta: The site's coefficients.
        layout: The design's columns.

    Returns:
        An array of shape (bands, temperatures): the contribution when every trailing mean of
        temperature equals the temperature, so the hinge is the same in all four windows.
    """
    heating = theta[layout.slices["heating"]].reshape(len(SMOOTHING_HOURS), N_BANDS)
    hinge = np.clip(HEATING_THRESHOLD_C - HEATING_TEMPERATURES, 0.0, None) / 10.0
    return heating.sum(axis=0)[:, None] * hinge[None, :]


def weekday_profile(*, theta: np.ndarray, layout: GamLayout) -> np.ndarray:
    """Return the weekday daily profile, 48 half-hours, in points."""
    profile = layout.slices["profile"]
    return theta[profile][:HALF_HOURS_PER_DAY]


def fleet_fit(*, cut_name: str, strength: PoolingStrength) -> tuple[PooledFit, arms.Built]:
    """Fit the pooled model N on the full windows at a cut, for the cards."""
    cache, vintages = arms.load_source(source=arms.Source())
    config = arms.ArmConfig(name="cards", strength=strength)
    built = arms.build_arm(
        cache=cache,
        vintages=vintages,
        cut=common.CUTS[cut_name],
        config=config,
        held_out=None,
        history=None,
    )
    joint = fit_pooled(
        cache=built.cache, sites=built.sites, cut=common.CUTS[cut_name], strengths=[strength]
    )
    return joint, built


def pull_table(*, joint: PooledFit, unpooled: PooledFit) -> pl.DataFrame:
    """Return how far the penalty pulled each family of each site towards the fleet mean.

    The pull is one minus the norm of the pooled deviation from the fleet mean over the norm of
    the per-site deviation from the same fleet mean, so 0 means the site kept its own coefficients
    and 1 means it sits on the fleet mean. The profile's band means are free and are left out.
    """
    layout = next(iter(joint.fits.values())).layout
    rows = []
    offset = 0
    mu = joint.fleet_mean[0]
    for family in POOLED_FAMILIES:
        block = layout.slices[family]
        size = block.stop - block.start
        centre = centring_matrix(family=family, size=size)
        for site in SITES:
            pooled = centre @ (joint.fits[site].theta[0, block] - mu[offset : offset + size])
            alone = centre @ (unpooled.fits[site].theta[0, block] - mu[offset : offset + size])
            rows.append(
                {
                    "site": site,
                    "family": family,
                    "pull": float(1.0 - np.linalg.norm(pooled) / np.linalg.norm(alone)),
                }
            )
        offset += size
    return pl.DataFrame(rows)


def figure_heating(*, joint: PooledFit, unpooled: PooledFit) -> alt.TopLevelMixin:
    """The heating response of the fleet mean and of each site, by hour band."""
    layout = next(iter(joint.fits.values())).layout
    rows = []
    mean_theta = np.zeros(layout.n_columns)
    offset = 0
    for family in POOLED_FAMILIES:
        size = layout.slices[family].stop - layout.slices[family].start
        mean_theta[layout.slices[family]] = joint.fleet_mean[0][offset : offset + size]
        offset += size
    curves = {
        "Fleet mean": heating_response(theta=mean_theta, layout=layout),
        **{
            f"{site} pooled": heating_response(theta=joint.fits[site].theta[0], layout=layout)
            for site in SITES
        },
        **{
            f"{site} per site": heating_response(theta=unpooled.fits[site].theta[0], layout=layout)
            for site in SITES
        },
    }
    for name, curve in curves.items():
        for band, label in enumerate(BAND_NAMES):
            for temperature, value in zip(HEATING_TEMPERATURES, curve[band], strict=True):
                rows.append(
                    {"series": name, "band": label, "temperature": temperature, "value": value}
                )
    table = pl.DataFrame(rows).with_columns(
        kind=pl.when(pl.col("series") == "Fleet mean")
        .then(pl.lit("fleet"))
        .when(pl.col("series").str.ends_with("pooled"))
        .then(pl.lit("pooled"))
        .otherwise(pl.lit("per site")),
        site=pl.col("series").str.split(" ").list.get(0),
    )
    chart = (
        alt.Chart(table.filter(pl.col("kind") != "per site"))  # ty: ignore[unresolved-attribute]
        .mark_line()
        .encode(
            x=alt.X("temperature:Q", title="Steady temperature (°C)"),
            y=alt.Y("value:Q", title="Heating contribution (points)"),
            color=alt.Color(
                "site:N",
                sort=["Fleet", *SITES],
                scale=alt.Scale(domain=["Fleet", *SITES], range=["#202020", *SITE_COLOURS]),
                legend=alt.Legend(orient="top", title=None),
            ),
            strokeWidth=alt.condition(alt.datum.kind == "fleet", alt.value(3), alt.value(1.4)),
        )
        .properties(width=WIDTH // 4 - 12, height=150)
        .facet(column=alt.Column("band:N", sort=list(BAND_NAMES), title=None))
    )
    return chart.properties(
        title=alt.TitleParams(
            text="Pooling pulls each substation's heating response towards the fleet mean",
            subtitle=[
                (
                    "Pooled model N at the 2025-07-01 cut, 2 years of history per substation. "
                    "Black: fleet mean. Coloured: each"
                ),
                "substation under its D-label. Four hour bands; the four smoothing windows summed.",
            ],
            anchor="start",
            fontSize=14,
            subtitleFontSize=11,
        )
    )


def capacity_rows(*, vintage_points: dict[str, float], held: dict[str, dict[str, float]]) -> list:
    """Return the capacity figure's rows: vintage, and fitted from each short window."""
    rows = []
    for site in SITES:
        rows.append({"site": site, "source": "Real-time vintage", "points": vintage_points[site]})
        for source, value in held[site].items():
            rows.append({"site": site, "source": source, "points": value})
    return rows


def figure_capacity(*, rows: list) -> alt.TopLevelMixin:
    """Capacity of each decision site by source, in points of effective capacity."""
    table = pl.DataFrame(rows)
    order = ["Real-time vintage", "N-4w fit", "G-4w fit", "N-3m fit", "G-3m fit"]
    return (
        alt.Chart(table)  # ty: ignore[unresolved-attribute]
        .mark_point(filled=True, size=70)
        .encode(
            x=alt.X(
                "points:Q",
                title="Solar capacity (points of the site's whole-history effective capacity)",
            ),
            y=alt.Y("site:N", sort=list(SITES), title=None),
            color=alt.Color(
                "source:N",
                sort=order,
                scale=alt.Scale(
                    domain=order,
                    range=[
                        "#202020",
                        ocf.DATA_BLUE,
                        ocf.BRAND_ORANGE,
                        ocf.DATA_SKY,
                        ocf.DATA_PURPLE,
                    ],
                ),
                legend=alt.Legend(orient="top", title=None),
            ),
        )
        .properties(
            width=WIDTH,
            height=160,
            title=alt.TitleParams(
                text="Capacity from a short window against the real-time vintage",
                subtitle=[
                    (
                        "Fits at the 2025-07-01 cut where the substation keeps only its last 4 "
                        "weeks or 3 months, against"
                    ),
                    "the capacity fitted to its whole history by the real-time vintage.",
                ],
                anchor="start",
                fontSize=14,
                subtitleFontSize=11,
            ),
        )
    )


def short_capacity(*, strength: PoolingStrength, ridge: dict[str, float]) -> dict:
    """Return each decision site's fitted capacity from 4 weeks and 3 months, pooled and per site.

    Args:
        strength: `lambda*` for the pooled fits.
        ridge: The per-site comparator's ridge for each history, from `tuned.json`.

    Returns:
        For each site, the capacity of the last season-year the fit trained on, converted back to
        points of the site's whole-history effective capacity, for each of the four fits.
    """
    cache, vintages = arms.load_source(source=arms.Source())
    cut = common.SCORED_CUT
    zero = PoolingStrength.uniform(value=0.0)
    out: dict[str, dict[str, float]] = {site: {} for site in SITES}
    for site in SITES:
        for history, name in ((common.FOUR_WEEKS, "4w"), (common.THREE_MONTHS, "3m")):
            for label, model_strength, alone in (("N", strength, False), ("G", zero, True)):
                config = arms.ArmConfig(
                    name=f"{label}{name}",
                    strength=model_strength,
                    history=history,
                    alone=alone,
                    short_ridge=ridge[name] if alone else None,
                )
                built = arms.build_arm(
                    cache=cache,
                    vintages=vintages,
                    cut=cut,
                    config=config,
                    held_out=site,
                    history=history,
                )
                joint = fit_pooled(
                    cache=built.cache, sites=built.sites, cut=cut, strengths=[model_strength]
                )
                fit: GamFit = joint.fits[site]
                scale = built.transform.scale if built.transform is not None else 1.0
                out[site][f"{label}-{name} fit"] = (
                    fit.capacity_points()[fit.last_season_year] / scale
                )
    return out


def main() -> None:
    """Draw the cards and write their numbers."""
    torch.set_num_threads(4)
    tuned = json.loads((common.WORK_DIR / "tuned.json").read_text())
    strength = PoolingStrength.uniform(value=tuned["lambda_star"])
    joint, built = fleet_fit(cut_name="scored", strength=strength)
    unpooled, _ = fleet_fit(cut_name="scored", strength=PoolingStrength.uniform(value=0.0))
    pulls = pull_table(joint=joint, unpooled=unpooled)
    vintage_points = {
        site: float(built.sites[[s.site for s in built.sites].index(site)].pinned.held_points)  # ty: ignore[unresolved-attribute]
        for site in SITES
    }
    held = short_capacity(strength=strength, ridge=tuned["g_short_ridge"])
    rows = capacity_rows(vintage_points=vintage_points, held=held)
    _save(figure_heating(joint=joint, unpooled=unpooled), name="figure_08_heating")
    _save(figure_capacity(rows=rows), name="figure_09_capacity")
    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    (DOCS_DIR / "cards.json").write_text(
        json.dumps(
            {
                "pull": pulls.to_dicts(),
                "capacity": rows,
                "capacity_sources": built.capacity_sources,
            },
            indent=2,
        )
        + "\n"
    )
    print(pulls.pivot(on="family", index="site", values="pull").write_csv(float_precision=3))


if __name__ == "__main__":
    main()
