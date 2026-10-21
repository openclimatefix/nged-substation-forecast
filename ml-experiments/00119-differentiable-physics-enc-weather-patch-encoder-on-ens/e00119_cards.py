"""Draw experiment 119's substation cards, the patch-field figure, and the hard-day calendar.

Cards are provisional: ENC-c's F1 fit, forecast on ENS. Each card shows, by D-label, the number of
distinct ENS boxes in the patch, the learned weight of each box (the change in the median shift per
kelvin added to that box's temperature, averaged over the scored year's rows, with the five seeds'
range), the median shift against lead, the cold-snap scores, and four example weeks fixed before
any score was read. Generator data is not used.

    uv run python ml-experiments/00119-*/e00119_cards.py
"""

import json
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Final, cast

import altair as alt
import e00119_bundle as bundles
import e00119_charts as charts
import e00119_common as common
import e00119_cv as cv
import e00119_data as data
import e00119_gamk as gamk
import e00119_model as model
import e00119_score as score
import e00119_stats as stats
import numpy as np
import plotting.ocf_theme as ocf
import polars as pl
import torch

WEEK_STARTS: Final[tuple[str, ...]] = ("2025-07-21", "2025-10-20", "2026-01-19", "2026-04-20")
N_SAMPLE_ROWS: Final[int] = 20000
SAMPLE_SEED: Final[int] = 5
CARD_LINE: Final[str] = (
    "Provisional: fitted to past weather, forecast on ENS, screening run. Patch weights are not "
    "identified individually (neighbouring cells share an ENS box); read the ring and east-west "
    "contrast, not one cell. No physical direction is claimed from one site."
)


def box_groups(*, patch: np.ndarray, present: np.ndarray) -> list[list[int]]:
    """Group a patch's cells that always hold identical values (one ENS box) over sampled rows."""
    groups: list[list[int]] = []
    seen: set[int] = set()
    for cell in range(common.N_PATCH_CELLS):
        if cell in seen or not present[cell]:
            continue
        same = [
            other
            for other in range(common.N_PATCH_CELLS)
            if present[other] and np.array_equal(patch[:, cell, 0], patch[:, other, 0])
        ]
        seen.update(same)
        groups.append(same)
    return groups


def box_weights(*, site: str) -> dict:
    """Return each distinct box's weight for one site, averaged over rows, with the seeds' range."""
    dataset = data.load_dataset()
    rows = bundles.rows_with_sensitivities(dataset=dataset)
    tensors = bundles.make_tensors(dataset=dataset)
    arm = model.ARMS["ENC-c"]
    bundle = bundles.make_bundle(
        rows=rows,
        tensors=tensors,
        arm=arm,
        distant=bundles.distant_sites(dataset=dataset),
        foreign=bundles.foreign_runs(dataset=dataset),
    )
    site_index = common.SCORED_SITES.index(site)
    scored = np.flatnonzero(
        cv.run_mask(rows=rows, period=cv.F1.scored) & (rows["site_index"].to_numpy() == site_index)
    )
    generator = np.random.default_rng(SAMPLE_SEED)
    sample = np.sort(generator.choice(scored, size=min(N_SAMPLE_ROWS, len(scored)), replace=False))
    part = bundle.subset(cv.to_device(sample))
    base = part.patch[part.columns["src_site"], part.columns["src_run"], part.columns["step"]]
    patch_np = model.to_numpy(base)
    groups = box_groups(patch=patch_np, present=dataset.present[site_index])
    decisions = json.loads((common.WORK_DIR / "decisions.json").read_text())
    shrink = decisions["F1.ENC-c"]["shrink"]
    weights = np.zeros((len(cv.SEEDS), len(groups)))
    for seed_index, seed in enumerate(cv.SEEDS):
        saved = torch.load(
            common.WORK_DIR / "final" / f"weights_F1_ENC-c_{seed}.pt", weights_only=False
        )
        encoder = model.Encoder(kind="c", n_sites=len(common.SCORED_SITES)).to(model.DEVICE)
        encoder.load_state_dict(saved["state"])
        encoder.eval()
        std = saved["standardiser"]

        def median_shift(
            patch: torch.Tensor, *, _e: model.Encoder = encoder, _s: model.Standardiser = std
        ) -> torch.Tensor:
            inputs = model.encoder_inputs(
                bundle=part, std=_s, arm=arm, drop=False, generator=None, patch_override=patch
            )
            return _e(**inputs)[:, 5]

        with torch.no_grad():
            reference = median_shift(base)
            for g, cells in enumerate(groups):
                moved = base.clone()
                moved[:, cells, 0] += 1.0
                weights[seed_index, g] = float((shrink * (median_shift(moved) - reference)).mean())
    geometry = dataset.geometry[site_index]
    return {
        "groups": groups,
        "mean": weights.mean(axis=0).tolist(),
        "low": weights.min(axis=0).tolist(),
        "high": weights.max(axis=0).tolist(),
        "dx": geometry[:, 0].tolist(),
        "dy": geometry[:, 1].tolist(),
    }


def hex_panel(*, info: dict, site: str) -> alt.TopLevelMixin:
    """Draw the patch as hexagon-spaced dots coloured by the box weight (no coordinates)."""
    rows = [
        {
            "x": info["dx"][cell] / 0.4,
            "y": info["dy"][cell] / 0.4,
            "weight": info["mean"][g],
            "label": (f"{info['mean'][g]:+.3f} ({info['low'][g]:+.3f} to {info['high'][g]:+.3f})"),
        }
        for g, cells in enumerate(info["groups"])
        for cell in cells
    ]
    bound = max(abs(r["weight"]) for r in rows) or 1.0  # ty: ignore[invalid-argument-type]
    return (
        alt.Chart(pl.DataFrame(rows))  # ty: ignore[unresolved-attribute]
        .mark_circle(size=330, stroke="#444444", strokeWidth=0.6)
        .encode(
            x=alt.X("x:Q", axis=None),
            y=alt.Y("y:Q", axis=None),
            color=alt.Color(
                "weight:Q",
                scale=alt.Scale(
                    domain=[-bound, 0, bound], range=[ocf.DATA_BLUE, "#FFFFFF", ocf.BRAND_ORANGE]
                ),
                legend=alt.Legend(title="Points per kelvin", orient="bottom", gradientLength=140),
            ),
            tooltip=["label:N"],
        )
        .properties(
            width=180,
            height=170,
            title=charts.title(
                text=f"{site}: weight of each ENS box",
                subtitle=(
                    f"{len(info['groups'])} distinct boxes; the median forecast changes by this "
                    "much per kelvin added to a box. Mean of five seeds."
                ),
            ),
        )
    )


def shift_by_lead(*, scored: score.Scored, site_index: int) -> alt.TopLevelMixin:
    """Draw the mean median shift (ENC-c minus GAM-K) against lead."""
    rows = scored.rows
    mask = rows["site_index"].to_numpy() == site_index
    shift = scored.quantiles["ENC-c"][mask, 5] - scored.quantiles["K0"][mask, 5]
    frame = pl.DataFrame(
        {"lead": np.floor(rows["lead_hours"].to_numpy()[mask] / 6) * 6, "shift": shift}
    )
    table = frame.group_by("lead").agg(pl.col("shift").mean()).sort("lead")
    return (
        alt.Chart(table)  # ty: ignore[unresolved-attribute]
        .mark_line(point=True, color=ocf.BRAND_ORANGE, strokeWidth=2)
        .encode(
            x=alt.X("lead:Q", title="Hours after the forecast origin"),
            y=alt.Y("shift:Q", title="Mean median shift (points)"),
        )
        .properties(width=220, height=140, title="Median shift against lead")
    )


def example_weeks(*, scored: score.Scored, site_index: int, site: str) -> alt.TopLevelMixin:
    """Draw four example weeks: K0 and ENC-c median and bands against the observed series."""
    k0_all = pl.read_parquet(gamk.k0_path(site=site))
    rows = scored.rows
    mask = rows["site_index"].to_numpy() == site_index
    enc = pl.DataFrame(
        {
            "run": rows["run"].to_numpy()[mask],
            "valid_time": rows["valid_time"].filter(pl.Series(mask)),
            **{
                f"enc_{c}": scored.quantiles["ENC-c"][mask, i]
                for i, c in enumerate(common.QUANTILE_COLUMNS)
            },
        }
    )
    dates = [d for d in common.run_dates() if d < common.HELD_OUT_START]
    run_of_day = {d.date(): i for i, d in enumerate(dates)}
    present = set(k0_all["run"].unique().to_list())

    def run_for_day(day: date) -> int:
        """Return the latest present run issued 1 to 3 days before a valid day, or -1."""
        for back in (1, 2, 3):
            candidate = run_of_day.get(day - timedelta(days=back))
            if candidate is not None and candidate in present:
                return candidate
        return -1

    panels = []
    for label in WEEK_STARTS:
        first = datetime.fromisoformat(label).replace(tzinfo=UTC)
        week = k0_all.filter(
            (pl.col("valid_time") >= first) & (pl.col("valid_time") < first + timedelta(days=7))
        ).with_columns(
            day_run=pl.col("valid_time").dt.date().map_elements(run_for_day, return_dtype=pl.Int64)
        )
        week = week.filter(pl.col("run") == pl.col("day_run"))
        week = week.join(enc, on=["run", "valid_time"], how="left")
        base = alt.Chart(week).encode(
            x=alt.X("valid_time:T", title=None, axis=alt.Axis(format="%a", tickCount=7))
        )
        bands = alt.layer(
            base.mark_area(opacity=0.2, color=ocf.DATA_BLUE).encode(y="q10:Q", y2="q90:Q"),  # ty: ignore[unresolved-attribute]
            base.mark_area(opacity=0.2, color=ocf.BRAND_ORANGE).encode(  # ty: ignore[unresolved-attribute]
                y="enc_q10:Q", y2="enc_q90:Q"
            ),
            base.mark_line(color=ocf.DATA_BLUE, strokeWidth=1.2).encode(y="q50:Q"),  # ty: ignore[unresolved-attribute]
            base.mark_line(color=ocf.BRAND_ORANGE, strokeWidth=1.2).encode(y="enc_q50:Q"),  # ty: ignore[unresolved-attribute]
            base.mark_line(color="#222222", strokeWidth=0.9, strokeDash=[2, 1]).encode(  # ty: ignore[unresolved-attribute]
                y=alt.Y("actual:Q", title="Points of capacity")
            ),
        ).properties(
            width=230,
            height=110,
            title=f"Week from {first:%d %B %Y}",
        )
        panels.append(bands)
    return alt.vconcat(alt.hconcat(*panels[:2]), alt.hconcat(*panels[2:]))


def card(*, site: str) -> alt.TopLevelMixin:
    """Return one substation's card."""
    scored = score.load(direction=cv.F1)
    site_index = common.SCORED_SITES.index(site)
    info = box_weights(site=site)
    masks = stats.hard_day_masks(rows=scored.rows)
    at_site = scored.rows["site_index"].to_numpy() == site_index
    cold = masks["coldest"] & at_site
    n_days = int(
        scored.rows["run"].to_numpy()[cold].size
        and len(np.unique(scored.rows["run"].to_numpy()[cold]))
    )
    top = alt.hconcat(
        hex_panel(info=info, site=site),  # ty: ignore[invalid-argument-type]
        shift_by_lead(scored=scored, site_index=site_index),  # ty: ignore[invalid-argument-type]
    )
    cold_text = f"Cold snap (coldest 5% of days, {n_days} days in this scored year): " + (
        "too few days to score."
        if n_days < 10
        else (
            f"mean score GAM-K {scored.score['K0'][cold].mean():.2f}, "
            f"ENC-c {scored.score['ENC-c'][cold].mean():.2f} points."
        )
    )
    return alt.vconcat(
        top,
        example_weeks(scored=scored, site_index=site_index, site=site),  # ty: ignore[invalid-argument-type]
    ).properties(
        title=charts.title(
            text=f"Substation card {site} (ENC-c, direction F1)",
            subtitle=(
                f"{CARD_LINE} {cold_text} Weeks: GAM-K blue, ENC-c orange, "
                "observed dashed black; median and 80% band."
            ),
        )
    )


def patch_fields(*, site: str = "D10") -> alt.TopLevelMixin:
    """Draw the patch's day-mean temperature and precipitation on three days, by hexagon layout."""
    dataset = data.load_dataset()
    spec = json.loads((Path(__file__).resolve().parent / "hard_days.json").read_text())
    index = common.SCORED_SITES.index(site)
    dates = [f"{d:%Y-%m-%d}" for d in dataset.run_dates]
    steps = [common.STEP_HOURS.index(h) for h in range(9, 34, 3)]
    frames = np.load(common.frames_path())
    union = [int(c) for c in frames["cells"]]
    day_temperature = {
        d: float(np.nanmean(dataset.patch[index, dates.index(d), steps, 0, 0]))
        for d in spec["coldest"]
    }
    coldest = min(day_temperature, key=lambda d: day_temperature[d])
    showery = spec["showery"][site][0]
    pool = [d for d in dates if "2024-07-01" <= d <= "2026-06-30"]
    ordinary_temperatures = {
        d: float(np.nanmean(dataset.patch[index, dates.index(d), steps, 0, 0])) for d in pool
    }
    median = float(np.median(list(ordinary_temperatures.values())))
    ordinary = min(pool, key=lambda d: abs(ordinary_temperatures[d] - median))
    del union
    rows = []
    for name, day in (
        ("coldest day", coldest),
        ("showery day", showery),
        ("ordinary day", ordinary),
    ):
        run = dates.index(day)
        for _field, channel, label, scale in (
            ("temperature", 0, "Day-mean 2 m temperature (degrees Celsius)", 1.0),
            ("precipitation", 3, "Day-mean precipitation rate (mm per hour)", 3600.0),
        ):
            values = np.nanmean(dataset.patch[index, run, steps, :, channel], axis=0) * scale
            rows.extend(
                {
                    "day": f"{name} ({day})",
                    "field": label,
                    "x": float(dataset.geometry[index, cell, 0] / 0.4),
                    "y": float(dataset.geometry[index, cell, 1] / 0.4),
                    "value": float(values[cell]),
                }
                for cell in range(common.N_PATCH_CELLS)
                if dataset.present[index, cell]
            )
    table = pl.DataFrame(rows)
    panels = []
    for label, colours in (
        (
            "Day-mean 2 m temperature (degrees Celsius)",
            [ocf.DATA_BLUE, "#FFFFFF", ocf.BRAND_ORANGE],
        ),
        ("Day-mean precipitation rate (mm per hour)", ["#FFFFFF", ocf.DATA_SKY, ocf.DATA_BLUE]),
    ):
        part = table.filter(pl.col("field") == label)
        low = float(cast(float, part["value"].min()))
        high = float(cast(float, part["value"].max()))
        domain = [low, (low + high) / 2, high] if len(colours) == 3 else [low, high]
        scale = alt.Scale(
            domain=domain,
            range=colours[: len(domain)] if len(colours) == 3 else [colours[0], colours[2]],
        )
        panels.append(
            alt.Chart(part)  # ty: ignore[unresolved-attribute]
            .mark_circle(size=230, stroke="#444444", strokeWidth=0.5)
            .encode(
                x=alt.X("x:Q", axis=None),
                y=alt.Y("y:Q", axis=None),
                color=alt.Color(
                    "value:Q",
                    scale=scale,
                    legend=alt.Legend(
                        title=label, orient="bottom", gradientLength=200, titleLimit=400
                    ),
                ),
                column=alt.Column(
                    "day:N", title=None, sort=list(dict.fromkeys(table["day"].to_list()))
                ),
            )
            .properties(width=150, height=140)
        )
    return (
        alt.vconcat(*panels)
        .resolve_scale(color="independent")
        .properties(
            title=charts.title(
                text=(
                    "Neighbouring ENS boxes differ on cold and showery days: "
                    f"the patch around {site}"
                ),
                subtitle=(
                    "Each dot is one of the 37 hexagonal cells around the substation (the centre "
                    "dot is its own cell, laid out as on the ground but without coordinates), "
                    "coloured by the ensemble-mean field averaged over the day's valid times "
                    "(run hours 9 to 33). Days: the coldest of the coldest-5% days, the most "
                    "showery day for this substation, and the day nearest the median temperature."
                ),
            )
        )
    )


def hard_day_calendar() -> alt.TopLevelMixin:
    """Draw the dates of the three hard-day subsets over the two years."""
    spec = json.loads((Path(__file__).resolve().parent / "hard_days.json").read_text())
    rows = [
        {"date": d, "subset": "Coldest 5% of days (all sites)", "y": 0} for d in spec["coldest"]
    ]
    showery = {d for days in spec["showery"].values() for d in days}
    largest = {d for days in spec["largest_error"].values() for d in days}
    rows += [{"date": d, "subset": "Most showery 5% (any site)", "y": 1} for d in sorted(showery)]
    rows += [
        {"date": d, "subset": "Largest GAM-K errors 5% (any site)", "y": 2} for d in sorted(largest)
    ]
    order = [
        "Coldest 5% of days (all sites)",
        "Most showery 5% (any site)",
        "Largest GAM-K errors 5% (any site)",
    ]
    return (
        alt.Chart(pl.DataFrame(rows).with_columns(pl.col("date").str.to_date()))  # ty: ignore[unresolved-attribute]
        .mark_tick(thickness=2, size=22)
        .encode(
            x=alt.X("date:T", title="Run date, 1 July 2024 to 30 June 2026"),
            y=alt.Y("subset:N", sort=order, title=None, axis=alt.Axis(labelLimit=240)),
            color=alt.Color(
                "subset:N",
                scale=alt.Scale(
                    domain=order, range=[ocf.DATA_BLUE, ocf.DATA_SKY, ocf.BRAND_ORANGE]
                ),
                legend=None,
            ),
        )
        .properties(
            width=charts.WIDTH - 250,
            height=110,
            title=charts.title(
                text="The hard days were fixed before any score was read, and cluster in winter",
                subtitle=(
                    "Each tick is a run date in one of the three subsets. Coldest 5% and most "
                    "showery 5% come from the ensemble's control member only; the largest-error "
                    "subset comes from GAM-K's own errors. Showery and largest-error days are "
                    "the union over the six substations."
                ),
            ),
        )
    )


def main() -> None:
    """Draw the cards, the patch figure, and the calendar."""
    charts.save(patch_fields(), name="patch_fields")
    charts.save(hard_day_calendar(), name="hard_day_calendar")
    for site in common.DECISION_SITES:
        charts.save(card(site=site), name=f"card_{site.lower()}")


if __name__ == "__main__":
    main()
