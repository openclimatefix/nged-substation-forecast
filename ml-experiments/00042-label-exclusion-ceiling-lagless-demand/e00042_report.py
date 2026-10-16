"""Score experiment 00042's fits and print every number the page quotes.

Reads the planned contrasts.

Reads `work/fits/` (see `e00042_fit.py`), scores each arm outside and inside the labelled
switching of the validation year, and writes `work/report.md`, `work/results/*.parquet`, and the
figures under `work/charts/`. A series that an arm does not refit uses the `none` fit.

- Absolute scores of every arm, with a month-by-half-hour climatology and `manual_heuristic` for
  scale.
- Planned contrasts C1 to C4 per source on the quantile score outside switching, over the series
  the exclusion changes. C2 and C4 are read at 98.75% (Bonferroni across four); C1 and C3 are
  reported without a verdict. Every contrast is also given at 95%.
- C2 and C4 per random draw, the draws' spread, the descriptive per-series and per-event gaps, the
  minimum detectable gap from the no-exclusion seed spread, and the pooled equivalent of the
  threshold.

    uv run python ml-experiments/00042-*/e00042_report.py [--smoke] [--no-charts]
"""

import argparse
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final, TypedDict

import altair as alt
import e00042_common as common
import numpy as np
import plotting.ocf_theme as ocf
import polars as pl

from studies import bootstrap, experiment_store

RESULTS_DIR: Final[Path] = common.WORK_DIR / "results"
CHART_DIR: Final[Path] = common.WORK_DIR / "charts"
N_RESAMPLES: Final[int] = 20_000
PLANNED_LEVEL: Final[float] = 100.0 * (1.0 - 0.05 / 4)
"""98.75%: Bonferroni across the four planned contrasts C2 and C4 of two sources."""
THRESHOLD: Final[float] = 0.10
"""Points of capacity: the smallest gap worth a detector (the plan's decision rule)."""
MIN_TRUSTED_EVENTS: Final[int] = 3
BASELINES: Final[tuple[str, ...]] = ("climatology", "manual_heuristic")
REAL_ARMS: Final[tuple[str, ...]] = ("none", "trusted", "allvis")
CONTRASTS: Final[dict[str, tuple[common.LabelSetType, str, str]]] = {
    "C1": ("trusted", "trusted", "none"),
    "C2": ("trusted", "trusted", "rand_trusted"),
    "C3": ("allvis", "allvis", "none"),
    "C4": ("allvis", "allvis", "rand_allvis"),
}
"""Contrast id: (label set that picks the affected series, treatment arm, reference arm)."""
EXPECTED_RUNS: Final[int] = 73
EXPECTED_MEMBERS: Final[int] = 51
ARM_LABELS: Final[dict[str, str]] = {
    "none": "No exclusion",
    "trusted": "Trusted switching removed",
    "rand_trusted": "Random half-hours removed (as trusted)",
    "allvis": "All visible switching removed",
    "rand_allvis": "Random half-hours removed (as all visible)",
    "climatology": "Climatology",
    "manual_heuristic": "manual_heuristic",
}
SOURCE_LABELS: Final[dict[str, str]] = {
    "era5": "Trained on ERA5 weather, 2022-01 to 2025-06",
    "ens": "Trained on ENS control member, 2024-04 to 2025-06",
}


def present_series(*, root: Path) -> list[int]:
    """Return the series that have a `none` fit for every source and seed 0."""
    ids = common.series_table()["time_series_id"].to_list()
    return [
        i
        for i in ids
        if all(
            common.fit_path(source=s, arm="none", seed=0, series_id=i, root=root).exists()
            for s in common.SOURCES
        )
    ]


def read_arm_file(
    *, root: Path, source: common.SourceType, arm: str, seed: int, series_id: int
) -> pl.DataFrame:
    """Return one series' predictions for an arm.

    The `none` predictions stand in only for a series that the arm's label set does not change.

    Raises:
        FileNotFoundError: If the arm's file is missing for a series that the arm must refit.
    """
    path = common.fit_path(source=source, arm=arm, seed=seed, series_id=series_id, root=root)
    if path.exists():
        return pl.read_parquet(path)
    label_set = common.arm_label_set(arm=arm)
    if label_set is None or series_id in common.affected_series(source=source, label_set=label_set):
        msg = f"{path} is missing for a series that arm {arm} changes"
        raise FileNotFoundError(msg)
    return pl.read_parquet(
        common.fit_path(source=source, arm="none", seed=seed, series_id=series_id, root=root)
    )


def actual_table(*, root: Path, series_ids: list[int]) -> pl.DataFrame:
    """Return every scored case with the observed power over capacity and the switching flag.

    The cases are the keys of each series' `none` fit; `inside` is true for a half-hour inside a
    visible labelled interval of the series.
    """
    series = common.series_table().filter(pl.col("time_series_id").is_in(series_ids))
    arrays = common.load_series_arrays(series=series)
    pieces = []
    for row in series.iter_rows(named=True):
        keys = pl.read_parquet(
            common.fit_path(
                source="era5", arm="none", seed=0, series_id=row["time_series_id"], root=root
            )
        ).select(common.KEYS)
        index = np.asarray(common.grid_index(times=keys["valid_time"]))
        switching = common.switching_mask(series_id=row["time_series_id"])
        pieces.append(
            keys.with_columns(
                actual=pl.Series(common.take(values=arrays[row["site"]].power, index=index)),
                inside=pl.Series(switching[index]),
            )
        )
    return pl.concat(pieces)


def check_case_counts(*, source: str, predictions: pl.DataFrame, series_ids: list[int]) -> None:
    """Require 73 runs per series and as many cases as the validation inputs hold (51 members).

    Raises:
        ValueError: If a series has the wrong number of runs or cases.
    """
    for series_id in series_ids:
        part = predictions.filter(pl.col("time_series_id") == series_id)
        inputs = pl.read_parquet(
            common.VAL_INPUTS_DIR / f"{series_id}.parquet", columns=["ensemble_member"]
        ).height
        runs, cases = part["power_fcst_init_time"].n_unique(), part.height
        if runs != EXPECTED_RUNS or cases != inputs // EXPECTED_MEMBERS:
            msg = (
                f"{source} series {series_id}: {runs} runs, {cases} cases; expected {EXPECTED_RUNS}"
            )
            raise ValueError(msg)


def case_losses(
    *, predictions: pl.DataFrame, actual: pl.DataFrame, label: str | None = None
) -> pl.DataFrame:
    """Return the quantile score and the median's absolute error per case, in points of capacity."""
    joined = predictions.join(actual, on=list(common.KEYS), how="inner", validate="m:1")
    quantiles = np.sort(joined.select(common.LEVEL_COLUMNS).to_numpy().astype(np.float64), axis=1)
    observed = joined["actual"].to_numpy().astype(np.float64)
    finite = np.isfinite(quantiles).all(axis=1) & np.isfinite(observed)
    if label is not None:
        late = int((joined["valid_time"] >= common.HOLDOUT).sum())
        print(
            f"{label}: {predictions.height} predicted cases, {joined.height} joined to an "
            f"observation, {int((~finite).sum())} dropped (non-finite quantile or observation), "
            f"{late} at or after the held-out date",
            flush=True,
        )
    frame = joined.select(*common.KEYS, "seed", "inside").with_columns(
        qs=pl.Series(common.pinball_quantile_score(quantiles=quantiles, actual=observed) * 100.0),
        mae=pl.Series(np.abs(quantiles[:, common.MEDIAN] - observed) * 100.0),
    )
    return frame.filter(pl.Series(finite) & (pl.col("valid_time") < common.HOLDOUT)).with_columns(
        pl.col("seed").cast(pl.Int32)
    )


def source_frames(
    *, root: Path, source: common.SourceType, series_ids: list[int], actual: pl.DataFrame
) -> dict[str, pl.DataFrame]:
    """Return each arm's per-case losses for one source: the arms, the pooled draws, each draw."""
    frames: dict[str, pl.DataFrame] = {}
    for arm in common.arm_names():
        pieces = []
        for seed in common.arm_seeds(arm=arm):
            if seed not in {0, 1, 2}:
                continue
            probe = common.fit_path(
                source=source, arm="none", seed=seed, series_id=series_ids[0], root=root
            )
            if not probe.exists():
                continue
            predictions = pl.concat(
                [
                    read_arm_file(root=root, source=source, arm=arm, seed=seed, series_id=i)
                    for i in series_ids
                ]
            )
            if arm == "none":
                check_case_counts(source=source, predictions=predictions, series_ids=series_ids)
            label = f"{source} none seed {seed}" if arm == "none" else None
            pieces.append(case_losses(predictions=predictions, actual=actual, label=label))
        if pieces:
            frames[arm] = pl.concat(pieces)
    for label_set in common.LABEL_SETS:
        draws = [frames[a] for a in frames if a.startswith(f"rand_{label_set}_d")]
        if draws:
            frames[f"rand_{label_set}"] = (
                pl.concat(draws)
                .group_by(*common.KEYS, "seed", "inside")
                .agg(pl.col("qs", "mae").mean())
            )
    return frames


def baseline_frames(*, actual: pl.DataFrame, seeds: list[int]) -> dict[str, pl.DataFrame]:
    """Return the stored baselines' per-case losses, repeated once per seed."""
    frames = {}
    for name in BASELINES:
        stored = pl.read_parquet(common.ARMS_DIR_00013 / f"{name}.parquet").select(
            *common.KEYS, *common.LEVEL_COLUMNS
        )
        scored = case_losses(predictions=stored.with_columns(seed=pl.lit(0)), actual=actual).drop(
            "seed"
        )
        frames[name] = pl.concat(
            [scored.with_columns(seed=pl.lit(s, dtype=pl.Int32)) for s in seeds]
        )
    return frames


def shared_rows(*, frames: dict[str, pl.DataFrame]) -> dict[str, pl.DataFrame]:
    """Keep only the cases every arm holds, for every seed each arm has."""
    keys = None
    for frame in frames.values():
        cases = frame.select(*common.KEYS).unique()
        keys = cases if keys is None else keys.join(cases, on=list(common.KEYS), how="inner")
    if keys is None:
        return frames
    return {a: f.join(keys, on=list(common.KEYS), how="semi") for a, f in frames.items()}


def scoped(*, frame: pl.DataFrame, series_ids: list[int], inside: bool | None) -> pl.DataFrame:
    """Restrict an arm's losses to some series, and to outside or inside switching (None: both)."""
    part = frame.filter(pl.col("time_series_id").is_in(series_ids))
    return part if inside is None else part.filter(pl.col("inside") == inside)


def bootstrap_losses(*, arms: dict[str, pl.DataFrame], seeds: set[int]) -> pl.DataFrame:
    """Return `studies.bootstrap`'s loss frame for arms restricted to the seeds they share."""
    restricted = {
        a: f.filter(pl.col("seed").is_in(sorted(seeds))).select(*common.KEYS, "seed", "qs", "mae")
        for a, f in arms.items()
    }
    return experiment_store.losses_for_bootstrap(arm_losses=restricted)


def contrast_rows(
    *,
    source: str,
    contrast_id: str,
    frames: dict[str, pl.DataFrame],
    treatment: str,
    reference: str,
    series_ids: list[int],
    scope: str,
    seeds: set[int],
) -> list[dict[str, Any]]:
    """Return one contrast's rows: the quantile score at two levels, and MAE at 95%."""
    pair = {
        a: scoped(frame=frames[a], series_ids=series_ids, inside=False)
        for a in (treatment, reference)
    }
    shared = set.intersection(*(set(f["seed"].unique().to_list()) for f in pair.values())) & seeds
    losses = bootstrap_losses(arms=pair, seeds=shared)
    rows = []
    for metric, label, level in (
        ("qs", "planned", PLANNED_LEVEL),
        ("qs", "95", 95.0),
        ("mae", "95", 95.0),
    ):
        differences, _months = bootstrap.paired_differences(
            losses=losses, treatment=treatment, reference=reference, metric=metric
        )
        lower, upper = bootstrap.bootstrap_difference_at_level(
            losses=losses,
            treatment=treatment,
            reference=reference,
            metric=metric,
            level=level,
            n_resamples=N_RESAMPLES,
        )
        rows.append(
            {
                "source": source,
                "id": contrast_id,
                "scope": scope,
                "treatment": treatment,
                "reference": reference,
                "metric": metric,
                "level_label": label,
                "level": level,
                "difference": float(differences.mean()),
                "lower": lower,
                "upper": upper,
                "cases_per_seed": int(differences.shape[1]),
                "seeds": len(shared),
            }
        )
    return rows


def fmt(value: float, digits: int = 3) -> str:
    """Format a number with a fixed number of decimals."""
    return f"{value:.{digits}f}"


def markdown(*, frame: pl.DataFrame, digits: int = 3) -> list[str]:
    """Return a frame as markdown table lines."""
    lines = ["| " + " | ".join(frame.columns) + " |", "|" + "---|" * len(frame.columns)]
    for record in frame.iter_rows():
        cells = [fmt(v, digits) if isinstance(v, float) else str(v) for v in record]
        lines.append("| " + " | ".join(cells) + " |")
    return lines


def absolute_rows(
    *,
    source: str,
    frames: dict[str, pl.DataFrame],
    scopes: Mapping[str, tuple[list[int], bool | None]],
) -> list[dict[str, Any]]:
    """Return every arm's mean quantile score and MAE, with a month-block interval on the score."""
    rows = []
    for scope, (ids, inside) in scopes.items():
        for arm, frame in frames.items():
            part = scoped(frame=frame, series_ids=ids, inside=inside)
            if part.is_empty():
                continue
            losses = bootstrap_losses(arms={arm: part}, seeds=set(part["seed"].unique().to_list()))
            interval = bootstrap.bootstrap_absolute(losses=losses, arm=arm, metric="qs")
            rows.append(
                {
                    "source": source,
                    "scope": scope,
                    "arm": arm,
                    "qs": interval["value"],
                    "qs_lower": interval["lower_95"],
                    "qs_upper": interval["upper_95"],
                    "mae": float(part["mae"].to_numpy().mean()),
                    "cases_per_seed": part.height // max(1, part["seed"].n_unique()),
                }
            )
    return rows


def per_draw_rows(
    *, source: str, frames: dict[str, pl.DataFrame], affected: Mapping[str, list[int]]
) -> list[dict[str, Any]]:
    """Return the real-minus-random difference for each random draw (its own model seed)."""
    rows = []
    for label_set in common.LABEL_SETS:
        for draw in range(common.N_RANDOM_DRAWS):
            arm = f"rand_{label_set}_d{draw}"
            if arm not in frames or not affected[label_set]:
                continue
            seed = draw % len(common.SEEDS)
            result = contrast_rows(
                source=source,
                contrast_id=f"{'C2' if label_set == 'trusted' else 'C4'}_d{draw}",
                frames=frames,
                treatment=label_set,
                reference=arm,
                series_ids=affected[label_set],
                scope=f"{label_set}-affected series, outside switching",
                seeds={seed},
            )
            rows += [r for r in result if r["metric"] == "qs" and r["level_label"] == "planned"]
    return rows


def series_gap_rows(
    *,
    source: str,
    frames: dict[str, pl.DataFrame],
    affected: Mapping[str, list[int]],
    sites: dict[int, str],
) -> pl.DataFrame:
    """Return the descriptive real-minus-random gap per affected series and scoring month."""
    pieces = []
    for label_set in common.LABEL_SETS:
        treatment, reference = frames[label_set], frames[f"rand_{label_set}"]
        for series_id in affected[label_set]:
            both = (
                scoped(frame=treatment, series_ids=[series_id], inside=False)
                .select(*common.KEYS, "seed", treatment=pl.col("qs"))
                .join(
                    scoped(frame=reference, series_ids=[series_id], inside=False).select(
                        *common.KEYS, "seed", reference=pl.col("qs")
                    ),
                    on=[*common.KEYS, "seed"],
                    how="inner",
                )
            )
            pieces.append(
                both.with_columns(
                    source=pl.lit(source),
                    label_set=pl.lit(label_set),
                    site=pl.lit(sites[series_id]),
                    month=pl.col("valid_time").dt.strftime("%Y-%m"),
                    gap=pl.col("treatment") - pl.col("reference"),
                ).select("source", "label_set", "site", "time_series_id", "month", "gap")
            )
    return pl.concat(pieces) if pieces else pl.DataFrame()


def event_month_rows(*, gaps: pl.DataFrame, sites: dict[int, str]) -> pl.DataFrame:
    """Return the mean gap in each month in which the series has a visible labelled event."""
    rows = []
    events = common.visible_events()
    for record in events.iter_rows(named=True):
        first = max(record["onset_utc"], common.VALIDATION_START)
        last = min(record["end_utc"], common.HOLDOUT)
        if last <= first:
            continue
        months = pl.datetime_range(
            first.replace(day=1, hour=0, minute=0), last, interval="1mo", eager=True
        )
        month_labels = [m.strftime("%Y-%m") for m in months]
        for series_id in record["series"]:
            if series_id not in sites:
                continue
            part = gaps.filter(
                (pl.col("time_series_id") == series_id) & pl.col("month").is_in(month_labels)
            )
            for (source, label_set), group in part.group_by("source", "label_set"):
                rows.append(
                    {
                        "card_id": record["card_id"],
                        "site": sites[series_id],
                        "months": ",".join(month_labels),
                        "source": source,
                        "label_set": label_set,
                        "gap": float(group["gap"].to_numpy().mean()),
                        "cases": group.height,
                    }
                )
    return pl.DataFrame(rows) if rows else pl.DataFrame()


def minimum_detectable_gap(*, frames: dict[str, pl.DataFrame], series_ids: list[int]) -> float:
    """Return the 95% half-width of a difference of two single-seed means of `none`'s score.

    It is `1.96 * sqrt(2) * s`, where `s` is the sample standard deviation across the model seeds
    of `none`'s mean quantile score outside switching on the given series. The paired bootstrap
    draws one seed per resample, so a gap smaller than this half-width cannot be told from seed
    noise by a design that fitted each arm once per seed.
    """
    part = scoped(frame=frames["none"], series_ids=series_ids, inside=False)
    per_seed = part.group_by("seed").agg(pl.col("qs").mean())["qs"].to_numpy()
    if per_seed.size < 2:
        return float("nan")
    return float(1.96 * np.sqrt(2.0) * per_seed.std(ddof=1))


def verdict(*, contrasts: pl.DataFrame, trusted_events: Mapping[str, int]) -> list[str]:
    """Apply the decision rule to C2 of the ERA5-trained source (C4 if it has too few events)."""
    source = "era5"
    contrast_id = "C2" if trusted_events.get(source, 0) >= MIN_TRUSTED_EVENTS else "C4"
    row = contrasts.filter(
        (pl.col("source") == source)
        & (pl.col("id") == contrast_id)
        & (pl.col("metric") == "qs")
        & (pl.col("level_label") == "planned")
    )
    if row.is_empty():
        return [f"No verdict: {contrast_id} of the {source} source was not computed."]
    r = row.row(0, named=True)
    if r["upper"] < -THRESHOLD:
        word = "supported"
    elif r["upper"] < 0:
        word = "small gain"
    elif r["lower"] > 0:
        word = "refuted"
    else:
        word = "inconclusive"
    return [
        (
            f"Verdict by the decision rule, read on {contrast_id} of the ERA5-trained source: "
            f"**{word}** "
            f"(difference {fmt(r['difference'])}, {fmt(r['level'], 2)}% interval "
            f"{fmt(r['lower'])} to {fmt(r['upper'])}; threshold -{THRESHOLD:.2f})."
        ),
    ]


def figures(
    *,
    chart_dir: Path,
    contrasts: pl.DataFrame,
    absolute: pl.DataFrame,
    draws: pl.DataFrame,
    gaps: pl.DataFrame,
) -> None:
    """Draw the four figures from the saved tables (demand series only, labelled D1 to D22)."""
    chart_dir.mkdir(parents=True, exist_ok=True)
    colours = {"era5": ocf.BRAND_ORANGE, "ens": ocf.DATA_BLUE}
    scope = "Demand series, validation year July 2025 to June 2026, outside labelled switching."

    named = (
        pl.col("id")
        + pl.lit(": ")
        + pl.col("treatment").replace_strict(ARM_LABELS, default=pl.col("treatment"))
        + pl.lit(" minus ")
        + pl.col("reference").replace_strict(ARM_LABELS, default=pl.col("reference"))
    )
    primary = contrasts.filter(
        (pl.col("metric") == "qs")
        & (pl.col("id").str.len_chars() == 2)
        & (
            (pl.col("id").is_in(["C2", "C4"]) & (pl.col("level_label") == "planned"))
            | (pl.col("id").is_in(["C1", "C3"]) & (pl.col("level_label") == "95"))
        )
    ).with_columns(label=named)
    base = alt.Chart(primary).encode(
        y=alt.Y("label:N", sort=None, title=None, axis=alt.Axis(labelLimit=420)),
        color=alt.Color(
            "source:N",
            scale=alt.Scale(domain=list(colours), range=list(colours.values())),
            legend=alt.Legend(
                title=None,
                orient="top",
                labelExpr=(
                    "datum.value == 'era5' ? 'Trained on ERA5 weather' "
                    ": 'Trained on ENS control member'"
                ),
            ),
        ),
        yOffset="source:N",
    )
    zero = alt.Chart(pl.DataFrame({"x": [0.0]})).mark_rule(color="#999999").encode(x="x:Q")  # ty: ignore[unresolved-attribute]
    figure_one = alt.layer(
        zero,
        base.mark_rule().encode(x="lower:Q", x2="upper:Q"),  # ty: ignore[unresolved-attribute]
        base.mark_point(filled=True, size=70).encode(  # ty: ignore[unresolved-attribute]
            x=alt.X(
                "difference:Q",
                title="Difference in quantile score (points of capacity; negative is better)",
            )
        ),
    ).properties(
        width=520,
        height=200,
        title=alt.TitleParams(
            "Removing labelled switching from training, against removing random half-hours",
            subtitle=[
                scope,
                (
                    "C2 and C4 are planned (98.75% interval). C1 and C3 are against no "
                    "exclusion (95% interval)."
                ),
                "Dot: estimate. Line: interval from re-drawing whole months and seeds.",
            ],
            anchor="start",
            fontSize=14,
            subtitleFontSize=11,
        ),
    )
    figure_one.save(chart_dir / "figure_1_contrasts.svg")

    shown = absolute.filter(pl.col("scope") == "all visible-affected series, outside").with_columns(
        label=pl.col("arm").replace_strict(ARM_LABELS, default=pl.col("arm"))
    )
    order = list(ARM_LABELS.values())
    leaderboard = alt.layer(
        alt.Chart(shown)
        .mark_rule()
        .encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y("label:N", sort=order, title=None, axis=alt.Axis(labelLimit=300)),
            x="qs_lower:Q",
            x2="qs_upper:Q",
            color="source:N",
            yOffset="source:N",
        ),
        alt.Chart(shown)
        .mark_point(filled=True, size=60)
        .encode(  # ty: ignore[unresolved-attribute]
            y=alt.Y("label:N", sort=order, title=None),
            x=alt.X(
                "qs:Q",
                title="Quantile score (points of capacity; smaller is better)",
                scale=alt.Scale(zero=False),
            ),
            color=alt.Color(
                "source:N",
                scale=alt.Scale(domain=list(colours), range=list(colours.values())),
                legend=alt.Legend(title=None, orient="top"),
            ),
            yOffset="source:N",
        ),
    ).properties(
        width=520,
        height=260,
        title=alt.TitleParams(
            "Quantile score of every arm on the series the all-visible exclusion changes",
            subtitle=[
                scope,
                "Dot: mean over seeds. Line: 95% interval from re-drawing whole months.",
            ],
            anchor="start",
            fontSize=14,
            subtitleFontSize=11,
        ),
    )
    leaderboard.save(chart_dir / "figure_2_absolute.svg")

    if not draws.is_empty():
        draw_data = draws.with_columns(
            draw=pl.col("id").str.extract(r"_d(\d)", 1).cast(pl.Int32),
            family=pl.col("id").str.slice(0, 2),
        )
        draw_chart = (
            alt.layer(
                alt.Chart().mark_rule(color="#999999").encode(x=alt.datum(0)),  # ty: ignore[unresolved-attribute]
                alt.Chart()
                .mark_rule()
                .encode(  # ty: ignore[unresolved-attribute]
                    y=alt.Y("draw:O", title="Random draw"),
                    x="lower:Q",
                    x2="upper:Q",
                    color="source:N",
                ),
                alt.Chart()
                .mark_point(filled=True, size=60)
                .encode(  # ty: ignore[unresolved-attribute]
                    y=alt.Y("draw:O", title="Random draw"),
                    x=alt.X(
                        "difference:Q",
                        title="Real minus one random draw (points of capacity; negative is better)",
                    ),
                    color=alt.Color(
                        "source:N",
                        scale=alt.Scale(domain=list(colours), range=list(colours.values())),
                        legend=alt.Legend(title=None, orient="top"),
                    ),
                ),
            )
            .properties(width=520, height=130)
            .facet(row=alt.Row("family:N", title=None), data=draw_data)
            .properties(
                title=alt.TitleParams(
                    "The gap against each of the six random draws",
                    subtitle=[
                        scope,
                        "C2 (trusted), C4 (all visible); one seed per draw. Line: 98.75% interval.",
                    ],
                    anchor="start",
                    fontSize=14,
                    subtitleFontSize=11,
                )
            )
        )
        draw_chart.save(chart_dir / "figure_3_per_draw.svg")

    if not gaps.is_empty():
        per_series = (
            gaps.group_by("source", "label_set", "site").agg(gap=pl.col("gap").mean()).sort("site")
        )
        series_chart = (
            alt.Chart(per_series)  # ty: ignore[unresolved-attribute]
            .mark_point(filled=True, size=60)
            .encode(
                y=alt.Y("site:N", title="Series", sort=None),
                x=alt.X(
                    "gap:Q", title="Real minus random (points of capacity; negative is better)"
                ),
                color=alt.Color(
                    "source:N",
                    scale=alt.Scale(domain=list(colours), range=list(colours.values())),
                    legend=alt.Legend(title=None, orient="top"),
                ),
                row=alt.Row("label_set:N", title=None),
            )
            .properties(
                width=520,
                height=160,
                title=alt.TitleParams(
                    "The gap is not spread evenly over the affected series",
                    subtitle=[
                        scope,
                        "Mean over cases and seeds, with no interval: a descriptive view.",
                    ],
                    anchor="start",
                    fontSize=14,
                    subtitleFontSize=11,
                ),
            )
        )
        series_chart.save(chart_dir / "figure_4_per_series.svg")


class SourceResult(TypedDict):
    """Everything one training source contributes to the report."""

    contrasts: list[dict[str, Any]]
    absolute: list[dict[str, Any]]
    draws: list[dict[str, Any]]
    gaps: pl.DataFrame
    notes: list[str]
    trusted_events: int


def threshold_notes(
    *,
    source: str,
    frames: dict[str, pl.DataFrame],
    affected: Mapping[str, list[int]],
    every: list[int],
) -> list[str]:
    """Return the minimum detectable gap and the pooled threshold per label set."""
    notes = []
    outside_all = scoped(frame=frames["none"], series_ids=every, inside=False).height
    for label_set in common.LABEL_SETS:
        if not affected[label_set]:
            continue
        gap = minimum_detectable_gap(frames=frames, series_ids=affected[label_set])
        outside_affected = scoped(
            frame=frames["none"], series_ids=affected[label_set], inside=False
        ).height
        notes.append(
            f"- {source}, {label_set}: minimum detectable gap {fmt(gap)} points on the "
            f"{len(affected[label_set])} affected series; the threshold -{THRESHOLD:.2f} is "
            f"-{fmt(THRESHOLD * outside_affected / outside_all)} on all {len(every)} series "
            "(pooled equivalent, case-weighted)."
        )
    return notes


def score_source(
    *,
    root: Path,
    source: common.SourceType,
    series_ids: list[int],
    actual: pl.DataFrame,
    baselines: dict[str, pl.DataFrame],
    sites: dict[int, str],
) -> SourceResult:
    """Score one training source: absolute scores, the four contrasts, draws, and gaps."""
    frames = shared_rows(
        frames={
            **source_frames(root=root, source=source, series_ids=series_ids, actual=actual),
            **baselines,
        }
    )
    affected: dict[str, list[int]] = {
        ls: [i for i in common.affected_series(source=source, label_set=ls) if i in series_ids]
        for ls in common.LABEL_SETS
    }
    counts = common.event_count_table().filter(pl.col("source") == source)
    scopes = {
        "trusted-affected series, outside": (affected["trusted"], False),
        "all visible-affected series, outside": (affected["allvis"], False),
        "all series, outside": (series_ids, False),
        "all visible-affected series, inside": (affected["allvis"], True),
    }
    result = SourceResult(
        contrasts=[],
        absolute=absolute_rows(source=source, frames=frames, scopes=scopes),
        draws=per_draw_rows(source=source, frames=frames, affected=affected),
        gaps=pl.DataFrame(),
        notes=threshold_notes(source=source, frames=frames, affected=affected, every=series_ids),
        trusted_events=int(counts.filter(pl.col("label_set") == "trusted")["events"][0]),
    )
    for contrast_id, (label_set, treatment, reference) in CONTRASTS.items():
        if not affected[label_set] or reference not in frames or treatment not in frames:
            result["notes"].append(f"{source} {contrast_id}: no affected series or arm; skipped.")
            continue
        for scope_ids, scope_name, suffix in (
            (affected[label_set], f"{label_set}-affected series, outside switching", ""),
            (series_ids, "all series, outside switching", "_pooled"),
        ):
            if suffix and contrast_id not in ("C2", "C4"):
                continue
            result["contrasts"] += contrast_rows(
                source=source,
                contrast_id=contrast_id + suffix,
                frames=frames,
                treatment=treatment,
                reference=reference,
                series_ids=scope_ids,
                scope=scope_name,
                seeds=set(common.SEEDS),
            )
    if all(f"rand_{ls}" in frames for ls in common.LABEL_SETS) and any(affected.values()):
        result["gaps"] = series_gap_rows(
            source=source, frames=frames, affected=affected, sites=sites
        )
    return result


def gap_lines(*, gaps: pl.DataFrame, sites: dict[int, str]) -> list[str]:
    """Return the descriptive per-series, per-month, and per-event gap tables."""
    by_series = (
        gaps.group_by("source", "label_set", "site")
        .agg(gap=pl.col("gap").mean(), cases=pl.len())
        .sort("source", "label_set", "site")
    )
    by_month = (
        gaps.group_by("source", "label_set", "month")
        .agg(gap=pl.col("gap").mean(), cases=pl.len())
        .sort("source", "label_set", "month")
    )
    lines = ["## Descriptive per-series gaps (real minus random, outside switching)", ""]
    lines += [*markdown(frame=by_series), "", "### By scoring month", ""]
    lines += [*markdown(frame=by_month), ""]
    per_event = event_month_rows(gaps=gaps, sites=sites)
    if not per_event.is_empty():
        ordered = per_event.sort("source", "label_set", "card_id")
        lines += ["### Gap in the scoring months of each labelled event", ""]
        lines += [*markdown(frame=ordered), ""]
    return lines


def draw_lines(*, draws: pl.DataFrame) -> list[str]:
    """Return the per-draw gap table and its spread across the six draws."""
    spread = draws.group_by("source", pl.col("id").str.slice(0, 2).alias("family")).agg(
        mean=pl.col("difference").mean(),
        sd=pl.col("difference").std(),
        low=pl.col("difference").min(),
        high=pl.col("difference").max(),
    )
    table = draws.select("source", "id", "difference", "lower", "upper")
    return [
        "## Per-draw gaps (one seed each, planned level)",
        "",
        *markdown(frame=table),
        "",
        "### Spread across the six draws",
        "",
        *markdown(frame=spread.sort("source", "family")),
        "",
    ]


def main() -> None:
    """Write and print the report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="read the scratch fits")
    parser.add_argument("--no-charts", action="store_true")
    args = parser.parse_args()
    root = common.WORK_DIR / ("fits_smoke" if args.smoke else "fits")
    results = RESULTS_DIR / ("smoke" if args.smoke else "")
    results.mkdir(parents=True, exist_ok=True)
    series_ids = present_series(root=root)
    sites = dict(common.series_table().select("time_series_id", "site").iter_rows())
    actual = actual_table(root=root, series_ids=series_ids)
    seeds = sorted({int(p.name.split("seed")[-1]) for p in (root / "era5").glob("none_seed*")})
    baselines = baseline_frames(actual=actual, seeds=seeds)
    by_source = {
        source: score_source(
            root=root,
            source=source,
            series_ids=series_ids,
            actual=actual,
            baselines=baselines,
            sites=sites,
        )
        for source in common.SOURCES
    }
    contrasts = pl.DataFrame([r for s in by_source.values() for r in s["contrasts"]])
    absolute = pl.DataFrame([r for s in by_source.values() for r in s["absolute"]])
    draws = pl.DataFrame([r for s in by_source.values() for r in s["draws"]])
    gaps = pl.concat([s["gaps"] for s in by_source.values() if not s["gaps"].is_empty()])
    absolute.write_parquet(results / "absolute.parquet")
    lines = ["# Experiment 00042: report", ""]
    lines += ["### Minimum detectable gap, the pooled equivalent of the threshold, and skips", ""]
    lines += [note for s in by_source.values() for note in s["notes"]]
    exclusion = common.WORK_DIR / "exclusion_report.md"
    if exclusion.exists():
        lines += ["", "## Exclusions (from `e00042_common.py`)", "", exclusion.read_text()]
    lines += ["## Absolute scores", "", *markdown(frame=absolute), ""]
    if not contrasts.is_empty():
        contrasts.write_parquet(results / "contrasts.parquet")
        table = contrasts.drop("level", "treatment", "reference")
        lines += ["## Planned and reported contrasts (points of capacity)", ""]
        lines += [*markdown(frame=table), ""]
        if not draws.is_empty():
            draws.write_parquet(results / "per_draw.parquet")
            lines += draw_lines(draws=draws)
        trusted_events: dict[str, int] = {s: r["trusted_events"] for s, r in by_source.items()}
        verdict_lines = verdict(contrasts=contrasts, trusted_events=trusted_events)
        lines += ["## Verdict", "", *verdict_lines, ""]
    if not gaps.is_empty():
        lines += gap_lines(gaps=gaps, sites=sites)
    text = "\n".join(lines)
    (results / "report.md").write_text(text + "\n")
    print(text)
    if not args.no_charts and not contrasts.is_empty():
        figures(
            chart_dir=CHART_DIR / ("smoke" if args.smoke else ""),
            contrasts=contrasts,
            absolute=absolute,
            draws=draws,
            gaps=gaps,
        )


if __name__ == "__main__":
    main()
