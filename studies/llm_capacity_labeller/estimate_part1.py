"""Part 1: tune and score HYST2, HM2, and CX2S, the estimators with the seasonal correction.

Part of the study in <https://github.com/openclimatefix/nged-substation-forecast/issues/1147>.

**Tuning uses the first study's tuning draw only (seed 11430) and a lexicographic rule:** the
setting with the lowest recovery error among those whose false movement on the 23 quiet
plant-months is at most `FALSE_MOVEMENT_GATE`. A weighted sum would mix a recovery error of about
0.02 to 0.05 with a 0-to-1 share, so the table also lists the choice a weight of 0 and a weight of
0.5 on the false-movement share would make. **CX2S is not tuned:** it runs the first study's tuned
CX2 setting on the corrected ratio, because the linear program is the expensive part.

**Scoring** is the first study's: `score_events` over the planted record's scale and the original
record's, and `false_movement_share` on the 23 quiet plant-months. The two fresh draws carry the
comparison; the first study's ten selection draws re-score HYST2 and HM2 only, and the page labels
that exploratory. S0 (the static power) is the do-nothing reference.

Outputs under `<root>/part1/`: `tuning.parquet`, `scores.parquet`, `summary.parquet`, `chosen.json`,
`cx2s_raw.npz` (the linear program's levels, resumable), and `scales_original.npz`.

Run it with `uv run python studies/llm_capacity_labeller/estimate_part1.py`.
"""

import argparse
import json
import logging
import sys
import warnings
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from labeller_paths import (
    FIRST_STUDY_BUILT,
    check_stamp,
    claim_root,
    fingerprint,
    load_first_events,
    load_first_record,
    load_fresh_draws,
    write_json_atomic,
)
from studies.capacity_convex import estimate_convex_plant
from studies.capacity_estimators import (
    CapacityRecord,
    IssueScale,
    RawEstimate,
    reference_day_indices,
    reference_levels,
)
from studies.capacity_recovery import (
    FALSE_MOVEMENT_GATE,
    PlantedEvent,
    draw_primary_metric,
    fallback_share_on_events,
    false_movement_days,
    false_movement_share,
    plant_losses,
    score_events,
)
from studies.capacity_seasonal import (
    corrected_record,
    cx2s_scale,
    hm2_scale,
    hyst2_ratio,
    hyst2_scale,
    seasonal_reference,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_LOG: Final[logging.Logger] = logging.getLogger("estimate_part1")

HYST2_GRID: Final[tuple[dict[str, float], ...]] = tuple(
    {"drop": drop, "persist_days": persist}
    for drop in (0.05, 0.075, 0.10, 0.15, 0.20)
    for persist in (3, 5, 7, 10, 14)
)
"""HYST2's settings: the share below the static power that counts as a drop, and the persistence."""

HM2_GRID: Final[tuple[dict[str, float], ...]] = tuple(
    {"stay_probability": stay, "observation_sd": sd}
    for stay in (0.9, 0.95, 0.98, 0.995)
    for sd in (0.10, 0.15, 0.20)
)
"""HM2's settings: the daily probability of staying at a level, and the observation spread."""

FALSE_MOVEMENT_WEIGHTS: Final[tuple[float, ...]] = (0.0, 0.5)
"""The weights on the false-movement share in the sensitivity of the tuning choice."""

CX2_TUNED_PATH: Final[Path] = FIRST_STUDY_BUILT.parent / "tuned" / "tuned_CX2.json"
"""The first study's tuned CX2 setting."""

SMOKE_TUNING_SETTINGS: Final[int] = 2
"""A smoke run tunes over this many settings of each estimator."""


@dataclass(frozen=True)
class RecordBundle:
    """The arrays of one record that the estimators share.

    Attributes:
        record: The record (planted or original).
        corrected: The record with the seasonal reference applied.
        ratio: HYST2's smoothed ratio over the original record's reference level.
    """

    record: CapacityRecord
    corrected: CapacityRecord
    ratio: np.ndarray


def bundle_of(
    *, record: CapacityRecord, kf_reference: np.ndarray | None
) -> tuple[RecordBundle, np.ndarray]:
    """Compute a record's corrected copy and HYST2's input.

    Args:
        record: The record.
        kf_reference: The original record's reference level of the smoothed ratio, or `None` to take
            it from this record (the original).

    Returns:
        The bundle, and the reference level used.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        corrected = corrected_record(record=record, reference=seasonal_reference(record=record))
        ratio, level = hyst2_ratio(corrected=corrected, reference_level=kf_reference)
    return RecordBundle(record=record, corrected=corrected, ratio=ratio), level


EstimatorFunction = Callable[[RecordBundle], IssueScale]
"""Maps a record bundle to the issue-time scale of one estimator at one setting."""


def static_scale(*, record: CapacityRecord) -> IssueScale:
    """Return the do-nothing scale: the static power on every day."""
    plants, days = len(record.sites), len(record.dates)
    return IssueScale(
        scale_mw=np.repeat(record.static_p99_mw[:, None], days, axis=1),
        relative_sd=np.full((plants, days), np.nan),
        fallback_reason=np.zeros((plants, days), dtype=np.int8),
    )


def estimator_function(*, name: str, params: Mapping[str, float]) -> EstimatorFunction:
    """Return the scale function of HYST2 or HM2 at one setting."""
    if name == "HYST2":
        return lambda bundle: hyst2_scale(
            record=bundle.record,
            ratio=bundle.ratio,
            drop=params["drop"],
            persist_days=int(params["persist_days"]),
        )
    return lambda bundle: hm2_scale(record=bundle.record, corrected=bundle.corrected, params=params)


def score_setting(
    *,
    function: EstimatorFunction,
    events: list[PlantedEvent],
    planted: RecordBundle,
    original: RecordBundle,
) -> pl.DataFrame:
    """Score one estimator setting on one draw's events."""
    return score_events(
        events=events,
        planted=function(planted),
        original=function(original),
        days=len(original.record.dates),
    )


def lexicographic_choice(*, table: pl.DataFrame) -> int:
    """Return the row index of the setting with the lowest error among those within the gate.

    Args:
        table: One row per setting with `tuning_error` and `false_movement`.

    Returns:
        The row index; if no setting is within the gate, the setting with the least false movement.
    """
    within = table.with_row_index("row").filter(pl.col("false_movement") <= FALSE_MOVEMENT_GATE)
    if within.height == 0:
        return int(table.with_row_index("row").sort("false_movement")["row"][0])
    return int(within.sort("tuning_error")["row"][0])


def weighted_choice(*, table: pl.DataFrame, weight: float) -> int:
    """Return the row index minimising `tuning_error + weight * false_movement`."""
    return int(
        table.with_row_index("row")
        .with_columns(total=pl.col("tuning_error") + weight * pl.col("false_movement"))
        .sort("total")["row"][0]
    )


def tune(
    *,
    name: str,
    grid: tuple[dict[str, float], ...],
    tuning_events: list[PlantedEvent],
    windows: list[tuple[int, int, int]],
    planted: RecordBundle,
    original: RecordBundle,
) -> tuple[pl.DataFrame, dict[str, float]]:
    """Tune one estimator on the tuning draw.

    Args:
        name: `HYST2` or `HM2`.
        grid: The settings.
        tuning_events: The tuning draw's events.
        windows: The quiet plant-months.
        planted: The tuning draw's planted record.
        original: The original record.

    Returns:
        The tuning table and the lexicographically chosen setting.
    """
    rows = []
    for params in grid:
        function = estimator_function(name=name, params=params)
        scores = score_setting(
            function=function, events=tuning_events, planted=planted, original=original
        )
        rows.append(
            {
                "estimator": name,
                "setting": json.dumps(params, sort_keys=True),
                "tuning_error": draw_primary_metric(scores=scores),
                "false_movement": false_movement_share(
                    windows=windows, original=function(original)
                ),
            }
        )
    table = pl.DataFrame(rows)
    chosen = lexicographic_choice(table=table)
    marks = {"chosen_lexicographic": chosen}
    marks |= {
        f"chosen_weight_{w}": weighted_choice(table=table, weight=w) for w in FALSE_MOVEMENT_WEIGHTS
    }
    table = table.with_row_index("row").with_columns(
        **{key: pl.col("row") == index for key, index in marks.items()}
    )
    return table.drop("row"), grid[chosen]


def cx2s_days(
    *,
    original: CapacityRecord,
    fresh: list[list[PlantedEvent]],
    windows: list[tuple[int, int, int]],
) -> dict[str, list[np.ndarray]]:
    """Return the estimate days the linear program needs on each record, per plant.

    Args:
        original: The original record.
        fresh: The two fresh draws' events.
        windows: The quiet plant-months.

    Returns:
        For `original` and each `draw01`, `draw02`, one sorted array of day indices per plant: the
        events' scored target days minus 2, and for the original also the false-movement days and
        the reference year's calibration days.
    """
    plants = len(original.sites)

    def event_days(events: list[PlantedEvent]) -> list[set[int]]:
        out: list[set[int]] = [set() for _ in range(plants)]
        for event in events:
            start, stop = event.scored_days
            out[event.plant] |= set(range(start - 2, stop - 2))
        return out

    per_draw = [event_days(events) for events in fresh]
    original_days = [set().union(*(draw[p] for draw in per_draw)) for p in range(plants)]
    for (plant, _, _), days in zip(windows, false_movement_days(windows=windows), strict=True):
        original_days[plant] |= set(days.tolist())
    reference = set(reference_day_indices(record=original).tolist())
    return {
        "original": [np.array(sorted(d | reference)) for d in original_days],
        "draw01": [np.array(sorted(d)) for d in per_draw[0]],
        "draw02": [np.array(sorted(d)) for d in per_draw[1]],
    }


def cx2s_raw(
    *, bundle: RecordBundle, days: list[np.ndarray], params: dict[str, float]
) -> RawEstimate:
    """Run CX2 on a corrected record at each plant's days."""
    parts = [
        estimate_convex_plant(
            name="CX2", record=bundle.corrected, plant=plant, params=params, days=days[plant]
        )
        for plant in range(len(bundle.record.sites))
    ]
    return RawEstimate(level=np.stack([part.level for part in parts]), sd=None)


def summarise(*, scores: pl.DataFrame) -> pl.DataFrame:
    """Average the event scores of each estimator over the fresh draws and over the old draws."""
    return (
        scores.group_by("estimator", "group")
        .agg(
            mean_error=pl.col("error").mean(),
            events=pl.len(),
            mean_step_delay_days=pl.col("delay").mean(),
        )
        .sort("group", "estimator")
    )


def main() -> int:  # noqa: PLR0915
    """Tune and score Part 1."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="Few settings, under _smoke/.")
    arguments = parser.parse_args()
    root = claim_root(smoke=arguments.smoke)
    part1 = root / "part1"
    check_stamp(directory=part1, stamp=fingerprint(extra_files=(Path(__file__),)))
    first = load_first_events()
    original_record = load_first_record()
    windows = first.false_movement_windows
    original, kf_reference = bundle_of(record=original_record, kf_reference=None)
    tuning = bundle_of(
        record=plant_losses(record=original_record, events=first.tuning_draw),
        kf_reference=kf_reference,
    )[0]
    grids = {"HYST2": HYST2_GRID, "HM2": HM2_GRID}
    if arguments.smoke:
        grids = {name: grid[:SMOKE_TUNING_SETTINGS] for name, grid in grids.items()}
    tables, chosen = [], {}
    for name, grid in grids.items():
        table, chosen[name] = tune(
            name=name,
            grid=grid,
            tuning_events=first.tuning_draw,
            windows=windows,
            planted=tuning,
            original=original,
        )
        tables.append(table)
        _LOG.info("%s chose %s", name, chosen[name])
    pl.concat(tables).write_parquet(part1 / "tuning.parquet")
    cx2_params = json.loads(CX2_TUNED_PATH.read_text())["params"]
    chosen["CX2S"] = cx2_params
    write_json_atomic(path=part1 / "chosen.json", payload=chosen)
    fresh = load_fresh_draws(root=root)
    groups: list[tuple[str, str, list[PlantedEvent]]] = [
        ("fresh", "draw01", fresh[0]),
        ("fresh", "draw02", fresh[1]),
        *(
            ("old", f"old{index:02d}", events)
            for index, events in enumerate(first.draws[1:], start=1)
        ),
    ]
    bundles = {
        label: bundle_of(
            record=plant_losses(record=original_record, events=events), kf_reference=kf_reference
        )[0]
        for _, label, events in groups
    }
    functions: dict[str, EstimatorFunction] = {
        "S0": lambda bundle: static_scale(record=bundle.record),
        **{name: estimator_function(name=name, params=chosen[name]) for name in grids},
    }
    cx2_path = part1 / "cx2s_raw.npz"
    days = cx2s_days(original=original_record, fresh=fresh, windows=windows)
    if not cx2_path.exists():
        raws = {
            "original": cx2s_raw(bundle=original, days=days["original"], params=cx2_params),
            "draw01": cx2s_raw(bundle=bundles["draw01"], days=days["draw01"], params=cx2_params),
            "draw02": cx2s_raw(bundle=bundles["draw02"], days=days["draw02"], params=cx2_params),
        }
        np.savez_compressed(cx2_path, **{key: raw.level for key, raw in raws.items()})  # ty: ignore[invalid-argument-type]
        _LOG.info("CX2S levels written")
    with np.load(cx2_path) as stored:
        cx2_levels = {key: stored[key] for key in stored.files}
    reference = reference_levels(
        raw=RawEstimate(level=cx2_levels["original"], sd=None), record=original.corrected
    )
    cx2_scales = {
        key: cx2s_scale(
            record=(original if key == "original" else bundles[key]).record,
            raw=RawEstimate(level=level, sd=None),
            reference=reference,
        )
        for key, level in cx2_levels.items()
    }
    score_rows: list[pl.DataFrame] = []
    fallback: dict[str, float] = {}
    for estimator, function in functions.items():
        for group, label, events in groups:
            scores = score_setting(
                function=function, events=events, planted=bundles[label], original=original
            )
            score_rows.append(
                scores.with_columns(
                    estimator=pl.lit(estimator), group=pl.lit(group), draw=pl.lit(label)
                )
            )
    for group, label, events in groups[:2]:
        scores = score_events(
            events=events,
            planted=cx2_scales[label],
            original=cx2_scales["original"],
            days=len(original_record.dates),
        )
        score_rows.append(
            scores.with_columns(estimator=pl.lit("CX2S"), group=pl.lit(group), draw=pl.lit(label))
        )
    all_scores = pl.concat(score_rows)
    all_scores.write_parquet(part1 / "scores.parquet")
    original_scales = {name: function(original) for name, function in functions.items()}
    original_scales["CX2S"] = cx2_scales["original"]
    summary_rows = []
    for estimator, scale in original_scales.items():
        fresh_scores = all_scores.filter(
            (pl.col("estimator") == estimator) & (pl.col("group") == "fresh")
        )
        fallback[estimator] = float(
            np.mean(
                [
                    fallback_share_on_events(
                        events=events,
                        planted=(
                            cx2_scales[label]
                            if estimator == "CX2S"
                            else functions[estimator](bundles[label])
                        ),
                    )
                    for _, label, events in groups[:2]
                ]
            )
        )
        summary_rows.append(
            {
                "estimator": estimator,
                "fresh_mean_error": float(fresh_scores["error"].mean()),  # ty: ignore[invalid-argument-type]
                "false_movement": false_movement_share(windows=windows, original=scale),
                "within_gate": false_movement_share(windows=windows, original=scale)
                <= FALSE_MOVEMENT_GATE,
                "fallback_share_on_event_days": fallback[estimator],
                "mean_step_delay_days": fresh_scores.filter(pl.col("shape") == "step")[
                    "delay"
                ].mean(),
            }
        )
    pl.DataFrame(summary_rows).write_parquet(part1 / "summary.parquet")
    summarise(scores=all_scores).write_parquet(part1 / "summary_by_group.parquet")
    for label in ("draw01", "draw02"):
        planted_scales = {
            name: function(bundles[label]).scale_mw for name, function in functions.items()
        }
        planted_scales["CX2S"] = cx2_scales[label].scale_mw
        np.savez_compressed(part1 / f"scales_{label}.npz", **planted_scales)  # ty: ignore[invalid-argument-type]
    np.savez_compressed(
        part1 / "scales_original.npz",
        **{name: scale.scale_mw for name, scale in original_scales.items()},  # ty: ignore[invalid-argument-type]
    )
    _LOG.info("wrote %s", part1)
    return 0


if __name__ == "__main__":
    sys.exit(main())
