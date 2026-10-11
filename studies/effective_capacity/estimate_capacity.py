"""Run the capacity estimators (all but DP1) on the original record and on every planted record.

Part of the study in <https://github.com/openclimatefix/nged-substation-forecast/issues/1143>. The
estimates are the raw daily levels of `studies.capacity_estimators` and `studies.capacity_convex`:
the estimate for day `d` reads days up to `d` only. Scoring them is `score_capacity_recovery.py`.

**Phases**, run in this order (each is resumable: a finished file is never recomputed):

- `tuning`: every setting of every estimator on the tuning draw (`draw00`) and on the original
  record at the days the tuning draw needs, so that `score_capacity_recovery.py --phase tuning` can
  choose a setting.
- `main`: the chosen setting of every estimator on the original record at every day any scoring
  needs, on draws 1 to 10, and on the positive control.
- `hindsight`: the shortlisted estimators on the time-reversed original and control records and on
  the ANM-blind original record, for the hindsight reference and the ANM dips.

**Checks that raise** before the first estimate: a recomputed NR that differs from the lag frame's
`rp_7d`; an estimator whose scale for a target day changes when every input from its issue day
onward is replaced by noise (a causal probe); and, inside a linear program, a status other than
optimal.

`--smoke` truncates each plant's day list to 12 days and writes under `_smoke/`. It is not a result.
The CPU worker count is `--workers`. Run it with
`uv run python studies/effective_capacity/estimate_capacity.py --phase tuning --workers 8`.
"""

import argparse
import json
import logging
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from capacity_paths import (
    CONTROL,
    ESTIMATOR_MODULES,
    LAG_FRAME_PATH,
    ORIGINAL,
    TUNING_GRIDS,
    check_stamp,
    claim_root,
    draw_id,
    estimate_path,
    estimates_dir,
    fingerprint,
    load_events,
    load_tuned,
    record_for,
    save_estimate,
    scored_days,
    setting_name,
)
from studies.capacity_convex import estimate_convex_plant
from studies.capacity_estimators import (
    CALIBRATED_ESTIMATORS,
    ESTIMATOR_FAMILY,
    ISSUE_LAG_DAYS,
    CapacityRecord,
    RawEstimate,
    assert_scale_ignores_the_issue_day_onward,
    issue_time_estimates,
    neighbour_ratio,
    reference_day_indices,
    reference_levels,
    run_cheap_estimator,
    target_day_indices,
)
from studies.capacity_recovery import false_movement_days

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
_LOG: Final[logging.Logger] = logging.getLogger("estimate_capacity")

SENSITIVITY_LAG_DAYS: Final[int] = 3
"""The T-3 sensitivity reads an estimate one day older than the primary."""

SMOKE_DAYS_PER_PLANT: Final[int] = 12
"""A smoke run estimates this many days per plant."""

CONVEX: Final[tuple[str, ...]] = ("CX1", "CX2")
CHEAP: Final[tuple[str, ...]] = tuple(
    name for name, family in ESTIMATOR_FAMILY.items() if family in ("cheap", "wacky", "reference")
)
"""The estimators this script runs: the linear programs, and the cheap and wacky ones."""

NR_TOLERANCE: Final[float] = 1e-4
"""NR must equal the lag frame's `rp_7d` to this relative tolerance (the frame sums in Float32)."""


def _event_days(*, events: list, plants: int) -> list[np.ndarray]:
    """Return, per plant, the estimate days its events' scored target days read (T-2 and T-3)."""
    per_plant: list[list[np.ndarray]] = [[] for _ in range(plants)]
    for event in events:
        start, stop = event.scored_days
        per_plant[event.plant].append(
            np.arange(start - SENSITIVITY_LAG_DAYS, stop - ISSUE_LAG_DAYS)
        )
    return [np.unique(np.concatenate(p)) if p else np.array([], dtype=int) for p in per_plant]


def days_for(*, root: Path, record_id: str, record: CapacityRecord, phase: str) -> list[np.ndarray]:
    """Return the estimate days each plant needs on a record.

    Args:
        root: The output root.
        record_id: The record's identifier.
        record: The record.
        phase: The phase, which sets how much of the original record is needed.

    Returns:
        One sorted array of day indices per plant.
    """
    events = load_events(root=root)
    plants = len(record.sites)
    target_days = scored_days(record=record, events=events)
    stage_b = np.unique(target_days - ISSUE_LAG_DAYS)
    base = record_id.removesuffix("__reversed")
    if base.startswith("draw"):
        return _event_days(events=events.draws[int(base[4:])], plants=plants)
    if base == CONTROL and not record_id.endswith("__reversed"):
        return [stage_b] * plants
    if base == ORIGINAL or record_id.startswith("original"):
        if phase == "tuning":
            groups = [_event_days(events=events.draws[0], plants=plants)]
        else:
            groups = [_event_days(events=draw, plants=plants) for draw in events.draws]
            windows = events.false_movement_windows
            groups.append(
                [
                    np.unique(
                        np.concatenate(
                            [
                                d
                                for (plant, _, _), d in zip(
                                    windows, false_movement_days(windows=windows), strict=True
                                )
                                if plant == p
                            ]
                            or [np.array([], dtype=int)]
                        )
                    )
                    for p in range(plants)
                ]
            )
            groups.append([stage_b] * plants)
        reference = reference_day_indices(record=record)
        return [
            np.unique(np.concatenate([g[p] for g in groups] + [reference])) for p in range(plants)
        ]
    return [np.array([], dtype=int)] * plants


def hindsight_days(
    *, record: CapacityRecord, root: Path, reversed_record: bool
) -> list[np.ndarray]:
    """Return the days the hindsight reference reads on the forward or the reversed record.

    Args:
        record: The (forward) record.
        root: The output root.
        reversed_record: Whether to give the reversed record's day indices.

    Returns:
        One array of day indices per plant (the same for every plant).
    """
    events = load_events(root=root)
    target = scored_days(record=record, events=events)
    total = len(record.dates)
    firsts = np.unique((target // 7) * 7)
    forward = firsts - 1
    backward = np.minimum(firsts + 7, total - 1)
    days = (total - 1 - backward) if reversed_record else forward
    days = np.unique(days[(days >= 0) & (days < total)])
    return [days] * len(record.sites)


def _truncate(*, days: list[np.ndarray], smoke: bool) -> list[np.ndarray]:
    """Keep only the first few days per plant in a smoke run."""
    return [d[:SMOKE_DAYS_PER_PLANT] for d in days] if smoke else days


def _convex_part(
    *,
    root_text: str,
    name: str,
    params: dict[str, float],
    record_id: str,
    plant: int,
    days: np.ndarray,
    part: str,
) -> str:
    """Worker: estimate one plant's days with a linear program and save the part."""
    root = Path(root_text)
    record = record_for(root=root, record_id=record_id)
    estimate = estimate_convex_plant(
        name=name, record=record, plant=plant, params=params, days=days
    )
    np.save(part, estimate.level)
    return part


def run_one(
    *,
    root: Path,
    name: str,
    params: dict[str, float],
    file_id: str,
    record_id: str,
    days: list[np.ndarray],
    pool: ProcessPoolExecutor | None,
) -> None:
    """Estimate one (estimator, setting, record) and save it unless it exists.

    Args:
        root: The output root.
        name: The estimator.
        params: Its settings.
        file_id: The name the estimates are saved under.
        record_id: The record the estimator runs on.
        days: The days each plant needs (used by the linear programs only).
        pool: The worker pool for the linear programs.
    """
    path = estimate_path(
        root=root, estimator=name, setting=setting_name(params=params), record_id=file_id
    )
    if path.exists():
        return
    record = record_for(root=root, record_id=record_id)
    if name in CONVEX:
        assert pool is not None
        parts = [
            path.with_name(f"{path.stem}__part{plant}.npy") for plant in range(len(record.sites))
        ]
        path.parent.mkdir(parents=True, exist_ok=True)
        futures = [
            pool.submit(
                _convex_part,
                root_text=str(root),
                name=name,
                params=params,
                record_id=record_id,
                plant=plant,
                days=days[plant],
                part=str(part),
            )
            for plant, part in enumerate(parts)
            if not part.exists() and len(days[plant])
        ]
        for future in futures:
            future.result()
        level = np.stack(
            [
                np.load(part) if part.exists() else np.full(len(record.dates), np.nan)
                for part in parts
            ]
        )
        save_estimate(path=path, level=level, sd=None)
        for part in parts:
            part.unlink(missing_ok=True)
    else:
        raw = run_cheap_estimator(name=name, record=record, params=params)
        save_estimate(path=path, level=raw.level, sd=raw.sd)
    _LOG.info("saved %s", path.relative_to(root))


def check_neighbour_ratio_matches_lag_frame(*, record: CapacityRecord) -> None:
    """Raise unless the recomputed NR equals the lag frame's `rp_7d` on every row.

    The lag frame's `rp_7d` for a row is the neighbour ratio at the latest whole day, two days
    before the row's target day. Agreement on all rows also checks the causal driver's two-day lag.

    Args:
        record: The original record.

    Raises:
        ValueError: If a value or a null differs.
    """
    frame = pl.read_parquet(LAG_FRAME_PATH, columns=["site", "time", "rp_7d"])
    plants, days = target_day_indices(record=record, rows=frame)
    mine = neighbour_ratio(record=record)[plants, days - ISSUE_LAG_DAYS]
    theirs = frame["rp_7d"].cast(pl.Float64).fill_null(np.nan).to_numpy()
    if not np.array_equal(np.isnan(mine), np.isnan(theirs)):
        differing = int((np.isnan(mine) != np.isnan(theirs)).sum())
        msg = f"NR and rp_7d are null on different rows ({differing})"
        raise ValueError(msg)
    finite = ~np.isnan(mine)
    if not np.allclose(mine[finite], theirs[finite], rtol=NR_TOLERANCE, atol=1e-9):
        worst = float(np.max(np.abs(mine[finite] - theirs[finite]) / np.abs(theirs[finite])))
        msg = f"NR differs from rp_7d by up to {worst:.2e} relative"
        raise ValueError(msg)
    _LOG.info("NR equals rp_7d on %d rows", int(finite.sum()))


def probe_causality(*, root: Path, record: CapacityRecord) -> None:
    """Raise if any estimator's scale for a target day reads its issue day or later.

    Args:
        root: The output root, for the target days of the probe.
        record: The original record.

    Raises:
        ValueError: From `assert_scale_ignores_the_issue_day_onward`.
    """
    events = load_events(root=root)
    target = scored_days(record=record, events=events)
    probes = target[[40, 200, 400]]
    reference_raw = {name: run_cheap_estimator(name=name, record=record) for name in CHEAP}
    reference_convex = {
        name: RawEstimate(
            level=np.stack(
                [
                    estimate_convex_plant(
                        name=name,
                        record=record,
                        plant=plant,
                        params=TUNING_GRIDS[name][0],
                        days=reference_day_indices(record=record)[:12],
                    ).level
                    for plant in range(len(record.sites))
                ]
            ),
            sd=None,
        )
        for name in CONVEX
    }
    references = {
        name: reference_levels(raw=raw, record=record)
        for name, raw in {**reference_raw, **reference_convex}.items()
        if name in CALIBRATED_ESTIMATORS
    }
    for name in (*CHEAP, *CONVEX):
        params = TUNING_GRIDS[name][0]

        def scale_function(
            candidate: CapacityRecord, days: np.ndarray, name: str = name, params: dict = params
        ) -> np.ndarray:
            if name in CONVEX:
                level = np.stack(
                    [
                        estimate_convex_plant(
                            name=name,
                            record=candidate,
                            plant=plant,
                            params=params,
                            days=days - ISSUE_LAG_DAYS,
                        ).level
                        for plant in range(len(candidate.sites))
                    ]
                )
                raw = RawEstimate(level=level, sd=None)
            else:
                raw = run_cheap_estimator(name=name, record=candidate, params=params)
            return issue_time_estimates(
                record=candidate, raw=raw, reference=references.get(name)
            ).scale_mw[:, days]

        assert_scale_ignores_the_issue_day_onward(
            record=record, scale_function=scale_function, target_days=probes[:2], seed=7
        )
        _LOG.info("causal probe passed for %s", name)


def resolve_job(
    *, root: Path, file_id: str, phase: str, smoke: bool
) -> tuple[str, list[np.ndarray]]:
    """Return the record an estimates file is computed on and the days each plant needs.

    Args:
        root: The output root.
        file_id: The estimates file's identifier: `original__tuning`, `original`, `drawNN`,
            `control`, `original__reversed`, `anm_blind`, or
            `hindsight_forward`.
        phase: The phase.
        smoke: Whether to truncate the day lists.

    Returns:
        The record identifier and one day array per plant.
    """
    original = record_for(root=root, record_id=ORIGINAL)
    events = load_events(root=root)
    if file_id == f"{ORIGINAL}__tuning":
        record_id, days = (
            ORIGINAL,
            days_for(root=root, record_id=ORIGINAL, record=original, phase="tuning"),
        )
    elif file_id == "hindsight_forward":
        record_id, days = (
            ORIGINAL,
            hindsight_days(record=original, root=root, reversed_record=False),
        )
    elif file_id.endswith("__reversed"):
        record_id = file_id
        days = hindsight_days(record=original, root=root, reversed_record=True)
    elif file_id == "anm_blind":
        record_id = file_id
        stage_b = np.unique(scored_days(record=original, events=events) - ISSUE_LAG_DAYS)
        days = [stage_b] * len(original.sites)
    else:
        record_id = file_id
        days = days_for(
            root=root,
            record_id=file_id,
            record=record_for(root=root, record_id=file_id),
            phase=phase,
        )
    return record_id, _truncate(days=days, smoke=smoke)


def main() -> int:
    """Run one phase."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("tuning", "main", "hindsight"), required=True)
    parser.add_argument(
        "--workers", type=int, default=4, help="CPU processes for the linear programs."
    )
    parser.add_argument(
        "--smoke", action="store_true", help="Truncate the day lists; not a result."
    )
    parser.add_argument("--skip-probes", action="store_true", help="Skip the two start-up checks.")
    arguments = parser.parse_args()
    if arguments.skip_probes and not arguments.smoke:
        parser.error("--skip-probes is allowed only with --smoke")
    root = claim_root(smoke=arguments.smoke)
    check_stamp(
        directory=estimates_dir(root=root),
        stamp=fingerprint(root=root, modules=ESTIMATOR_MODULES),
    )
    original = record_for(root=root, record_id=ORIGINAL)
    if not arguments.skip_probes:
        check_neighbour_ratio_matches_lag_frame(record=original)
        probe_causality(root=root, record=original)
    names = (*CHEAP, *CONVEX)
    if arguments.phase == "tuning":
        jobs = [
            (name, params, file_id)
            for name in names
            for params in TUNING_GRIDS[name]
            for file_id in (f"{ORIGINAL}__tuning", draw_id(index=0))
        ]
    elif arguments.phase == "main":
        draws = [draw_id(index=i) for i in range(len(load_events(root=root).draws))]
        jobs = [
            (name, load_tuned(root=root, estimator=name), file_id)
            for name in names
            for file_id in (ORIGINAL, *draws, CONTROL)
        ]
    else:
        shortlist = json.loads((root / "selection.json").read_text())["shortlist"]
        jobs = [
            (name, load_tuned(root=root, estimator=name), file_id)
            for name in shortlist
            if name in names
            for file_id in (
                f"{ORIGINAL}__reversed",
                "anm_blind",
                "hindsight_forward",
            )
        ]
    with ProcessPoolExecutor(max_workers=arguments.workers) as pool:
        for name, params, file_id in jobs:
            record_id, days = resolve_job(
                root=root, file_id=file_id, phase=arguments.phase, smoke=arguments.smoke
            )
            run_one(
                root=root,
                name=name,
                params=params,
                file_id=file_id,
                record_id=record_id,
                days=days,
                pool=pool,
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
