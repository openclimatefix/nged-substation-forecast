"""Fit the convex-model arms of experiment 107 at every cut and substation, and store their means.

One frozen fit per arm, substation, and cut, on data before the cut, scored on the 3 months after.
The arms differ in the training window (12 months or all years), the drift (none or D1), whether
the PV capacity is pinned to experiment 45's real-time vintage or fitted, the drift step, and
whether the two day-of-year harmonics are kept. The penalties are Part 2's (`tuned.json`). Spreads
are not applied here: `e00107_spread.py` builds them from these means, because a spread pool is
made of the means of earlier cuts.

Each job (substation, cut, variant) writes `work/gam/<variant>/<site>_<cut>.parquet` atomically and
is skipped if the file exists. `--variant d10_masked` refits D10 with its two unlabelled-jump
windows removed from training and scoring.

    uv run python ml-experiments/00107-*/e00107_gam.py [--workers 8] [--variant base|d10_masked]
"""

import argparse
import json
import os
from collections.abc import Generator
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Final

import e00107_common as common
import numpy as np
import polars as pl
from studies.dp_features import FeatureCache
from studies.dp_gam import (
    CapacityType,
    DriftType,
    GamSpec,
    TrainingWindow,
    fit_gam,
    predict_gam,
)
from studies.dp_pinned_capacity import pinned_capacity, read_vintages

from studies import dp_gam


@dataclass(frozen=True)
class Arm:
    """One convex-model arm.

    Attributes:
        name: The arm's name.
        drift: The drift form.
        months: The training window in months, or `None` for all years.
        capacity: Whether the PV capacity is pinned to the real-time vintage or fitted.
        drift_step: The drift's second-difference penalty weight, or `None` for the tuned one.
        harmonics: False to remove the two day-of-year harmonics from the design.
    """

    name: str
    drift: DriftType
    months: int | None
    capacity: CapacityType = "pinned"
    drift_step: float | None = None
    harmonics: bool = True


ARMS: Final[tuple[Arm, ...]] = (
    Arm(name="g1", drift="none", months=12),
    Arm(name="g_all", drift="none", months=None),
    Arm(name="g_all_d1", drift="d1", months=None),
    Arm(name="g1_fc", drift="none", months=12, capacity="fitted"),
    Arm(name="g_all_fc", drift="none", months=None, capacity="fitted"),
    Arm(name="g_all_d1_fc", drift="d1", months=None, capacity="fitted"),
    Arm(name="g_all_d1_s10k", drift="d1", months=None, drift_step=10_000.0),
    Arm(name="g1_noharm", drift="none", months=12, harmonics=False),
    Arm(name="g_all_d1_noharm", drift="d1", months=None, harmonics=False),
)
"""Nine arms. `_fc` is fitted capacity (Part 2's configuration), `_s10k` a drift step of 10,000."""


@contextmanager
def harmonics_removed(*, remove: bool) -> Generator[None]:
    """Zero the two day-of-year harmonics' design columns, inside the block, in this process.

    The ridge penalty on the harmonics then sets their coefficients to zero, so a fit has no
    harmonics. `studies.dp_gam` has no switch for this, so the experiment wraps its design matrix.

    Args:
        remove: False leaves the design alone.

    Yields:
        Nothing.
    """
    original = dp_gam.design_matrix
    if not remove:
        yield
        return

    def without_harmonics(**kwargs: Any) -> np.ndarray:
        design = original(**kwargs)
        design[:, kwargs["layout"].slices["harmonics"]] = 0.0
        return design

    dp_gam.design_matrix = without_harmonics  # ty: ignore[invalid-assignment]
    try:
        yield
    finally:
        dp_gam.design_matrix = original


def spec_for(*, arm: Arm, tuned: dict) -> GamSpec:
    """Return Part 2's tuned penalties for an arm, with its own capacity mode and drift step."""
    return GamSpec(
        drift=arm.drift,
        ridge=tuned["ridge"]["chosen"],
        drift_step=arm.drift_step if arm.drift_step is not None else tuned["drift_step"]["chosen"],
        capacity=arm.capacity,
    )


def fit_job(
    *, cache: FeatureCache, site: str, cut: datetime, vintages: pl.DataFrame, tuned: dict
) -> tuple[pl.DataFrame, list[dict]]:
    """Fit every arm of one substation at one cut and return the means and the capacity records.

    Args:
        cache: The feature cache (the masked variant for `d10_masked`).
        site: The substation's label.
        cut: The cut.
        vintages: Experiment 45's real-time capacity vintages.
        tuned: The contents of Part 2's `tuned.json`.

    Returns:
        A long frame with one row per arm and scored half-hour, and one capacity record per arm.
    """
    index = common.scoring_index(cache=cache, site=site, cut=cut)
    table = cache.sites[site].table
    pinned = pinned_capacity(vintages=vintages, table=table, cut=cut)
    means: dict[str, np.ndarray] = {}
    records: list[dict] = []
    for arm in ARMS:
        spec = spec_for(arm=arm, tuned=tuned)
        window = TrainingWindow(months=arm.months)
        with harmonics_removed(remove=not arm.harmonics):
            fit = fit_gam(
                cache=cache,
                site=site,
                specs=[spec],
                cut=cut,
                window=window,
                season_years=common.SEASON_YEARS,
                pinned=pinned if arm.capacity == "pinned" else None,
            )
            means[arm.name] = predict_gam(cache=cache, fit=fit, index=index)
        capacity = fit.capacity_points()
        records.append(
            {
                "site": site,
                "origin": common.cut_label(cut=cut),
                "arm": arm.name,
                "n_train_rows": fit.n_rows,
                "violation": float(fit.violation.max()),
                "pinned_source": pinned.source if arm.capacity == "pinned" else None,
                "pinned_vintage_origin": str(pinned.origin) if arm.capacity == "pinned" else None,
                "pinned_held_points": pinned.held_points if arm.capacity == "pinned" else None,
                "fitted_last_capacity_points": capacity[max(capacity)] if capacity else None,
            }
        )
    frames = [
        common.base_frame(
            cache=cache,
            site=site,
            cut=cut,
            index=index,
            extra={"arm": np.full(len(index), name), "mean": mean},
        )
        for name, mean in means.items()
    ]
    return pl.concat(frames), records


_STATE: dict[str, Any] = {}


def _worker(job: tuple[str, datetime, str]) -> str:
    """Fit one job in a worker process with one thread, and write its files."""
    import torch  # ty: ignore[unresolved-import]

    torch.set_num_threads(1)
    site, cut, variant = job
    folder = common.WORK_DIR / "gam" / variant
    target = folder / f"{site}_{common.cut_label(cut=cut)}.parquet"
    if target.exists():
        return f"skip {target.name}"
    if "cache" not in _STATE:
        cache = common.load_cache()
        _STATE["cache"] = common.d10_masked_cache(cache=cache) if variant == "d10_masked" else cache
        _STATE["vintages"] = read_vintages()
        _STATE["tuned"] = json.loads(common.TUNED_PATH.read_text())
    frame, records = fit_job(
        cache=_STATE["cache"],
        site=site,
        cut=cut,
        vintages=_STATE["vintages"],
        tuned=_STATE["tuned"],
    )
    common.write_atomic(
        frame=pl.DataFrame(records),
        path=folder / "records" / f"{site}_{common.cut_label(cut=cut)}.parquet",
    )
    common.write_atomic(frame=frame, path=target)
    return f"done {target.name}: {frame.height} rows"


def main() -> None:
    """Fit every job of a variant in parallel worker processes."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--variant", choices=("base", "d10_masked"), default="base")
    arguments = parser.parse_args()
    os.environ["OMP_NUM_THREADS"] = "1"
    common.load_cache()
    if arguments.variant == "base":
        jobs = [
            (site, cut, "base") for cut in common.CUTS for site in common.sites_for_cut(cut=cut)
        ]
    else:
        jobs = [("D10", cut, "d10_masked") for cut in common.CUTS]
    with ProcessPoolExecutor(max_workers=arguments.workers) as pool:
        for message in pool.map(_worker, jobs):
            print(message, flush=True)


if __name__ == "__main__":
    main()
