"""Fit the four convex-model arms of experiment 149 at every cut and substation, and store means.

Adapted from experiment 107's `e00107_gam.py`. One frozen fit per arm, substation, and cut, on data
before the cut, scored on the 3 months after. The arms differ only in the training window (24
months or every year from the substation's first usable date) and the drift (none or D1). The solar
capacity is experiment 45's pinned real-time vintage in every arm, and the penalties are Part 2's
(`tuned.json`). Spreads are built by `e00149_score.py` from these means, because a spread pool is
made of the means of earlier cuts.

Each job (substation, cut) writes `work/gam/<site>_<cut>.parquet` atomically and is skipped if the
file exists.

    uv run --no-sync python ml-experiments/00149-*/e00149_gam.py [--workers 8]
"""

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Final

import e00149_common as common
import numpy as np
import polars as pl
from studies.dp_features import FeatureCache
from studies.dp_gam import DriftType, GamSpec, TrainingWindow, fit_gam, predict_gam
from studies.dp_pinned_capacity import pinned_capacity, read_vintages


@dataclass(frozen=True)
class Arm:
    """One convex-model arm.

    Attributes:
        name: The arm's name.
        drift: The drift form.
        months: The training window in months, or `None` for every year.
    """

    name: str
    drift: DriftType
    months: int | None


ARMS: Final[tuple[Arm, ...]] = (
    Arm(name="a24", drift="none", months=24),
    Arm(name="a24_d1", drift="d1", months=24),
    Arm(name="aall", drift="none", months=None),
    Arm(name="aall_d1", drift="d1", months=None),
)
"""The four arms: 24 months or all years, each with and without the drift term D1."""


def spec_for(*, arm: Arm, tuned: dict) -> GamSpec:
    """Return Part 2's tuned penalties for an arm, with pinned capacity."""
    return GamSpec(
        drift=arm.drift,
        ridge=tuned["ridge"]["chosen"],
        drift_step=tuned["drift_step"]["chosen"],
        capacity="pinned",
    )


def fit_job(
    *, cache: FeatureCache, site: str, cut: datetime, vintages: pl.DataFrame, tuned: dict
) -> tuple[pl.DataFrame, list[dict]]:
    """Fit every arm of one substation at one cut and return the means and the records.

    Args:
        cache: The feature cache.
        site: The substation's label.
        cut: The cut.
        vintages: Experiment 45's real-time capacity vintages.
        tuned: The contents of Part 2's `tuned.json`.

    Returns:
        A long frame with one row per arm and scored half-hour, and one record per arm.
    """
    index = common.scoring_index(cache=cache, site=site, cut=cut)
    pinned = pinned_capacity(vintages=vintages, table=cache.sites[site].table, cut=cut)
    means: dict[str, np.ndarray] = {}
    records: list[dict] = []
    for arm in ARMS:
        fit = fit_gam(
            cache=cache,
            site=site,
            specs=[spec_for(arm=arm, tuned=tuned)],
            cut=cut,
            window=TrainingWindow(months=arm.months),
            season_years=common.SEASON_YEARS,
            pinned=pinned,
        )
        means[arm.name] = predict_gam(cache=cache, fit=fit, index=index)
        records.append(
            {
                "site": site,
                "origin": common.cut_label(cut=cut),
                "arm": arm.name,
                "n_train_rows": fit.n_rows,
                "violation": float(fit.violation.max()),
                "pinned_source": pinned.source,
                "pinned_vintage_origin": str(pinned.origin),
                "pinned_held_points": pinned.held_points,
                "history_months_at_cut": (cut - common.FIRST_USABLE[site]).days / 30.4375,
                "passes_history_rule": common.passes_history_rule(site=site, cut=cut),
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


def _worker(job: tuple[str, datetime]) -> str:
    """Fit one job in a worker process with one thread, and write its files."""
    import torch  # ty: ignore[unresolved-import]

    torch.set_num_threads(1)
    site, cut = job
    folder = common.WORK_DIR / "gam"
    target = folder / f"{site}_{common.cut_label(cut=cut)}.parquet"
    if target.exists():
        return f"skip {target.name}"
    if "cache" not in _STATE:
        _STATE["cache"] = common.load_cache()
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
    """Fit every job in parallel worker processes."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=8)
    arguments = parser.parse_args()
    os.environ["OMP_NUM_THREADS"] = "1"
    common.load_cache()
    jobs = [(site, cut) for cut in common.CUTS for site in common.sites_for_cut(cut=cut)]
    with ProcessPoolExecutor(max_workers=arguments.workers) as pool:
        for message in pool.map(_worker, jobs):
            print(message, flush=True)


if __name__ == "__main__":
    main()
