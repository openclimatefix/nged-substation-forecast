"""Fit the T-all arms of experiment 167 at every cut and substation, and store their means.

One frozen fit per arm, substation, and cut, on data before the cut, scored on the 3 months after.
Every arm is experiment 136's T-all (the heating split by day type and by half of the hour band,
the lighting split by day type) with the PV capacity pinned to experiment 45's real-time vintage.
The arms differ only in the training window (the last 24 months or all years) and the drift term
(none or D1). The penalties are experiment 104's tuned ones. Spreads are not applied here:
`e00167_score.py` builds them from these means, because a spread pool is made of the means of
earlier cuts. D10's two unlabelled-jump windows are removed from training and scoring.

Each job (substation, cut) writes `work/means/<site>_<cut>.parquet` atomically and is skipped if the
file exists. `--variant unmasked` refits D10 alone with the jump windows left in, into
`work/means_unmasked/`, to test whether the mask explains D10's share of a contrast.

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00167-*/e00167_fit.py --workers 6
"""

import argparse
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Final

import e00167_common as common
import e00167_design as design
import numpy as np
import polars as pl
from studies.dp_features import FeatureCache
from studies.dp_gam import DriftType, GamSpec, TrainingWindow
from studies.dp_pinned_capacity import read_vintages


@dataclass(frozen=True)
class Arm:
    """One arm.

    Attributes:
        name: The arm's name.
        drift: The drift form.
        months: The training window in months, or `None` for all years.
    """

    name: str
    drift: DriftType
    months: int | None


ARMS: Final[tuple[Arm, ...]] = (
    Arm(name="m24", drift="none", months=24),
    Arm(name="all", drift="none", months=None),
    Arm(name="all_d1", drift="d1", months=None),
    Arm(name="m24_d1", drift="d1", months=24),
)
"""`m24_d1` is exploratory; the three pre-registered arms are `m24`, `all`, and `all_d1`."""


def fit_job(
    *, cache: FeatureCache, site: str, cut: datetime, vintages: pl.DataFrame, tuned: dict
) -> tuple[pl.DataFrame, list[dict]]:
    """Fit every arm of one substation at one cut and return the means and the fit records.

    Args:
        cache: The feature cache with D10's jump windows removed.
        site: The substation's label.
        cut: The cut.
        vintages: Experiment 45's real-time capacity vintages.
        tuned: The contents of experiment 104's `tuned.json`.

    Returns:
        A long frame with one row per arm and scored half-hour, and one record per arm.
    """
    index = common.scoring_index(cache=cache, site=site, cut=cut)
    means: dict[str, np.ndarray] = {}
    records: list[dict] = []
    for arm in ARMS:
        spec = GamSpec(
            drift=arm.drift,
            ridge=tuned["ridge"]["chosen"],
            drift_step=tuned["drift_step"]["chosen"],
            capacity="pinned",
            final_knot="free",
        )
        base = design.build_base(
            cache=cache,
            vintages=vintages,
            site=site,
            cut=cut,
            spec=spec,
            window=TrainingWindow(months=arm.months),
        )
        tall = design.interaction_arm(name="t_all", penalty_weight=common.T_ALL_LAMBDA)
        fit = design.fit_arm(cache=cache, base=base, arm=tall)
        means[arm.name] = design.predict_arm(cache=cache, base=base, fit=fit, index=index)
        assert base.pinned is not None
        records.append(
            {
                "site": site,
                "origin": common.cut_label(cut=cut),
                "arm": arm.name,
                "n_train_rows": base.n,
                "history_months": common.history_months(cache=cache, site=site, cut=cut),
                "pinned_source": base.pinned.source,
                "pinned_vintage_origin": str(base.pinned.origin),
                "penalised_loss": fit.penalised_loss,
                "unpenalised_loss": fit.unpenalised_loss,
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
    import torch

    torch.set_num_threads(1)
    site, cut, variant = job
    label = common.cut_label(cut=cut)
    folder = common.WORK_DIR / ("means" if variant == "masked" else "means_unmasked")
    target = folder / f"{site}_{label}.parquet"
    if target.exists():
        return f"skip {target.name}"
    if "cache" not in _STATE:
        cache = common.load_cache()
        _STATE["cache"] = common.d10_masked_cache(cache=cache) if variant == "masked" else cache
        _STATE["vintages"] = read_vintages()
        _STATE["tuned"] = json.loads(common.TUNED_PATH.read_text())
    started = time.monotonic()
    frame, records = fit_job(
        cache=_STATE["cache"],
        site=site,
        cut=cut,
        vintages=_STATE["vintages"],
        tuned=_STATE["tuned"],
    )
    common.write_atomic(
        frame=pl.DataFrame(records), path=folder / "records" / f"{site}_{label}.parquet"
    )
    common.write_atomic(frame=frame, path=target)
    return f"done {target.name}: {frame.height} rows, {time.monotonic() - started:.0f} s"


def main() -> None:
    """Fit every (site, cut) job in parallel worker processes."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument(
        "--variant",
        choices=("masked", "unmasked"),
        default="masked",
        help="`unmasked` refits D10 alone with its two jump windows left in.",
    )
    parser.add_argument("--only-first", action="store_true", help="Time a single job and stop.")
    arguments = parser.parse_args()
    os.environ["OMP_NUM_THREADS"] = "1"
    cache = common.d10_masked_cache(cache=common.load_cache())
    sites = ("D10",) if arguments.variant == "unmasked" else None
    jobs = [
        (site, cut, arguments.variant)
        for cut in common.CUTS
        for site in common.sites_for_cut(cache=cache, cut=cut)
        if sites is None or site in sites
    ]
    if arguments.only_first:
        jobs = jobs[-1:]
    print(f"{len(jobs)} jobs", flush=True)
    with ProcessPoolExecutor(max_workers=arguments.workers) as pool:
        for message in pool.map(_worker, jobs):
            print(message, flush=True)


if __name__ == "__main__":
    main()
