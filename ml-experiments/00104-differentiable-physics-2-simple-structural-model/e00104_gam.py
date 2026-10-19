"""Fit the four convex-model arms per substation and write their quantiles, capacities, and drift.

One frozen fit per arm and substation on data before the cut, scored on every row with a reading
after it. The quantiles are the fitted mean plus empirical quantiles of the residuals of a replica
of the scored forecast (a fit 12 months before the cut, predicting the following 12 months), per
hour band and calendar quarter (`studies.dp_gam.replica_residuals`). The penalties come from
`work/tuned.json` (`e00104_tune.py`).

Arms: `g1` (12-month window, no drift), `g_all` (all years, no drift), `g_all_d1` (all years,
drift D1), and `g_all_d1_steps` (all years, D1 plus steps). `--cold` runs the cold block instead:
the first three arms with the cut at 2022-11-01, scored November 2022 to February 2023, named
`cold_<arm>`.

Substations are fitted in parallel worker processes with one thread each.

    uv run python ml-experiments/00104-*/e00104_gam.py [--cold] [--workers 6]
"""

import argparse
import json
import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Final

import e00104_common as common
import numpy as np
import polars as pl
from studies.dp_gam import (
    N_BANDS,
    DriftType,
    GamFit,
    GamSpec,
    ResidualSpread,
    TrainingWindow,
    fit_gam,
    predict_gam,
    replica_residuals,
)


@dataclass(frozen=True)
class Arm:
    """One convex-model arm.

    Attributes:
        name: The prediction set's name.
        drift: The drift form.
        window: The training-window rule.
    """

    name: str
    drift: DriftType
    window: TrainingWindow


ARMS: Final[tuple[Arm, ...]] = (
    Arm(name="g1", drift="none", window=TrainingWindow(months=12)),
    Arm(name="g_all", drift="none", window=TrainingWindow()),
    Arm(name="g_all_d1", drift="d1", window=TrainingWindow()),
    Arm(name="g_all_d1_steps", drift="d1_steps", window=TrainingWindow()),
)
COLD_ARMS: Final[tuple[str, ...]] = ("g1", "g_all", "g_all_d1")


def spec_for(*, arm: Arm, tuned: dict) -> GamSpec:
    """Return the tuned penalties for an arm.

    Args:
        arm: The arm.
        tuned: The contents of `work/tuned.json`.

    Returns:
        The spec.
    """
    return GamSpec(
        drift=arm.drift,
        ridge=tuned["ridge"]["chosen"],
        drift_step=tuned["drift_step"]["chosen"],
        step_penalty=tuned["step_penalty"]["chosen"],
    )


def drift_rows(*, fit: GamFit, site: str, arm: str, cut: datetime) -> list[dict]:
    """Return the drift path of a fit as rows, one per hour band and knot.

    Args:
        fit: The fitted model.
        site: The substation's label.
        arm: The arm's name.
        cut: The cut, which is the last knot.

    Returns:
        Rows with the knot's day number, the smooth drift level, and the cumulative step level.
    """
    layout = fit.layout
    if "drift" not in layout.slices:
        return []
    n_knots = len(layout.knot_days)
    drift = fit.theta[0, layout.slices["drift"]].reshape(N_BANDS, n_knots)
    steps = np.zeros((N_BANDS, n_knots))
    if "steps" in layout.slices:
        block = fit.theta[0, layout.slices["steps"]]
        signed = (block[: block.size // 2] - block[block.size // 2 :]).reshape(N_BANDS, n_knots - 1)
        steps[:, 1:] = np.cumsum(signed, axis=1)
    return [
        {
            "site": site,
            "arm": arm,
            "cut": cut.isoformat(),
            "band": band,
            "day": float(layout.knot_days[knot]),
            "drift": float(drift[band, knot]),
            "step_level": float(steps[band, knot]),
        }
        for band in range(N_BANDS)
        for knot in range(n_knots)
    ]


def fit_site(*, site: str, cold: bool) -> dict[str, Any]:
    """Fit every arm of one substation and return its predictions, capacities, and drift paths.

    Args:
        site: The substation's label.
        cold: True for the cold block.

    Returns:
        A dict with `frames` (arm name to predictions), `capacities`, and `drift` rows.
    """
    cache = common.load_cache()
    tuned = json.loads((common.WORK_DIR / "tuned.json").read_text())
    cut = common.COLD_CUT if cold else common.FINAL_CUT
    through = common.COLD_END if cold else common.SCORING_END
    index = common.scoring_index(cache=cache, site=site, after=cut, through=through)
    frames: dict[str, pl.DataFrame] = {}
    capacities: list[dict] = []
    drift: list[dict] = []
    for arm in ARMS:
        if cold and arm.name not in COLD_ARMS:
            continue
        spec = spec_for(arm=arm, tuned=tuned)
        fit = fit_gam(
            cache=cache,
            site=site,
            specs=[spec],
            cut=cut,
            window=arm.window,
            season_years=common.SEASON_YEARS,
        )
        replica = replica_residuals(
            cache=cache,
            site=site,
            spec=spec,
            cut=cut,
            window=arm.window,
            season_years=common.SEASON_YEARS,
        )
        spread = ResidualSpread.fit(
            residuals=replica.residuals,
            band=replica.band,
            quarter=replica.quarter,
            centre=False,
        )
        mean = predict_gam(cache=cache, fit=fit, index=index)
        quantiles = spread.quantiles(
            mean=mean, band=cache.calendar.band[index], quarter=cache.calendar.quarter[index]
        )
        name = f"cold_{arm.name}" if cold else arm.name
        frames[name] = common.prediction_frame(
            cache=cache, site=site, index=index, quantiles=quantiles
        ).with_columns(mean=pl.Series(mean.astype(np.float32)))
        effective = cache.sites[site].table.effective_capacity_mw
        capacities += [
            {
                "site": site,
                "arm": name,
                "season_year": year,
                "capacity_points": points,
                "capacity_mw": points * effective / 100.0,
            }
            for year, points in fit.capacity_points().items()
        ]
        drift += drift_rows(fit=fit, site=site, arm=name, cut=cut)
        print(f"{site} {name}: {fit.n_rows} rows, violation {fit.violation.max():.1e}", flush=True)
    return {"frames": frames, "capacities": capacities, "drift": drift}


def _worker(job: tuple[str, bool]) -> dict[str, Any]:
    """Run `fit_site` in a worker process with one thread."""
    import torch

    torch.set_num_threads(1)
    site, cold = job
    return fit_site(site=site, cold=cold)


def main() -> None:
    """Fit all substations and write the prediction sets and result files."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--cold", action="store_true")
    parser.add_argument("--workers", type=int, default=6)
    arguments = parser.parse_args()
    os.environ["OMP_NUM_THREADS"] = "1"
    sites = common.COLD_SITES if arguments.cold else common.SCORED_SITES
    common.WORK_DIR.mkdir(parents=True, exist_ok=True)
    common.load_cache()
    with ProcessPoolExecutor(max_workers=arguments.workers) as pool:
        results = list(pool.map(_worker, [(site, arguments.cold) for site in sites]))
    names = [name for result in results for name in result["frames"]]
    for name in dict.fromkeys(names):
        frame = pl.concat(
            [result["frames"][name] for result in results if name in result["frames"]]
        )
        common.store_arm(name=name, frame=frame)
    suffix = "cold" if arguments.cold else "scored"
    pl.DataFrame([row for result in results for row in result["capacities"]]).write_parquet(
        common.WORK_DIR / f"gam_capacities_{suffix}.parquet"
    )
    pl.DataFrame([row for result in results for row in result["drift"]]).write_parquet(
        common.WORK_DIR / f"gam_drift_{suffix}.parquet"
    )


if __name__ == "__main__":
    main()
