"""Equivalence check: can the DP, configured like the convex model, reproduce its score?

The per-site DP is set up to match Part 2's convex model G as closely as it can: no interaction
network, no shared daily shapes (each substation has its own profile), a weak penalty on the
deviation, and the capacity pinned to the convex model's capacity path fitted at the same cut.
It is fitted on every usable year (as G-all) with the same replica spread, and scored on the same
rows. If its score matches G-all (2.726 at the five decision sites) the gap between the DP and G
elsewhere is the DP's extra freedom or the pooling; if not, the gap is optimisation, the
heating-threshold term the convex model does not have, or a bug.

    uv run --no-sync python ml-experiments/00105-*/e00105_equiv.py
"""

import json
from datetime import datetime

import e00105_arms as arms
import e00105_common as common
import numpy as np
import torch
from studies.dp_features import FeatureCache
from studies.dp_gam import GamSpec, TrainingWindow, fit_gam
from studies.dp_pooled import DpConfig, PooledBatch, PopulationScales

WEAK_DEVIATION_SCALE = 100.0


def convex_override(
    *, cache: FeatureCache, site: str, cut: datetime, batch: PooledBatch
) -> torch.Tensor:
    """Return the convex model capacity path of `site` fitted at `cut`, shape (1, years)."""
    tuned = json.loads((common.PART_2_DIR / "work" / "tuned.json").read_text())
    spec = GamSpec(
        drift="none",
        ridge=tuned["ridge"]["chosen"],
        drift_step=tuned["drift_step"]["chosen"],
        step_penalty=tuned["step_penalty"]["chosen"],
    )
    fit = fit_gam(
        cache=cache,
        site=site,
        specs=[spec],
        cut=cut,
        window=TrainingWindow(),
        season_years=common.SEASON_YEARS,
    )
    by_year = {int(year): float(value) for year, value in fit.capacity_points().items()}
    path = np.full((1, batch.n_season_years), np.nan)
    previous = 0.0
    for offset in range(batch.n_season_years):
        previous = by_year.get(batch.first_season_year + offset, previous)
        path[0, offset] = previous
    return torch.as_tensor(path, dtype=torch.float32)


def main() -> None:
    """Fit the equivalence arm and store it as `s_equiv_all`."""
    cache = common.load_cache()
    config = DpConfig(
        shared_shapes=0,
        population_scales=PopulationScales(deviation=WEAK_DEVIATION_SCALE),
    )
    forecasts, record = arms.per_site_forecasts(
        cache=cache,
        config=config,
        sites=common.SCORED_SITES,
        months=None,
        cut=common.FINAL_CUT,
        through=common.SCORING_END,
        override_for=lambda site, cut, batch: convex_override(
            cache=cache, site=site, cut=cut, batch=batch
        ),
    )
    common.store_arm(name="s_equiv_all", frame=arms.frames_from(cache=cache, forecasts=forecasts))
    print(record, flush=True)


if __name__ == "__main__":
    main()
