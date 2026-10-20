"""G-2y: Part 2's convex model fitted per substation on the 2-year window, no drift.

G-2y is the convex anchor of experiment 112. It is `studies.dp_gam`'s model with Part 2's tuned
penalties, fitted on the 2 years before the cut, with the spread from the replica fitted 12 months
earlier by the same rule and centred on the median (`ResidualSpread` with `centre=True`). The
replicate year (cut 2024-07-01) and the scored year (cut 2025-07-01) are both written; the
replicate year's set carries the suffix `_rep`.

    uv run --no-sync python ml-experiments/00112-*/e00112_convex.py
"""

import json
from datetime import datetime

import e00112_common as common
import numpy as np
import polars as pl
from studies.dp_features import FeatureCache
from studies.dp_gam import (
    GamSpec,
    ResidualSpread,
    TrainingWindow,
    fit_gam,
    predict_gam,
    replica_residuals,
    training_rows,
)

WINDOW = TrainingWindow(months=common.WINDOW_MONTHS)
YEARS: dict[str, tuple[datetime, datetime]] = {
    "_rep": (common.REPLICATE_CUT, common.FINAL_CUT),
    "": (common.FINAL_CUT, common.SCORING_END),
}
"""For each set-name suffix: the cut and the end of the scored rows."""


def tuned_spec() -> GamSpec:
    """Return Part 2's convex model with its tuned penalties and no drift."""
    tuned = json.loads((common.PART_2_DIR / "work" / "tuned.json").read_text())
    return GamSpec(
        drift="none",
        ridge=tuned["ridge"]["chosen"],
        drift_step=tuned["drift_step"]["chosen"],
        step_penalty=tuned["step_penalty"]["chosen"],
    )


def site_replica(*, cache: FeatureCache, site: str, cut: datetime, spec: GamSpec):  # noqa: ANN201
    """Return G-2y's replica residuals of a substation, or `None` if it has none."""
    try:
        return replica_residuals(
            cache=cache,
            site=site,
            spec=spec,
            cut=cut,
            window=WINDOW,
            season_years=common.SEASON_YEARS,
        )
    except ValueError:
        return None


def year_frames(
    *, cache: FeatureCache, cut: datetime, through: datetime, spec: GamSpec
) -> list[pl.DataFrame]:
    """Return G-2y's predictions of every reported substation for one year.

    A substation with no training rows at the cut has no forecast. One with training rows but none
    at the replica cut (its history began later) takes the spread of the other substations'
    replica residuals pooled, as the gradient per-site arms do.
    """
    sites = [
        site
        for site in common.REPORT_SITES
        if len(training_rows(features=cache.sites[site], cache=cache, cut=cut, window=WINDOW))
    ]
    replicas = {site: site_replica(cache=cache, site=site, cut=cut, spec=spec) for site in sites}
    frames = []
    for site in sites:
        own = replicas[site]
        pool = [r for other, r in replicas.items() if other != site and r is not None]
        use = [own] if own is not None else pool
        spread = ResidualSpread.fit(
            residuals=np.concatenate([r.residuals for r in use]),
            band=np.concatenate([r.band for r in use]),
            quarter=np.concatenate([r.quarter for r in use]),
            centre=True,
        )
        fit = fit_gam(
            cache=cache,
            site=site,
            specs=[spec],
            cut=cut,
            window=WINDOW,
            season_years=common.SEASON_YEARS,
        )
        index = common.scoring_index(cache=cache, site=site, after=cut, through=through)
        mean = predict_gam(cache=cache, fit=fit, index=index)
        quantiles = spread.quantiles(
            mean=mean, band=cache.calendar.band[index], quarter=cache.calendar.quarter[index]
        )
        frames.append(
            common.prediction_frame(
                cache=cache, site=site, index=index, quantiles=quantiles
            ).with_columns(mean=pl.Series(mean.astype(np.float32)))
        )
    return frames


def main() -> None:
    """Write G-2y's prediction sets for the replicate year and the scored year."""
    cache = common.load_cache()
    spec = tuned_spec()
    for suffix, (cut, through) in YEARS.items():
        frames = year_frames(cache=cache, cut=cut, through=through, spec=spec)
        common.store_arm(name=f"g_2y{suffix}", frame=pl.concat(frames))
        print(f"stored g_2y{suffix}", flush=True)


if __name__ == "__main__":
    main()
