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


def site_frame(
    *, cache: FeatureCache, site: str, cut: datetime, through: datetime, spec: GamSpec
) -> pl.DataFrame | None:
    """Return G-2y's predictions of one substation, or `None` if it has no training rows."""
    if len(training_rows(features=cache.sites[site], cache=cache, cut=cut, window=WINDOW)) == 0:
        return None
    fit = fit_gam(
        cache=cache,
        site=site,
        specs=[spec],
        cut=cut,
        window=WINDOW,
        season_years=common.SEASON_YEARS,
    )
    replica = replica_residuals(
        cache=cache,
        site=site,
        spec=spec,
        cut=cut,
        window=WINDOW,
        season_years=common.SEASON_YEARS,
    )
    spread = ResidualSpread.fit(
        residuals=replica.residuals, band=replica.band, quarter=replica.quarter, centre=True
    )
    index = common.scoring_index(cache=cache, site=site, after=cut, through=through)
    mean = predict_gam(cache=cache, fit=fit, index=index)
    quantiles = spread.quantiles(
        mean=mean, band=cache.calendar.band[index], quarter=cache.calendar.quarter[index]
    )
    return common.prediction_frame(
        cache=cache, site=site, index=index, quantiles=quantiles
    ).with_columns(mean=pl.Series(mean.astype(np.float32)))


def main() -> None:
    """Write G-2y's prediction sets for the replicate year and the scored year."""
    cache = common.load_cache()
    spec = tuned_spec()
    for suffix, (cut, through) in YEARS.items():
        frames = [
            frame
            for site in common.REPORT_SITES
            if (frame := site_frame(cache=cache, site=site, cut=cut, through=through, spec=spec))
            is not None
        ]
        common.store_arm(name=f"g_2y{suffix}", frame=pl.concat(frames))
        print(f"stored g_2y{suffix}", flush=True)


if __name__ == "__main__":
    main()
