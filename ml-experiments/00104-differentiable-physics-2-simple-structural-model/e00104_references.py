"""Write the climatology and manual-heuristic reference rows on the scored rows.

These two references hold the same information as the production baselines of the same names and
are scored on the same rows as every arm, with the same day-ahead initialisation.

- `climatology`: the 11 quantiles of the training window's power in the row's local month,
  half-hour of day, and weekend flag, pooled with the neighbouring months and half-hours (the
  nine-cell neighbourhood of `baseline_forecasters.climatology`). The training window is every
  usable year before the cut (rows in `default_rows`), as for `g_all`.
- `manual_heuristic`: the quantiles of an ensemble of observed powers at the same time of day, the
  six weekly analogues one to six weeks back and the seven annual analogues 49 to 55 weeks back
  (`baseline_forecasters.manual_heuristic`). At the day-ahead lead every analogue is old enough.
  A row with no analogue observed falls back to the climatology quantiles, and the report counts
  those rows.

    uv run python ml-experiments/00104-*/e00104_references.py
"""

import json
from typing import Final

import e00104_common as common
import numpy as np
import polars as pl
from studies.dp_features import FeatureCache
from studies.dp_gam import QUANTILE_LEVELS, TrainingWindow, training_rows

WEEKLY_ANALOGUES: Final[tuple[int, ...]] = (1, 2, 3, 4, 5, 6)
ANNUAL_ANALOGUES: Final[tuple[int, ...]] = (49, 50, 51, 52, 53, 54, 55)
STEPS_PER_WEEK: Final[int] = common.STEPS_PER_WEEK
N_MONTHS: Final[int] = 12
N_HALF_HOURS: Final[int] = 48


def cell_ids(*, cache: FeatureCache, index: np.ndarray) -> np.ndarray:
    """Return the climatology cell (month, half-hour, weekend flag) of each row, as one integer."""
    month = cache.calendar.local_date[index].astype("datetime64[M]").astype(np.int64) % N_MONTHS
    weekend = np.isin(cache.calendar.day_type[index], (1, 2)).astype(np.int64)
    return (month * N_HALF_HOURS + cache.calendar.half_hour[index].astype(np.int64)) * 2 + weekend


def climatology_table(*, cache: FeatureCache, site: str) -> np.ndarray:
    """Return the pooled quantiles of every cell, shape (cells, 11), from the training window.

    Args:
        cache: The feature cache.
        site: The substation's label.

    Returns:
        A table indexed by `cell_ids`; cells with no sample hold `NaN`.
    """
    features = cache.sites[site]
    train = training_rows(
        features=features, cache=cache, cut=common.FINAL_CUT, window=TrainingWindow()
    )
    cells = cell_ids(cache=cache, index=train)
    n_cells = N_MONTHS * N_HALF_HOURS * 2
    samples: list[np.ndarray] = [features.y[train][cells == cell] for cell in range(n_cells)]
    table = np.full((n_cells, len(QUANTILE_LEVELS)), np.nan)
    for cell in range(n_cells):
        month, rest = divmod(cell // 2, N_HALF_HOURS)
        weekend = cell % 2
        pooled = []
        for month_offset in (-1, 0, 1):
            for half_hour_offset in (-1, 0, 1):
                neighbour = (
                    ((month + month_offset) % N_MONTHS) * N_HALF_HOURS
                    + (rest + half_hour_offset) % N_HALF_HOURS
                ) * 2 + weekend
                pooled.append(samples[neighbour])
        values = np.concatenate(pooled)
        if len(values):
            table[cell] = np.quantile(values, QUANTILE_LEVELS)
    return table


def manual_heuristic_quantiles(
    *, cache: FeatureCache, site: str, index: np.ndarray
) -> tuple[np.ndarray, int]:
    """Return the quantiles of the analogue ensemble of each row, and how many rows had none.

    Args:
        cache: The feature cache.
        site: The substation's label.
        index: Grid rows.

    Returns:
        An array of shape (len(index), 11) with `NaN` where no analogue was observed, and the
        number of such rows.
    """
    features = cache.sites[site]
    observed = np.where(features.present, features.y, np.nan)
    lags = [STEPS_PER_WEEK * weeks for weeks in (*WEEKLY_ANALOGUES, *ANNUAL_ANALOGUES)]
    members = np.stack([observed[np.clip(index - lag, 0, None)] for lag in lags], axis=1)
    members[index[:, None] - np.array(lags)[None, :] < 0] = np.nan
    out = np.full((len(index), len(QUANTILE_LEVELS)), np.nan)
    has_member = np.isfinite(members).any(axis=1)
    out[has_member] = np.nanquantile(members[has_member], QUANTILE_LEVELS, axis=1).T
    return out, int((~has_member).sum())


def main() -> None:
    """Write both reference prediction sets and a small report of the fallbacks."""
    cache = common.load_cache()
    frames: dict[str, list[pl.DataFrame]] = {"climatology": [], "manual_heuristic": []}
    fallback: dict[str, int] = {}
    for site in common.SCORED_SITES:
        index = common.scoring_index(
            cache=cache, site=site, after=common.FINAL_CUT, through=common.SCORING_END
        )
        table = climatology_table(cache=cache, site=site)
        climatology = table[cell_ids(cache=cache, index=index)]
        heuristic, missing = manual_heuristic_quantiles(cache=cache, site=site, index=index)
        fallback[site] = missing
        heuristic = np.where(np.isfinite(heuristic), heuristic, climatology)
        for name, quantiles in (("climatology", climatology), ("manual_heuristic", heuristic)):
            frames[name].append(
                common.prediction_frame(cache=cache, site=site, index=index, quantiles=quantiles)
            )
    for name, parts in frames.items():
        common.store_arm(name=name, frame=pl.concat(parts))
    common.WORK_DIR.mkdir(parents=True, exist_ok=True)
    (common.WORK_DIR / "references.json").write_text(
        json.dumps({"manual_heuristic_rows_without_analogue": fallback}, indent=2) + "\n"
    )
    print("manual-heuristic rows with no analogue, by site:", fallback)


if __name__ == "__main__":
    main()
