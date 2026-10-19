"""Write the climatology and manual-heuristic reference rows on the 12 scored origins.

Both references hold the same information as the production baselines of the same names and are
scored on the same rows as every arm. They are refitted at each cut.

- `climatology`: the 11 quantiles of the training window's power in the row's local month,
  half-hour of day, and weekend flag, pooled with the neighbouring months and half-hours (the
  nine-cell neighbourhood of `baseline_forecasters.climatology`). The training window is every
  usable year before the cut (rows in `default_rows`), as for `g_all`.
- `manual_heuristic`: the quantiles of an ensemble of observed powers at the same time of day, the
  six weekly analogues one to six weeks back and the seven annual analogues 49 to 55 weeks back
  (`baseline_forecasters.manual_heuristic`). A frozen forecast is issued at the cut, so an analogue
  counts only if it ends on or before the cut: late in the window the weekly analogues drop out and
  the annual analogues remain. A row with no analogue falls back to the climatology quantiles, and
  the report counts those rows.

The files are written to `work/scored/` with spread `own`, beside the arms.

    uv run python ml-experiments/00107-*/e00107_references.py
"""

from datetime import datetime
from typing import Final

import e00107_common as common
import numpy as np
import polars as pl
from studies.dp_features import FeatureCache
from studies.dp_gam import QUANTILE_COLUMNS, QUANTILE_LEVELS, TrainingWindow, training_rows

WEEKLY_ANALOGUES: Final[tuple[int, ...]] = (1, 2, 3, 4, 5, 6)
ANNUAL_ANALOGUES: Final[tuple[int, ...]] = (49, 50, 51, 52, 53, 54, 55)
STEPS_PER_WEEK: Final[int] = 7 * 48
N_MONTHS: Final[int] = 12
N_HALF_HOURS: Final[int] = 48
KEY_COLUMNS: Final[tuple[str, ...]] = (
    "site",
    "time_series_id",
    "origin",
    "valid_time",
    "power_fcst_init_time",
    "lead_hours",
    "actual",
    "switching_mask",
    "jump_window",
)


def cell_ids(*, cache: FeatureCache, index: np.ndarray) -> np.ndarray:
    """Return the climatology cell (month, half-hour, weekend flag) of each row, as one integer."""
    month = cache.calendar.local_date[index].astype("datetime64[M]").astype(np.int64) % N_MONTHS
    weekend = np.isin(cache.calendar.day_type[index], (1, 2)).astype(np.int64)
    return (month * N_HALF_HOURS + cache.calendar.half_hour[index].astype(np.int64)) * 2 + weekend


def climatology_table(*, cache: FeatureCache, site: str, cut: datetime) -> np.ndarray:
    """Return the pooled quantiles of every cell, shape (cells, 11), from rows before the cut."""
    features = cache.sites[site]
    train = training_rows(features=features, cache=cache, cut=cut, window=TrainingWindow())
    cells = cell_ids(cache=cache, index=train)
    n_cells = N_MONTHS * N_HALF_HOURS * 2
    samples = [features.y[train][cells == cell] for cell in range(n_cells)]
    table = np.full((n_cells, len(QUANTILE_LEVELS)), np.nan)
    for cell in range(n_cells):
        month, rest = divmod(cell // 2, N_HALF_HOURS)
        weekend = cell % 2
        pooled = [
            samples[
                (((month + dm) % N_MONTHS) * N_HALF_HOURS + (rest + dh) % N_HALF_HOURS) * 2
                + weekend
            ]
            for dm in (-1, 0, 1)
            for dh in (-1, 0, 1)
        ]
        values = np.concatenate(pooled)
        if len(values):
            table[cell] = np.quantile(values, QUANTILE_LEVELS)
    return table


def manual_heuristic_quantiles(
    *, cache: FeatureCache, site: str, index: np.ndarray, cut: datetime
) -> tuple[np.ndarray, int]:
    """Return the analogue ensemble's quantiles per row, and how many rows had no analogue."""
    features = cache.sites[site]
    observed = np.where(features.present, features.y, np.nan)
    cut_position = int(np.searchsorted(cache.end_times, common.to_64(moment=cut), side="right")) - 1
    lags = np.array([STEPS_PER_WEEK * weeks for weeks in (*WEEKLY_ANALOGUES, *ANNUAL_ANALOGUES)])
    positions = index[:, None] - lags[None, :]
    members = observed[np.clip(positions, 0, None)]
    members[(positions < 0) | (positions > cut_position)] = np.nan
    out = np.full((len(index), len(QUANTILE_LEVELS)), np.nan)
    has_member = np.isfinite(members).any(axis=1)
    out[has_member] = np.nanquantile(members[has_member], QUANTILE_LEVELS, axis=1).T
    return out, int((~has_member).sum())


def reference_frame(
    *,
    cache: FeatureCache,
    site: str,
    cut: datetime,
    index: np.ndarray,
    quantiles: np.ndarray,
    name: str,
) -> pl.DataFrame:
    """Return one reference's rows in the layout `e00107_assemble.py` writes."""
    ordered = np.sort(quantiles, axis=1)
    extra = {column: ordered[:, position] for position, column in enumerate(QUANTILE_COLUMNS)}
    extra["mean"] = ordered[:, QUANTILE_LEVELS.index(0.5)]
    frame = common.base_frame(cache=cache, site=site, cut=cut, index=index, extra=extra)
    return frame.select(*KEY_COLUMNS, "mean", *QUANTILE_COLUMNS).with_columns(
        arm=pl.lit(name), spread=pl.lit("own")
    )


def main() -> None:
    """Write both reference files and print the manual heuristic's fallback counts."""
    cache = common.load_cache()
    frames: dict[str, list[pl.DataFrame]] = {"climatology": [], "manual_heuristic": []}
    fallback: dict[str, int] = {}
    for cut in common.SCORED_ORIGINS:
        for site in common.sites_for_cut(cut=cut):
            if site == "D9" and cut < common.add_months(moment=common.D9_FIRST_CUT, months=12):
                continue
            index = common.scoring_index(cache=cache, site=site, cut=cut)
            climatology = climatology_table(cache=cache, site=site, cut=cut)[
                cell_ids(cache=cache, index=index)
            ]
            heuristic, missing = manual_heuristic_quantiles(
                cache=cache, site=site, index=index, cut=cut
            )
            fallback[f"{site}@{common.cut_label(cut=cut)}"] = missing
            heuristic = np.where(np.isfinite(heuristic), heuristic, climatology)
            for name, quantiles in (("climatology", climatology), ("manual_heuristic", heuristic)):
                frames[name].append(
                    reference_frame(
                        cache=cache, site=site, cut=cut, index=index, quantiles=quantiles, name=name
                    )
                )
    for name, parts in frames.items():
        frame = pl.concat(parts)
        common.write_atomic(frame=frame, path=common.WORK_DIR / "scored" / f"{name}__own.parquet")
        print(f"{name}: {frame.height} rows")
    print("manual-heuristic rows with no analogue:", {k: v for k, v in fallback.items() if v})


if __name__ == "__main__":
    main()
