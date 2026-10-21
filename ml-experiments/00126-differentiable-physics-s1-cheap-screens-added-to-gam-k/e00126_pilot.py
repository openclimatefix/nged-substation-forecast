"""Pilot half-widths of experiment 126 (gate 2), written before any arm result is read.

**The pilot contrast is experiment 114's B0-K minus T-all-K on the tuning year**, from 114's stored
predictions, on the five decision sites and rows outside the hand-filtered switching events. For
each row subset the script bootstraps whole calendar months (20,000 resamples) and prints the
half-width of the interval at 99.44% (Bonferroni over 9) and 98.75% (over 4). A contrast of the
experiment whose half-width exceeds M = 0.03 points is `undecidable` whatever it scores.

Subsets: November to March (the primary rows), cold-snap days, school-holiday weekdays, dark winter
evenings (15:30 to 19:00 local, November to February), and November and December (W-stn's rows).
The result goes to `work/pilot.json` with a SHA-256 hash.

    uv run --no-sync python ml-experiments/00126-*/e00126_pilot.py
"""

import hashlib
import json
from typing import Final

import e00126_common as common
import e00126_design as design
import e00126_fmix as fmix
import numpy as np
import polars as pl
from studies.bootstrap import bootstrap_difference_at_level
from studies.dp_features import FeatureCache
from studies.dp_gam import QUANTILE_COLUMNS
from studies.experiment_store import losses_for_bootstrap

PREDICTIONS_114: Final = (
    common.ML_EXPERIMENTS_DATA_DIR
    / "00114-differentiable-physics-n3-n4-interactions-and-heating-threshold"
    / "predictions"
)
LEVELS: Final[dict[str, float]] = {"99.44": 99.44, "98.75": 98.75}
KEYS: Final[list[str]] = ["time_series_id", "power_fcst_init_time", "valid_time", "seed"]


def load(*, name: str) -> pl.DataFrame:
    """Read a stored 114 tuning-year set with its per-row quantile score."""
    frame = pl.read_parquet(PREDICTIONS_114 / f"tuning_{name}" / "predictions.parquet").filter(
        ~pl.col("switching_mask")
    )
    quantiles = frame.select(QUANTILE_COLUMNS).to_numpy().astype(np.float64)
    actual = frame["actual"].to_numpy().astype(np.float64)
    return frame.with_columns(
        qs=pl.Series(common.quantile_score(actual=actual, quantiles=quantiles))
    )


def subset_masks(*, frame: pl.DataFrame, cache: FeatureCache) -> dict[str, np.ndarray]:
    """Return the boolean mask of each row subset on a stored frame."""
    valid = frame["valid_time"].dt.replace_time_zone(None).to_numpy().astype("datetime64[us]")
    position = np.searchsorted(cache.end_times, valid)
    ids = frame["time_series_id"].to_numpy()
    site_of = {cache.sites[s].table.time_series_id: s for s in common.SCORED_SITES}
    month = frame["valid_time"].dt.month().to_numpy()
    half_hour = cache.calendar.half_hour[position]
    cold = np.zeros(len(frame), dtype=bool)
    school = np.zeros(len(frame), dtype=bool)
    for tsid, site in site_of.items():
        rows = np.flatnonzero(ids == tsid)
        regimes = fmix.regime_map(cache=cache, site=site).regime
        cold[rows] = regimes[position[rows]] == fmix.COLD_SNAP
        school[rows] = design.site_aux(cache=cache, site=site).school_holiday[position[rows]]
    winter = np.isin(month, fmix.WINTER_MONTHS)
    return {
        "Nov-Mar": winter,
        "cold-snap days (Nov-Mar)": winter & cold,
        "school-holiday weekdays": school,
        "dark winter evenings (Nov-Feb, 15:30-19:00)": np.isin(month, (11, 12, 1, 2))
        & (half_hour >= 31)
        & (half_hour <= 37),
        "Nov-Dec (W-stn rows)": np.isin(month, (11, 12)),
    }


def main() -> None:
    """Print and store the pilot half-widths."""
    cache = common.load_cache()
    reference = load(name="b0_k")
    treatment = load(name="t_all_k")
    shared = reference.select(KEYS).join(treatment.select(KEYS), on=KEYS, how="inner")
    reference = reference.join(shared, on=KEYS, how="inner").sort(KEYS)
    treatment = treatment.join(shared, on=KEYS, how="inner").sort(KEYS)
    masks = subset_masks(frame=reference, cache=cache)
    record: dict[str, dict[str, float]] = {}
    for name, mask in masks.items():
        arms = {
            "b0_k": reference.filter(pl.Series(mask)).select(*KEYS, "qs"),
            "t_all_k": treatment.filter(pl.Series(mask)).select(*KEYS, "qs"),
        }
        arms = {k: v.with_columns(ae=pl.col("qs")) for k, v in arms.items()}
        losses = losses_for_bootstrap(arm_losses=arms)
        entry: dict[str, float] = {"rows": float(mask.sum())}
        for label, level in LEVELS.items():
            low, high = bootstrap_difference_at_level(
                losses=losses,
                treatment="b0_k",
                reference="t_all_k",
                metric="qs",
                level=level,
                n_resamples=common.N_RESAMPLES,
            )
            entry[f"half_width_{label}"] = 0.5 * (high - low)
        record[name] = entry
        print(f"{name}: {entry}")
    text = json.dumps(record, indent=2, sort_keys=True) + "\n"
    (common.WORK_DIR / "pilot.json").write_text(text)
    (common.WORK_DIR / "pilot.sha256").write_text(hashlib.sha256(text.encode()).hexdigest() + "\n")


if __name__ == "__main__":
    main()
