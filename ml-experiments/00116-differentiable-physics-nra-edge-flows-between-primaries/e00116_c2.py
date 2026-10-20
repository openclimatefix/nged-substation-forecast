"""C2 of experiment 116: does the label-free NRA model do no harm outside events?

Arms A0 (trained on every row), A1 (trained without the hand-filtered switching events), and A4
(A0 refitted on power with the hindsight edge flows taken out) each forecast day D+1 from 09:00 UTC
origins on real data, corrected by the plain Kalman filter, with quantiles from the arm's own
replica (`studies.dp_assimilation.post_filter_lead_spread`). Rows inside the hand-filtered events
are left out of the scored rows, and the main contrast is A4 minus A0. The decision is a
non-inferiority test with a margin of 0.05 points at the 98.33% level, in the scored year and the
replicate year.

    uv run python ml-experiments/00116-*/e00116_c2.py
"""

import json
from functools import partial
from typing import Any, Final

import e00116_causal as causal
import e00116_common as common
import e00116_gam as gam
import e00116_score as score
import numpy as np
import polars as pl
from studies.bootstrap import bootstrap_difference_at_level
from studies.dp_assimilation import (
    DEFAULT_DELAY_HOURS,
    FilterHyperparameters,
    correction_at,
    day_ahead_initialisation,
    post_filter_lead_spread,
    run_kalman,
    usable_cutoff_positions,
)
from studies.dp_features import FeatureCache
from studies.dp_gam import QUANTILE_COLUMNS, ReplicaResiduals

SITES: Final[tuple[str, ...]] = ("D5", "D9", "D10", "D11", "D14", "D16")
DECISION_SITES: Final[tuple[str, ...]] = ("D5", "D10", "D11", "D14", "D16")
ARMS: Final[tuple[str, ...]] = ("a0", "a1", "a4")
MARGIN_POINTS: Final[float] = 0.05
"""The non-inferiority margin: about 2% of GAM-K's day-ahead score."""
LEVEL: Final[float] = score.BONFERRONI_LEVEL
RESULT_PATH: Final = common.WORK_DIR / "c2.json"
LOSSES_PATH: Final = common.WORK_DIR / "c2_losses.parquet"
PREVIOUS: Final[dict[str, str]] = {"replicate": "tuning", "scored": "replicate"}


def stored_arm(*, arm: str, cut_name: str) -> gam.ArmType:
    """Return the arm whose stored fit exists at a cut: A4 is A0 at the tuning cut."""
    return "a0" if arm == "a4" and cut_name == "tuning" else arm  # ty: ignore[invalid-return-type]


def read_fit(*, arm: str, site: str, cut_name: str) -> pl.DataFrame | None:
    """Read an arm's stored fit of a site, or `None` if the site had too little history."""
    path = gam.job_path(
        arm=stored_arm(arm=arm, cut_name=cut_name), site=site, cut=common.CUTS[cut_name]
    )
    return pl.read_parquet(path) if path.exists() else None


def replica_of(
    *, cache: FeatureCache, arm: str, site: str, cut_name: str
) -> ReplicaResiduals | None:
    """Build the replica residuals of an arm from the previous cut's out-of-sample year."""
    previous = PREVIOUS[cut_name]
    frame = read_fit(arm=arm, site=site, cut_name=previous)
    if frame is None:
        return None
    year = common.year_slice(cache=cache, cut=common.CUTS[previous])
    position = frame["position"].to_numpy()
    keep = (position >= year.start) & (position < year.stop)
    index = position[keep]
    return ReplicaResiduals(
        residuals=(frame["y"] - frame["mean"]).to_numpy()[keep],
        band=cache.calendar.band[index],
        quarter=cache.calendar.quarter[index],
        index=index,
        fit_cut=common.CUTS[previous],
        switching=frame["switching_mask"].to_numpy()[keep],
    )


def arm_losses(
    *,
    cache: FeatureCache,
    arm: str,
    site: str,
    cut_name: str,
    hyper: FilterHyperparameters,
) -> pl.DataFrame | None:
    """Forecast day D+1 over a scored year with one arm and score it."""
    frame = read_fit(arm=arm, site=site, cut_name=cut_name)
    replica = replica_of(cache=cache, arm=arm, site=site, cut_name=cut_name)
    if frame is None or replica is None:
        return None
    cut = common.CUTS[cut_name]
    year = common.year_slice(cache=cache, cut=cut)
    position = frame["position"].to_numpy()
    keep = (position >= year.start) & (position < year.stop)
    index = position[keep]
    mean = frame["mean"].to_numpy()[keep]
    y = frame["y"].to_numpy()[keep]
    stream = np.full(len(cache.end_times), np.nan)
    stream[index] = y - mean
    band = cache.calendar.band
    trace = run_kalman(residual=stream, band=band, hyper=hyper)
    initialisation = day_ahead_initialisation(end_times=cache.end_times[index], days_ahead=1)
    cutoff = usable_cutoff_positions(
        end_times=cache.end_times, initialisation=initialisation, delay_hours=DEFAULT_DELAY_HOURS
    )
    correction = correction_at(trace=trace, cutoff=cutoff, target=index, band=band[index])
    spread = post_filter_lead_spread(
        end_times=cache.end_times, replica=replica, run=partial(run_kalman, hyper=hyper)
    )
    corrected = mean + correction.total
    quantiles = spread.quantiles(
        mean=corrected, band=band[index], quarter=cache.calendar.quarter[index], lead_band=0
    )
    quantiles = np.sort(quantiles, axis=1)
    features = cache.sites[site]
    time = cache.end_times[index]
    d10_jump = np.zeros(len(index), dtype=bool)
    if site == "D10":
        for start, stop in common.D10_JUMP_WINDOWS:
            d10_jump |= (time > common.to_64(moment=start)) & (time <= common.to_64(moment=stop))
    return pl.DataFrame(
        {
            "arm": arm,
            "site": site,
            "cut": cut_name,
            "seed": 0,
            "time": pl.Series(time).dt.replace_time_zone("UTC"),
            "month": time.astype("datetime64[M]").astype(int),
            "actual": y,
            "mean": corrected,
            "score": score.quantile_score(actual=y, quantiles=quantiles),
            "abs_error": np.abs(y - corrected),
            "signed_error": y - corrected,
            "in_event": features.switching_mask[index],
            "jump_window": d10_jump,
            **{c: quantiles[:, k] for k, c in enumerate(QUANTILE_COLUMNS)},
        }
    )


def hyper_for(*, cache: FeatureCache, arm: str) -> dict[str, FilterHyperparameters]:
    """Return an arm's hyperparameters: A4 shares A0's, which are fitted on the same stream."""
    return causal.fit_hyper(cache=cache, arm="a1" if arm == "a1" else "a0")


def contrast(
    *, losses: pl.DataFrame, treatment: str, reference: str, metric: str
) -> dict[str, Any]:
    """Return the paired contrast of two arms with its Bonferroni interval and site counts."""
    lower, upper = bootstrap_difference_at_level(
        losses=losses, treatment=treatment, reference=reference, metric=metric, level=LEVEL
    )
    per_site = (
        losses.pivot(on="arm", index=["site", "time"], values=metric)
        .group_by("site")
        .agg((pl.col(treatment) - pl.col(reference)).mean().alias("difference"))
    )
    difference = (
        losses.filter(pl.col("arm") == treatment)[metric].mean()
        - losses.filter(pl.col("arm") == reference)[metric].mean()
    )
    return {
        "difference": float(difference),
        "lower": lower,
        "upper": upper,
        "sites_better": int((per_site["difference"] < 0).sum()),
        "sites": per_site.height,
        "by_site": dict(
            zip(per_site["site"].to_list(), per_site["difference"].to_list(), strict=True)
        ),
    }


def main() -> None:
    """Score the three arms in both years and write the contrasts."""
    cache = common.load_cache()
    frames = []
    for cut_name in ("replicate", "scored"):
        for arm in ARMS:
            hyper = hyper_for(cache=cache, arm=arm)
            for site in SITES:
                found = arm_losses(
                    cache=cache, arm=arm, site=site, cut_name=cut_name, hyper=hyper[site]
                )
                if found is not None:
                    frames.append(found)
    losses = pl.concat(frames)
    LOSSES_PATH.parent.mkdir(parents=True, exist_ok=True)
    losses.write_parquet(LOSSES_PATH)
    results: dict[str, Any] = {}
    for cut_name in ("replicate", "scored"):
        for label, filters in {
            "outside_events": (~pl.col("in_event")),
            "outside_events_and_jump_windows": (~pl.col("in_event")) & (~pl.col("jump_window")),
            "all_rows": pl.lit(True),
        }.items():
            scope = losses.filter(
                pl.col("cut") == cut_name, filters, pl.col("site").is_in(DECISION_SITES)
            )
            results[f"{cut_name}:{label}"] = {
                "a4_minus_a0": contrast(
                    losses=scope, treatment="a4", reference="a0", metric="score"
                ),
                "a4_minus_a1": contrast(
                    losses=scope, treatment="a4", reference="a1", metric="score"
                ),
                "a1_minus_a0": contrast(
                    losses=scope, treatment="a1", reference="a0", metric="score"
                ),
                "a4_minus_a0_mean_error": contrast(
                    losses=scope, treatment="a4", reference="a0", metric="abs_error"
                ),
                "absolute": {
                    arm: float(scope.filter(pl.col("arm") == arm)["score"].mean()) for arm in ARMS
                },
            }
    score.write_json(path=RESULT_PATH, payload=results)
    print(json.dumps(results["scored:outside_events"]["a4_minus_a0"], indent=2))


if __name__ == "__main__":
    main()
