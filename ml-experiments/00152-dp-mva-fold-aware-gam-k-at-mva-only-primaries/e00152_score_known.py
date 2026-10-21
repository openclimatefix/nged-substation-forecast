"""Score stage 1: the known-answer checks K1 to K5.

Reads `work/preds/known_*` and the stored fits; writes `work/known.json`. Only ratios, scores,
and coverage shares are written, never a farm's series.
"""

import json
import pickle
from typing import Final

import e00152_common as common
import e00152_inject as inject
import e00152_score as score
import numpy as np
import polars as pl
from studies.dp_features import FeatureCache

LEVELS: Final[tuple[float, ...]] = (0.0, 1.0, 2.0)
MIN_TRUTH_SHARE: Final[float] = 0.05
"""K1 scores rows with `|P_true|` of at least this share of the site's 99th percentile reading."""


def variant_frame(*, site: str, level: float, mode: str) -> pl.DataFrame:
    """Return a variant's scored-year predictions with the score and calendar month."""
    path = common.WORK_DIR / "preds" / f"known_{site}_L{level}_{mode}" / f"scored_{site}.parquet"
    return score.with_score(frame=pl.read_parquet(path).with_columns(pl.lit(site).alias("site")))


def truth(*, cache: FeatureCache, site: str, level: float) -> pl.DataFrame:
    """Return the signed truth per valid time, and the floor on `|P_true|` for K1."""
    signed, _ = inject.signed_and_folded(cache=cache, site=site, level=level, power_factor=False)
    original = cache.sites[site].y
    floor = MIN_TRUTH_SHARE * float(np.nanpercentile(np.abs(original), 99))
    return pl.DataFrame(
        {
            "valid_time": pl.Series(cache.end_times).dt.replace_time_zone("UTC"),
            "signed": signed,
            "floor": np.full(len(signed), floor),
        }
    )


def balanced_accuracy(*, flow: np.ndarray, signed: np.ndarray) -> float:
    """Return the mean of the true-positive and true-negative rates of the flow's sign."""
    predicted_negative = flow < 0.0
    actual_negative = signed < 0.0
    if actual_negative.sum() == 0 or (~actual_negative).sum() == 0:
        return float("nan")
    sensitivity = predicted_negative[actual_negative].mean()
    specificity = (~predicted_negative)[~actual_negative].mean()
    return float((sensitivity + specificity) / 2.0)


def capacity_ratio(*, site: str, cache: FeatureCache) -> dict[str, float]:
    """Return the fitted capacity's rise at level 1.0, over the injected capacity (K4)."""
    injected = inject.midday_median_absolute(cache=cache, site=site)
    last = {}
    for level in (0.0, 1.0):
        fits = pickle.loads(
            (
                common.WORK_DIR / "fits" / f"known_{site}_L{level}_q0" / f"{site}_meter.pkl"
            ).read_bytes()
        )
        capacities = fits[-1].fit.capacity_points()
        last[level] = float(capacities[max(capacities)])
    return {
        "rise": last[1.0] - last[0.0],
        "injected": injected,
        "ratio": (last[1.0] - last[0.0]) / injected,
    }


def main() -> None:
    """Compute K1 to K5 and write `known.json`."""
    cache = common.load_cache()
    out: dict = {"k1": {}, "k2": {}, "k3": {}, "k4": {}, "k5": {}}
    folded = {
        (s, lv, m): variant_frame(site=s, level=lv, mode=m)
        for s in common.DECISION_SITES
        for lv, m in [(0.0, "q0"), (1.0, "q0"), (1.0, "pf97"), (2.0, "q0"), (2.0, "pf97")]
    }
    signed_frames = {
        (s, lv): variant_frame(site=s, level=lv, mode="signed")
        for s in common.DECISION_SITES
        for lv in LEVELS
    }
    for site in common.DECISION_SITES:
        out["k4"][site] = capacity_ratio(site=site, cache=cache)
    for level in (1.0, 2.0):
        for mode in ("q0", "pf97"):
            accuracy = {}
            for site in common.DECISION_SITES:
                frame = folded[(site, level, mode)].filter(pl.col("arm") == "F_fold")
                joined = frame.join(truth(cache=cache, site=site, level=level), on="valid_time")
                joined = joined.filter(pl.col("signed").abs() >= pl.col("floor"))
                accuracy[site] = balanced_accuracy(
                    flow=joined["flow"].to_numpy(), signed=joined["signed"].to_numpy()
                )
            out["k1"][f"L{level}_{mode}"] = accuracy
    for level, mode in [(0.0, "q0"), (1.0, "q0"), (1.0, "pf97"), (2.0, "q0"), (2.0, "pf97")]:
        frame = pl.concat([folded[(s, level, mode)] for s in common.DECISION_SITES])
        out["k2"][f"L{level}_{mode}"] = score.contrast(
            frame=frame, treatment="F_fold", reference="N", sites=list(common.DECISION_SITES)
        )
        out["k2"][f"L{level}_{mode}_F_minus_A"] = score.contrast(
            frame=frame, treatment="F_fold", reference="A", sites=list(common.DECISION_SITES)
        )
        tail = list(
            score.tail_calibration(
                frame=frame.filter(pl.col("sunny_midday") & (pl.col("arm") == "F_fold"))
            )
        )
        out["k5"][f"L{level}_{mode}"] = {
            "below_q05": float(
                np.average([r["q05"] for r in tail], weights=[r["n"] for r in tail])
            ),
            "above_q95": float(
                np.average([r["q95"] for r in tail], weights=[r["n"] for r in tail])
            ),
        }
    for level in (2.0,):
        errors: dict[str, list[np.ndarray]] = {"F_fold": [], "N": [], "S": []}
        for site in common.DECISION_SITES:
            t = truth(cache=cache, site=site, level=level).filter(pl.col("signed") < 0.0)
            for arm, frame in (
                ("F_fold", folded[(site, level, "q0")]),
                ("N", folded[(site, level, "q0")]),
                ("S", signed_frames[(site, level)]),
            ):
                joined = frame.filter(pl.col("arm") == arm).join(t, on="valid_time")
                forecast = joined["flow"] if arm != "N" else joined["q50"]
                errors[arm].append((forecast - joined["signed"]).abs().to_numpy())
        mae = {arm: float(np.concatenate(v).mean()) for arm, v in errors.items()}
        out["k3"][f"L{level}"] = {
            **mae,
            "F_over_S": mae["F_fold"] / mae["S"],
            "N_over_S": mae["N"] / mae["S"],
        }
    out["absolute_score_natural"] = {
        f"{s}|{a}": float(g["score"].mean())  # ty: ignore[invalid-argument-type]
        for s in common.DECISION_SITES
        for (a,), g in folded[(s, 0.0, "q0")].group_by("arm")
    }
    out["signed_ceiling_score"] = {
        f"{s}|L{lv}": float(signed_frames[(s, lv)]["score"].mean())  # ty: ignore[invalid-argument-type]
        for s in common.DECISION_SITES
        for lv in LEVELS
    }
    (common.WORK_DIR / "known.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
