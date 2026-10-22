"""Score every arm against arm D on the scored year, with month-blocked paired intervals.

The unit of resampling is the calendar month of the run's origin (12 months in the scored year). A
contrast is the mean over seeds of (arm's quantile score minus arm D's), pooled over the rows of a
group; each of 20,000 resamples draws 12 months with replacement and pools their sums and counts.
The planned intervals are 97.5% (four planned contrasts: arms 1 to 4); the blind control (arm 0)
is post hoc.

    uv run --no-sync python ml-experiments/00201-*/e00201_score.py
"""

import json
from typing import Final

import e00201_common as common
import e00201_data as data
import e00201_nn as nn_
import numpy as np
import polars as pl

N_RESAMPLES: Final[int] = 20_000
LEVEL: Final[float] = 0.975
SEEDS: Final[tuple[int, ...]] = (0, 1, 2)
RNG_SEED: Final[int] = 2010
COLD_NIGHT_C: Final[float] = 2.0
SUNNY_W_M2: Final[float] = 400.0
SUNNY_ELEVATION_DEG: Final[float] = 20.0


def group_masks(*, frame: pl.DataFrame) -> dict[str, np.ndarray]:
    """Return the row groups of the write-up."""
    lead = frame["lead_hours"].to_numpy()
    hh = frame["local_hh"].to_numpy()
    night = (hh <= 13) | (hh >= 40)
    return {
        "all": np.ones(frame.height, dtype=bool),
        "d_plus_1": (lead >= common.D1_FIRST_LEAD) & (lead <= common.D1_LAST_LEAD),
        "lead_under_24h": lead < 24,
        "lead_24_72h": (lead >= 24) & (lead < 72),
        "lead_72_168h": lead >= 72,
        "cold_nights": night & (frame["true_temp_c"].to_numpy() < COLD_NIGHT_C),
        "sunny_daylight": (frame["drv_sw_w_m2"].to_numpy() > SUNNY_W_M2)
        & (frame["elevation_deg"].to_numpy() > SUNNY_ELEVATION_DEG),
        "christmas": frame["christmas"].to_numpy(),
        "bank_holidays": frame["bank_holiday"].to_numpy() & ~frame["christmas"].to_numpy(),
        "economy_7_nights": frame["e7"].to_numpy(),
    }


def blocked_interval(
    *, diff: np.ndarray, mask: np.ndarray, month: np.ndarray, rng: np.random.Generator
) -> tuple[float, float, float]:
    """Return the pooled mean difference over `mask` and its 97.5% month-blocked interval."""
    months = np.unique(month)
    sums = np.array([diff[mask & (month == m)].sum() for m in months])
    counts = np.array([(mask & (month == m)).sum() for m in months], dtype=np.float64)
    draws = rng.integers(0, len(months), size=(N_RESAMPLES, len(months)))
    pooled = sums[draws].sum(axis=1) / np.maximum(counts[draws].sum(axis=1), 1.0)
    tail = (1.0 - LEVEL) / 2.0 * 100
    return (
        float(sums.sum() / counts.sum()),
        float(np.percentile(pooled, tail)),
        float(np.percentile(pooled, 100 - tail)),
    )


def level_pinball(*, actual: np.ndarray, q: np.ndarray, index: int) -> float:
    """Return twice the mean pinball loss at one quantile level."""
    error = actual - q[:, index]
    level = nn_.LEVELS[index]
    return float(2.0 * np.maximum(level * error, (level - 1.0) * error).mean())


def main() -> None:
    """Score all arms and write `work/score.json`."""
    frame = data.load_rows()
    frame = frame.filter(pl.col("period") == "scored")
    q_d = nn_.quantile_matrix(frame=frame)
    actual = frame["actual"].to_numpy().astype(np.float64)
    month = frame["month_key"].to_numpy()
    masks = group_masks(frame=frame)
    rng = np.random.default_rng(RNG_SEED)
    score_d = nn_.row_scores(actual=actual, quantiles=q_d)
    tails = {"q01": 0, "q05": 1, "q95": 9, "q99": 10}
    result: dict[str, dict] = {"D": {}}
    result["D"]["score"] = {g: float(score_d[m].mean()) for g, m in masks.items()}
    result["D"]["coverage"] = {k: float((actual <= q_d[:, i]).mean()) for k, i in tails.items()}
    result["D"]["median_minus_mean"] = {
        s: float(
            frame.filter(pl.col("site") == s).select((pl.col("q50") - pl.col("mean")).mean()).item()
        )
        for s in common.SCORED_SITES
    }
    result["D"]["mae_median"] = float(np.abs(actual - q_d[:, nn_.MEDIAN_INDEX]).mean())
    for arm in (*data.ARMS, *data.ABLATION_ARMS):
        quantiles = []
        for seed in SEEDS:
            preds = pl.read_parquet(common.WORK_DIR / "preds" / f"arm{arm}_seed{seed}.parquet")
            quantiles.append(preds.select(common.QUANTILES).to_numpy())
        scores = np.mean([nn_.row_scores(actual=actual, quantiles=q) for q in quantiles], axis=0)
        diff = scores - score_d
        entry: dict = {"contrast": {}, "score": {}, "coverage": {}, "by_site": {}, "leave_out": {}}
        for group, mask in masks.items():
            mean_diff, low, high = blocked_interval(diff=diff, mask=mask, month=month, rng=rng)
            entry["contrast"][group] = [mean_diff, low, high]
            entry["score"][group] = float(scores[mask].mean())
        for key, index in tails.items():
            entry["coverage"][key] = float(
                np.mean([(actual <= q[:, index]).mean() for q in quantiles])
            )
        pin = {
            key: float(np.mean([level_pinball(actual=actual, q=q, index=i) for q in quantiles]))
            for key, i in tails.items()
        }
        entry["tail_pinball"] = pin
        for site in common.SCORED_SITES:
            site_mask = (frame["site"] == site).to_numpy()
            entry["by_site"][site] = float(diff[site_mask].mean())
            entry["leave_out"][site] = blocked_interval(
                diff=diff, mask=~site_mask, month=month, rng=rng
            )[0]
        entry["mae_median"] = float(
            np.mean([np.abs(actual - q[:, nn_.MEDIAN_INDEX]).mean() for q in quantiles])
        )
        result[f"arm{arm}"] = entry
    result["D"]["tail_pinball"] = {
        key: level_pinball(actual=actual, q=q_d, index=i) for key, i in tails.items()
    }
    (common.WORK_DIR / "score.json").write_text(json.dumps(result, indent=2))
    for arm in (*data.ARMS, *data.ABLATION_ARMS):
        c = result[f"arm{arm}"]["contrast"]
        print(
            f"arm {arm}: all {c['all'][0]:+.3f} [{c['all'][1]:+.3f}, {c['all'][2]:+.3f}]  "
            f"D+1 {c['d_plus_1'][0]:+.3f}  cold {c['cold_nights'][0]:+.3f}  "
            f"sunny {c['sunny_daylight'][0]:+.3f}  xmas {c['christmas'][0]:+.3f}  "
            f"bank {c['bank_holidays'][0]:+.3f}  e7 {c['economy_7_nights'][0]:+.3f}"
        )
    print("D score", result["D"]["score"]["all"], "coverage", result["D"]["coverage"])


if __name__ == "__main__":
    main()
