# ruff: noqa: PLR0915
"""Score R0, POST, PRE, and PRE-T on the scored year, with month-resampled paired intervals.

- `R0`: arm D of experiment 155 (rebuilt here; identical to the stored arm D).
- `POST`: experiment 201's arm 2 network applied after the filter (three seeds, each row's score
  averaged over the seeds).
- `PRE`: the network-before-filter arm with POOL's filter noise (`pre_s1.00`).
- `PRE-T`: the arm of `NOISE_SCALES` with the lowest pooled D+1 quantile score on the tuning months.

The resampling unit is the calendar month of the run's origin (12 months in the scored year); each
of 20,000 resamples draws 12 months with replacement and pools their sums and counts. Contrasts are
read on the five decision substations (D5, D10, D11, D14, D16); D9 appears beside them in the
tables. Rows are outside the hand-filtered switching events.

    uv run --no-sync python ml-experiments/00222-*/e00222_score.py
"""

import json
from typing import Final

import e00222_build as build
import e00222_common as common
import e00222_nn as nn_
import numpy as np
import polars as pl

N_RESAMPLES: Final[int] = 20_000
RNG_SEED: Final[int] = 2222
LEVELS: Final[dict[str, float]] = {"95%": 0.95, "99%": 0.99}
POST_SEEDS: Final[tuple[int, ...]] = (0, 1, 2)
D1_FIRST_LEAD: Final[float] = 15.5
D1_LAST_LEAD: Final[float] = 39.0
TAILS: Final[dict[str, int]] = {"q01": 0, "q05": 1, "q95": 9, "q99": 10}
COLD_C: Final[float] = 2.0


def arm_rows(*, name: str, period: str) -> pl.DataFrame:
    """Return an arm's rows of one period at all six sites, outside switching events."""
    frames = [
        pl.read_parquet(build.ARMS_DIR / f"{name}_{site}.parquet")
        .filter((pl.col("period") == period) & ~pl.col("switching_mask"))
        .with_columns(site=pl.lit(site))
        for site in common.SCORED_SITES
    ]
    return pl.concat(frames).sort("site", "power_fcst_init_time", "valid_time")


def mean_score(*, frame: pl.DataFrame, lead_min: float, lead_max: float) -> float:
    """Return the mean quantile score of a frame's rows with leads in a band."""
    rows = frame.filter(pl.col("lead_hours").is_between(lead_min, lead_max))
    return float(
        nn_.row_scores(
            actual=rows["actual"].to_numpy().astype(np.float64),
            quantiles=nn_.quantile_matrix(frame=rows),
        ).mean()
    )


def choose_noise_scale() -> tuple[float, dict[str, float]]:
    """Return the factor of `NOISE_SCALES` with the lowest pooled D+1 tuning-month score."""
    scores = {
        f"{scale:.2f}": mean_score(
            frame=arm_rows(name=f"pre_s{scale:.2f}", period="tuning"),
            lead_min=D1_FIRST_LEAD,
            lead_max=D1_LAST_LEAD,
        )
        for scale in build.NOISE_SCALES
    }
    best = min(scores, key=lambda key: scores[key])
    return float(best), scores


def post_quantiles(*, rows: pl.DataFrame) -> np.ndarray:
    """Return experiment 201's arm 2 mean row score over three seeds, aligned to `rows`."""
    keys = rows.select("site", "run", "valid_time").with_row_index("row")
    scores = []
    for seed in POST_SEEDS:
        preds = pl.read_parquet(
            common.E201_WORK_DIR / "preds" / f"arm2_seed{seed}.parquet",
            columns=["site", "run", "valid_time", *common.QUANTILES],
        )
        joined = keys.join(preds, on=["site", "run", "valid_time"], how="left").sort("row")
        if joined["q50"].null_count() > 0:
            msg = "experiment 201's arm 2 does not cover every scored row"
            raise ValueError(msg)
        scores.append(
            nn_.row_scores(
                actual=rows["actual"].to_numpy().astype(np.float64),
                quantiles=nn_.quantile_matrix(frame=joined),
            )
        )
    return np.mean(scores, axis=0)


def blocked_interval(
    *, diff: np.ndarray, mask: np.ndarray, month: np.ndarray, rng: np.random.Generator
) -> dict[str, float | list[float]]:
    """Return the pooled mean difference over `mask` and its 95% and 99% month-blocked intervals."""
    months = np.unique(month)
    sums = np.array([diff[mask & (month == m)].sum() for m in months])
    counts = np.array([(mask & (month == m)).sum() for m in months], dtype=np.float64)
    draws = rng.integers(0, len(months), size=(N_RESAMPLES, len(months)))
    pooled = sums[draws].sum(axis=1) / np.maximum(counts[draws].sum(axis=1), 1.0)
    out: dict[str, float | list[float]] = {"mean": float(sums.sum() / counts.sum())}
    for label, level in LEVELS.items():
        tail = (1.0 - level) / 2.0 * 100
        out[label] = [float(np.percentile(pooled, tail)), float(np.percentile(pooled, 100 - tail))]
    return out


def level_pinball(*, actual: np.ndarray, q: np.ndarray, index: int) -> float:
    """Return twice the mean pinball loss at one quantile level."""
    error = actual - q[:, index]
    level = nn_.LEVELS[index]
    return float(2.0 * np.maximum(level * error, (level - 1.0) * error).mean())


def main() -> None:
    """Score all arms and write `work/score.json` and `work/scored_rows.parquet`."""
    scale, tuning_scores = choose_noise_scale()
    print(f"PRE-T noise scale {scale:.2f}; tuning D+1 scores {tuning_scores}", flush=True)
    r0 = arm_rows(name="r0", period="scored")
    arms = {
        "R0": r0,
        "PRE": arm_rows(name="pre_s1.00", period="scored"),
        "PRE-T": arm_rows(name=f"pre_s{scale:.2f}", period="scored"),
    }
    key = ["site", "run", "valid_time"]
    for name, frame in arms.items():
        if not frame.select(key).equals(r0.select(key)):
            msg = f"{name}: rows differ from arm D's"
            raise ValueError(msg)
    actual = r0["actual"].to_numpy().astype(np.float64)
    lead = r0["lead_hours"].to_numpy()
    month = r0["power_fcst_init_time"].dt.strftime("%Y-%m").to_numpy()
    features = pl.concat(
        [
            pl.read_parquet(
                common.E201_WORK_DIR / "features" / f"{site}.parquet",
                columns=["run", "valid_time", "drv_temp_c", "true_temp_c"],
            ).with_columns(site=pl.lit(site))
            for site in common.SCORED_SITES
        ]
    )
    joined = r0.select(key).join(features, on=key, how="left")
    temp = joined["drv_temp_c"].to_numpy()
    true_temp = joined["true_temp_c"].to_numpy()
    scores = {
        name: nn_.row_scores(actual=actual, quantiles=nn_.quantile_matrix(frame=f))
        for name, f in arms.items()
    }
    scores["POST"] = post_quantiles(rows=r0)
    site = r0["site"].to_numpy()
    decision = np.isin(site, common.DECISION_SITES)
    d1 = (lead >= D1_FIRST_LEAD) & (lead <= D1_LAST_LEAD)
    hh = r0["local_hh"].to_numpy()
    night = (hh <= 13) | (hh >= 40)
    groups = {
        "d_plus_1": d1,
        "all_leads": np.ones(len(lead), dtype=bool),
        "lead_under_24h": lead < 24,
        "lead_24_72h": (lead >= 24) & (lead < 72),
        "lead_72_168h": lead >= 72,
        "cold_nights_d1": d1 & night & (true_temp < COLD_C),
    }
    rng = np.random.default_rng(RNG_SEED)
    contrasts = {"C1 PRE-POST": ("PRE", "POST"), "C2 PRE-R0": ("PRE", "R0"),
                 "C3 PRE-T minus PRE": ("PRE-T", "PRE"), "POST-R0": ("POST", "R0"),
                 "PRE-T minus POST": ("PRE-T", "POST")}  # fmt: skip
    result: dict[str, object] = {
        "pre_t_noise_scale": scale,
        "tuning_d1_scores": tuning_scores,
        "rows": len(lead),
        "d1_rows_decision": int((d1 & decision).sum()),
    }
    out_scores: dict[str, dict[str, float]] = {}
    for name, per_row in scores.items():
        out_scores[name] = {
            f"{g}{suffix}": float(per_row[m & mask].mean())
            for g, m in groups.items()
            for suffix, mask in (("", decision), ("_six_sites", np.ones_like(decision)))
        }
    result["score"] = out_scores
    out_contrast: dict[str, dict] = {}
    for label, (a, b) in contrasts.items():
        diff = scores[a] - scores[b]
        entry: dict[str, object] = {"by_group": {}, "by_site": {}, "leave_one_out_d1": {}}
        for g, m in groups.items():
            entry["by_group"][g] = blocked_interval(
                diff=diff, mask=m & decision, month=month, rng=rng
            )
        for s in common.SCORED_SITES:
            entry["by_site"][s] = float(diff[d1 & (site == s)].mean())
        for s in common.DECISION_SITES:
            entry["leave_one_out_d1"][s] = float(diff[d1 & decision & (site != s)].mean())
        out_contrast[label] = entry
    result["contrast"] = out_contrast
    # Tails and the error of the mean and the median beside each score (D+1, decision sites).
    tails: dict[str, dict] = {}
    mask = d1 & decision
    for name, frame in {**arms, "POST": None}.items():
        if frame is None:
            continue
        q = nn_.quantile_matrix(frame=frame)
        tails[name] = {
            "coverage": {k: float((actual[mask] <= q[mask, i]).mean()) for k, i in TAILS.items()},
            "pinball": {
                k: level_pinball(actual=actual[mask], q=q[mask], index=i) for k, i in TAILS.items()
            },
            "mae_median": float(np.abs(actual[mask] - q[mask, nn_.MEDIAN_INDEX]).mean()),
            "mae_mean": float(np.abs(actual[mask] - frame["mean"].to_numpy()[mask]).mean()),
            "median_minus_mean_by_site": {
                s: float(
                    (
                        q[d1 & (site == s), nn_.MEDIAN_INDEX]
                        - frame["mean"].to_numpy()[d1 & (site == s)]
                    ).mean()
                )
                for s in common.SCORED_SITES
            },
        }
    result["tails"] = tails
    # Prediction 2: the filter's level by temperature decile (D+1, decision sites).
    edges = np.nanpercentile(temp[mask], np.linspace(0, 100, 11))
    decile = np.clip(np.searchsorted(edges, temp, side="right") - 1, 0, 9)
    result["level_by_decile"] = {
        name: [
            float(np.abs(arms[name]["level"].to_numpy()[mask & (decile == k)]).mean())
            for k in range(10)
        ]
        for name in ("R0", "PRE")
    }
    result["level_signed_by_decile"] = {
        name: [
            float(arms[name]["level"].to_numpy()[mask & (decile == k)].mean()) for k in range(10)
        ]
        for name in ("R0", "PRE")
    }
    result["decile_edges_c"] = [float(e) for e in edges]
    # Per-decile contrast of PRE minus POST with month-blocked 95% intervals.
    diff_c1 = scores["PRE"] - scores["POST"]
    result["c1_by_decile"] = [
        blocked_interval(diff=diff_c1, mask=mask & (decile == k), month=month, rng=rng)
        for k in range(10)
    ]
    out = common.WORK_DIR
    (out / f"score{common.VARIANT}.json").write_text(json.dumps(result, indent=2))
    rows = r0.select(*key, "power_fcst_init_time", "lead_hours", "actual").with_columns(
        pl.Series("temp", temp),
        pl.Series("decile", decile),
        pl.Series("decision", decision),
        *[pl.Series(f"score_{n}", s) for n, s in scores.items()],
        *[pl.Series(f"level_{n}", arms[n]["level"].to_numpy()) for n in ("R0", "PRE")],
        pl.Series("delta_pre", arms["PRE"]["delta"].to_numpy()),
    )
    rows.write_parquet(out / f"scored_rows{common.VARIANT}.parquet")
    for label, entry in out_contrast.items():
        c = entry["by_group"]["d_plus_1"]
        a = entry["by_group"]["all_leads"]
        print(f"{label}: D+1 {c['mean']:+.3f} {c['99%']} | all leads {a['mean']:+.3f} {a['99%']}")
    print({n: out_scores[n]["d_plus_1"] for n in out_scores})


if __name__ == "__main__":
    main()
