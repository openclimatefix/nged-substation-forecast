"""Gates 4 and 4' of experiment 121: can the night-step arms find an injected step, and where?

**The synthetic substations.** The truth is B0's real fit at the 2024-07-01 cut plus an injected
step. The noise is whole donor months of the other decision substations' B0 residuals
(`e00121_noise`), on the training rows and on the held-out year alike, so no 2024-25 residual
enters. The synthetic target is net power plus the pinned solar output, so the solar term cancels.
Three truths, each with 100 draws (20 per decision substation), all on the same noise:

- **`utc`**: a flat 4-point step in the half-hour starting 00:30 UTC (the 01:00 step), all year.
- **`local`**: the same step in the half-hour starting 00:30 on the local clock, all year.
- **`cold`**: a step in the half-hour starting 00:30 UTC equal to `c` times the 15.5 degree hinge
  of the first smoothed temperature, with `c` set so the mean step over November to February rows
  in that half-hour is 4 points. It is zero in summer, as storage-heating demand is.

**Every arm is fitted to every truth** (unfiltered, as in gate 2 of experiment 114), scored with
B0's spread rule, and scored on all rows, on British Summer Time nights (April to October, 22:00
to 08:00 local), and on winter nights (November to February, 22:00 to 08:00 local).

**Gate 4 as written in the plan.** (a) on each of `utc` and `local`, the matching arm of NS-L and
NS-U beats the other on British Summer Time nights in at least 80% of draws; (b) on winter nights,
NS-L and NS-U both beat B0 on both truths in at least 90% of draws; (c) on each truth, the
mismatched arm's fitted level at the injected half-hour is under half the injection.

**Gate 4' (clarification before any replicate or scored fit).** (a) on `utc`, NS-U beats B0 on
British Summer Time nights in at least 80% of draws, and on `local`, B0 beats NS-U there in at
least 80% of draws; (b) on `cold`, NS-HW beats B0 on winter nights in at least 90% of draws and the
95% interval of the mean difference lies wholly below zero.

    uv run python ml-experiments/00121-*/e00121_gate4.py run [--workers 5]
    uv run python ml-experiments/00121-*/e00121_gate4.py summary
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any, Final

import e00121_common as common
import e00121_design as design
import e00121_noise as noise
import numpy as np
import polars as pl
import torch
from studies.dp_features import FeatureCache
from studies.dp_gam import GamLayout, ResidualSpread, design_matrix
from studies.dp_pinned_capacity import read_vintages

N_DRAWS_PER_SITE: Final[int] = 20
INJECTED_POINTS: Final[float] = 4.0
INJECTED_HALF_HOUR: Final[int] = 1
"""The half-hour of day (start 00:30) of the 01:00 step."""
ARM_NAMES: Final[tuple[str, ...]] = ("b0", "ns_l", "ns_u", "ns_h", "ns_hw")
TRUTHS: Final[tuple[str, ...]] = ("utc", "local", "cold")
SUBSETS: Final[tuple[str, ...]] = ("all", "bst_night", "winter_night")
GATE_SEED_TAG: Final[int] = 4
BST_MONTHS: Final[tuple[int, ...]] = (4, 5, 6, 7, 8, 9, 10)
WINTER_MONTHS: Final[tuple[int, ...]] = (11, 12, 1, 2)
N_BOOTSTRAP_DRAWS: Final[int] = 2000

_STATE: dict[Any, Any] = {}


def night_and_month(*, cache: FeatureCache, index: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return, per row, whether it starts 22:00 to 08:00 local, and its local calendar month."""
    half = design.start_half_hour(cache=cache, index=index, local=True)
    night = (half >= common.NIGHT_START_HALF_HOUR) | (
        half < common.NIGHT_HALF_HOURS - (design.HALF_HOURS_PER_DAY - common.NIGHT_START_HALF_HOUR)
    )
    start = pl.Series(cache.end_times[index] - np.timedelta64(30, "m"))
    month = (
        start.dt.replace_time_zone("UTC")
        .dt.convert_time_zone(design.LOCAL_ZONE)
        .dt.month()
        .to_numpy()
    )
    return night, month


@dataclass
class SiteSetup:
    """Everything about one substation that does not change from draw to draw.

    Attributes:
        site: The substation's label.
        layout: B0's columns.
        theta_b0: B0's real coefficients at the main cut.
        inflation: The ratio of B0's out-of-sample to in-sample RMSE on the tuning year.
        rows: The union of every row set, ascending.
        positions: The position in `rows` of each row set, by name.
        matrices: Each arm's design on each row set (and on `all`).
        grams: Each arm's Gram matrix on the two training sets.
        penalties: Each arm's penalty on the two training sets.
        bands: Hour band and quarter of each test set's rows.
        keep: True where a test row is outside the switching masks.
        masks: Row subsets of the main test year (`bst_night`, `winter_night`), after `keep`.
        steps: The injected step on every row of `rows`, by truth.
    """

    site: str
    layout: GamLayout
    theta_b0: np.ndarray
    inflation: float
    rows: np.ndarray
    positions: dict[str, np.ndarray]
    matrices: dict[tuple[str, str], np.ndarray]
    grams: dict[tuple[str, str], torch.Tensor]
    penalties: dict[tuple[str, str], design.ArmPenalty]
    bands: dict[str, tuple[np.ndarray, np.ndarray]]
    keep: dict[str, np.ndarray]
    masks: dict[str, np.ndarray]
    steps: dict[str, np.ndarray]


def injected_steps(*, cache: FeatureCache, site: str, rows: np.ndarray) -> dict[str, np.ndarray]:
    """Return the three injected steps on the grid rows `rows`."""
    utc = design.start_half_hour(cache=cache, index=rows, local=False) == INJECTED_HALF_HOUR
    local = design.start_half_hour(cache=cache, index=rows, local=True) == INJECTED_HALF_HOUR
    hinge = design.cache_hinge(cache=cache, site=site, index=rows)[:, 0]
    _, month = night_and_month(cache=cache, index=rows)
    winter_rows = utc & np.isin(month, WINTER_MONTHS)
    scale = INJECTED_POINTS / float(hinge[winter_rows].mean())
    if not (np.isfinite(hinge).all() and np.isfinite(scale)):
        msg = f"{site}: the cold step is not finite"
        raise ValueError(msg)
    return {
        "utc": INJECTED_POINTS * utc.astype(np.float64),
        "local": INJECTED_POINTS * local.astype(np.float64),
        "cold": scale * hinge * utc,
    }


def build_setup(*, cache: FeatureCache, vintages: Any, site: str) -> SiteSetup:
    """Build one substation's fixed synthetic-data inputs."""
    main = design.build_base(cache=cache, vintages=vintages, site=site, cut=common.REPLICATE_CUT)
    rep = design.build_base(cache=cache, vintages=vintages, site=site, cut=common.TUNING_CUT)
    theta_b0 = design.fit_arm(cache=cache, base=main, arm=design.B0).theta
    sets = {
        "train_main": main.index,
        "train_rep": rep.index,
        "test_main": common.scoring_index(
            cache=cache,
            site=site,
            after=common.REPLICATE_CUT,
            through=common.year_end(cut=common.REPLICATE_CUT),
        ),
        "test_rep": common.scoring_index(
            cache=cache,
            site=site,
            after=common.TUNING_CUT,
            through=common.year_end(cut=common.TUNING_CUT),
        ),
    }
    rows = np.unique(np.concatenate(list(sets.values())))
    positions = {name: np.searchsorted(rows, index) for name, index in sets.items()}
    features = cache.sites[site]
    arms = {name: design.all_arms()[name] for name in ARM_NAMES}
    matrices: dict[tuple[str, str], np.ndarray] = {}
    grams: dict[tuple[str, str], torch.Tensor] = {}
    penalties: dict[tuple[str, str], design.ArmPenalty] = {}
    all_x0 = design_matrix(cache=cache, features=features, layout=main.layout, index=rows)
    for arm_name, arm in arms.items():
        for name, index in sets.items():
            x0 = design_matrix(cache=cache, features=features, layout=main.layout, index=index)
            matrices[(arm_name, name)] = design.arm_matrix(
                cache=cache, layout=main.layout, site=site, index=index, x0=x0, arm=arm
            )
        for name, base in (("train_main", main), ("train_rep", rep)):
            x = matrices[(arm_name, name)]
            xt = torch.as_tensor(x)
            grams[(arm_name, name)] = xt.T @ xt
            penalties[(arm_name, name)] = design.penalty_for(cache=cache, base=base, arm=arm, x=x)
    matrices[("b0", "all")] = all_x0
    bands = {
        name: (cache.calendar.band[sets[name]], cache.calendar.quarter[sets[name]])
        for name in ("test_main", "test_rep")
    }
    keep = {name: ~features.switching_mask[sets[name]] for name in ("test_main", "test_rep")}
    night, month = night_and_month(cache=cache, index=sets["test_main"])
    masks = {
        "all": keep["test_main"],
        "bst_night": keep["test_main"] & night & np.isin(month, BST_MONTHS),
        "winter_night": keep["test_main"] & night & np.isin(month, WINTER_MONTHS),
    }
    return SiteSetup(
        site=site,
        layout=main.layout,
        theta_b0=theta_b0,
        inflation=noise.tuning_inflation(cache=cache, vintages=vintages, site=site),
        rows=rows,
        positions=positions,
        matrices=matrices,
        grams=grams,
        penalties=penalties,
        bands=bands,
        keep=keep,
        masks=masks,
        steps=injected_steps(cache=cache, site=site, rows=rows),
    )


def fit_and_score(*, setup: SiteSetup, arm_name: str, z: np.ndarray) -> dict[str, float]:
    """Fit an arm to the synthetic target at both cuts; return the score on each subset.

    Args:
        setup: The substation's inputs.
        arm_name: A member of `ARM_NAMES`.
        z: The synthetic target on every row of the setup.

    Returns:
        The mean quantile score on the held-out year, by subset, plus `level` (the fitted level
        of night column 5, the injected half-hour, for the night level arms; NaN otherwise).
    """
    thetas = {}
    for cut_name in ("rep", "main"):
        x_train = setup.matrices[(arm_name, f"train_{cut_name}")]
        target = z[setup.positions[f"train_{cut_name}"]]
        gram = setup.grams[(arm_name, f"train_{cut_name}")][None]
        rhs = torch.as_tensor(x_train.T @ target)[None]
        thetas[cut_name] = design.solve(
            gram=gram, rhs=rhs, penalty=setup.penalties[(arm_name, f"train_{cut_name}")]
        )[0]
    rep_mean = setup.matrices[(arm_name, "test_rep")] @ thetas["rep"]
    main_mean = setup.matrices[(arm_name, "test_main")] @ thetas["main"]
    keep_rep = setup.keep["test_rep"]
    residual = z[setup.positions["test_rep"]] - rep_mean
    band_rep, quarter_rep = setup.bands["test_rep"]
    spread = ResidualSpread.fit(
        residuals=residual[keep_rep],
        band=band_rep[keep_rep],
        quarter=quarter_rep[keep_rep],
        centre=True,
    )
    band_main, quarter_main = setup.bands["test_main"]
    quantiles = spread.quantiles(mean=main_mean, band=band_main, quarter=quarter_main)
    actual = z[setup.positions["test_main"]]
    per_row = common.quantile_score(actual=actual, quantiles=quantiles)
    out = {subset: float(per_row[mask].mean()) for subset, mask in setup.masks.items()}
    level_index = 5
    out["level"] = (
        float(thetas["main"][-common.NIGHT_HALF_HOURS + level_index])
        if arm_name in ("ns_l", "ns_u")
        else float("nan")
    )
    return out


def one_draw(
    *, setup: SiteSetup, pool: noise.ResidualPool, site_position: int, draw: int
) -> dict[str, float]:
    """Run one draw under the three truths with the same noise and return the scores."""
    rng = np.random.default_rng([common.EXPERIMENT_ID, GATE_SEED_TAG, site_position, draw])
    noise_values = setup.inflation * pool.draw(site=setup.site, index=setup.rows, rng=rng)
    base_mean = setup.matrices[("b0", "all")] @ setup.theta_b0
    out: dict[str, float] = {}
    for truth in TRUTHS:
        z = base_mean + setup.steps[truth] + noise_values
        for arm_name in ARM_NAMES:
            scores = fit_and_score(setup=setup, arm_name=arm_name, z=z)
            for key, value in scores.items():
                out[f"{truth}_{arm_name}_{key}"] = value
    return out


def _init_worker() -> None:
    """Load the cache, vintages, and donors once per worker process."""
    common.configure_threads()
    cache = common.load_cache()
    _STATE["cache"] = cache
    _STATE["vintages"] = read_vintages()
    _STATE["pool"] = noise.build_pool(cache=cache)


def _site_job(job: tuple[str, int]) -> list[dict[str, Any]]:
    """Run one substation's draws."""
    site, position = job
    setup = build_setup(cache=_STATE["cache"], vintages=_STATE["vintages"], site=site)
    rows = []
    for draw in range(N_DRAWS_PER_SITE):
        scores = one_draw(setup=setup, pool=_STATE["pool"], site_position=position, draw=draw)
        rows.append({"site": site, "draw": draw, **scores})
        print(f"{site} draw {draw} done", flush=True)
    return rows


def run(*, workers: int) -> None:
    """Run the 100 draws and store them."""
    common.load_cache()
    jobs = list(enumerate(common.DECISION_SITES))
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as pool:
        results = list(pool.map(_site_job, [(site, position) for position, site in jobs]))
    frame = pl.DataFrame([row for site_rows in results for row in site_rows])
    frame.write_parquet(common.WORK_DIR / "gate4_draws.parquet")


def _share_better(
    *, frame: pl.DataFrame, truth: str, first: str, second: str, subset: str
) -> float:
    """Return the share of draws in which `first` scores lower than `second`."""
    a = frame[f"{truth}_{first}_{subset}"].to_numpy()
    b = frame[f"{truth}_{second}_{subset}"].to_numpy()
    return float((a < b).mean())


def summary() -> dict[str, Any]:
    """Print and return the criteria of gate 4 as written and of gate 4' from the stored draws."""
    frame = pl.read_parquet(common.WORK_DIR / "gate4_draws.parquet")
    rng = np.random.default_rng(common.EXPERIMENT_ID)
    report: dict[str, Any] = {"n_draws": frame.height}
    shares: dict[str, float] = {}
    for truth in TRUTHS:
        for arm in ("ns_l", "ns_u", "ns_h", "ns_hw"):
            for subset in SUBSETS:
                shares[f"{truth}:{arm}_beats_b0:{subset}"] = _share_better(
                    frame=frame, truth=truth, first=arm, second="b0", subset=subset
                )
        shares[f"{truth}:ns_u_beats_ns_l:bst_night"] = _share_better(
            frame=frame, truth=truth, first="ns_u", second="ns_l", subset="bst_night"
        )
        shares[f"{truth}:ns_l_beats_ns_u:bst_night"] = _share_better(
            frame=frame, truth=truth, first="ns_l", second="ns_u", subset="bst_night"
        )
        shares[f"{truth}:b0_beats_ns_u:bst_night"] = _share_better(
            frame=frame, truth=truth, first="b0", second="ns_u", subset="bst_night"
        )
    report["shares"] = shares
    levels = {
        f"{truth}:{arm}": float(np.nanmean(np.abs(frame[f"{truth}_{arm}_level"].to_numpy())))
        for truth in TRUTHS
        for arm in ("ns_l", "ns_u")
    }
    report["mean_abs_fitted_level_at_injected_half_hour"] = levels
    a = (
        shares["utc:ns_u_beats_ns_l:bst_night"] >= 0.8
        and shares["local:ns_l_beats_ns_u:bst_night"] >= 0.8
    )
    b = all(
        shares[f"{truth}:{arm}_beats_b0:winter_night"] >= 0.9
        for truth in ("utc", "local")
        for arm in ("ns_l", "ns_u")
    )
    c = levels["utc:ns_l"] < 0.5 * INJECTED_POINTS and levels["local:ns_u"] < 0.5 * INJECTED_POINTS
    report["gate4_as_written"] = {"a": a, "b": b, "c": c, "passed": bool(a and b and c)}
    difference = (frame["cold_ns_hw_winter_night"] - frame["cold_b0_winter_night"]).to_numpy()
    resampled = rng.choice(difference, size=(N_BOOTSTRAP_DRAWS, len(difference))).mean(axis=1)
    interval = [float(np.percentile(resampled, 2.5)), float(np.percentile(resampled, 97.5))]
    prime_a = (
        shares["utc:ns_u_beats_b0:bst_night"] >= 0.8
        and shares["local:b0_beats_ns_u:bst_night"] >= 0.8
    )
    prime_b = bool(shares["cold:ns_hw_beats_b0:winter_night"] >= 0.9 and interval[1] < 0.0)
    report["gate4_prime"] = {
        "a": prime_a,
        "b": prime_b,
        "ns_hw_minus_b0_winter_night_mean": float(difference.mean()),
        "ns_hw_minus_b0_winter_night_interval_95": interval,
        "passed": bool(prime_a and prime_b),
    }
    report["mean_scores"] = {
        column: common.as_float(frame[column].mean())
        for column in frame.columns
        if column.endswith(tuple(f"_{s}" for s in SUBSETS))
    }
    (common.WORK_DIR / "gate4.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: v for k, v in report.items() if k != "mean_scores"}, indent=2))
    return report


def main() -> None:
    """Run the draws or print the summary."""
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=["run", "summary"])
    parser.add_argument("--workers", type=int, default=5)
    arguments = parser.parse_args()
    if arguments.action == "run":
        run(workers=arguments.workers)
    summary()


if __name__ == "__main__":
    main()
