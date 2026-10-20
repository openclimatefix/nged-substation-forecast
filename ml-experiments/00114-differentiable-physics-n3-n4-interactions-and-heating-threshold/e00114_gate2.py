"""Gate 2 of experiment 114: does T-all beat B0 when an interaction exists, and not lose if not?

**The synthetic substations.** The truth is B0's real fit at the 2024-07-01 cut. **Truth I** then
lowers the weekend heating slopes to `1 - 0.4 a` times the weekday slopes and raises the weekend
lighting ramps to `1 + 0.5 a` times the weekday ramps, for an amplitude `a`. **Truth 0** is B0's fit
with `a = 0`. The noise is whole donor months of the other decision substations' B0 residuals
(`e00114_noise`), on the training rows and on the held-out year alike, so no 2024-25 residual
enters. The synthetic target is net power plus the pinned solar output, so the solar term cancels
in every residual.

**The amplitude is set before the 100 draws, on 5 pilot draws** (one per substation, seeds apart
from the main draws), so that the interaction's gain for the true-parameter forecast over B0's fit
on the held-out synthetic year is 0.03 points, the decision threshold. The gain is the gap between
B0's fit and the true-parameter forecast under truth I, minus the same gap under truth 0, because
the gap under truth 0 is the estimation error that any fitted arm pays.

**Each draw is scored with B0's spread rule.** Every arm is fitted at the 2023-07-01 cut on the
synthetic year before the main cut, its residuals on the following year give the centred residual
quantiles per hour band and calendar quarter, and the arm fitted at 2024-07-01 forecasts the
held-out year with that spread. The gate tests the unfiltered model only.

**Pass** if (a) on truth I, T-all at the tuned penalty weight beats B0 in at least 80% of the 100
draws, and (b) on truth 0, T-all is no worse than B0 by more than 0.01 points on average.

    uv run python ml-experiments/00114-*/e00114_gate2.py run [--workers 5]
    uv run python ml-experiments/00114-*/e00114_gate2.py summary
"""

import argparse
import hashlib
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from itertools import product
from typing import Any, Final

import e00114_common as common
import e00114_design as design
import e00114_noise as noise
import numpy as np
import polars as pl
import torch
from studies.dp_features import FeatureCache
from studies.dp_gam import GamLayout, ResidualSpread, design_matrix
from studies.dp_pinned_capacity import read_vintages

N_DRAWS_PER_SITE: Final[int] = 20
N_PILOT_DRAWS: Final[int] = 5
TARGET_GAIN_POINTS: Final[float] = 0.03
WEEKEND_HEATING_DROP: Final[float] = 0.4
WEEKEND_LIGHTING_RISE: Final[float] = 0.5
MIN_WIN_SHARE: Final[float] = 0.80
MAX_TRUTH_ZERO_LOSS: Final[float] = 0.01
AMPLITUDE_BRACKET: Final[tuple[float, float]] = (0.0, 6.0)
BISECTION_STEPS: Final[int] = 14
GATE_SEED_TAG: Final[int] = 2
PILOT_DRAW_ID: Final[int] = 99
N_BOOTSTRAP_DRAWS: Final[int] = 2000

_STATE: dict[Any, Any] = {}


def verified_decisions() -> dict[str, Any]:
    """Return `decisions.json` after checking its hash against `decisions.sha256`.

    Returns:
        The decisions.

    Raises:
        ValueError: If the file no longer matches its recorded hash.
    """
    text = (common.WORK_DIR / "decisions.json").read_text()
    recorded = (common.WORK_DIR / "decisions.sha256").read_text().strip()
    if hashlib.sha256(text.encode()).hexdigest() != recorded:
        msg = "decisions.json no longer matches the hash recorded before the gates"
        raise ValueError(msg)
    return json.loads(text)


@dataclass
class SiteSetup:
    """Everything about one substation that does not change from draw to draw.

    Attributes:
        site: The substation's label.
        layout: B0's columns.
        theta_b0: B0's real coefficients at the main cut.
        t_all: The T-all arm, whose columns the truth is written in.
        rows: The union of every row set, ascending.
        positions: The position in `rows` of each row set, by name.
        matrices: B0's and T-all's design on each row set.
        grams: B0's and T-all's Gram matrix on the two training sets.
        penalties: B0's and T-all's penalty on the two training sets.
        bands: Hour band and quarter of each test set's rows.
        keep: True where a test row is outside the switching masks.
    """

    site: str
    layout: GamLayout
    theta_b0: np.ndarray
    t_all: design.ArmSpec
    rows: np.ndarray
    positions: dict[str, np.ndarray]
    matrices: dict[tuple[str, str], np.ndarray]
    grams: dict[tuple[str, str], torch.Tensor]
    penalties: dict[tuple[str, str], design.ArmPenalty]
    bands: dict[str, tuple[np.ndarray, np.ndarray]]
    keep: dict[str, np.ndarray]


def build_setup(
    *, cache: FeatureCache, vintages: Any, site: str, t_all: design.ArmSpec
) -> SiteSetup:
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
    matrices: dict[tuple[str, str], np.ndarray] = {}
    grams: dict[tuple[str, str], torch.Tensor] = {}
    penalties: dict[tuple[str, str], design.ArmPenalty] = {}
    for arm_name, arm in (("b0", design.B0), ("t_all", t_all)):
        for name, index in sets.items():
            x0 = design_matrix(cache=cache, features=features, layout=main.layout, index=index)
            matrices[(arm_name, name)] = design.arm_matrix(
                cache=cache, layout=main.layout, site=site, index=index, x0=x0, arm=arm
            )
        for name, base in (("train_main", main), ("train_rep", rep)):
            x = matrices[(arm_name, name)]
            xt = torch.as_tensor(x)
            grams[(arm_name, name)] = xt.T @ xt
            penalties[(arm_name, name)] = design.arm_penalty(
                layout=base.layout, arm=arm, n_rows=base.n
            )
    bands = {
        name: (cache.calendar.band[sets[name]], cache.calendar.quarter[sets[name]])
        for name in ("test_main", "test_rep")
    }
    keep = {name: ~features.switching_mask[sets[name]] for name in ("test_main", "test_rep")}
    # The truth is written in T-all's columns, so the full-row matrix of T-all is kept too.
    all_x0 = design_matrix(cache=cache, features=features, layout=main.layout, index=rows)
    matrices[("t_all", "all")] = design.arm_matrix(
        cache=cache, layout=main.layout, site=site, index=rows, x0=all_x0, arm=t_all
    )
    return SiteSetup(
        site=site,
        layout=main.layout,
        theta_b0=theta_b0,
        t_all=t_all,
        rows=rows,
        positions=positions,
        matrices=matrices,
        grams=grams,
        penalties=penalties,
        bands=bands,
        keep=keep,
    )


def truth_theta(*, setup: SiteSetup, amplitude: float) -> np.ndarray:
    """Return the truth's coefficients in T-all's columns.

    Args:
        setup: The substation's inputs.
        amplitude: The interaction amplitude `a`; 0 gives B0's fit.

    Returns:
        T-all's coefficients: B0's for the columns T-all keeps, and for each split column its
        pieces, with the weekend pieces scaled.
    """
    layout = setup.layout
    heating = layout.slices["heating"]
    replaced = design.replaced_columns(layout=layout, arm=setup.t_all)
    drop = {column for column, _ in replaced}
    keep = [j for j in range(len(setup.theta_b0)) if j not in drop]
    pieces: list[float] = list(setup.theta_b0[keep])
    for column, factors in replaced:
        weekend_scale = (
            1.0 - WEEKEND_HEATING_DROP * amplitude
            if heating.start <= column < heating.stop
            else 1.0 + WEEKEND_LIGHTING_RISE * amplitude
        )
        for combo in product((0, 1), repeat=len(factors)):
            is_weekend = combo[factors.index("daytype")] == 1
            pieces.append(float(setup.theta_b0[column]) * (weekend_scale if is_weekend else 1.0))
    return np.array(pieces)


def fit_and_score(*, setup: SiteSetup, arm_name: str, z: np.ndarray) -> float:
    """Fit an arm to the synthetic target at the replica and main cuts and return its mean score.

    Args:
        setup: The substation's inputs.
        arm_name: `b0` or `t_all`.
        z: The synthetic target on every row of the setup.

    Returns:
        The mean quantile score on the held-out year outside the switching masks.
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
    return spread_score(setup=setup, z=z, rep_mean=rep_mean, main_mean=main_mean)


def spread_score(
    *, setup: SiteSetup, z: np.ndarray, rep_mean: np.ndarray, main_mean: np.ndarray
) -> float:
    """Return the mean quantile score of forecasts whose spread comes from their replica."""
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
    keep = setup.keep["test_main"]
    actual = z[setup.positions["test_main"]]
    return common.as_float(
        common.quantile_score(actual=actual[keep], quantiles=quantiles[keep]).mean()
    )


def oracle_score(
    *, setup: SiteSetup, truth_mean: np.ndarray, noise_values: np.ndarray, z: np.ndarray
) -> float:
    """Return the true-parameter forecast's score, with a spread from the noise alone."""
    rep_mean = z[setup.positions["test_rep"]] - noise_values[setup.positions["test_rep"]]
    main_mean = truth_mean[setup.positions["test_main"]]
    return spread_score(setup=setup, z=z, rep_mean=rep_mean, main_mean=main_mean)


def one_draw(
    *, setup: SiteSetup, pool: noise.ResidualPool, site_position: int, draw: int, amplitude: float
) -> dict[str, float]:
    """Run one draw under truth I and truth 0 with the same noise and return the scores."""
    rng = np.random.default_rng([common.EXPERIMENT_ID, GATE_SEED_TAG, site_position, draw])
    noise_values = pool.draw(site=setup.site, index=setup.rows, rng=rng)
    out: dict[str, float] = {"noise_sd": float(noise_values.std())}
    for kind, a in (("i", amplitude), ("zero", 0.0)):
        truth_mean = setup.matrices[("t_all", "all")] @ truth_theta(setup=setup, amplitude=a)
        z = truth_mean + noise_values
        out[f"{kind}_b0"] = fit_and_score(setup=setup, arm_name="b0", z=z)
        out[f"{kind}_t_all"] = fit_and_score(setup=setup, arm_name="t_all", z=z)
        out[f"{kind}_oracle"] = oracle_score(
            setup=setup, truth_mean=truth_mean, noise_values=noise_values, z=z
        )
    return out


def _init_worker() -> None:
    """Load the cache, vintages, and donors once per worker process."""
    common.configure_threads()
    cache = common.load_cache()
    _STATE["cache"] = cache
    _STATE["vintages"] = read_vintages()
    _STATE["pool"] = noise.build_pool(cache=cache)


def _setup(*, site: str, lam: float) -> SiteSetup:
    """Build one substation's setup at the T-all weight `lam`, cached for this process."""
    key = (site, lam)
    if key not in _STATE:
        _STATE[key] = build_setup(
            cache=_STATE["cache"],
            vintages=_STATE["vintages"],
            site=site,
            t_all=design.interaction_arm(name="t_all", penalty_weight=lam),
        )
    return _STATE[key]


def _pilot_gap(*, lam: float, amplitude: float) -> float:
    """Return the pilot draws' mean interaction gain at an amplitude, net of truth 0's gap."""
    gaps = []
    for position, site in enumerate(common.DECISION_SITES):
        scores = one_draw(
            setup=_setup(site=site, lam=lam),
            pool=_STATE["pool"],
            site_position=position,
            draw=PILOT_DRAW_ID,
            amplitude=amplitude,
        )
        gaps.append(
            (scores["i_b0"] - scores["i_oracle"]) - (scores["zero_b0"] - scores["zero_oracle"])
        )
    return float(np.mean(gaps))


def calibrate_amplitude(*, lam: float) -> dict[str, float]:
    """Return the amplitude at which the pilot gain equals `TARGET_GAIN_POINTS` (bisection)."""
    low, high = AMPLITUDE_BRACKET
    gain_high = _pilot_gap(lam=lam, amplitude=high)
    if gain_high < TARGET_GAIN_POINTS:
        msg = f"even amplitude {high} gives a pilot gain of only {gain_high:.4f} points"
        raise ValueError(msg)
    for _ in range(BISECTION_STEPS):
        middle = 0.5 * (low + high)
        if _pilot_gap(lam=lam, amplitude=middle) < TARGET_GAIN_POINTS:
            low = middle
        else:
            high = middle
    amplitude = 0.5 * (low + high)
    return {"amplitude": amplitude, "pilot_gain": _pilot_gap(lam=lam, amplitude=amplitude)}


def _site_job(job: tuple[str, int, float, float]) -> list[dict[str, Any]]:
    """Run one substation's main draws."""
    site, position, lam, amplitude = job
    setup = _setup(site=site, lam=lam)
    rows = []
    for draw in range(N_DRAWS_PER_SITE):
        scores = one_draw(
            setup=setup, pool=_STATE["pool"], site_position=position, draw=draw, amplitude=amplitude
        )
        rows.append({"site": site, "draw": draw, **scores})
        print(
            f"{site} draw {draw}: " + " ".join(f"{k}={v:.3f}" for k, v in scores.items()),
            flush=True,
        )
    return rows


def _fit_lambda_job(job: tuple[str, int, float, float]) -> list[dict[str, Any]]:
    """Run one substation's draws for one penalty weight (for the all-weights table)."""
    return _site_job(job)


def run(*, workers: int) -> None:
    """Calibrate the amplitude, run the 100 draws at every penalty weight, and store them."""
    decisions = verified_decisions()
    tuned = float(decisions["lambda"]["t_all"])
    common.load_cache()
    with ProcessPoolExecutor(max_workers=1, initializer=_init_worker) as one:
        calibration = one.submit(calibrate_amplitude, lam=tuned).result()
    print(
        f"amplitude {calibration['amplitude']:.4f} gives a pilot gain of "
        f"{calibration['pilot_gain']:.4f}"
    )
    (common.WORK_DIR / "gate2_amplitude.json").write_text(json.dumps(calibration, indent=2) + "\n")
    frames = []
    for lam in common.LAMBDA_GRID:
        jobs = [
            (site, position, lam, calibration["amplitude"])
            for position, site in enumerate(common.DECISION_SITES)
        ]
        with ProcessPoolExecutor(max_workers=workers, initializer=_init_worker) as pool:
            results = list(pool.map(_fit_lambda_job, jobs))
        frame = pl.DataFrame([row for site_rows in results for row in site_rows])
        frames.append(frame.with_columns(penalty_weight=pl.lit(lam)))
    pl.concat(frames).write_parquet(common.WORK_DIR / "gate2_draws.parquet")


def summary() -> dict[str, Any]:
    """Print and return gate 2's criteria from the stored draws."""
    tuned = float(verified_decisions()["lambda"]["t_all"])
    frame = pl.read_parquet(common.WORK_DIR / "gate2_draws.parquet")
    calibration = json.loads((common.WORK_DIR / "gate2_amplitude.json").read_text())
    rng = np.random.default_rng(common.EXPERIMENT_ID)
    report: dict[str, Any] = {"tuned_penalty_weight": tuned, **calibration, "by_weight": {}}
    for lam in common.LAMBDA_GRID:
        part = frame.filter(pl.col("penalty_weight") == lam)
        entry: dict[str, Any] = {}
        for kind in ("i", "zero"):
            difference = (part[f"{kind}_t_all"] - part[f"{kind}_b0"]).to_numpy()
            resampled = rng.choice(difference, size=(N_BOOTSTRAP_DRAWS, len(difference))).mean(
                axis=1
            )
            entry[kind] = {
                "mean_t_all_minus_b0": common.as_float(difference.mean()),
                "interval_95": [
                    float(np.percentile(resampled, 2.5)),
                    float(np.percentile(resampled, 97.5)),
                ],
                "share_t_all_better": common.as_float((difference < 0.0).mean()),
                "wins_by_site": {
                    site: int(
                        (
                            part.filter(pl.col("site") == site)[f"{kind}_t_all"]
                            < part.filter(pl.col("site") == site)[f"{kind}_b0"]
                        ).sum()
                    )
                    for site in common.DECISION_SITES
                },
                "oracle_gap_b0": common.as_float(
                    (part[f"{kind}_b0"] - part[f"{kind}_oracle"]).mean()
                ),
                "oracle_gap_t_all": common.as_float(
                    (part[f"{kind}_t_all"] - part[f"{kind}_oracle"]).mean()
                ),
            }
        report["by_weight"][f"{lam:g}"] = entry
    chosen = report["by_weight"][f"{tuned:g}"]
    report["a_pass"] = chosen["i"]["share_t_all_better"] >= MIN_WIN_SHARE
    report["b_pass"] = chosen["zero"]["mean_t_all_minus_b0"] <= MAX_TRUTH_ZERO_LOSS
    report["passed"] = bool(report["a_pass"] and report["b_pass"])
    report["noise_sd_mean"] = common.as_float(frame["noise_sd"].mean())
    (common.WORK_DIR / "gate2.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
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
