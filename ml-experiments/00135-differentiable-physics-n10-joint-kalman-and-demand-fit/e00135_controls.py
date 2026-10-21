"""Controls and diagnostics of experiment 135: the true-loss check, shape distance, and absorption.

- **P3, the solved loss is the true loss.** At the cuts given, the joint fit's penalised loss, with
  the correction taken from `run_kalman` on the fitted residuals, is no larger than the two-stage
  fit's, and a perturbed copy of the joint fit has a larger one.
- **Shape distance.** For every monthly cut, the root-mean-square difference between each joint
  arm's fitted curves and the two-stage curves, per term, in points and as a share of the two-stage
  curve's root mean square about its mean, and with the mean difference removed (a constant shift
  of a term, which the filter's level can absorb, is not a change of shape). A curve is a term's
  columns times its coefficients over the training rows.
- **Absorption control.** Over the last 180 days before a cut, and over a middle 180 days, a level
  change of 2 points and a PV-shaped change (30% of the pinned solar output) are added to the
  decision substations' targets. The two-stage and joint fits are refitted with and without the
  addition, and the largest change of each term's curve is reported against a noise floor: the
  same change when the cut moves back by 7 days. A joint arm fails when a term moves by more than
  the larger of 0.10 points and twice its own noise floor, and by more than twice the larger of the
  two-stage fit's movement and noise floor, because a plain fit should move when the data really
  change. A term whose own noise floor exceeds 1 point is unidentified. PV is pinned in this
  design, so no fitted PV curve exists, and the PV-shaped change can only move the other terms.

    uv run --no-sync python ml-experiments/00135-*/e00135_controls.py [--sites D5 D10]
"""

import argparse
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from typing import Final

import e00135_arms as arms
import e00135_common as common
import e00135_fits as fits_module
import numpy as np
from studies.dp_assimilation import FilterHyperparameters, correction_at, run_kalman
from studies.dp_features import FeatureCache
from studies.dp_gam import (
    GamFit,
    TrainingWindow,
    design_matrix,
    fit_gam,
    penalty_matrix,
    pinned_solar_points,
    prepare_fit,
)
from studies.dp_joint_fit import (
    fit_joint_gam,
    loss_cases,
    stream_design,
    stream_rows,
)
from studies.dp_pinned_capacity import PinnedCapacity, pinned_capacity, read_vintages

TERMS: Final[tuple[str, ...]] = ("profile", "holiday", "tent", "harmonics", "heating", "lighting")
ABSORPTION_CUTS: Final[tuple[datetime, ...]] = (
    datetime(2025, 1, 1, tzinfo=UTC),
    datetime(2025, 7, 1, tzinfo=UTC),
    datetime(2026, 1, 1, tzinfo=UTC),
)
INJECTED_DAYS: Final[int] = 180
LEVEL_CHANGE_POINTS: Final[float] = 2.0
PV_SHARE: Final[float] = 0.30
FAIL_FLOOR_POINTS: Final[float] = 0.10
NOISE_FLOOR_MULTIPLE: Final[float] = 2.0
UNIDENTIFIED_POINTS: Final[float] = 1.0
RIDGE_FOR_J2: Final[float] = 1.0
WINDOW: Final[TrainingWindow] = TrainingWindow(months=fits_module.WINDOW_MONTHS)


def curves(*, cache: FeatureCache, fit: GamFit, index: np.ndarray) -> dict[str, np.ndarray]:
    """Return each term's fitted curve in points over the rows `index`."""
    features = cache.sites[fit.site]
    design = design_matrix(cache=cache, features=features, layout=fit.layout, index=index)
    return {
        term: design[:, fit.layout.slices[term]] @ fit.theta[0][fit.layout.slices[term]]
        for term in TERMS
    }


def shape_distance(
    *, cache: FeatureCache, joint: GamFit, two_stage: GamFit, index: np.ndarray
) -> dict[str, dict[str, float]]:
    """Return the curves' root-mean-square difference in points and as a share of the reference."""
    joint_curves = curves(cache=cache, fit=joint, index=index)
    reference = curves(cache=cache, fit=two_stage, index=index)
    out = {}
    for term in TERMS:
        difference = np.sqrt(np.mean((joint_curves[term] - reference[term]) ** 2))
        spread = np.sqrt(np.mean((reference[term] - reference[term].mean()) ** 2))
        offset = joint_curves[term] - reference[term]
        out[term] = {
            "rms_difference_points": float(difference),
            "rms_difference_centred_points": float(np.sqrt(np.mean((offset - offset.mean()) ** 2))),
            "share_of_reference": float(difference / spread) if spread > 0 else float("nan"),
        }
    return out


def true_penalised_loss(
    *, cache: FeatureCache, fit: GamFit, hyper: np.ndarray, pinned: PinnedCapacity | None
) -> float:
    """Return the joint loss of a fit with the correction taken from `run_kalman`.

    The loss is the sum of the squared corrected errors at the loss cases, plus the quadratic and
    linear penalties of `fit_gam`.
    """
    layout, index = prepare_fit(
        cache=cache,
        site=fit.site,
        spec=fit.layout.spec,
        cut=fit.cut,
        window=WINDOW,
        season_years=common.SEASON_YEARS,
    )
    start = cache.end_times[index[0]] - np.timedelta64(30, "m")
    rows = stream_rows(cache=cache, site=fit.site, cut=fit.cut, start=start)
    columns, target = stream_design(
        cache=cache, site=fit.site, layout=layout, rows=rows, pinned=pinned
    )
    residual = np.full(len(cache.end_times), np.nan)
    residual[rows] = target - columns @ fit.theta[0]
    trace = run_kalman(
        residual=residual,
        band=cache.calendar.band,
        hyper=FilterHyperparameters.from_array(values=hyper),
    )
    case_target, case_cutoff = loss_cases(cache=cache, index=index, rows=rows)
    correction = correction_at(
        trace=trace,
        cutoff=case_cutoff,
        target=case_target,
        band=cache.calendar.band[case_target],
    ).total
    error = residual[case_target] - correction
    matrix, cost = penalty_matrix(layout=layout, n_rows=len(index))
    theta = fit.theta[0]
    return float(error @ error + theta @ matrix @ theta + cost @ theta)


def true_loss_check(*, site: str, cut_index: int) -> dict[str, float]:
    """Return the joint, two-stage, and perturbed joint losses at one cut of one site."""
    cache = common.load_cache()
    two_stage = fits_module.load_fits(arm="gamk", site=site)[cut_index]
    joint = fits_module.load_fits(arm="j1", site=site)[cut_index]
    hyper = arms.read_hyper(site=site)
    pinned = pinned_capacity(
        vintages=read_vintages(), table=cache.sites[site].table, cut=two_stage.cut
    )
    rng = np.random.default_rng(0)
    perturbed = replace(
        joint,
        theta=np.where(joint.layout.constrained, np.abs(joint.theta), joint.theta)
        * (1.0 + 0.01 * rng.normal(size=joint.theta.shape)),
    )
    return {
        name: true_penalised_loss(cache=cache, fit=fit, hyper=hyper, pinned=pinned)
        for name, fit in (("joint", joint), ("two_stage", two_stage), ("perturbed", perturbed))
    }


def injected_cache(
    *,
    cache: FeatureCache,
    site: str,
    cut: datetime,
    kind: str,
    where: str,
    pinned: PinnedCapacity,
) -> FeatureCache:
    """Return the cache with a 180-day change added to one substation's target."""
    features = cache.sites[site]
    end = np.datetime64(cut.replace(tzinfo=None), "us")
    last = end - np.timedelta64(INJECTED_DAYS, "D") if where == "middle" else end
    window = (cache.end_times > last - np.timedelta64(INJECTED_DAYS, "D")) & (
        cache.end_times <= last
    )
    rows = np.flatnonzero(window & np.isfinite(features.y))
    change = np.zeros(len(features.y))
    if kind == "level":
        change[rows] = LEVEL_CHANGE_POINTS
    else:
        change[rows] = -PV_SHARE * pinned_solar_points(
            cache=cache, features=features, pinned=pinned, index=rows
        )
    return replace(cache, sites={**cache.sites, site: replace(features, y=features.y + change)})


def fit_three(
    *, cache: FeatureCache, site: str, cut: datetime, hyper: np.ndarray
) -> dict[str, GamFit]:
    """Fit the two-stage, J1, and J2 (ridge 1) arms at one cut of one site."""
    pinned = pinned_capacity(vintages=read_vintages(), table=cache.sites[site].table, cut=cut)
    spec = fits_module.gamk_spec()
    two_stage = fit_gam(
        cache=cache,
        site=site,
        specs=[spec],
        cut=cut,
        window=WINDOW,
        season_years=common.SEASON_YEARS,
        pinned=pinned,
    )
    j1, j2 = fit_joint_gam(
        cache=cache,
        site=site,
        spec=spec,
        cut=cut,
        window=WINDOW,
        season_years=common.SEASON_YEARS,
        hyper=hyper,
        pinned=pinned,
        anchor=two_stage,
        ridges=(0.0, RIDGE_FOR_J2),
    )
    return {"ts": two_stage, "j1": j1, "j2": j2}


def movement(
    *, cache: FeatureCache, before: dict[str, GamFit], after: dict[str, GamFit], index: np.ndarray
) -> dict[str, dict[str, float]]:
    """Return the largest absolute change of each term's curve for each arm, in points."""
    return {
        arm: {
            term: float(
                np.abs(
                    curves(cache=cache, fit=after[arm], index=index)[term]
                    - curves(cache=cache, fit=before[arm], index=index)[term]
                ).max()
            )
            for term in TERMS
        }
        for arm in before
    }


def absorption_control(*, site: str, cut: datetime) -> dict:
    """Return the noise floor and the movement under each injection at one cut of one site."""
    cache = common.load_cache()
    hyper = arms.read_hyper(site=site)
    base = fit_three(cache=cache, site=site, cut=cut, hyper=hyper)
    _, index = prepare_fit(
        cache=cache,
        site=site,
        spec=fits_module.gamk_spec(),
        cut=cut,
        window=WINDOW,
        season_years=common.SEASON_YEARS,
    )
    floor_fits = fit_three(cache=cache, site=site, cut=cut - timedelta(days=7), hyper=hyper)
    out = {"noise_floor": movement(cache=cache, before=base, after=floor_fits, index=index)}
    pinned = pinned_capacity(vintages=read_vintages(), table=cache.sites[site].table, cut=cut)
    for kind in ("level", "pv"):
        for where in ("end", "middle"):
            changed = injected_cache(
                cache=cache, site=site, cut=cut, kind=kind, where=where, pinned=pinned
            )
            fits = fit_three(cache=changed, site=site, cut=cut, hyper=hyper)
            out[f"{kind}_{where}"] = movement(cache=cache, before=base, after=fits, index=index)
    return out


def verdict(*, result: dict) -> dict[str, dict[str, bool]]:
    """Return, for each joint arm, whether it fails the control or has an unidentified term.

    A term fails in a scenario when its curve moves by more than the larger of `FAIL_FLOOR_POINTS`
    and `NOISE_FLOOR_MULTIPLE` times the arm's own noise floor in the term, and by more than
    `NOISE_FLOOR_MULTIPLE` times the larger of the two-stage fit's movement and noise floor in the
    term. The two-stage fit is the reference because a plain least-squares fit should move when the
    data really change. A term is unidentified when the arm's own noise floor exceeds
    `UNIDENTIFIED_POINTS`, so a shift of its curve cannot be read at all.
    """
    out = {}
    for arm in ("j1", "j2"):
        failed = False
        unidentified = False
        for term in TERMS:
            floor = result["noise_floor"][arm][term]
            unidentified |= floor > UNIDENTIFIED_POINTS
            for scenario, moves in result.items():
                if scenario == "noise_floor":
                    continue
                reference = max(moves["ts"][term], result["noise_floor"]["ts"][term])
                failed |= (
                    moves[arm][term] > max(FAIL_FLOOR_POINTS, NOISE_FLOOR_MULTIPLE * floor)
                    and moves[arm][term] > NOISE_FLOOR_MULTIPLE * reference
                )
        out[arm] = {"fails": bool(failed), "has_unidentified_term": bool(unidentified)}
    return out


def main() -> None:
    """Run the three controls and write `work/controls.json`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sites", nargs="+", default=list(common.DECISION_SITES))
    parser.add_argument("--skip-absorption", action="store_true")
    args = parser.parse_args()
    cache = common.load_cache()
    out: dict = {"true_loss": {}, "shape_distance": {}, "absorption": {}}
    for site in args.sites:
        two_stage = fits_module.load_fits(arm="gamk", site=site)
        out["true_loss"][site] = true_loss_check(site=site, cut_index=len(two_stage) // 2)
        out["shape_distance"][site] = {}
        for arm in ("j1", *common.J2_ARM_NAMES):
            fits = fits_module.load_fits(arm=arm, site=site)
            out["shape_distance"][site][arm] = [
                {
                    "cut": fit.cut.isoformat(),
                    **shape_distance(
                        cache=cache,
                        joint=fit,
                        two_stage=reference,
                        index=prepare_fit(
                            cache=cache,
                            site=site,
                            spec=fit.layout.spec,
                            cut=fit.cut,
                            window=WINDOW,
                            season_years=common.SEASON_YEARS,
                        )[1],
                    ),
                }
                for fit, reference in zip(fits, two_stage, strict=True)
            ]
        if not args.skip_absorption:
            out["absorption"][site] = {}
            for cut in ABSORPTION_CUTS:
                result = absorption_control(site=site, cut=cut)
                out["absorption"][site][cut.isoformat()] = {
                    **result,
                    "pass": verdict(result=result),
                }
    common.write_json_atomic(path=common.WORK_DIR / "controls.json", content=out)
    print(f"wrote {common.WORK_DIR / 'controls.json'}")


if __name__ == "__main__":
    main()
