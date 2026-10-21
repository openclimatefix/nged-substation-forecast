r"""Build each arm's correction, spread, spread factors, and forecast frame (experiment 134).

An arm is K0's level and transient plus a profile shift (`e00134_profiles`). Its spread is fitted
on the tuning rows' corrected-mean residuals, and one factor per lead band is chosen on the tuning
rows by the grid of experiment 115, so every arm is built as recipe R was. K0 and R are linked from
experiment 115's frames, which this experiment only reads.

    systemd-run --user --scope -p MemoryMax=8G -p OOMPolicy=continue \\
        uv run --no-sync python ml-experiments/00134-*/e00134_frames.py [--workers 3]
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from typing import Final

import e00134_arms as arms
import e00134_common as common
import e00134_profiles as profiles
import e00134_run as run
import e00134_site as site_module
import e00134_tune as tune
import numpy as np
from studies.dp_ens import LeadHourSpread
from studies.dp_ens_weather import N_LEAD_BANDS

ARM_FAMILIES: Final[dict[str, str]] = {
    "Rh": "Rh",
    "L1": "L1",
    "RW": "RW",
    "RB": "RB",
    "RB-day-types": "RBdt",
}
FACTORS_PATH: Final = common.WORK_DIR / "factors.json"


def _correction_and_spread(
    *, arm: str, site: str
) -> tuple[site_module.SiteData, arms.Correction, LeadHourSpread]:
    data = run.get_site(site)
    k0, q50 = run.load_k0(site=site)
    spec = run.chosen_spec(family=ARM_FAMILIES[arm])
    correction = profiles.profile_correction(data=data, k0=k0, k0_q50=q50, spec=spec)
    spread = arms.fit_spread(data=data, correction=correction.total)
    assert spread is not None, site
    return data, correction, spread


def _factor_job(job: tuple[str, str]) -> dict:
    arm, site = job
    data, correction, spread = _correction_and_spread(arm=arm, site=site)
    scores = {
        str(factor): arms.tuning_band_scores(
            data=data,
            correction=correction.total,
            spread=spread.with_factors(factors=[factor] * N_LEAD_BANDS),
        ).tolist()
        for factor in arms.FACTOR_GRID
    }
    arms.save_correction(arm=arm, site=site, correction=correction)
    arms.save_spread(arm=arm, site=site, spread=spread)
    return {"arm": arm, "site": site, "scores": scores}


def _frame_job(job: tuple[str, str, list[float]]) -> str:
    arm, site, factors = job
    data = run.get_site(site)
    correction = arms.load_correction(arm=arm, site=site)
    spread = arms.load_spread(arm=arm, site=site).with_factors(factors=factors)
    frame = arms.frame_for_arm(data=data, arm=arm, correction=correction, spread=spread)
    frame = arms.replace_period_masks(frame=frame, data=data)
    target = tune.frame_path(arm=arm, site=site)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(target)
    return f"{arm} {site}: {frame.height:,} rows"


def link_references() -> None:
    """Link experiment 115's K0 and R frames into this experiment's frames folder."""
    for arm in ("K0", "R"):
        for site in common.SCORED_SITES:
            target = tune.frame_path(arm=arm, site=site)
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                target.symlink_to(run.SRC_FRAMES_DIR / f"{arm}_{site}.parquet")


def main() -> None:
    """Build every arm's frames."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    link_references()
    jobs = [(arm, site) for arm in ARM_FAMILIES for site in common.SCORED_SITES]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(_factor_job, jobs))
        factors: dict[str, list[float]] = {}
        for arm in ARM_FAMILIES:
            per_site = {r["site"]: r["scores"] for r in results if r["arm"] == arm}
            factors[arm] = tune.choose_factors(scores=per_site)
        common.write_json_atomic(path=FACTORS_PATH, content=factors)
        print(json.dumps(factors))
        for line in pool.map(_frame_job, [(a, s, factors[a]) for a, s in jobs]):
            print(line, flush=True)


if __name__ == "__main__":
    np.seterr(all="warn")
    main()
