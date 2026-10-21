r"""Build each arm's correction, spread, spread factors, and forecast frame (experiment 145).

An arm is K0's level and transient plus an analog profile shift (`e00145_analog`). Its spread is
fitted on the tuning rows' corrected-mean residuals, and one factor per lead band is chosen on the
tuning rows by the grid of experiment 115, so every arm is built as recipe R was. K0 and R are
linked from experiment 115's frames and RB from experiment 134's, which this experiment only reads.

    OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 \\
    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \\
        uv run --no-sync python ml-experiments/00145-*/e00145_frames.py [--workers 4]
"""

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Final

import e00145_analog as analog
import e00145_arms as arms
import e00145_common as common
import e00145_run as run
import e00145_site as site_module
import numpy as np
from studies.dp_ens import LeadHourSpread
from studies.dp_ens_weather import N_LEAD_BANDS

ARM_FAMILIES: Final[dict[str, str]] = {
    "ANALOG-k": "ANALOG-k",
    "ANALOG+R": "ANALOG+R",
    "ANALOG-holiday": "ANALOG-holiday",
    "ANALOG+R-notemp": "ANALOG+R-notemp",
    "ANALOG+R-recent": "ANALOG+R-recent",
}
FACTORS_PATH: Final[Path] = common.WORK_DIR / "factors.json"
LINKED: Final[tuple[tuple[str, Path, str], ...]] = (
    ("K0", run.SRC_FRAMES_DIR, "K0"),
    ("R", run.SRC_FRAMES_DIR, "R"),
    ("RB", common.SRC_134_WORK_DIR / "frames", "RB"),
    ("RB-day-types", common.SRC_134_WORK_DIR / "frames", "RB-day-types"),
)


def _correction_and_spread(
    *, arm: str, site: str
) -> tuple[site_module.SiteData, arms.Correction, LeadHourSpread]:
    data = run.get_site(site)
    k0, _q50 = run.load_k0(site=site)
    days = run.site_days(site=site)
    spec = run.chosen_spec(family=ARM_FAMILIES[arm])
    shift = analog.shift_at_rows(data=data, by_run=analog.analog_shift_by_run(days=days, spec=spec))
    correction = arms.Correction(level=k0.level, anomaly=k0.anomaly, shape=shift)
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
    target = arms.frame_path(arm=arm, site=site)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.write_parquet(target)
    return f"{arm} {site}: {frame.height:,} rows"


def choose_factors(*, scores: dict[str, dict[str, list]]) -> list[float]:
    """Return, per lead band, the factor with the lowest pooled tuning score at decision sites."""
    totals = np.zeros((len(arms.FACTOR_GRID), N_LEAD_BANDS, 2))
    for site, per_factor in scores.items():
        if site not in common.DECISION_SITES:
            continue
        for position, factor in enumerate(arms.FACTOR_GRID):
            totals[position] += np.array(per_factor[str(factor)])
    mean = totals[:, :, 0] / totals[:, :, 1]
    return [float(arms.FACTOR_GRID[i]) for i in mean.argmin(axis=0)]


def link_references() -> None:
    """Link the stored frames of K0, R, RB, and RB-day-types into this experiment's folder."""
    for arm, folder, source_arm in LINKED:
        for site in common.SCORED_SITES:
            target = arms.frame_path(arm=arm, site=site)
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                target.symlink_to(folder / f"{source_arm}_{site}.parquet")


def main() -> None:
    """Build every arm's frames."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    link_references()
    jobs = [(arm, site) for arm in ARM_FAMILIES for site in common.SCORED_SITES]
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(_factor_job, jobs))
        factors: dict[str, list[float]] = {}
        for arm in ARM_FAMILIES:
            per_site = {r["site"]: r["scores"] for r in results if r["arm"] == arm}
            factors[arm] = choose_factors(scores=per_site)
        common.write_json_atomic(path=FACTORS_PATH, content=factors)
        print(json.dumps(factors))
        for line in pool.map(_frame_job, [(a, s, factors[a]) for a, s in jobs]):
            print(line, flush=True)


if __name__ == "__main__":
    np.seterr(all="warn")
    main()
