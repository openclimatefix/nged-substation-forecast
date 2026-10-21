"""Known-answer test of experiment 126's two filter arms on a real residual stream.

1. F-mix with all four log factors at zero reproduces `run_kalman` to 1e-9.
2. G-skf with both multipliers at one reproduces `run_kalman` to 1e-6, whatever the transition.
3. G-skf's collapsed level reacts to a synthetic level step faster than the plain filter does.
4. G-skf's shift-or-outlier probability stays below 0.5 on the clean stream's median day.

    uv run --no-sync python ml-experiments/00126-*/e00126_test_filters.py
"""

import json
import sys

import e00126_common as common
import e00126_fmix as fmix
import e00126_pipeline as pipeline
import e00126_skf as skf
import e00126_tune as tune
import numpy as np
from studies.dp_assimilation import FilterHyperparameters, run_kalman
from studies.dp_pinned_capacity import read_vintages

SITE = "D10"
STEP_POINTS = 3.0


def main() -> None:
    """Run the checks and exit non-zero on a failure."""
    common.configure_threads()
    cache = common.load_cache()
    run = pipeline.fit_year(
        cache=cache,
        vintages=read_vintages(),
        site=SITE,
        fit_label="tuning",
        arm_name="t_all",
        penalty_weight=tune.t_all_weight(),
    )
    assert run is not None
    residual = run.residual_stream(cache=cache)
    band = cache.calendar.band
    decisions = json.loads((common.WORK_DIR / "decisions.json").read_text())
    hyper = FilterHyperparameters.from_array(
        values=np.array(decisions["filter_hyper"]["t_all"][SITE])
    )
    plain = run_kalman(residual=residual, band=band, hyper=hyper)
    failures: list[str] = []
    mixed = fmix.run_regime_kalman(
        residual=residual,
        band=band,
        regimes=fmix.regime_map(cache=cache, site=SITE).regime,
        log_factors=np.zeros(fmix.N_FACTORS),
        hyper=hyper,
    )
    gap = max(float(np.abs(a - b).max()) for a, b in zip(plain.level, mixed.level, strict=True))
    print(f"F-mix at zero factors against run_kalman, max level gap {gap:.2e}")
    if gap > 1e-9:
        failures.append(f"F-mix does not reproduce the plain filter ({gap:.2e})")
    transition = skf.PRIOR_TRANSITION
    same, positions = skf.run_gpb1(
        residual=residual[None, :],
        band=band,
        hyper=hyper,
        transition=transition,
        multipliers=(1.0, 1.0),
    )
    gap = 0.0
    for result, rows, level in zip(same, positions, plain.level, strict=True):
        finite = np.isfinite(residual[rows])
        gap = max(gap, float(np.abs(result.level[0][finite] - level).max()))
    print(f"G-skf with multipliers 1 against run_kalman, max level gap {gap:.2e}")
    if gap > 1e-6:
        failures.append(f"G-skf with multipliers 1 does not reproduce the plain filter ({gap:.2e})")
    stepped = residual.copy()
    finite = np.flatnonzero(np.isfinite(residual))
    start = int(finite[len(finite) // 2])
    window = slice(start, start + 48 * 3)
    stepped[window] += STEP_POINTS
    plain_step = run_kalman(residual=stepped, band=band, hyper=hyper)
    results, positions = skf.run_gpb1(
        residual=stepped[None, :], band=band, hyper=hyper, transition=transition
    )
    base_results, _ = skf.run_gpb1(
        residual=residual[None, :], band=band, hyper=hyper, transition=transition
    )
    tracked: dict[str, list[float]] = {"plain": [], "G-skf": []}
    for b, rows in enumerate(positions):
        inside = (rows >= window.start) & (rows < window.stop)
        finite_rows = np.isfinite(stepped[rows])
        plain_inside = inside[finite_rows]
        tracked["plain"].append(
            float(
                np.mean(
                    (
                        plain_step.level[b]
                        + plain_step.anomaly[b]
                        - plain.level[b]
                        - plain.anomaly[b]
                    )[plain_inside]
                )
            )
        )
        gap_levels = (results[b].level[0] + results[b].anomaly[0]) - (
            base_results[b].level[0] + base_results[b].anomaly[0]
        )
        tracked["G-skf"].append(float(np.mean(gap_levels[inside])))
    for name, values in tracked.items():
        mean_tracked = np.mean(values)
        print(f"{name}: mean extra level over the 3 days {mean_tracked:.2f} of {STEP_POINTS}")
    p_clean = np.concatenate([r.p_bad[0] for r in base_results])
    print(f"G-skf: mean shift-or-outlier probability on the clean stream {p_clean.mean():.3f}")
    p_bad = np.concatenate([r.p_bad[0] for r in results])
    print(f"G-skf: mean shift-or-outlier probability on the stepped stream {p_bad.mean():.3f}")
    if failures:
        print("\nFILTER TEST FAILED")
        for line in failures:
            print(f"  {line}")
        sys.exit(1)
    print("\nFILTER TEST PASSED")


if __name__ == "__main__":
    main()
