"""Experiment 113's analytic diagnostics: absorption ratios, collinearity share, spectral check.

For each substation and window, with the full sample's Gram matrix `G` and penalty `P`
(`A = G + P`, restricted to a fit's columns):

- **Absorption ratio** `r = v^T A0^-1 v / v^T A^-1 v`: the exact variance of the functional
  `c = v^T theta` in the smaller fit (A0) over the full fit (A), under iid noise.
- **Collinearity share** `rho = (v^T v)^2 / (v^T A v . v^T A^-1 v)`, descriptive only.
- **Spectral check**: the share of the partialled column's variance at periods under 10 days.
  The partialled column is `X v` minus its penalised projection on all other columns of the fit.
"""

import json
from dataclasses import replace

import e00113_common as common
import e00113_design as design_module
import numpy as np
import polars as pl

SPECTRAL_PERIOD_DAYS = 10.0
HALF_HOURS_PER_DAY = 48


def _arms_at(*, penalty: float) -> dict[str, design_module.Arm]:
    return {
        "full": replace(design_module.ARMS["g_full"], drift_step=penalty),
        "g_0": replace(design_module.ARMS["g_0"], drift_step=penalty),
        "g_h": replace(design_module.ARMS["g_h"], drift_step=penalty),
        "g_d": replace(design_module.ARMS["g_d"], drift_step=penalty),
    }


def _inverse_quadratic(*, design: design_module.Design, arm: design_module.Arm) -> np.ndarray:
    keep = design_module.selection(design=design, arm=arm)
    chosen = np.flatnonzero(keep)
    gram = design_module.summed_gram(design=design).numpy()
    matrix = gram[np.ix_(chosen, chosen)] + design.penalties[arm.drift_step][np.ix_(chosen, chosen)]
    v = design.functionals[chosen]
    return np.einsum("pf,pf->f", v, np.linalg.solve(matrix, v))


def spectral_share(*, design: design_module.Design, column: np.ndarray) -> float:
    """Return the share of a zero-filled series' variance at periods shorter than 10 days."""
    low = design.index[0]
    series = np.zeros(design.index[-1] - low + 1)
    series[design.index - low] = column - column.mean()
    power = np.abs(np.fft.rfft(series)) ** 2
    frequency = np.fft.rfftfreq(len(series), d=1.0)
    cutoff = 1.0 / (SPECTRAL_PERIOD_DAYS * HALF_HOURS_PER_DAY)
    return float(power[frequency > cutoff].sum() / power[1:].sum())


def partialled_column(
    *, design: design_module.Design, arm: design_module.Arm, functional: int
) -> np.ndarray:
    """Return `X v` minus its penalised projection on the arm's other columns."""
    keep = design_module.selection(design=design, arm=arm)
    v = design.functionals[:, functional]
    own = v != 0.0
    others = np.flatnonzero(keep & ~own)
    gram = design_module.summed_gram(design=design).numpy()
    matrix = gram[np.ix_(others, others)] + design.penalties[arm.drift_step][np.ix_(others, others)]
    cross = gram[others] @ v
    coefficients = np.linalg.solve(matrix, cross)
    return design.x @ v - design.x[:, others] @ coefficients


def analytics_for(*, design: design_module.Design) -> list[dict]:
    """Return the analytic rows of one design: r, rho, and the spectral shares."""
    rows = []
    gram = design_module.summed_gram(design=design).numpy()
    for penalty in (common.DRIFT_STEP_PART_2, common.DRIFT_STEP_PART_4):
        arms = _arms_at(penalty=penalty)
        full = _inverse_quadratic(design=design, arm=arms["full"])
        matrix = gram + design.penalties[penalty]
        v = design.functionals
        quad = np.einsum("pf,pq,qf->f", v, matrix, v)
        rho = np.einsum("pf,pf->f", v, v) ** 2 / (quad * full)
        for name in ("g_0", "g_h", "g_d"):
            small = _inverse_quadratic(design=design, arm=arms[name])
            for functional, label in enumerate(design_module.FUNCTIONAL_NAMES):
                rows.append(
                    {
                        "site": design.site,
                        "window": design.window,
                        "drift_step": penalty,
                        "removed": name,
                        "functional": label,
                        "r": float(small[functional] / full[functional]),
                        "rho": float(rho[functional]),
                    }
                )
    for name in ("full", "g_0"):
        arm = _arms_at(penalty=common.DRIFT_STEP_PART_2)[name]
        for functional, label in enumerate(design_module.FUNCTIONAL_NAMES):
            column = partialled_column(design=design, arm=arm, functional=functional)
            rows.append(
                {
                    "site": design.site,
                    "window": design.window,
                    "drift_step": common.DRIFT_STEP_PART_2,
                    "removed": f"spectral_{name}",
                    "functional": label,
                    "r": float("nan"),
                    "rho": float("nan"),
                    "short_period_share": spectral_share(design=design, column=column),
                    "partialled_variance_share": float(
                        column.var() / (design.x @ design.functionals[:, functional]).var()
                    ),
                }
            )
    return rows


def main() -> None:
    """Compute the analytic rows for every decision substation and window."""
    common.configure_threads(threads=1)
    cache = common.load_cache()
    rows = []
    for site in common.DECISION_SITES:
        for window in ("all", "2y"):
            design = design_module.build_design(
                cache=cache, site=site, cut=common.REPLICATE_CUT, window=window
            )
            rows += analytics_for(design=design)
            print("done", site, window, flush=True)
    common.WORK_DIR.mkdir(parents=True, exist_ok=True)
    pl.DataFrame(rows, infer_schema_length=None).write_parquet(common.WORK_DIR / "analytic.parquet")
    (common.WORK_DIR / "analytic.json").write_text(json.dumps(rows[:3], indent=1))


if __name__ == "__main__":
    main()
