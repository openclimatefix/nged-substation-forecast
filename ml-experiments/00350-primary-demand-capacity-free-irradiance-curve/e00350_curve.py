"""The curve arms' model of experiment 350: ramps, fit record, prediction, and solvers.

The module holds everything a pickled fit needs, so a fit file loads in any script. `e00350_fits.py`
runs the fits.
"""

from dataclasses import dataclass
from datetime import datetime

import e00136_common as c136  # ty: ignore[unresolved-import]
import e00136_design as design  # ty: ignore[unresolved-import]
import e00350_common as common
import numpy as np
import torch
from studies.dp_features import FeatureCache
from studies.dp_gam import GamLayout, design_matrix, solve_bounded_quadratic

N_RAMPS = len(common.IRRADIANCE_KNOTS_W_M2)


def ramps(*, ghi_w_m2: np.ndarray) -> np.ndarray:
    """Return the saturating ramps of irradiance, shape (rows, 8): 0 below a knot, 1 above."""
    left = np.array(common.IRRADIANCE_KNOTS_W_M2)
    right = np.array((*common.IRRADIANCE_KNOTS_W_M2[1:], common.IRRADIANCE_TOP_W_M2))
    return np.clip((ghi_w_m2[:, None] - left[None, :]) / (right - left)[None, :], 0.0, 1.0)


@dataclass(frozen=True)
class CurveFit:
    """A curve-arm fit at one site and cut; the last 8 coefficients are the PV curve's.

    Attributes:
        site: The substation's label.
        cut: The cut; no training row ends after it.
        layout: B0's columns.
        arm: The interaction arm (T-all) that built the non-PV columns.
        theta: The coefficients: T-all's columns, then the 8 non-negative curve coefficients (the
            curve is subtracted from net power).
        n_rows: The number of training rows.
        penalised_loss: Squared error plus penalty at the solution.
    """

    site: str
    cut: datetime
    layout: GamLayout
    arm: design.ArmSpec
    theta: np.ndarray
    n_rows: int
    penalised_loss: float


def predict_curve(*, cache: FeatureCache, fit: CurveFit, index: np.ndarray) -> np.ndarray:
    """Predict net power in points at the grid rows `index`: T-all's columns minus the curve."""
    features = cache.sites[fit.site]
    x0 = design_matrix(cache=cache, features=features, layout=fit.layout, index=index)
    x = design.arm_matrix(
        cache=cache, layout=fit.layout, site=fit.site, index=index, x0=x0, arm=fit.arm
    )
    ghi = features.ghi_w_m2[index]
    if not (np.isfinite(x).all() and np.isfinite(ghi).all()):
        msg = f"{fit.site} has a non-finite input among the rows to predict"
        raise ValueError(msg)
    return np.hstack([x, -ramps(ghi_w_m2=ghi)]) @ fit.theta


@dataclass(frozen=True)
class SiteProblem:
    """One site's design at a cut: the T-all columns, the ramps, the target, and the penalty."""

    base: design.BaseFit
    arm: design.ArmSpec
    x_tall: np.ndarray
    ramps: np.ndarray
    penalty_tall: np.ndarray
    constrained_tall: np.ndarray

    @staticmethod
    def build(*, cache: FeatureCache, site: str, cut: datetime, vintages) -> SiteProblem:  # noqa: ANN001
        """Build the site's design with no pinned capacity (the target is the net power)."""
        base = design.build_base(cache=cache, vintages=vintages, site=site, cut=cut, fitted=True)
        arm = design.interaction_arm(name="t_all", penalty_weight=c136.T_ALL_LAMBDA)
        x = design.arm_matrix(
            cache=cache, layout=base.layout, site=site, index=base.index, x0=base.x0, arm=arm
        )
        penalty = design.arm_penalty(layout=base.layout, arm=arm, n_rows=base.n)
        ghi = cache.sites[site].ghi_w_m2[base.index]
        if not (np.isfinite(x).all() and np.isfinite(ghi).all() and np.isfinite(base.target).all()):
            msg = f"{site} has a non-finite training input at {cut}"
            raise ValueError(msg)
        return SiteProblem(
            base=base,
            arm=arm,
            x_tall=x,
            ramps=ramps(ghi_w_m2=ghi),
            penalty_tall=penalty.matrix,
            constrained_tall=penalty.constrained,
        )

    def solve(self, *, pv_columns: np.ndarray) -> tuple[np.ndarray, float]:
        """Fit T-all's columns plus the given non-negative PV columns (each is subtracted)."""
        n_pv = pv_columns.shape[1]
        x = np.hstack([self.x_tall, -pv_columns])
        p0 = self.x_tall.shape[1]
        penalty = np.zeros((p0 + n_pv, p0 + n_pv))
        penalty[:p0, :p0] = self.penalty_tall
        penalty[p0:, p0:] = common.PV_RIDGE * np.eye(n_pv)
        constrained = np.concatenate([self.constrained_tall, np.ones(n_pv, dtype=bool)])
        xt = torch.as_tensor(x)
        gram = (xt.T @ xt)[None] + torch.as_tensor(penalty)
        rhs = torch.as_tensor(x.T @ self.base.target)[None]
        theta, _ = solve_bounded_quadratic(
            matrix=gram, rhs=rhs, constrained=torch.as_tensor(constrained)
        )
        theta_np = theta.numpy()[0]
        residual = self.base.target - x @ theta_np
        return theta_np, float(residual @ residual + theta_np @ penalty @ theta_np)


def fit_arm_c(*, problems: dict[str, SiteProblem]) -> dict[str, tuple[np.ndarray, float]]:
    """Fit each site with its own curve."""
    return {site: problem.solve(pv_columns=problem.ramps) for site, problem in problems.items()}


def fit_arm_b(
    *, problems: dict[str, SiteProblem]
) -> tuple[dict[str, tuple[np.ndarray, float]], list[float]]:
    """Alternate the per-site scales and the shared curve; return the fits and the losses."""
    start = fit_arm_c(problems=problems)
    pooled = np.sum([theta[-N_RAMPS:] for theta, _ in start.values()], axis=0)
    shape = pooled / pooled.sum()
    losses: list[float] = []
    fits: dict[str, tuple[np.ndarray, float]] = {}
    for _ in range(common.SHAPE_ITERATIONS):
        fits = {
            site: problem.solve(pv_columns=(problem.ramps @ shape)[:, None])
            for site, problem in problems.items()
        }
        losses.append(sum(loss for _, loss in fits.values()))
        gram = np.zeros((N_RAMPS, N_RAMPS))
        rhs = np.zeros(N_RAMPS)
        for site, problem in problems.items():
            theta, _ = fits[site]
            scale = theta[-1]
            net_without_pv = problem.x_tall @ theta[:-1]
            wanted = net_without_pv - problem.base.target
            design_site = scale * problem.ramps
            gram += design_site.T @ design_site
            rhs += design_site.T @ wanted
        if not np.any(gram):
            break
        solved, _ = solve_bounded_quadratic(
            matrix=torch.as_tensor(gram + common.PV_RIDGE * np.eye(N_RAMPS))[None],
            rhs=torch.as_tensor(rhs)[None],
            constrained=torch.ones(N_RAMPS, dtype=torch.bool),
        )
        updated = solved.numpy()[0]
        shape = updated / updated.sum()
    fits = {
        site: problem.solve(pv_columns=(problem.ramps @ shape)[:, None])
        for site, problem in problems.items()
    }
    losses.append(sum(loss for _, loss in fits.values()))
    expanded = {
        site: (np.concatenate([theta[:-1], theta[-1] * shape]), loss)
        for site, (theta, loss) in fits.items()
    }
    return expanded, losses
