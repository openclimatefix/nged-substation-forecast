"""Fit one solar farm's plant to its metered output: a grid search plus a learned smooth response.

Written for the per-farm plant fit of the metered solar farms. The plant turns the irradiance on the
panels' plane into AC power in three steps. The cell temperature is the air temperature plus
`CELL_TEMPERATURE_RISE_K` times the plane's irradiance in units of 1000 W/m². The DC power is a
scale times that irradiance times `1 - heat_per_kelvin * (cell temperature - 25 C)`. A soft clip, a
p-norm minimum with a fitted AC limit, then keeps the output below the inverters' limit.

Everything is in units of the farm's capacity, so a scale of 1 means DC power equal to capacity at
1000 W/m². The search is exhaustive over tilt, azimuth, and heat coefficient. At each grid point the
scale and the AC limit are fitted by bounded least squares, which has two parameters and so needs no
random restarts.

The learned response is a table of output over irradiance and sun elevation, monotone non-decreasing
in irradiance, and penalised towards a prior table. Its fit is a non-negative least-squares problem
in the increments of the table along the irradiance axis, which is a convex bounded quadratic.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Final

import numpy as np
from scipy import sparse
from scipy.linalg import cho_factor, solve_triangular
from scipy.optimize import least_squares, lsq_linear

from studies.pv_physics import STANDARD_IRRADIANCE_W_M2

CELL_TEMPERATURE_RISE_K: Final[float] = 25.0
"""Cell temperature above the air at 1000 W/m² on the plane."""
REFERENCE_CELL_TEMPERATURE_C: Final[float] = 25.0
CLIP_SHARPNESS: Final[float] = 6.0
"""The exponent `p` of the soft clip's p-norm minimum. Larger is closer to a hard clip."""
MIN_SCALE: Final[float] = 0.3
MAX_SCALE: Final[float] = 3.0
MIN_AC_LIMIT: Final[float] = 0.4
MAX_AC_LIMIT: Final[float] = 1.8


@dataclass(frozen=True)
class FittedPlant:
    """The scale and AC limit fitted at one grid point, and the error it leaves.

    Attributes:
        scale: DC power over capacity at 1000 W/m² and 25 C cell temperature.
        ac_limit: The soft clip's limit, over capacity. `inf` means no clip.
        squared_error: The sum of squared errors over the fitted rows, in capacity squared.
    """

    scale: float
    ac_limit: float
    squared_error: float

    @property
    def dc_ac_ratio(self) -> float:
        """The DC scale over the AC limit."""
        return self.scale / self.ac_limit


def unclipped_unit_power(
    *, poa_w_m2: np.ndarray, temperature_c: np.ndarray, heat_per_kelvin: float
) -> np.ndarray:
    """Return DC power per unit scale: irradiance on the plane times the heat derating.

    Args:
        poa_w_m2: Irradiance on the panels' plane.
        temperature_c: Air temperature at the same half-hours.
        heat_per_kelvin: The fractional power loss per kelvin of cell temperature above 25 C.

    Returns:
        The unclipped power over scale, never negative.
    """
    unit = poa_w_m2 / STANDARD_IRRADIANCE_W_M2
    cell = temperature_c + CELL_TEMPERATURE_RISE_K * unit
    derating = np.clip(1.0 - heat_per_kelvin * (cell - REFERENCE_CELL_TEMPERATURE_C), 0.0, None)
    return unit * derating


def soft_minimum(*, power: np.ndarray, limit: float) -> np.ndarray:
    """Return the p-norm minimum of `power` and `limit`, smooth near the limit.

    Args:
        power: Non-negative power.
        limit: The curve's upper asymptote (the output at `power == limit` is about 0.89 of it);
            `inf` returns `power` unchanged.

    Returns:
        `power / (1 + (power / limit) ** p) ** (1 / p)` with `p = CLIP_SHARPNESS`.
    """
    if not np.isfinite(limit):
        return power
    return power / (1.0 + (power / limit) ** CLIP_SHARPNESS) ** (1.0 / CLIP_SHARPNESS)


def plant_output(*, unclipped: np.ndarray, scale: float, ac_limit: float) -> np.ndarray:
    """Return the soft-clipped AC power of a plant with a given scale and AC limit.

    Args:
        unclipped: The output of `unclipped_unit_power`.
        scale: DC power over capacity at the standard irradiance.
        ac_limit: The AC limit over capacity; `inf` for no clip.

    Returns:
        AC power over capacity.
    """
    return soft_minimum(power=scale * unclipped, limit=ac_limit)


def fit_scale_only(*, unclipped: np.ndarray, output: np.ndarray) -> FittedPlant:
    """Fit the scale with no clip: the closed-form least-squares scale.

    Args:
        unclipped: The output of `unclipped_unit_power`.
        output: The metered output over capacity.

    Returns:
        The fit, with an infinite AC limit.
    """
    denominator = float(np.sum(unclipped**2))
    scale = float(np.sum(unclipped * output)) / denominator if denominator > 0 else 0.0
    error = float(np.sum((output - scale * unclipped) ** 2))
    return FittedPlant(scale=scale, ac_limit=float("inf"), squared_error=error)


def fit_scale_and_clip(*, unclipped: np.ndarray, output: np.ndarray) -> FittedPlant:
    """Fit the scale and the AC limit by bounded least squares.

    Starts from the no-clip scale and a limit at the output's 99.9th percentile. If the clip does
    not reduce the error, the no-clip fit is returned, so the clipped fit is never worse.

    Args:
        unclipped: The output of `unclipped_unit_power`.
        output: The metered output over capacity.

    Returns:
        The fit.
    """
    baseline = fit_scale_only(unclipped=unclipped, output=output)
    start_limit = float(np.clip(np.quantile(output, 0.999), MIN_AC_LIMIT, MAX_AC_LIMIT))
    start_scale = float(np.clip(baseline.scale, MIN_SCALE, MAX_SCALE))

    def residuals(theta: np.ndarray) -> np.ndarray:
        return plant_output(unclipped=unclipped, scale=theta[0], ac_limit=theta[1]) - output

    solution = least_squares(
        residuals,
        x0=np.array([start_scale, start_limit]),
        bounds=([MIN_SCALE, MIN_AC_LIMIT], [MAX_SCALE, MAX_AC_LIMIT]),
        x_scale=np.array([1.0, 1.0]),
    )
    error = float(np.sum(solution.fun**2))
    if error >= baseline.squared_error:
        return baseline
    return FittedPlant(
        scale=float(solution.x[0]), ac_limit=float(solution.x[1]), squared_error=error
    )


def fraction_of_output_fitted(*, output: np.ndarray, predicted: np.ndarray) -> float:
    """Return 1 minus the squared error over the squared output (a zero baseline, not the mean).

    Args:
        output: The metered output.
        predicted: The model's output at the same half-hours.

    Returns:
        The fraction; 1 is a perfect fit.
    """
    return 1.0 - float(np.sum((output - predicted) ** 2)) / float(np.sum(output**2))


@dataclass(frozen=True)
class ResponseGrid:
    """The irradiance and elevation nodes of a learned response table.

    Attributes:
        irradiance_w_m2: Increasing irradiance nodes.
        elevation_deg: Increasing sun-elevation nodes.
    """

    irradiance_w_m2: np.ndarray
    elevation_deg: np.ndarray

    @property
    def n_nodes(self) -> int:
        """The number of table entries."""
        return len(self.irradiance_w_m2) * len(self.elevation_deg)


def _hat_weights(*, values: np.ndarray, nodes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return the lower node index and the weight on the upper node for linear interpolation."""
    clipped = np.clip(values, nodes[0], nodes[-1])
    lower = np.clip(np.searchsorted(nodes, clipped, side="right") - 1, 0, len(nodes) - 2)
    span = nodes[lower + 1] - nodes[lower]
    return lower, (clipped - nodes[lower]) / span


def interpolation_matrix(
    *, irradiance_w_m2: np.ndarray, elevation_deg: np.ndarray, grid: ResponseGrid
) -> sparse.csr_matrix:
    """Return the sparse bilinear weights that read a table at each half-hour.

    Args:
        irradiance_w_m2: Irradiance at each half-hour.
        elevation_deg: Sun elevation at each half-hour.
        grid: The table's nodes.

    Returns:
        A matrix of shape (half-hours, nodes); the table is flattened with irradiance varying
        slowest, so entry `i * n_elevation + j` is the node at irradiance `i` and elevation `j`.
    """
    n_rows = len(irradiance_w_m2)
    n_elevation = len(grid.elevation_deg)
    i0, wi = _hat_weights(values=irradiance_w_m2, nodes=grid.irradiance_w_m2)
    j0, wj = _hat_weights(values=elevation_deg, nodes=grid.elevation_deg)
    rows = np.repeat(np.arange(n_rows), 4)
    cols = np.stack(
        [
            i0 * n_elevation + j0,
            i0 * n_elevation + j0 + 1,
            (i0 + 1) * n_elevation + j0,
            (i0 + 1) * n_elevation + j0 + 1,
        ],
        axis=1,
    ).ravel()
    weights = np.stack([(1 - wi) * (1 - wj), (1 - wi) * wj, wi * (1 - wj), wi * wj], axis=1).ravel()
    return sparse.csr_matrix((weights, (rows, cols)), shape=(n_rows, grid.n_nodes))


def _cumulative_matrix(*, grid: ResponseGrid) -> sparse.csr_matrix:
    """Return the matrix that turns table increments along irradiance into table values."""
    n_irradiance = len(grid.irradiance_w_m2)
    lower_triangle = sparse.csr_matrix(np.tril(np.ones((n_irradiance, n_irradiance))))
    return sparse.kron(lower_triangle, sparse.identity(len(grid.elevation_deg))).tocsr()


def _smoothing_matrix(*, grid: ResponseGrid) -> sparse.csr_matrix:
    """Return second differences of the table along both axes."""
    n_irradiance = len(grid.irradiance_w_m2)
    n_elevation = len(grid.elevation_deg)

    def second_difference(size: int) -> sparse.csr_matrix:
        return sparse.diags([1.0, -2.0, 1.0], [0, 1, 2], shape=(size - 2, size)).tocsr()

    along_irradiance = sparse.kron(second_difference(n_irradiance), sparse.identity(n_elevation))
    along_elevation = sparse.kron(sparse.identity(n_irradiance), second_difference(n_elevation))
    return sparse.vstack([along_irradiance, along_elevation]).tocsr()


@dataclass(frozen=True)
class ResponseTable:
    """A fitted learned response: output over capacity at each node.

    Attributes:
        grid: The nodes.
        values: The table, shape (irradiance nodes, elevation nodes).
    """

    grid: ResponseGrid
    values: np.ndarray

    def predict(self, *, irradiance_w_m2: np.ndarray, elevation_deg: np.ndarray) -> np.ndarray:
        """Return the table read at each half-hour.

        Args:
            irradiance_w_m2: Irradiance at each half-hour.
            elevation_deg: Sun elevation at each half-hour.

        Returns:
            Output over capacity.
        """
        matrix = interpolation_matrix(
            irradiance_w_m2=irradiance_w_m2, elevation_deg=elevation_deg, grid=self.grid
        )
        return matrix @ self.values.ravel()


def prior_table(
    *,
    grid: ResponseGrid,
    irradiance_w_m2: np.ndarray,
    elevation_deg: np.ndarray,
    prior_output: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the weighted cell means of a prior prediction and the weight of each cell.

    Args:
        grid: The nodes.
        irradiance_w_m2: Irradiance at each half-hour.
        elevation_deg: Sun elevation at each half-hour.
        prior_output: The prior arm's prediction at each half-hour.

    Returns:
        The cell means (zero where the weight is zero) and the weights, both flattened.
    """
    matrix = interpolation_matrix(
        irradiance_w_m2=irradiance_w_m2, elevation_deg=elevation_deg, grid=grid
    )
    weight = np.asarray(matrix.sum(axis=0)).ravel()
    total = matrix.T @ prior_output
    mean = np.divide(total, weight, out=np.zeros_like(total), where=weight > 1e-9)
    return mean, weight


def fit_response_table(
    *,
    grid: ResponseGrid,
    irradiance_w_m2: np.ndarray,
    elevation_deg: np.ndarray,
    output: np.ndarray,
    prior_output: np.ndarray,
    penalty: float,
    smoothing: float = 1e-3,
) -> ResponseTable:
    """Fit a monotone-in-irradiance table, penalised towards the prior's cell means.

    The objective is the half-hours' squared error plus `penalty` times the cell-weighted squared
    distance of the table from the prior's cell means plus `smoothing` times the table's squared
    second differences per half-hour. The increments along irradiance are non-negative, which makes
    the table monotone and the problem convex. It is solved exactly as a bounded least squares on
    the Cholesky factor of the normal equations.

    Args:
        grid: The nodes.
        irradiance_w_m2: Irradiance at each half-hour.
        elevation_deg: Sun elevation at each half-hour.
        output: The metered output over capacity.
        prior_output: The prior arm's prediction, whose cell means anchor the table.
        penalty: The weight on the distance from the prior; 0 leaves the data to speak.
        smoothing: The weight on the second differences.

    Returns:
        The fitted table.
    """
    interpolation = interpolation_matrix(
        irradiance_w_m2=irradiance_w_m2, elevation_deg=elevation_deg, grid=grid
    )
    cumulative = _cumulative_matrix(grid=grid)
    design = interpolation @ cumulative
    mean, weight = prior_table(
        grid=grid,
        irradiance_w_m2=irradiance_w_m2,
        elevation_deg=elevation_deg,
        prior_output=prior_output,
    )
    anchor = sparse.diags(np.sqrt(penalty * weight)) @ cumulative
    anchor_target = np.sqrt(penalty * weight) * mean
    smooth = np.sqrt(smoothing * len(output)) * (_smoothing_matrix(grid=grid) @ cumulative)
    hessian = (design.T @ design + anchor.T @ anchor + smooth.T @ smooth).toarray()
    hessian += 1e-9 * np.eye(grid.n_nodes)
    linear = design.T @ output + anchor.T @ anchor_target
    factor = np.triu(cho_factor(hessian, lower=False)[0])
    target = solve_triangular(factor, linear, trans="T", lower=False)
    solution = lsq_linear(factor, target, bounds=(0.0, np.inf), method="bvls")
    values = (cumulative @ solution.x).reshape(len(grid.irradiance_w_m2), len(grid.elevation_deg))
    return ResponseTable(grid=grid, values=values)


def block_folds(*, block_of_row: np.ndarray, n_folds: int) -> Sequence[np.ndarray]:
    """Split rows into folds of contiguous blocks, for cross-validation that respects time.

    Args:
        block_of_row: A sortable block label (such as year times 12 plus month) for each row.
        n_folds: How many folds.

    Returns:
        For each fold, a boolean mask of its held-out rows.
    """
    blocks = np.unique(block_of_row)
    chunks = np.array_split(blocks, n_folds)
    return [np.isin(block_of_row, chunk) for chunk in chunks]
