"""Fit the smooth-in-lead band arms on the tuning months and write every arm's forecasts.

| Arm | Base offsets from the mean |
|---|---|
| `a` | experiment 169's arm A, as stored (reference) |
| `pb` | offset = scale(substation, hour band, quarter) x growth(hour band, lead) (first attempt) |
| `p` | the same with one growth curve per hour band and quarter, pooled over substations |
| `u` | the same with one growth curve per substation, hour band, and quarter |
| `s` | experiment 160's table with each cell shrunk towards the table pooled across substations |
| `ps` | `p` with its cell scales shrunk the same way |

The growth is non-decreasing and piecewise linear with a knot every 24 hours, fitted by linear
programme on the pinball loss of the rows outside the switching events, alternating three times
with a weighted-quantile refit of every cell scale. Fix O's stored multipliers are applied on top of
every arm. The mean forecast, and the offsets at q40, q50, and q60, are never changed.

    OMP_NUM_THREADS=1 systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00176-*/e00176_fit.py [--workers 4]
"""

import argparse
import json
import os
import warnings
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Final

import e00176_common as common
import numpy as np
import polars as pl
from scipy import sparse
from scipy.optimize import linprog

MAX_FIT_ROWS: Final[int] = 40_000
SEED: Final[int] = 176
ROUNDS: Final[int] = 3
SCALE_FLOOR: Final[float] = 1e-3
SHRINKAGE_GRID: Final[tuple[float, ...]] = (0.0, 5.0, 20.0, 80.0, 320.0, 1280.0)
N_FOLDS: Final[int] = 3
FIX_O_TOLERANCE: Final[float] = 1e-2


def load(*, arm: str, period: str, site: str) -> pl.DataFrame:
    """Return one stored arm's forecasts, sorted so a row's position is stable."""
    return pl.read_parquet(common.stored_path(arm=arm, period=period, site=site)).select(
        list(common.BASE_COLUMNS)
    )


def offsets_of(*, frame: pl.DataFrame) -> np.ndarray:
    """Return the frame's quantile offsets from the mean, shape (rows, 11)."""
    quantiles = frame.select(list(common.QUANTILES)).to_numpy().astype(np.float64)
    return quantiles - frame["mean"].to_numpy().astype(np.float64)[:, None]


def fit_rows_mask(*, frame: pl.DataFrame) -> np.ndarray:
    """Return which rows a fit may use: a reading, outside the switching events, lead <= 168 h."""
    return (
        (~frame["switching_mask"]).to_numpy()
        & np.isfinite(frame["actual"].to_numpy())
        & (frame["lead_hours"].to_numpy() <= common.HORIZON_HOURS)
    )


def fix_o_multipliers(*, ref: pl.DataFrame, arm_a: pl.DataFrame) -> dict[int, float]:
    """Return experiment 164's fix-O multiplier at each upper level, read from the stored arms.

    Args:
        ref: The reference forecasts (experiment 160's arm I) of the tuning months.
        arm_a: Experiment 169's arm A for the same rows.

    Returns:
        The multiplier at each position of the upper levels in `QUANTILES`.

    Raises:
        ValueError: If the multiplier is not constant on the forecast-cold rows of a level.
    """
    cold = np.nan_to_num(arm_a["ens_mean_c"].to_numpy(), nan=99.0) < common.COLD_GATE_C
    base = offsets_of(frame=ref)
    scaled = offsets_of(frame=arm_a)
    out = {}
    for position in common.UPPER_FIX_POSITIONS:
        usable = cold & (base[:, position] > 0.5)
        ratio = scaled[usable, position] / base[usable, position]
        if np.ptp(ratio) > FIX_O_TOLERANCE:
            msg = f"fix O is not one multiplier at position {position}: {ratio.min()} {ratio.max()}"
            raise ValueError(msg)
        out[position] = float(np.median(ratio))
    return out


def growth_basis(*, lead: np.ndarray) -> np.ndarray:
    """Return the design matrix of the piecewise-linear growth: a constant, then 7 ramps."""
    ramps = [np.clip((lead - 24.0 * k) / 24.0, 0.0, 1.0) for k in range(7)]
    return np.stack([np.ones_like(lead), *ramps], axis=1)


def fit_growth(*, y: np.ndarray, scale: np.ndarray, lead: np.ndarray, tau: float) -> np.ndarray:
    """Return the growth curve's values at the knots, minimising the scale-weighted pinball loss.

    The loss of a row is `scale * pinball_tau(y / scale - growth(lead))`. The constant and the
    ramp slopes are non-negative, so the growth is non-decreasing in lead. The curve is rescaled to
    a mean of 1 over the 0 to 168 hour grid.
    """
    design = growth_basis(lead=lead)
    n_rows, n_terms = design.shape
    identity = sparse.identity(n_rows, format="csr")
    equality = sparse.hstack([sparse.csr_matrix(design), identity, -identity], format="csr")
    cost = np.concatenate([np.zeros(n_terms), scale * tau, scale * (1.0 - tau)])
    bounds = [(0.01, None)] + [(0.0, None)] * (n_terms - 1) + [(0.0, None)] * (2 * n_rows)
    result = linprog(cost, A_eq=equality, b_eq=y / scale, bounds=bounds, method="highs")
    if not result.success:
        msg = f"growth fit failed: {result.message}"
        raise RuntimeError(msg)
    weights = result.x[:n_terms]
    at_knots = growth_basis(lead=common.KNOTS_HOURS) @ weights
    grid = np.linspace(0.0, common.HORIZON_HOURS, 337)
    return at_knots / np.interp(grid, common.KNOTS_HOURS, at_knots).mean()


def weighted_quantile(*, values: np.ndarray, weights: np.ndarray, tau: float) -> float:
    """Return the weighted quantile, the smallest value whose cumulative weight reaches tau."""
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    index = min(int(np.searchsorted(cumulative, tau * cumulative[-1])), len(values) - 1)
    return float(values[order][index])


def _growth_job(task: tuple) -> tuple[tuple, np.ndarray]:
    key, y, scale, lead, tau = task
    return key, fit_growth(y=y, scale=scale, lead=lead, tau=tau)


@dataclass
class LevelFit:
    """The state of one quantile level's fit: growth curves and cell scales."""

    position: int
    sign: float
    scale: np.ndarray
    growth: np.ndarray


class SiteData:
    """The tuning rows a fit may use at one substation, as arrays."""

    def __init__(self, *, frame: pl.DataFrame) -> None:
        """Keep the usable rows of the arm-A tuning frame."""
        keep = fit_rows_mask(frame=frame)
        used = frame.filter(pl.Series(keep))
        self.lead = used["lead_hours"].to_numpy().astype(np.float64)
        self.band = used["hour_band"].to_numpy().astype(np.int64)
        self.quarter = used["quarter"].to_numpy().astype(np.int64)
        self.run = used["run"].to_numpy().astype(np.int64)
        self.month = used["power_fcst_init_time"].dt.strftime("%Y-%m").to_numpy()
        self.residual = (used["actual"] - used["mean"]).to_numpy().astype(np.float64)
        self.raw_residual = self.residual.copy()
        edges = np.array(common.LEAD_BAND_UPPER_HOURS)
        lead_band = np.minimum(
            np.searchsorted(edges, self.lead, side="left"), common.N_LEAD_BANDS - 1
        )
        for band in range(common.N_HOUR_BANDS):
            for quarter in range(common.N_QUARTERS):
                for k in range(common.N_LEAD_BANDS):
                    chosen = (self.band == band) & (self.quarter == quarter) & (lead_band == k)
                    if chosen.any():
                        self.residual[chosen] -= np.median(self.raw_residual[chosen])
        self.offsets = offsets_of(frame=used)


def interpolated_offsets(
    *, table: np.ndarray, band: np.ndarray, quarter: np.ndarray, lead_hours: np.ndarray
) -> np.ndarray:
    """Return each row's offsets, linearly interpolated between the lead-band centres."""
    centres = np.array(common.LEAD_BAND_CENTRES_HOURS)
    low = np.clip(np.searchsorted(centres, lead_hours, side="right") - 1, 0, len(centres) - 2)
    weight = np.clip((lead_hours - centres[low]) / (centres[low + 1] - centres[low]), 0.0, 1.0)
    return (1.0 - weight)[:, None] * table[band, quarter, low] + weight[:, None] * table[
        band, quarter, low + 1
    ]


def runs_per_cell(*, data: SiteData, by_lead_band: bool) -> np.ndarray:
    """Return the number of distinct runs in each (hour band, quarter[, lead band]) cell."""
    edges = np.array(common.LEAD_BAND_UPPER_HOURS)
    lead_band = np.minimum(np.searchsorted(edges, data.lead, side="left"), common.N_LEAD_BANDS - 1)
    shape = (common.N_HOUR_BANDS, common.N_QUARTERS) + (
        (common.N_LEAD_BANDS,) if by_lead_band else ()
    )
    out = np.zeros(shape)
    for index in np.ndindex(shape):
        chosen = (data.band == index[0]) & (data.quarter == index[1])
        if by_lead_band:
            chosen &= lead_band == index[2]
        out[index] = len(np.unique(data.run[chosen]))
    return out


SHAPE_MODES: Final[tuple[str, ...]] = ("band", "cell", "site_cell")
"""How a growth curve is shared: across a hour band, across substations within an hour band and
quarter, or not at all (one curve per substation, hour band, and quarter)."""


def shape_groups(*, mode: str, n_sites: int) -> list[list[tuple[int, int, int]]]:
    """Return the cells (substation, hour band, quarter) that share one growth curve."""
    cells = [
        (i, b, q)
        for i in range(n_sites)
        for b in range(common.N_HOUR_BANDS)
        for q in range(common.N_QUARTERS)
    ]
    keys = {
        "band": lambda c: (c[1],),
        "cell": lambda c: (c[1], c[2]),
        "site_cell": lambda c: c,
    }[mode]
    groups: dict[tuple, list[tuple[int, int, int]]] = {}
    for cell in cells:
        groups.setdefault(keys(cell), []).append(cell)
    return list(groups.values())


def fit_smooth_arms(  # noqa: PLR0912
    *, data: dict[str, SiteData], tables: dict[str, np.ndarray], workers: int
) -> dict:
    """Fit the growth curves and cell scales at the eight levels for each way of sharing a curve.

    Args:
        data: The tuning rows of each substation.
        tables: Experiment 160's stored spread table of each substation, which starts the scales.
        workers: The number of processes that fit growth curves in parallel.

    Returns:
        `{mode: {position: {"growth": array, "scale": array}}}`, where `growth` has shape
        (substations, hour bands, quarters, knots) and `scale` (substations, hour bands, quarters).
    """
    sites = list(common.SCORED_SITES)
    n_sites = len(sites)
    rng = np.random.default_rng(SEED)
    cell_rows = {}
    for i, site in enumerate(sites):
        for b in range(common.N_HOUR_BANDS):
            for q in range(common.N_QUARTERS):
                rows = np.flatnonzero((data[site].band == b) & (data[site].quarter == q))
                cell_rows[i, b, q] = rng.permutation(rows)
    state = {}
    for mode in SHAPE_MODES:
        for position in common.FIT_LEVEL_POSITIONS:
            sign = 1.0 if common.LEVELS[position] > 0.5 else -1.0
            scale = np.zeros((n_sites, common.N_HOUR_BANDS, common.N_QUARTERS))
            for i, site in enumerate(sites):
                scale[i] = np.maximum(sign * tables[site][..., position].mean(axis=2), SCALE_FLOOR)
            growth = np.ones(
                (n_sites, common.N_HOUR_BANDS, common.N_QUARTERS, len(common.KNOTS_HOURS))
            )
            state[mode, position] = LevelFit(
                position=position, sign=sign, scale=scale, growth=growth
            )
    for round_number in range(ROUNDS):
        tasks = []
        for (mode, position), current in state.items():
            tau = (
                float(common.LEVELS[position])
                if current.sign > 0
                else 1.0 - float(common.LEVELS[position])
            )
            for group in shape_groups(mode=mode, n_sites=n_sites):
                share = MAX_FIT_ROWS // len(group)
                y, lead, weight = [], [], []
                for i, b, q in group:
                    rows = cell_rows[i, b, q][:share]
                    y.append(current.sign * data[sites[i]].residual[rows])
                    lead.append(data[sites[i]].lead[rows])
                    weight.append(np.full(len(rows), current.scale[i, b, q]))
                tasks.append(
                    (
                        (mode, position, group),
                        np.concatenate(y),
                        np.concatenate(weight),
                        np.concatenate(lead),
                        tau,
                    )
                )
        with ProcessPoolExecutor(max_workers=workers) as pool:
            for (mode, position, group), curve in pool.map(_growth_job, tasks, chunksize=4):
                for i, b, q in group:
                    state[mode, position].growth[i, b, q] = curve
        for current in state.values():
            tau = float(common.LEVELS[current.position])
            tau = tau if current.sign > 0 else 1.0 - tau
            for i, site in enumerate(sites):
                site_data = data[site]
                for b in range(common.N_HOUR_BANDS):
                    for q in range(common.N_QUARTERS):
                        chosen = (site_data.band == b) & (site_data.quarter == q)
                        h = np.interp(
                            site_data.lead[chosen], common.KNOTS_HOURS, current.growth[i, b, q]
                        )
                        ratio = current.sign * site_data.residual[chosen] / h
                        current.scale[i, b, q] = max(
                            weighted_quantile(values=ratio, weights=h, tau=tau), SCALE_FLOOR
                        )
        print(f"round {round_number + 1} of {ROUNDS} done", flush=True)
    out: dict = {mode: {} for mode in SHAPE_MODES}
    for (mode, position), current in state.items():
        out[mode][position] = {"growth": current.growth, "scale": current.scale}
    return out


def smooth_offsets(
    *,
    fitted: dict,
    site_index: int,
    band: np.ndarray,
    quarter: np.ndarray,
    lead: np.ndarray,
    scale_override: dict | None = None,
) -> np.ndarray:
    """Return rows' offsets at the eight fitted levels and zero at the other three."""
    out = np.zeros((len(lead), len(common.QUANTILES)))
    for position, parts in fitted.items():
        sign = 1.0 if common.LEVELS[position] > 0.5 else -1.0
        scale = parts["scale"] if scale_override is None else scale_override[position]
        h = np.zeros(len(lead))
        for b in range(common.N_HOUR_BANDS):
            for q in range(common.N_QUARTERS):
                chosen = (band == b) & (quarter == q)
                h[chosen] = np.interp(
                    lead[chosen], common.KNOTS_HOURS, parts["growth"][site_index, b, q]
                )
        out[:, position] = sign * scale[site_index, band, quarter] * h
    return out


def site_widths(*, tables: dict[str, np.ndarray]) -> dict[str, float]:
    """Return each substation's mean q05 to q95 width over the table's cells."""
    return {site: float((table[..., 9] - table[..., 1]).mean()) for site, table in tables.items()}


def shrink_toward_pool(
    *,
    values: dict[str, np.ndarray],
    counts: dict[str, np.ndarray],
    widths: dict[str, float],
    k: float,
) -> dict[str, np.ndarray]:
    """Shrink each substation's cells towards the pooled cells with weight n / (n + k).

    The pooled cell is the mean over substations of the cell divided by the substation's width,
    rescaled by the substation's own width. An empty cell reads the pooled value.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        pooled = np.nanmean([values[s] / widths[s] for s in values], axis=0)
    out = {}
    for site, cell in values.items():
        n = counts[site]
        if cell.ndim > n.ndim:
            n = n[..., None]
        weight = n / (n + k) if k > 0 else np.where(n > 0, 1.0, 0.0)
        filled = np.where(np.isfinite(cell), cell, 0.0)
        out[site] = weight * filled + (1.0 - weight) * widths[site] * pooled
    return out


def empirical_table(*, data: SiteData, rows: np.ndarray) -> np.ndarray:
    """Return the median-centred empirical offsets by cell and level; NaN where a cell is empty."""
    edges = np.array(common.LEAD_BAND_UPPER_HOURS)
    lead_band = np.minimum(
        np.searchsorted(edges, data.lead[rows], side="left"), common.N_LEAD_BANDS - 1
    )
    out = np.full(
        (common.N_HOUR_BANDS, common.N_QUARTERS, common.N_LEAD_BANDS, len(common.QUANTILES)), np.nan
    )
    for index in np.ndindex(out.shape[:3]):
        chosen = (
            (data.band[rows] == index[0])
            & (data.quarter[rows] == index[1])
            & (lead_band == index[2])
        )
        if chosen.sum() >= 2:
            cell = data.raw_residual[rows][chosen]
            out[index] = np.quantile(cell, common.LEVELS) - np.median(cell)
    return out


def pinball_of_table(*, data: SiteData, rows: np.ndarray, table: np.ndarray) -> float:
    """Return the mean pinball loss over the eight fitted levels of an interpolated table.

    A row whose cell the table leaves empty is skipped, the same rows for every constant tried.
    """
    offsets = interpolated_offsets(
        table=table, band=data.band[rows], quarter=data.quarter[rows], lead_hours=data.lead[rows]
    )[:, list(common.FIT_LEVEL_POSITIONS)]
    error = data.raw_residual[rows][:, None] - offsets
    levels = common.LEVELS[list(common.FIT_LEVEL_POSITIONS)][None, :]
    loss = np.maximum(levels * error, (levels - 1.0) * error)
    return float(np.nanmean(loss))


def choose_shrinkage(*, data: dict[str, SiteData], tables: dict[str, np.ndarray]) -> dict:
    """Return the shrinkage constant (in runs) with the lowest held-out pinball loss.

    Three folds of alternate initialisation months inside the tuning months: each table is
    rebuilt from two folds as the empirical quantiles of the residuals, shrunk, and scored on
    the third fold.
    """
    months = sorted({m for d in data.values() for m in d.month})
    fold_of = {m: i % N_FOLDS for i, m in enumerate(months)}
    losses = dict.fromkeys(SHRINKAGE_GRID, 0.0)
    for fold in range(N_FOLDS):
        train, test = {}, {}
        for site, d in data.items():
            fold_ids = np.array([fold_of[m] for m in d.month])
            train[site], test[site] = (
                np.flatnonzero(fold_ids != fold),
                np.flatnonzero(fold_ids == fold),
            )
        raw = {s: empirical_table(data=data[s], rows=train[s]) for s in data}
        counts = {}
        for s, d in data.items():
            sub = SiteData.__new__(SiteData)
            sub.lead, sub.band, sub.quarter, sub.run = (
                getattr(d, a)[train[s]] for a in ("lead", "band", "quarter", "run")
            )
            counts[s] = runs_per_cell(data=sub, by_lead_band=True)
        widths = {s: float(np.nanmean(t[..., 9] - t[..., 1])) for s, t in raw.items()}
        for k in SHRINKAGE_GRID:
            shrunk = shrink_toward_pool(values=raw, counts=counts, widths=widths, k=k)
            losses[k] += sum(
                pinball_of_table(data=data[s], rows=test[s], table=shrunk[s]) for s in data
            )
    best = min(losses, key=lambda k: losses[k])
    return {"losses": {str(k): v for k, v in losses.items()}, "k": best}


def apply_arm(
    *, ref: pl.DataFrame, offsets: np.ndarray, multipliers: dict[int, float]
) -> pl.DataFrame:
    """Return the frame with quantiles rebuilt as mean plus offsets, fix O applied, and sorted."""
    cold = np.nan_to_num(ref["ens_mean_c"].to_numpy(), nan=99.0) < common.COLD_GATE_C
    scaled = offsets.copy()
    for position, multiplier in multipliers.items():
        scaled[:, position] = np.where(
            cold, offsets[:, position] * multiplier, offsets[:, position]
        )
    mean = ref["mean"].to_numpy().astype(np.float64)
    quantiles = np.sort(mean[:, None] + scaled, axis=1)
    return ref.with_columns(
        pl.Series(q, quantiles[:, position].astype(np.float32))
        for position, q in enumerate(common.QUANTILES)
    )


def write_arm(*, frame: pl.DataFrame, arm: str, period: str, site: str) -> None:
    """Write one arm's forecasts atomically."""
    target = common.out_path(arm=arm, period=period, site=site)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.name}.{os.getpid()}.tmp")
    frame.with_columns(arm=pl.lit(arm)).write_parquet(temporary)
    temporary.rename(target)


def main() -> None:
    """Fit on the tuning months, build the arms for every period and substation, write them."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--reuse-fit", action="store_true", help="read the growth curves and scales already fitted"
    )
    args = parser.parse_args()
    common.WORK_DIR.mkdir(parents=True, exist_ok=True)
    sites = list(common.SCORED_SITES)
    tables = {s: np.load(common.table_path(site=s)) for s in sites}
    tuning = {s: load(arm="a", period="tuning", site=s) for s in sites}
    tuning_ref = {s: load(arm="ref", period="tuning", site=s) for s in sites}
    multipliers = fix_o_multipliers(
        ref=pl.concat(list(tuning_ref.values())), arm_a=pl.concat(list(tuning.values()))
    )
    print("fix O multipliers", multipliers, flush=True)
    data = {s: SiteData(frame=tuning[s]) for s in sites}
    parameters_path = common.WORK_DIR / "fit_parameters.json"
    if args.reuse_fit:
        stored = json.loads(parameters_path.read_text())
        fitted = {
            mode: {
                int(p): {
                    "growth": np.array(stored["fitted"][mode]["growth"][p]),
                    "scale": np.array(stored["fitted"][mode]["scale"][p]),
                }
                for p in stored["fitted"][mode]["growth"]
            }
            for mode in SHAPE_MODES
        }
    else:
        fitted = fit_smooth_arms(data=data, tables=tables, workers=args.workers)
    choice = choose_shrinkage(data=data, tables=tables)
    k = choice["k"]
    print("shrinkage", choice, flush=True)
    widths = site_widths(tables=tables)
    table_counts = {s: runs_per_cell(data=data[s], by_lead_band=True) for s in sites}
    shrunk_tables = shrink_toward_pool(values=tables, counts=table_counts, widths=widths, k=k)
    cell_counts = {s: runs_per_cell(data=data[s], by_lead_band=False) for s in sites}
    shrunk_scales = {}
    for position in common.FIT_LEVEL_POSITIONS:
        raw = {s: fitted["cell"][position]["scale"][i] for i, s in enumerate(sites)}
        shrunk = shrink_toward_pool(values=raw, counts=cell_counts, widths=widths, k=k)
        shrunk_scales[position] = np.stack([shrunk[s] for s in sites])
    common.write_json_atomic(
        path=common.WORK_DIR / "fit_parameters.json",
        content={
            "fix_o_multipliers": multipliers,
            "shrinkage": choice,
            "widths": widths,
            "fitted": {
                mode: {
                    "growth": {str(p): v["growth"].tolist() for p, v in fitted[mode].items()},
                    "scale": {str(p): v["scale"].tolist() for p, v in fitted[mode].items()},
                }
                for mode in fitted
            },
            "scale_pooled_shrunk": {str(p): v.tolist() for p, v in shrunk_scales.items()},
        },
    )
    fit_positions = list(common.FIT_LEVEL_POSITIONS)
    for period in common.PERIODS:
        for index, site in enumerate(sites):
            ref = load(arm="ref", period=period, site=site)
            stored_a = load(arm="a", period=period, site=site)
            band = ref["hour_band"].to_numpy().astype(np.int64)
            quarter = ref["quarter"].to_numpy().astype(np.int64)
            lead = ref["lead_hours"].to_numpy().astype(np.float64)
            base = offsets_of(frame=ref)
            arms = {}
            arms["a"] = apply_arm(ref=ref, offsets=base, multipliers=multipliers)
            reproduction = float(
                np.abs(
                    arms["a"].select(list(common.QUANTILES)).to_numpy()
                    - stored_a.select(list(common.QUANTILES)).to_numpy()
                ).max()
            )
            for arm, mode, override in (
                ("pb", "band", None),
                ("p", "cell", None),
                ("u", "site_cell", None),
                ("ps", "cell", shrunk_scales),
            ):
                offsets = base.copy()
                new = smooth_offsets(
                    fitted=fitted[mode],
                    site_index=index,
                    band=band,
                    quarter=quarter,
                    lead=lead,
                    scale_override=override,
                )
                offsets[:, fit_positions] = new[:, fit_positions]
                arms[arm] = apply_arm(ref=ref, offsets=offsets, multipliers=multipliers)
            offsets = base.copy()
            offsets[:, fit_positions] = interpolated_offsets(
                table=shrunk_tables[site], band=band, quarter=quarter, lead_hours=lead
            )[:, fit_positions]
            arms["s"] = apply_arm(ref=ref, offsets=offsets, multipliers=multipliers)
            for arm, frame in arms.items():
                write_arm(frame=frame, arm=arm, period=period, site=site)
            print(period, site, f"written; arm A reproduction gap {reproduction:.2e}", flush=True)


if __name__ == "__main__":
    main()
