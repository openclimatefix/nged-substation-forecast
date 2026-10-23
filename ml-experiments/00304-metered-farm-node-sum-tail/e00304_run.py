"""Experiment 304: forms the quantiles of the node's summed farm output four ways and scores them.

Writes `node_quantiles.parquet` (one row per node row, arm, and form) and `farm_residuals.parquet`
under the experiment's private data folder. Run with one thread:
`OMP_NUM_THREADS=1 POLARS_MAX_THREADS=1 uv run --no-sync python e00304_run.py`.
"""

import e00304_data as data
import numpy as np
import polars as pl
from studies.quantile_sums import (
    copula_sum_quantiles,
    independent_sum_quantiles,
    member_wise_sum_quantiles,
    pit_normal_scores,
    sum_of_quantiles,
)

N_DRAWS = 2040
"""Draws per row for the independent and copula forms, equal to the member-wise form's."""
SEED = 0
CSI_EDGES = (0.4, 0.75)
"""Clear-sky index band edges: overcast below 0.4, broken cloud to 0.75, clear above."""
CSI_BANDS = ("overcast", "broken", "clear")
LEAD_EDGES_HOURS = (24.0,)
FORMS = ("a_sum_of_quantiles", "b_independent", "c_member_wise", "c_shuffled", "d_copula")


def csi_band(*, csi: np.ndarray) -> np.ndarray:
    """Return the clear-sky band index (0 overcast, 1 broken, 2 clear) of each row."""
    return np.digitize(csi, CSI_EDGES)


def shuffle_across_days(*, members: np.ndarray, times: pl.Series, seed: int) -> np.ndarray:
    """Permute the member rows among the rows that share a half-hour of the day."""
    rng = np.random.default_rng(seed)
    key = (times.dt.hour() * 2 + times.dt.minute() // 30).to_numpy()
    out = members.copy()
    for value in np.unique(key):
        idx = np.flatnonzero(key == value)
        out[idx] = members[rng.permutation(idx)]
    return out


def fitted_correlations(*, scores: np.ndarray, band: np.ndarray, month: np.ndarray) -> dict:
    """Return each (held-out month, band) correlation matrix of the farms' normal scores.

    The matrix for month m uses every row of the same band outside month m.
    """
    out: dict[tuple[int, int], np.ndarray] = {}
    for m in np.unique(month):
        for b in range(len(CSI_BANDS)):
            use = (month != m) & (band == b)
            corr = np.corrcoef(scores[use].T)
            values, vectors = np.linalg.eigh(corr)
            values = np.clip(values, 1e-3, None)
            corr = (vectors * values) @ vectors.T
            scale = np.sqrt(np.diag(corr))
            out[(int(m), b)] = corr / np.outer(scale, scale)
    return out


def node_forms(*, farms: list[data.FarmRows]) -> tuple[dict[str, np.ndarray], np.ndarray, dict]:
    """Return each form's `(n, 5)` quantiles of the node sum, the node actuals, and diagnostics."""
    parts = [farm.quantiles for farm in farms]
    caps = [data.CAPACITY_CAP_FRACTION * farm.capacity_mw for farm in farms]
    times = farms[0].keys["time"]
    month = (times.dt.year() * 12 + times.dt.month()).to_numpy()
    band = csi_band(csi=np.mean([farm.csi for farm in farms], axis=0))
    actual = np.sum([farm.actual for farm in farms], axis=0)
    scores = np.column_stack(
        [
            pit_normal_scores(
                quantiles=farm.quantiles, levels=data.LEVELS, actual=farm.actual, seed=SEED + i
            )
            for i, farm in enumerate(farms)
        ]
    )
    part_levels, levels = data.LEVELS, data.OUTPUT_LEVELS
    forms = {
        "a_sum_of_quantiles": sum_of_quantiles(parts=parts, part_levels=part_levels, levels=levels),
        "b_independent": independent_sum_quantiles(
            parts=parts,
            n_draws=N_DRAWS,
            seed=SEED,
            upper=caps,
            part_levels=part_levels,
            levels=levels,
        ),
        "c_member_wise": member_wise_sum_quantiles(
            parts=parts,
            members=[f.members for f in farms],
            seed=SEED,
            upper=caps,
            part_levels=part_levels,
            levels=levels,
        ),
    }
    shuffled = [farms[0].members] + [
        shuffle_across_days(members=f.members, times=times, seed=SEED + i + 1)
        for i, f in enumerate(farms[1:])
    ]
    forms["c_shuffled"] = member_wise_sum_quantiles(
        parts=parts, members=shuffled, seed=SEED, upper=caps, part_levels=part_levels, levels=levels
    )
    correlations = fitted_correlations(scores=scores, band=band, month=month)
    copula = np.empty_like(forms["a_sum_of_quantiles"])
    for (m, b), corr in correlations.items():
        use = np.flatnonzero((month == m) & (band == b))
        if use.size == 0:
            continue
        copula[use] = copula_sum_quantiles(
            parts=[p[use] for p in parts],
            correlation=corr,
            n_draws=N_DRAWS,
            seed=SEED,
            upper=caps,
            part_levels=part_levels,
            levels=levels,
        )
    forms["d_copula"] = copula
    return forms, actual, {"scores": scores, "band": band, "month": month}


def single_farm_forms(*, farm: data.FarmRows) -> dict[str, np.ndarray]:
    """Return the one-farm control: every form reduces to the farm's own quantiles."""
    columns = [data.LEVELS.index(level) for level in data.OUTPUT_LEVELS]
    return {"a_sum_of_quantiles": farm.quantiles[:, columns]}


def main() -> None:
    """Run every node and arm, and write the private tables."""
    data.DATA_DIR.mkdir(parents=True, exist_ok=True)
    tables = []
    residuals = []
    for node in data.NODES:
        for arm in data.ARMS:
            farms = data.node_rows(node=node, arm=arm)
            forms, actual, diag = node_forms(farms=farms)
            times = farms[0].keys
            base = times.select("time", "init_time", "lead_hours").with_columns(
                node=pl.lit(node),
                arm=pl.lit(arm),
                actual=pl.Series(actual),
                month=pl.Series(diag["month"]),
                band=pl.Series(diag["band"]),
            )
            for form, q in forms.items():
                columns = {
                    f"q{round(100 * lv):02d}": q[:, i] for i, lv in enumerate(data.OUTPUT_LEVELS)
                }
                tables.append(
                    base.with_columns(
                        form=pl.lit(form), **{k: pl.Series(v) for k, v in columns.items()}
                    )
                )
            median_col = data.LEVELS.index(0.50)
            for site, farm in zip(data.NODES[node], farms, strict=True):
                residuals.append(
                    times.select("lead_hours").with_columns(
                        node=pl.lit(node),
                        arm=pl.lit(arm),
                        site=pl.lit(site),
                        time=times["time"],
                        band=pl.Series(diag["band"]),
                        residual_fraction=pl.Series(
                            (farm.actual - farm.quantiles[:, median_col]) / farm.capacity_mw
                        ),
                    )
                )
            print(node, arm, "rows", times.height, flush=True)
    pl.concat(tables).write_parquet(data.DATA_DIR / "node_quantiles.parquet")
    pl.concat(residuals).write_parquet(data.DATA_DIR / "farm_residuals.parquet")


if __name__ == "__main__":
    main()
