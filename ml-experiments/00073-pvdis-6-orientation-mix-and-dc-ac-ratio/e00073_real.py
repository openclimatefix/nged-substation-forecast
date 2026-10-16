"""Fit the orientation arms to the real series, bootstrap the pooled shares, and hold out blocks.

Per Primary (one worker each):

- O1 shares per season-year (all summer rows) and O2 shares per season-year (clear days);
- the pooled arm O3 with ridge 0 (the arm the known answers chose), its shares, and an interval from
  200 resamples of 7-day blocks of clear days;
- contrast C3: fit on alternate 28-day blocks, score the other blocks, and the reverse. The
  comparator is the same fit with all capacity on the south-facing shape.

    POLARS_MAX_THREADS=6 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run --with holidays python \
        ml-experiments/00073-pvdis-6-orientation-mix-and-dc-ac-ratio/e00073_real.py
"""

from multiprocessing import Pool

import e00073_common as common
import e00073_fit as fit
import e00073_inputs as inputs
import numpy as np
import polars as pl

BOOTSTRAP_DRAWS_PER_PRIMARY = 200
CHOSEN_RIDGE = 0.0


def _normalise(*, capacities: np.ndarray) -> np.ndarray:
    total = capacities.sum()
    return capacities / total if total > 0 else np.full(3, np.nan)


def block_ids(*, days: np.ndarray, length: int) -> np.ndarray:
    """Return the id of the `length`-day block of each row's local date."""
    return (days - days.min()) // length


def refit_and_predict(
    *, w: inputs.YearWindow, train: np.ndarray, test: np.ndarray, u: np.ndarray, c: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Fix the capacities `c`, refit the calendar cells and nuisance on `train`, predict `test`.

    Returns:
        The predictions and the observed power on the test rows whose cell has enough train rows.
    """
    y_adj = w.power[train] + u[train] @ c
    x = w.nuisance[train]
    cells = w.cells[train]
    beta, *_ = np.linalg.lstsq(
        fit.demean(values=x, cells=cells), fit.demean(values=y_adj, cells=cells), rcond=None
    )
    alpha_rows = y_adj - x @ beta
    keys, inverse = np.unique(cells, return_inverse=True)
    counts = np.bincount(inverse)
    alpha = np.bincount(inverse, weights=alpha_rows) / counts
    position = np.searchsorted(keys, w.cells[test])
    position = np.clip(position, 0, len(keys) - 1)
    known = (keys[position] == w.cells[test]) & (counts[position] >= common.MIN_CELL_ROWS)
    prediction = alpha[position] + w.nuisance[test] @ beta - u[test] @ c
    return prediction[known], w.power[test][known]


def heldout_primary(*, windows: list[inputs.YearWindow]) -> list[dict[str, object]]:
    """Return the held-out RMSE of the pooled mix and of the south-only shape, per year and fold."""
    out = []
    usable = [w for w in windows if len(np.unique(w.days[w.clear])) >= common.MIN_CLEAR_DAYS]
    blocks = {w.year: block_ids(days=w.days, length=common.HELDOUT_BLOCK_DAYS) % 2 for w in usable}
    for fold in (0, 1):
        train_designs, train_y, train_u, trained = [], [], [], []
        for w in usable:
            train = (blocks[w.year] == fold) & w.clear
            if len(np.unique(w.days[train])) < 10:
                continue
            u = w.primary_basis(
                ratio=common.FLEET_RATIO, tilt=common.MAIN_TILT, mask=np.ones_like(w.clear)
            )
            train_designs.append(w.design(mask=train))
            train_y.append(w.power[train])
            train_u.append(u[train])
            trained.append(w)
        if not trained:
            continue
        m, scales = fit.fit_pooled_detail(
            designs=train_designs, ys=train_y, shapes=train_u, ridge=CHOSEN_RIDGE
        )
        for w, s in zip(trained, scales, strict=True):
            u = w.primary_basis(
                ratio=common.FLEET_RATIO, tilt=common.MAIN_TILT, mask=np.ones_like(w.clear)
            )
            train = blocks[w.year] == fold
            test = ~train
            rmse = {}
            for name, c in (("mix", s * m), ("south", np.array([0.0, s, 0.0]))):
                pred, obs = refit_and_predict(w=w, train=train, test=test, u=u, c=c)
                rmse[name] = float(np.sqrt(np.mean((pred - obs) ** 2)))
            out.append(
                {"site": w.site, "year": w.year, "fold": fold, "rmse_mix": rmse["mix"],
                 "rmse_south": rmse["south"], "change": rmse["mix"] / rmse["south"] - 1.0}
            )  # fmt: skip
    return out


def real_primary(*, prepared: pl.DataFrame, primary: str) -> tuple[list[dict], list[dict]]:
    """Return the share rows and the held-out rows of one Primary."""
    windows = inputs.year_windows(prepared=prepared, primary=primary)
    rows = []
    ones = np.ones
    for w in windows:
        allmask = ones(len(w.power), dtype=bool)
        u = w.primary_basis(ratio=common.FLEET_RATIO, tilt=common.MAIN_TILT, mask=allmask)
        c1 = fit.fit_orientation(design=w.design(mask=allmask), y=w.power, shapes=u)
        row = {"site": primary, "year": w.year, "arm": "O1", "total": c1.sum()}
        row.update(zip(("east", "south", "west"), _normalise(capacities=c1), strict=True))
        rows.append(row)
        if len(np.unique(w.days[w.clear])) >= common.MIN_CLEAR_DAYS:
            c2 = fit.fit_orientation(
                design=w.design(mask=w.clear), y=w.power[w.clear], shapes=u[w.clear]
            )
            row = {"site": primary, "year": w.year, "arm": "O2", "total": c2.sum()}
            row.update(zip(("east", "south", "west"), _normalise(capacities=c2), strict=True))
            rows.append(row)
    usable = [w for w in windows if len(np.unique(w.days[w.clear])) >= common.MIN_CLEAR_DAYS]
    if usable:
        designs = [w.design(mask=w.clear) for w in usable]
        ys = [w.power[w.clear] for w in usable]
        us = [
            w.primary_basis(ratio=common.FLEET_RATIO, tilt=common.MAIN_TILT, mask=w.clear)
            for w in usable
        ]
        m, scales = fit.fit_pooled_detail(designs=designs, ys=ys, shapes=us, ridge=CHOSEN_RIDGE)
        point = _normalise(capacities=m)
        draws = []
        rng = np.random.default_rng(common.BOOTSTRAP_SEED + sum(map(ord, primary)))
        for _ in range(BOOTSTRAP_DRAWS_PER_PRIMARY):
            d_designs, d_ys, d_us = [], [], []
            for w, u_clear in zip(usable, us, strict=True):
                days = w.days[w.clear]
                block = block_ids(days=days, length=common.BLOCK_DAYS)
                ids = np.unique(block)
                picks = rng.choice(ids, size=len(ids))
                index = np.concatenate([np.nonzero(block == b)[0] for b in picks])
                cells = w.cells[w.clear][index]
                d_designs.append(fit.Design.build(cells=cells, nuisance=w.nuisance[w.clear][index]))
                d_ys.append(w.power[w.clear][index])
                d_us.append(u_clear[index])
            md, _ = fit.fit_pooled_detail(
                designs=d_designs, ys=d_ys, shapes=d_us, ridge=CHOSEN_RIDGE
            )
            draws.append(_normalise(capacities=md))
        draws_array = np.array(draws)
        low = np.nanquantile(draws_array, 0.025, axis=0)
        high = np.nanquantile(draws_array, 0.975, axis=0)
        rows.append(
            {"site": primary, "year": -1, "arm": "O3_0.0",
             "total": float(np.mean(scales) * m.sum()),
             "east": point[0], "south": point[1], "west": point[2],
             "east_low": low[0], "east_high": high[0], "south_low": low[1],
             "south_high": high[1], "west_low": low[2], "west_high": high[2]}
        )  # fmt: skip
    return rows, heldout_primary(windows=windows)


def _job(args: tuple[pl.DataFrame, str]) -> tuple[list[dict], list[dict]]:
    prepared, primary = args
    return real_primary(prepared=prepared, primary=primary)


def main() -> None:
    """Run every Primary in parallel and write the share and held-out tables."""
    prepared = inputs.load_prepared()
    jobs = [(prepared.filter(pl.col("site") == p), p) for p in common.DIRECTIONAL_PRIMARIES]
    with Pool(6) as pool:
        results = pool.map(_job, jobs)
    pl.DataFrame([r for part in results for r in part[0]], infer_schema_length=None).write_parquet(
        common.REAL_PATH
    )
    pl.DataFrame([r for part in results for r in part[1]]).write_parquet(common.HELDOUT_PATH)
    print(pl.read_parquet(common.REAL_PATH).filter(pl.col("arm") == "O3_0.0"))
    print(pl.read_parquet(common.HELDOUT_PATH).group_by("site").agg(pl.col("change").mean()))


if __name__ == "__main__":
    main()
