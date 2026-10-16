"""Known-answer test that carries noise: inject into a demand series with day-shuffled residuals.

The increment test of `e00073_known_orientation.py` is blind to noise, because a least-squares fit
is linear and the increment of a fit with and without an injection is the fit of the injection
alone. This script instead builds, for each Primary and season-year, a series with no solar of its
own and the real series' noise:

    y_syn = (y + U c_O1 - r) + r_shuffled - injection

where `c_O1` is the free fit's capacities (so the Primary's own fitted solar is added back), `r` is
the part of the solar-free series left after removing its cell means and nuisance terms, and
`r_shuffled` swaps whole days of `r` within each month. The truth is the injection alone, and the
arms are scored on the fitted capacities directly. Three shuffles per unit.

    POLARS_MAX_THREADS=6 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run --with holidays python \
        ml-experiments/00073-pvdis-6-orientation-mix-and-dc-ac-ratio/e00073_known_noisy.py
"""

from multiprocessing import Pool

import e00073_common as common
import e00073_fit as fit
import e00073_inputs as inputs
import numpy as np
import polars as pl

SHUFFLES: int = 3


def day_donors(*, w: inputs.YearWindow, rng: np.random.Generator) -> np.ndarray:
    """Return, per row, the row index of the same half-hour on a donor day of the same month."""
    half_hour = (w.cells % 1000) // 10
    month = w.cells // 1000
    unique_days, day_index = np.unique(w.days, return_inverse=True)
    table = np.full((len(unique_days), 48), -1, dtype=int)
    table[day_index, half_hour] = np.arange(len(w.days))
    day_month = np.zeros(len(unique_days), dtype=int)
    day_month[day_index] = month
    donor_day = np.arange(len(unique_days))
    for m in np.unique(day_month):
        members = np.nonzero(day_month == m)[0]
        donor_day[members] = rng.permutation(members)
    donor = table[donor_day[day_index], half_hour]
    own = np.arange(len(w.days))
    return np.where(donor >= 0, donor, own)


def score_primary(*, prepared: pl.DataFrame, primary: str) -> list[dict[str, object]]:
    """Return the error rows of every noisy unit of one Primary."""
    windows = inputs.year_windows(prepared=prepared, primary=primary)
    ones = [np.ones(len(w.power), dtype=bool) for w in windows]
    usable = [
        k for k, w in enumerate(windows) if len(np.unique(w.days[w.clear])) >= common.MIN_CLEAR_DAYS
    ]
    designs_all = [w.design(mask=m) for w, m in zip(windows, ones, strict=True)]
    designs_clear = {k: windows[k].design(mask=windows[k].clear) for k in usable}
    shapes_all = [
        w.primary_basis(ratio=common.FLEET_RATIO, tilt=common.MAIN_TILT, mask=m)
        for w, m in zip(windows, ones, strict=True)
    ]
    structure, residual = [], []
    for k, w in enumerate(windows):
        c1 = fit.fit_orientation(design=designs_all[k], y=w.power, shapes=shapes_all[k])
        y0 = w.power + shapes_all[k] @ c1
        r = designs_all[k].residual(values=y0)
        structure.append(y0 - r)
        residual.append(r)
    out: list[dict[str, object]] = []
    for seed in range(SHUFFLES):
        rng = np.random.default_rng(common.BOOTSTRAP_SEED * 100 + seed + sum(map(ord, primary)))
        shuffled = [r[day_donors(w=w, rng=rng)] for w, r in zip(windows, residual, strict=True)]
        for tilt in common.TILTS:
            for farm in common.FARMS:
                for mix_id, mix in enumerate(common.MIXES):
                    truth = np.array(mix)
                    ys = []
                    for k, w in enumerate(windows):
                        inj = common.INJECTED_MW * (
                            w.farm_basis(
                                farm=farm, ratio=common.FLEET_RATIO, tilt=tilt, mask=ones[k]
                            )
                            @ truth
                        )
                        ys.append(structure[k] + shuffled[k] - inj)
                    base = {"site": primary, "farm": farm, "tilt": tilt, "mix_id": mix_id,
                            "seed": seed}  # fmt: skip
                    for k, w in enumerate(windows):
                        c = fit.fit_orientation(
                            design=designs_all[k], y=ys[k], shapes=shapes_all[k]
                        )
                        err, bound = fit.tv_error(increment=c, truth=truth)
                        out.append({**base, "arm": "O1", "year": w.year, "tv": err, "bound": bound})
                        if k in designs_clear:
                            cl = w.clear
                            c = fit.fit_orientation(
                                design=designs_clear[k], y=ys[k][cl], shapes=shapes_all[k][cl]
                            )
                            err, bound = fit.tv_error(increment=c, truth=truth)
                            out.append(
                                {**base, "arm": "O2", "year": w.year, "tv": err, "bound": bound}
                            )
                    if usable:
                        ds = [designs_clear[k] for k in usable]
                        sh = [shapes_all[k][windows[k].clear] for k in usable]
                        yc = [ys[k][windows[k].clear] for k in usable]
                        for lam in common.RIDGE_GRID:
                            c = fit.fit_pooled(designs=ds, ys=yc, shapes=sh, ridge=lam)
                            err, bound = fit.tv_error(increment=c, truth=truth)
                            out.append(
                                {**base, "arm": f"O3_{lam}", "year": -1, "tv": err, "bound": bound}
                            )
    return out


def _job(args: tuple[pl.DataFrame, str]) -> list[dict[str, object]]:
    prepared, primary = args
    return score_primary(prepared=prepared, primary=primary)


def main() -> None:
    """Run every Primary in parallel and write the noisy known-answer rows."""
    prepared = inputs.load_prepared()
    jobs = [(prepared.filter(pl.col("site") == p), p) for p in common.DIRECTIONAL_PRIMARIES]
    with Pool(6) as pool:
        rows = [row for part in pool.map(_job, jobs) for row in part]
    table = pl.DataFrame(rows)
    table.write_parquet(common.WORK_DIR / "known_noisy.parquet")
    print(
        table.filter(pl.col("tilt") == common.MAIN_TILT)
        .group_by("arm")
        .agg(pl.col("tv").median().alias("median_tv"))
        .sort("arm")
    )


if __name__ == "__main__":
    main()
