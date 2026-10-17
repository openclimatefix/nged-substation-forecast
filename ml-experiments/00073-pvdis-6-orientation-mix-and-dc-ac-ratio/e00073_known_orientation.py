"""Known-answer test for the orientation mix: inject made-up east-south-west output and recover it.

For each Primary, season-year, farm location, tilt, and true mix, the script adds
`3 MW x (wE uE + wS uS + wW uW)` (shapes from CAMS at the farm location, clipped at the fleet ratio)
to the real power of the Primary and fits the orientation arms to the series with and without the
injection. The recovered capacities are the increments of the fitted ones.

    POLARS_MAX_THREADS=6 OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 uv run --with holidays python \
        ml-experiments/00073-pvdis-6-orientation-mix-and-dc-ac-ratio/e00073_known_orientation.py
"""

from multiprocessing import Pool

import e00073_common as common
import e00073_fit as fit
import e00073_inputs as inputs
import e00073_selftest as selftest
import numpy as np
import polars as pl

ARMS: tuple[str, ...] = ("O1", "O2", *(f"O3_{lam}" for lam in common.RIDGE_GRID))


def score_primary(*, prepared: pl.DataFrame, primary: str) -> list[dict[str, object]]:
    """Return the error rows of every unit of one Primary."""
    windows = inputs.year_windows(prepared=prepared, primary=primary)
    all_masks = [np.ones(len(w.power), dtype=bool) for w in windows]
    usable = [
        k for k, w in enumerate(windows) if len(np.unique(w.days[w.clear])) >= common.MIN_CLEAR_DAYS
    ]
    designs_all = [w.design(mask=m) for w, m in zip(windows, all_masks, strict=True)]
    designs_clear = {k: windows[k].design(mask=windows[k].clear) for k in usable}
    out: list[dict[str, object]] = []
    for tilt in common.TILTS:
        for farm in common.FARMS:
            for mix_id, mix in enumerate(common.MIXES):
                truth = np.array(mix)
                injections = [
                    common.INJECTED_MW
                    * (
                        w.farm_basis(
                            farm=farm,
                            ratio=common.FLEET_RATIO,
                            tilt=tilt,
                            mask=np.ones_like(w.clear),
                        )
                        @ truth
                    )
                    for w in windows
                ]
                base = {
                    "site": primary, "farm": farm, "tilt": tilt, "mix_id": mix_id,
                }  # fmt: skip
                for k, w in enumerate(windows):
                    shapes = w.primary_basis(
                        ratio=common.FLEET_RATIO, tilt=common.MAIN_TILT, mask=all_masks[k]
                    )
                    inc = fit.fit_orientation(
                        design=designs_all[k], y=w.power - injections[k], shapes=shapes
                    ) - fit.fit_orientation(design=designs_all[k], y=w.power, shapes=shapes)
                    err, bound = fit.tv_error(increment=inc, truth=truth)
                    out.append({**base, "arm": "O1", "year": w.year, "tv": err, "bound": bound})
                    if k in designs_clear:
                        clear = w.clear
                        shapes_c = shapes[clear]
                        inc = fit.fit_orientation(
                            design=designs_clear[k], y=(w.power - injections[k])[clear],
                            shapes=shapes_c,
                        ) - fit.fit_orientation(
                            design=designs_clear[k], y=w.power[clear], shapes=shapes_c
                        )  # fmt: skip
                        err, bound = fit.tv_error(increment=inc, truth=truth)
                        out.append({**base, "arm": "O2", "year": w.year, "tv": err, "bound": bound})
                if usable:
                    ds = [designs_clear[k] for k in usable]
                    sh = [
                        windows[k].primary_basis(
                            ratio=common.FLEET_RATIO, tilt=common.MAIN_TILT,
                            mask=windows[k].clear,
                        )
                        for k in usable
                    ]  # fmt: skip
                    y0 = [windows[k].power[windows[k].clear] for k in usable]
                    y1 = [(windows[k].power - injections[k])[windows[k].clear] for k in usable]
                    for lam in common.RIDGE_GRID:
                        inc = fit.fit_pooled(
                            designs=ds, ys=y1, shapes=sh, ridge=lam
                        ) - fit.fit_pooled(designs=ds, ys=y0, shapes=sh, ridge=lam)
                        err, bound = fit.tv_error(increment=inc, truth=truth)
                        out.append(
                            {**base, "arm": f"O3_{lam}", "year": -1, "tv": err, "bound": bound}
                        )
    return out


def _job(args: tuple[pl.DataFrame, str]) -> list[dict[str, object]]:
    prepared, primary = args
    return score_primary(prepared=prepared, primary=primary)


def main() -> None:
    """Run the self-tests, then every Primary in parallel, and write the error rows."""
    selftest.run()
    common.WORK_DIR.mkdir(parents=True, exist_ok=True)
    prepared = inputs.load_prepared()
    jobs = [(prepared.filter(pl.col("site") == p), p) for p in common.DIRECTIONAL_PRIMARIES]
    with Pool(6) as pool:
        rows = [row for part in pool.map(_job, jobs) for row in part]
    table = pl.DataFrame(rows)
    table.write_parquet(common.KNOWN_ORIENTATION_PATH)
    summary = (
        table.filter(pl.col("tilt") == common.MAIN_TILT)
        .group_by("arm")
        .agg(pl.col("tv").median().alias("median_tv"), pl.col("bound").mean().alias("bound_share"))
        .sort("arm")
    )
    print(summary)


if __name__ == "__main__":
    main()
