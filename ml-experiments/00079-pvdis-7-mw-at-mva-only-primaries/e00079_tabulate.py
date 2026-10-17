"""Tabulate, before any fit runs, how much reverse flow each injection level creates.

It counts, per MW Primary window and injection level, the half-hours with negative power, and
lists the midday minimum of the magnitude at each MVA-only Primary. Nothing is fitted.

    POLARS_MAX_THREADS=4 OMP_NUM_THREADS=1 uv run --with holidays python \
        ml-experiments/00079-pvdis-7-mw-at-mva-only-primaries/e00079_tabulate.py
"""

import e00079_common as common
import e00079_inputs as inputs
import numpy as np
import polars as pl


def main() -> None:
    """Write `tabulate.txt` with the reverse-flow counts and the MVA midday minima."""
    prepared = inputs.load_prepared()
    farms = inputs.farm_series(prepared=prepared)
    rows = []
    for primary in common.MW_PRIMARIES:
        for w in inputs.year_windows(prepared=prepared, primary=primary):
            for farm, series in farms.items():
                mask, injected = inputs.aligned_injection(window=w, farm=series)
                scale = w.midday_scale(mask=mask)
                floor = common.SIGN_FLOOR_SHARE * scale
                for level in (0.0, *common.LEVELS):
                    y = w.power[mask] - level * scale * injected
                    rows.append(
                        {
                            "site": primary, "year": w.year, "farm": farm, "level": level,
                            "rows": int(mask.sum()),
                            "negative": int((y < -floor).sum()),
                            "positive": int((y > floor).sum()),
                        }
                    )  # fmt: skip
    table = pl.DataFrame(rows)
    summary = (
        table.group_by("site", "level")
        .agg(
            pl.len().alias("windows"),
            (pl.col("negative") >= common.MIN_CLASS_ROWS).sum().alias("windows_ge_min_negative"),
            pl.col("negative").median().alias("median_negative"),
            (pl.col("negative") / pl.col("rows")).median().alias("median_share"),
        )
        .sort("level", "site")
    )
    mva = []
    for primary in common.MVA_PRIMARIES:
        for w in inputs.year_windows(prepared=prepared, primary=primary):
            lo, hi = common.MIDDAY_HALF_HOURS
            mid = (w.half_hour >= lo) & (w.half_hour < hi)
            mva.append(
                {
                    "site": primary, "year": w.year,
                    "midday_min": float(w.power[mid].min()),
                    "midday_p05": float(np.quantile(w.power[mid], 0.05)),
                    "midday_median": float(np.median(w.power[mid])),
                }
            )  # fmt: skip
    mva_table = (
        pl.DataFrame(mva)
        .group_by("site")
        .agg(
            pl.col("midday_min").min(),
            pl.col("midday_p05").median(),
            pl.col("midday_median").median(),
        )
    )
    pl.Config.set_tbl_rows(200)
    text = f"{summary}\n\n{mva_table.sort('site')}\n"
    common.WORK_DIR.mkdir(parents=True, exist_ok=True)
    (common.WORK_DIR / "tabulate.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
