"""Post hoc: do extra starting sign patterns find reverse flow at Stickney and Wrangle T1?

The pre-registered starts flip rows by a capacity multiple of the window's median midday reading.
This script adds starts that flip every row whose unit solar output exceeds a threshold, and reports
the loss reduction against the naive fit, the fitted capacity, and the fitted reverse-flow share.
It changes no pre-registered number.

    POLARS_MAX_THREADS=4 OMP_NUM_THREADS=1 uv run --with holidays python \
        ml-experiments/00079-pvdis-7-mw-at-mva-only-primaries/e00079_post_hoc.py
"""

from typing import Final

import e00079_common as common
import e00079_fit as fit
import e00079_inputs as inputs
import numpy as np
import polars as pl

THRESHOLDS: Final[tuple[float, ...]] = (0.2, 0.3, 0.4, 0.5, 0.6)
SITES: Final[tuple[str, ...]] = ("D12", "D15", "D22", "D8")


def main() -> None:
    """Fit each window of the chosen Primaries with the extra starts and write `post_hoc.txt`."""
    prepared = inputs.load_prepared()
    rows = []
    for site in SITES:
        for w in inputs.year_windows(prepared=prepared, primary=site):
            design = fit.Design.build(cells=w.cells, nuisance=w.nuisance)
            a = np.abs(w.power)
            u = w.shape
            c_naive, m_naive = fit.naive_fit(design=design, a=a, u=u)
            loss_naive = fit.abs_loss(a=a, m=m_naive)
            best = (c_naive, m_naive, loss_naive)
            for tau in THRESHOLDS:
                sign = np.where(u >= tau, -1.0, 1.0)
                found, _ = fit._run(design=design, a=a, u=u, sign=sign)
                if found.loss < best[2]:
                    best = (found.capacity, found.mean, found.loss)
            rows.append(
                {
                    "site": site, "year": w.year, "c_naive": c_naive, "c_best": best[0],
                    "negative_share": float((best[1] < 0).mean()),
                    "loss_reduction": 1.0 - best[2] / loss_naive,
                }
            )  # fmt: skip
    table = pl.DataFrame(rows)
    pl.Config.set_tbl_rows(60)
    text = str(table.sort("site", "year"))
    (common.WORK_DIR / "post_hoc.txt").write_text(text)
    print(text)


if __name__ == "__main__":
    main()
