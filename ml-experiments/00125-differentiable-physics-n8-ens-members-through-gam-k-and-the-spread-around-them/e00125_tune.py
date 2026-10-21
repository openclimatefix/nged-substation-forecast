"""Tune experiment 125's factors and rescaling on the tuning months, before any other period.

Everything here reads the tuning months only (the odd months of 2024-07 to 2025-06, for origin and
valid time alike). The steps, in order, per site and then pooled over the five decision sites:

1. **Residual tables.** K0's table (ensemble-mean residuals) and MP's table (proxy-driven
   residuals), each fitted on the tuning rows outside the switching masks.
2. **Factors.** Per lead band, the factor of the wide grid with the lowest quantile score, for K0-w,
   MP, and MM (the table of K0 with the mean as the centre). K0 itself keeps experiment 111's
   stored factors.
3. **Rescaling s.** With MP's factors held, the s of the grid with the lowest mean pinball at q01,
   q05, q95, and q99 per lead band; then the factors re-tuned given that s.

The score surfaces are written whole, so the report can print the ridge between s and the factor.

    uv run python ml-experiments/00125-*/e00125_tune.py [--workers 3]
"""

import argparse
import time
from concurrent.futures import ProcessPoolExecutor

import e00125_arms as arms
import e00125_cache as cache_module
import e00125_common as common
import e00125_post as post
import numpy as np
from studies.dp_ens_weather import N_LEAD_BANDS, lead_band_of_hours


def band_sums(*, values: np.ndarray, rows: post.Rows) -> list[list[float]]:
    """Return the sum and count of per-row values in each lead band, outside switching masks."""
    band = lead_band_of_hours(lead_hours=rows.lead_hours)
    keep = ~rows.switching_mask
    return [
        [float(values[keep & (band == b)].sum()), int((keep & (band == b)).sum())]
        for b in range(N_LEAD_BANDS)
    ]


def tune_site(*, site: str) -> dict:
    """Score every factor and every s at one site on the tuning months.

    Args:
        site: The substation's label.

    Returns:
        Per-band score sums for each (arm, factor, s) cell, and the stored K0 factors.
    """
    runs = cache_module.read_cache(site=site, period="tuning")
    mask = post.stacked_months_mask(runs=runs, months=common.TUNING_MONTHS)
    k0_table = post.residual_table(runs=runs, centre="mean", row_filter=mask)
    mp_table = post.residual_table(runs=runs, centre="proxy", row_filter=mask)
    in_period = post.month_mask(
        valid_time=np.concatenate([run.valid_time for run in runs]), months=common.TUNING_MONTHS
    )
    out: dict = {"site": site, "quantile_score": {}, "tail_pinball": {}}

    def record(*, key: str, rows: post.Rows) -> None:
        rows = post.select(rows=rows, mask=in_period)
        out["quantile_score"][key] = band_sums(
            values=post.quantile_score(actual=rows.actual, quantiles=rows.quantiles), rows=rows
        )
        out["tail_pinball"][key] = band_sums(values=post.tail_pinball(rows=rows), rows=rows)

    for factor in post.FACTOR_GRID:
        flat = [factor] * N_LEAD_BANDS
        record(
            key=f"K0|{factor}",
            rows=post.build_rows(
                runs=runs, spread=k0_table.with_factors(factors=flat), centre="members"
            ),
        )
        record(
            key=f"MP|{factor}",
            rows=post.build_rows(
                runs=runs, spread=mp_table.with_factors(factors=flat), centre="members"
            ),
        )
        record(
            key=f"MM|{factor}",
            rows=post.build_rows(
                runs=runs, spread=k0_table.with_factors(factors=flat), centre="mean"
            ),
        )
    for s_value in post.S_GRID:
        for factor in post.FACTOR_GRID:
            flat = [factor] * N_LEAD_BANDS
            record(
                key=f"MPs|{s_value}|{factor}",
                rows=post.build_rows(
                    runs=runs,
                    spread=mp_table.with_factors(factors=flat),
                    centre="members",
                    s=[s_value] * N_LEAD_BANDS,
                ),
            )
    return out


def _job(site: str) -> dict:
    started = time.monotonic()
    result = tune_site(site=site)
    print(f"{site} tuned in {time.monotonic() - started:.0f} s", flush=True)
    return result


def pooled(*, results: list[dict], metric: str, key: str) -> np.ndarray:
    """Return the decision sites' pooled mean per lead band for one cell."""
    total = np.zeros((N_LEAD_BANDS, 2))
    for result in results:
        if result["site"] in common.DECISION_SITES:
            total += np.array(result[metric][key])
    return total[:, 0] / total[:, 1]


def main() -> None:
    """Tune over the six scored sites and write `work/tuned_n8.json`."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        results = list(pool.map(_job, common.SCORED_SITES))
    grid = post.FACTOR_GRID
    chosen: dict = {}
    for name in ("K0", "MP", "MM"):
        surface = np.stack(
            [pooled(results=results, metric="quantile_score", key=f"{name}|{f}") for f in grid]
        )
        chosen[name] = np.array(grid)[surface.argmin(axis=0)].tolist()
    # s first, factor held at MP's chosen value.
    s_surface = np.zeros((len(post.S_GRID), N_LEAD_BANDS))
    for i, s_value in enumerate(post.S_GRID):
        for band in range(N_LEAD_BANDS):
            key = f"MPs|{s_value}|{chosen['MP'][band]}"
            s_surface[i, band] = pooled(results=results, metric="tail_pinball", key=key)[band]
    s_chosen = np.array(post.S_GRID)[s_surface.argmin(axis=0)]
    factor_given_s = []
    for band in range(N_LEAD_BANDS):
        scores = [
            pooled(
                results=results,
                metric="quantile_score",
                key=f"MPs|{s_chosen[band]}|{f}",
            )[band]
            for f in grid
        ]
        factor_given_s.append(grid[int(np.argmin(scores))])
    surfaces = {
        "quantile_score": {
            key: pooled(results=results, metric="quantile_score", key=key).tolist()
            for key in results[0]["quantile_score"]
        },
        "tail_pinball": {
            key: pooled(results=results, metric="tail_pinball", key=key).tolist()
            for key in results[0]["tail_pinball"]
        },
    }
    common.write_json_atomic(
        path=common.WORK_DIR / "tuned_n8.json",
        content={
            "stored_k0_factors": arms.read_tuned()["spread_factors"]["gamk"],
            "k0w_factors": chosen["K0"],
            "mp_factors": chosen["MP"],
            "mm_factors": chosen["MM"],
            "mp_s": s_chosen.tolist(),
            "mp_s_factors": factor_given_s,
            "s_surface_tail_pinball": s_surface.tolist(),
            "surfaces": surfaces,
            "per_site": {r["site"]: r for r in results},
        },
    )
    print("K0-w", chosen["K0"], "MP", chosen["MP"], "MM", chosen["MM"], "s", s_chosen.tolist())


if __name__ == "__main__":
    main()
