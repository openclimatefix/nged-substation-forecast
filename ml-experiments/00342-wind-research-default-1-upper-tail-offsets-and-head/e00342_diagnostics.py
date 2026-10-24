"""Experiment 00342, diagnostics: why the hindcast cannot fix the upper tail.

For default seed 0, rebuild the default's hindcast quantiles (experiment 00340's recipe) and report,
per farm: the share of hindcast fit rows (valid before 2025-01-01), of the hindcast months of 2025,
and of the scored year with observed power above the default's q95 and q99; and what a constant
shift of q99 fitted on the scored year itself (an oracle, never adopted) would have gained. Also
check that the shuffled-input placebo arm differs from the base-only arm.

Output: `work/diagnostics.json`.

    uv run --no-sync python ml-experiments/00342-*/e00342_diagnostics.py   (PYTHONPATH: common)
"""

import json
import os

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import e00303_ens_final as ens_final  # noqa: E402  # isort: skip  # ty: ignore[unresolved-import]
import e00336_final as final336  # noqa: E402  # isort: skip  # ty: ignore[unresolved-import]
import e00342_common as common  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

c303 = common.c303


def pinball(*, q: np.ndarray, actual: np.ndarray, level: float) -> float:
    """Return the mean pinball loss (points of capacity) of one quantile forecast."""
    d = actual - q
    return float(np.where(d >= 0, level * d, (level - 1) * d).mean() * c303.POINTS_PER_FRACTION)


def main() -> None:
    """Write the diagnostics."""
    sites = c303.site_table()
    hind = ens_final.hindcast_cases(sites=sites)
    scored = ens_final.scored_cases(sites=sites)
    usable = hind["actual_norm"].is_finite().fill_null(value=False).to_numpy()
    hu = hind.filter(pl.Series(usable))
    all_cases = pl.concat(
        [
            hu.select(*c303.KEY_COLUMNS, "lead_hours", "actual_norm"),
            scored.select(*c303.KEY_COLUMNS, "lead_hours", "actual_norm"),
        ],
        how="vertical_relaxed",
    )
    raw_h = final336.load(arm="default", seed=0, set_name="hindcast")[usable]
    raw_s = final336.load(arm="default", seed=0, set_name="scored")
    adjusted = c303.apply_spread(
        cases=all_cases, mixture=np.concatenate([raw_h, raw_s]).astype(np.float64)
    )
    adj_h, adj_s = adjusted[: hu.height], adjusted[hu.height :]
    table, keys = ens_final.offsets_table(hindcast=hu, quantiles=adj_h)
    hind_q = ens_final.apply_offsets(cases=hu, quantiles=adj_h, table=table, keys=keys)
    scored_q = ens_final.apply_offsets(cases=scored, quantiles=adj_s, table=table, keys=keys)
    actual_h = hu["actual_norm"].to_numpy().astype(np.float64)
    actual_s = scored["actual_norm"].to_numpy().astype(np.float64)
    fit = (hu["valid_time"] < common.REPLICATE_START).to_numpy()
    h2025 = ~fit
    kept_s = (~scored["masked"]).to_numpy()
    out: dict = {}
    for farm in common.FARMS:
        sh = hu["site"].to_numpy() == farm
        ss = scored["site"].to_numpy() == farm
        entry = {}
        for name, rows, q, a in (
            ("hindcast_fit_in_sample", sh & fit, hind_q, actual_h),
            ("hindcast_2025_h1", sh & h2025, hind_q, actual_h),
            ("scored_year_kept", ss & kept_s, scored_q, actual_s),
        ):
            entry[name] = {
                "rows": int(rows.sum()),
                "above_q95_pct": float((a[rows] > q[rows, 9]).mean() * 100),
                "above_q99_pct": float((a[rows] > q[rows, 10]).mean() * 100),
                "p99_points": pinball(q=q[rows, 10], actual=a[rows], level=0.99),
            }
        rows = ss & kept_s
        residual = actual_s[rows] - scored_q[rows, 10]
        shift = float(np.quantile(residual, 0.99))
        entry["oracle_q99_shift_points"] = shift * c303.POINTS_PER_FRACTION
        entry["oracle_p99_points"] = pinball(
            q=scored_q[rows, 10] + shift, actual=actual_s[rows], level=0.99
        )
        out[farm] = entry
    base = np.load(common.FINAL_DIR / "x2base.npy")
    placebo = np.load(common.FINAL_DIR / "x2plc.npy")
    full = np.load(common.FINAL_DIR / "x2.npy")
    out["placebo_vs_base_max_abs_difference"] = float(np.abs(placebo - base).max())
    out["full_vs_base_max_abs_difference"] = float(np.abs(full - base).max())
    (common.WORK_DIR / "diagnostics.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
