"""Score the solar-factor refit cadences (C5): monthly against quarterly, on the common rows.

Gate: the rebuilt monthly arm must equal experiment 202's stored `scored_r2.npy` (max absolute
difference under 1e-3 points). The quarterly arm is added to experiment 220's arm A in place of the
monthly arm (`a_s`), with and without the stored head correction.

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00278-*/e00278_solar_score.py
"""

import json

import numpy as np
import polars as pl

import e00278_common as common  # isort: skip

SEED = 278
E202_ARMS = (
    common.ML_EXPERIMENTS_DATA_DIR / "00202-dp-ens-two-number-rolling-replay" / "work" / "arms"
)


def main() -> None:
    """Gate, score, and write the report."""
    rows = pl.read_parquet(common.E220_ARMS / "rows.parquet")
    index = rows["r2_index"].to_numpy()
    monthly = np.load(common.WORK_DIR / "arms" / "scored_monthly.npy").astype(np.float64)
    quarterly = np.load(common.WORK_DIR / "arms" / "scored_quarterly.npy").astype(np.float64)
    stored = np.load(E202_ARMS / "scored_r2.npy").astype(np.float64)
    assert np.abs(np.sort(monthly, axis=1) - stored).max() < 1e-3, "monthly gate: factors differ"
    report: dict = {
        "gate_max_abs_diff_monthly_vs_202": float(np.abs(np.sort(monthly, axis=1) - stored).max())
    }
    base = {
        n: np.load(common.E220_ARMS / f"{n}.npy").astype(np.float64) for n in ("a", "a_s", "a_hs")
    }
    delta = (base["a_hs"] - base["a_s"])[:, [5]]
    arms = {
        "s_monthly": base["a_s"],
        "s_quarterly": quarterly[index],
        "hs_monthly": base["a_hs"],
        "hs_quarterly": quarterly[index] + delta,
    }
    edges = common.edges_of(rows=rows)
    actual = rows["actual"].to_numpy().astype(np.float64)
    metrics = {
        k: common.per_row(actual=actual, q=np.sort(v, axis=1), edges=edges) for k, v in arms.items()
    }
    months = rows["month"].to_numpy()
    daylight = rows["daylight"].to_numpy()
    origin_month = rows["power_fcst_init_time"].dt.month().to_numpy()
    everything = np.ones(rows.height, dtype=bool)
    rng = np.random.default_rng(SEED)
    report["n_rows"] = int(rows.height)
    report["n_daylight_rows"] = int(daylight.sum())
    report["absolute"] = {
        k: {m: float(v[m][daylight].mean()) for m in ("qs", "mae")}
        | {"qs_all": float(v["qs"].mean())}
        for k, v in metrics.items()
    }
    report["contrasts"] = {}
    for stack in ("s", "hs"):
        d = metrics[f"{stack}_quarterly"]["qs"] - metrics[f"{stack}_monthly"]["qs"]
        subsets = {
            "daylight": daylight,
            "all": everything,
            "daylight_winter": daylight & np.isin(origin_month, list(common.WINTER_MONTHS)),
            "daylight_summer": daylight & np.isin(origin_month, list(common.SUMMER_MONTHS)),
        }
        report["contrasts"][f"C5 {stack} quarterly minus monthly"] = {
            name: {
                "99": common.mean_interval(values=d, mask=mask, blocks=months, level=99.0, rng=rng),
                "95": common.mean_interval(values=d, mask=mask, blocks=months, level=95.0, rng=rng),
            }
            for name, mask in subsets.items()
        }
        by_since = {}
        since = (origin_month - 1) % 3
        for k in (0, 1, 2):
            sel = daylight & (since == k)
            by_since[str(k)] = float(d[sel].mean())
        report["contrasts"][f"C5 {stack} quarterly minus monthly"][
            "daylight_by_months_since_refit"
        ] = by_since
        report["contrasts"][f"C5 {stack} quarterly minus monthly"]["by_origin_month"] = {
            m: float(d[(months == m) & daylight].mean()) for m in sorted(set(months.tolist()))
        }
    common.write_json_atomic(path=common.REPORT_DIR / "solar_report.json", content=report)
    print(json.dumps(report, indent=1, default=float))


if __name__ == "__main__":
    main()
