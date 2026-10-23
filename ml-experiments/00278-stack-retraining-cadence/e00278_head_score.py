# ruff: noqa: E501
"""Score the network head's retrain cadences, on arm A (with and without the solar summary).

Arms (all on experiment 220's rows, `a` = arm D with experiment 169's bands, `a_s` = A with the
two-number solar summary): `once` adds experiment 220's stored head correction (trained once before
2025-07); `quarterly` adds the correction of the head fitted at the start of the row's quarter
(July, October, January, April); `monthly` adds the head fitted at the start of the row's month.
Each correction is the mean over three seeds. Also scores BG2 refitted weekly and monthly.

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00278-*/e00278_head_score.py
"""

import json

import numpy as np
import polars as pl

import e00278_common as common  # isort: skip

SEED = 278
BG2 = {"bg2_weekly": "p_s6_bg2", "bg2_monthly": "p_a3_monthly_in_weekly_slot"}
QUARTER_START = {1: 1, 2: 1, 3: 1, 4: 4, 5: 4, 6: 4, 7: 7, 8: 7, 9: 7, 10: 10, 11: 10, 12: 10}


def head_delta(*, rows: pl.DataFrame, cadence: str) -> np.ndarray:
    """Return the mean-over-seeds median correction of a retrain cadence on each row, `NaN` if none."""
    origin = rows["power_fcst_init_time"]
    months = origin.dt.month().to_list()
    years = origin.dt.year().to_list()
    labels = []
    for y, m in zip(years, months, strict=True):
        start = m if cadence == "monthly" else QUARTER_START[m]
        labels.append(f"{y}-{start:02d}")
    delta = np.full(rows.height, np.nan)
    labels_arr = np.array(labels)
    for label in sorted(set(labels)):
        files = sorted((common.WORK_DIR / "head").glob(f"cut{label}_seed*.parquet"))
        if len(files) != 3:
            continue
        stacked = pl.concat(
            [pl.read_parquet(f, columns=["site", "run", "valid_time", "delta"]) for f in files]
        )
        mean = stacked.group_by(common.KEYS).agg(pl.col("delta").mean())
        index = np.flatnonzero(labels_arr == label)
        part = (
            rows[index]
            .select(common.KEYS)
            .with_row_index("i")
            .join(mean, on=common.KEYS, how="left")
            .sort("i")
        )
        delta[index] = part["delta"].to_numpy()
    return delta


def main() -> None:
    """Score, write the report, and print it."""
    rows = pl.read_parquet(common.E220_ARMS / "rows.parquet").with_row_index("row")
    base = {
        name: np.load(common.E220_ARMS / f"{name}.npy").astype(np.float64)
        for name in ("a", "a_s", "a_h", "a_hs")
    }
    delta_once = (base["a_h"] - base["a"])[:, [5]]
    stored = {}
    for name, folder in BG2.items():
        keys = ["site", "power_fcst_init_time", "valid_time"]
        frame = pl.read_parquet(
            common.E87_PREDICTIONS / folder / "predictions.parquet",
            columns=[*keys, *common.QUANTILES],
        ).unique(subset=keys)
        joined = (
            rows.select([*keys, "row"]).join(frame, on=keys, how="left", validate="1:1").sort("row")
        )
        stored[name] = joined.select(list(common.QUANTILES)).to_numpy().astype(np.float64) * 100.0
    arms = {
        "a_once": base["a"] + delta_once,
        "stack_once": base["a_s"] + delta_once,
        **stored,
    }
    for cadence in ("quarterly", "monthly"):
        delta = head_delta(rows=rows, cadence=cadence)[:, None]
        arms[f"a_{cadence}"] = base["a"] + delta
        arms[f"stack_{cadence}"] = base["a_s"] + delta
    arms["a_no_head"] = base["a"]
    arms["stack_no_head"] = base["a_s"]
    keep = np.ones(rows.height, dtype=bool)
    for q in arms.values():
        keep &= np.isfinite(q).all(axis=1)
    assert keep.all(), f"{int((~keep).sum())} rows have a missing arm"
    rows = rows.filter(pl.Series(keep))
    arms = {k: np.sort(v[keep], axis=1) for k, v in arms.items()}
    edges = common.edges_of(rows=rows)
    actual = rows["actual"].to_numpy().astype(np.float64)
    metrics = {k: common.per_row(actual=actual, q=v, edges=edges) for k, v in arms.items()}
    months = rows["month"].to_numpy()
    origin = rows["power_fcst_init_time"]
    origin_month = origin.dt.month().to_numpy()
    weeks = origin.dt.strftime("%G-%V").to_numpy()
    winter = np.isin(origin_month, list(common.WINTER_MONTHS))
    summer = np.isin(origin_month, list(common.SUMMER_MONTHS))
    last_quarter = np.isin(origin_month, [4, 5, 6])
    everything = np.ones(rows.height, dtype=bool)
    rng = np.random.default_rng(SEED)
    report: dict = {"n_rows": int(rows.height), "n_rows_last_quarter": int(last_quarter.sum())}
    report["absolute"] = {
        k: {
            m: float(v[m].mean())
            for m in ("qs", "mae", "tw95", "tw99", "pin_q01", "pin_q99", "cov_q01", "cov_q99")
        }
        for k, v in metrics.items()
    }
    report["contrasts"] = {}
    for stack in ("a", "stack"):
        for label, (x, y) in {
            "C3 once minus quarterly": (f"{stack}_once", f"{stack}_quarterly"),
            "C4 monthly minus quarterly": (f"{stack}_monthly", f"{stack}_quarterly"),
            "monthly minus once": (f"{stack}_monthly", f"{stack}_once"),
            "no head minus quarterly": (f"{stack}_no_head", f"{stack}_quarterly"),
        }.items():
            d = metrics[x]["qs"] - metrics[y]["qs"]
            entry = {}
            for subset_name, mask in (
                ("all", everything),
                ("last_quarter", last_quarter),
                ("winter", winter),
                ("summer", summer),
            ):
                entry[subset_name] = {
                    "month_blocks_99": common.mean_interval(
                        values=d, mask=mask, blocks=months, level=99.0, rng=rng
                    ),
                    "month_blocks_95": common.mean_interval(
                        values=d, mask=mask, blocks=months, level=95.0, rng=rng
                    ),
                    "week_blocks_99": common.mean_interval(
                        values=d, mask=mask, blocks=weeks, level=99.0, rng=rng
                    ),
                }
            entry["mae_all_95"] = common.mean_interval(
                values=metrics[x]["mae"] - metrics[y]["mae"],
                mask=everything,
                blocks=months,
                level=95.0,
                rng=rng,
            )
            entry["by_origin_month"] = {
                m: float(d[months == m].mean()) for m in sorted(set(months.tolist()))
            }
            report["contrasts"][f"{stack}: {label}"] = entry
    # Skill by months since the last retrain (quarterly: 0-2; once: months since 2025-07).
    since_once = (rows["power_fcst_init_time"].dt.year().to_numpy() - 2025) * 12 + origin_month - 7
    since_q = (origin_month - 1) % 3
    report["by_months_since"] = {
        "once_qs": {
            str(k): float(metrics["stack_once"]["qs"][since_once == k].mean()) for k in range(12)
        },
        "quarterly_qs": {
            str(k): float(metrics["stack_quarterly"]["qs"][since_q == k].mean()) for k in range(3)
        },
        "monthly_qs": float(metrics["stack_monthly"]["qs"].mean()),
        "once_minus_monthly_by_k": {
            str(k): float(
                (metrics["stack_once"]["qs"] - metrics["stack_monthly"]["qs"])[
                    since_once == k
                ].mean()
            )
            for k in range(12)
        },
        "quarterly_minus_monthly_by_k": {
            str(k): float(
                (metrics["stack_quarterly"]["qs"] - metrics["stack_monthly"]["qs"])[
                    since_q == k
                ].mean()
            )
            for k in range(3)
        },
    }
    fit_seconds = pl.concat(
        [
            pl.read_parquet(f, columns=["seconds", "epochs", "n_train"]).head(1)
            for f in sorted((common.WORK_DIR / "head").glob("cut*_seed*.parquet"))
        ]
    )
    report["head_fit"] = {
        "mean_seconds": fit_seconds.select(pl.col("seconds").mean()).item(),
        "mean_epochs": fit_seconds.select(pl.col("epochs").mean()).item(),
        "n_fits": fit_seconds.height,
    }
    common.write_json_atomic(path=common.REPORT_DIR / "head_report.json", content=report)
    print(json.dumps(report, indent=1, default=float))


if __name__ == "__main__":
    main()
