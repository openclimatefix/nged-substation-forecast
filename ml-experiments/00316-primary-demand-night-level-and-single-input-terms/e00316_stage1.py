# ruff: noqa: E501
r"""Stage 1 of experiment 316: which single-input term of arm D differs from BG2's twin?

Grafts the contributions of experiment 305's depth-1 XGBoost refit onto arm D one input family at a
time. Arm D's term contributions are not stored, so a depth-1 surrogate of arm D's median stands in
for them: family k is grafted by shifting all 11 of arm D's quantiles by the refit's family-k
contribution minus the surrogate's.

    PYTHONPATH=ml-experiments/00305-summer-gap-what-bg2-knows-that-arm-d-does-not \\
        OMP_NUM_THREADS=4 uv run --no-sync python ml-experiments/00316-*/e00316_stage1.py
"""

from typing import Final

import e00305_analyse as an  # isort: skip  # ty: ignore[unresolved-import]
import e00305_fit as fit  # isort: skip  # ty: ignore[unresolved-import]
import e00316_common as common
import numpy as np
import polars as pl
import xgboost as xgb

SITES: Final[tuple[str, ...]] = ("D5", "D9", "D10")
TARGETS: Final[tuple[str, ...]] = ("D5", "D9")
MONTHS: Final[tuple[str, ...]] = ("2026-04", "2026-05", "2026-06")
KEYS: Final[list[str]] = ["site", "power_fcst_init_time", "valid_time"]
SURROGATE_ROUNDS: Final[int] = 300
THREADS: Final[int] = 4
GRAFTS: Final[dict[str, tuple[str, ...]]] = {
    **{k: (k,) for k in an.FAMILIES},
    "solar+lags+calendar": ("solar", "lags", "calendar"),
    "all five": tuple(an.FAMILIES),
}


def build_frame() -> pl.DataFrame:
    """Return master rows with inputs, the refit's contributions, and the surrogate's."""
    master = pl.read_parquet(common.E266_MASTER).filter(
        pl.col("site").is_in(SITES) & pl.col("init_month").is_in(MONTHS)
    )
    d_q = np.load(common.E251_SCORED / "d.npy").astype(np.float64)
    index = (
        pl.read_parquet(common.E251_SCORED / "rows.parquet", columns=KEYS)
        .with_row_index("position")
        .filter(pl.col("site").is_in(SITES))
    )
    position = master.join(index, on=KEYS, how="left", validate="1:1")["position"].to_numpy()
    q_d = d_q[position]
    check = common.qs_per_row(actual=master["actual"].to_numpy(), q=q_d)
    assert np.allclose(check, master["qs_d"].to_numpy(), atol=1e-4), "d.npy rows do not match qs_d"
    master = master.with_columns(
        *[pl.Series(f"d_{q}", q_d[:, i]) for i, q in enumerate(common.QUANTILES)]
    )
    inputs = pl.concat(
        [pl.read_parquet(p) for p in sorted((common.E305_DIR / "inputs").glob("*.parquet"))]
    )
    refit = pl.concat(
        [
            pl.read_parquet(p)
            for p in sorted((common.E305_DIR / "preds" / fit.ADDITIVE).glob("*.parquet"))
        ]
    ).rename({q: f"r_{q}" for q in common.QUANTILES})
    frame = (
        master.join(inputs, on=KEYS, how="inner", validate="1:1")
        .join(refit, on=KEYS, how="inner", validate="1:1")
        .with_columns(
            week=pl.col("power_fcst_init_time").dt.iso_year() * 100
            + pl.col("power_fcst_init_time").dt.week()
        )
    )
    assert frame.height == master.height, "a master row lost its inputs or refit"
    return frame


def contributions(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Add per-family contributions of the refit (`cr_`) and of the arm D surrogate (`cs_`)."""
    columns = fit.fit.input_columns(a7=False)
    pieces = []
    stats: dict = {}
    for (site, month), group in frame.group_by(["site", "init_month"], maintain_order=True):
        x = group.select(columns).to_numpy().astype(np.float32)
        refit_model = xgb.Booster()
        refit_model.load_model(
            str(common.E305_DIR / "models" / fit.ADDITIVE / f"{site}_{month}.ubj")
        )
        cr = refit_model.predict(xgb.DMatrix(x), pred_contribs=True)[:, 5, :] * fit.SCALE
        y = group["med_d"].to_numpy().astype(np.float64)
        params = {"max_depth": 1, "eta": 0.1, "nthread": THREADS, "objective": "reg:squarederror", "seed": 0}  # fmt: skip
        surrogate = xgb.train(params, xgb.DMatrix(x, label=y), num_boost_round=SURROGATE_ROUNDS)
        cs = surrogate.predict(xgb.DMatrix(x), pred_contribs=True)
        fitted = cs.sum(axis=1)
        stats[f"{site}_{month}"] = {
            "r2": float(1.0 - ((y - fitted) ** 2).sum() / ((y - y.mean()) ** 2).sum()),
            "rmse": float(np.sqrt(((y - fitted) ** 2).mean())),
            "refit_sum_matches_q50_raw": float(
                np.abs(cr.sum(axis=1) - group["r_q50"].to_numpy()).max()
            ),
        }
        data = {}
        for name, cols in an.FAMILIES.items():
            idx = [columns.index(c) for c in cols]
            data[f"cr_{name}"] = cr[:, idx].sum(axis=1)
            data[f"cs_{name}"] = cs[:, idx].sum(axis=1)
        data["cr_bias"], data["cs_bias"] = cr[:, -1], cs[:, -1]
        data["s_fit"] = fitted
        pieces.append(group.with_columns(*[pl.Series(k, v) for k, v in data.items()]))
    common.write_json_atomic(path=common.REPORT_DIR / "stage1_surrogate.json", content=stats)
    return pl.concat(pieces)


def subsets(*, frame: pl.DataFrame) -> dict[str, pl.DataFrame]:
    """Return experiment 305's row groups: the worst quarter and all daylight rows."""
    out: dict[str, pl.DataFrame] = {}
    out["D5+D9 worst quarter"] = pl.concat([an.selected(rows=frame, site=s) for s in TARGETS])
    out["D5+D9 all daylight"] = frame.filter(pl.col("site").is_in(TARGETS) & pl.col("daylight"))
    for s in (*TARGETS, "D10"):
        out[f"{s} worst quarter"] = an.selected(rows=frame, site=s)
    out["D10 all daylight"] = frame.filter((pl.col("site") == "D10") & pl.col("daylight"))
    return out


def main() -> None:
    """Score the grafts and write the stage 1 report."""
    frame = contributions(frame=build_frame())
    frame.write_parquet(common.WORK_DIR / "stage1_frame.parquet")
    report: dict = {}
    for name, rows in subsets(frame=frame).items():
        actual = rows["actual"].to_numpy().astype(np.float64)
        q_d = rows.select([f"d_{q}" for q in common.QUANTILES]).to_numpy().astype(np.float64)
        q_r = np.sort(
            rows.select([f"r_{q}" for q in common.QUANTILES]).to_numpy().astype(np.float64), axis=1
        )
        qs_d, qs_r = (
            common.qs_per_row(actual=actual, q=q_d),
            common.qs_per_row(actual=actual, q=q_r),
        )
        blocks = rows["week"].to_numpy()
        entry: dict = {
            "rows": rows.height,
            "qs_d": float(qs_d.mean()),
            "qs_refit": float(qs_r.mean()),
            "qs_bg2_stored": float(np.mean(rows["qs_bg2"].to_numpy())),
            "gap_med_refit_minus_d": float(np.mean((rows["r_q50"] - rows["med_d"]).to_numpy())),
            "S": {},
        }
        for graft, families in GRAFTS.items():
            delta = sum(rows[f"cr_{f}"].to_numpy() - rows[f"cs_{f}"].to_numpy() for f in families)
            qs_g = common.qs_per_row(actual=actual, q=q_d + delta[:, None])
            entry["S"][graft] = {
                **common.ratio_interval(num=qs_d - qs_g, den=qs_d - qs_r, blocks=blocks, seed=11),
                "qs_graft": float(qs_g.mean()),
                "mean_delta": float(np.mean(delta)),
            }
        med_d, s_fit = rows["med_d"].to_numpy(), rows["s_fit"].to_numpy()
        entry["surrogate_r2_rows"] = float(
            1.0 - ((med_d - s_fit) ** 2).sum() / ((med_d - med_d.mean()) ** 2).sum()
        )
        report[name] = entry
    common.write_json_atomic(path=common.REPORT_DIR / "stage1.json", content=report)
    for name, entry in report.items():
        print(name, entry["rows"], round(entry["qs_d"], 3), round(entry["qs_refit"], 3), {k: round(v["est"], 3) for k, v in entry["S"].items()})  # fmt: skip


if __name__ == "__main__":
    common.WORK_DIR.mkdir(parents=True, exist_ok=True)
    main()
