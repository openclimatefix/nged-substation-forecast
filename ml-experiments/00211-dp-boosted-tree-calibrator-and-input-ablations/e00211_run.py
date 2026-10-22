"""Experiment 211: input ablations of the network without error inputs, and a boosted-tree model.

`--part nets` retrains the network without error inputs (the reference) and the same network with
lead, half-hour, arm D's forecast level, or the ensemble-era flag zeroed, one seed each, on CPU.
`--part xgb` fits XGBoost with the pinball loss at the 11 quantiles on the same inputs.
`--part report` scores every arm on experiment 200's scored rows and prints the contrasts.
Run: `uv run --no-sync python e00211_run.py --part nets` (4 threads).
"""

import argparse
import json
import time
from typing import Final

import e00211_data as data_module
import e00211_model as model_module
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]

TRAIN_END: Final[np.datetime64] = np.datetime64("2025-06-28", "us")
VALID: Final[tuple[np.datetime64, np.datetime64]] = (
    np.datetime64("2025-04-28", "us"),
    np.datetime64("2025-06-24", "us"),
)
PURGE: Final[np.timedelta64] = np.timedelta64(15, "D")
SCORED_START: Final[np.datetime64] = np.datetime64("2025-07-02", "us")
LEVELS: Final[np.ndarray] = np.array(
    [0.01, 0.05, 0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90, 0.95, 0.99]
)
DECISION: Final[tuple[str, ...]] = ("D5", "D10", "D11", "D14", "D16")
N_RESAMPLES: Final[int] = 20_000
ARMS_DIR = data_module.WORK_DIR / "arms"
ERROR_COLUMNS: Final[tuple[int, ...]] = (*range(18), 29, 30)
"""The 18 error columns, the Kalman level, and the Kalman anomaly: zeroed in every network here."""
ABLATIONS: Final[dict[str, tuple[int, ...]]] = {
    "reference": (),
    "no_lead": (18,),
    "no_half_hour": (19, 20),
    "no_level": (28,),
    "no_era": (31,),
}
"""Columns of `model_module.design` zeroed on top of the error columns."""


def score(*, q: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Twice the mean pinball loss over the 11 quantiles, per row."""
    diff = y[:, None] - q
    return 2 * np.maximum(LEVELS * diff, (LEVELS - 1) * diff).mean(axis=1)


def interval(
    *, diff: np.ndarray, months: np.ndarray, rng: np.random.Generator
) -> tuple[float, float, float]:
    """Mean difference with a 99% month-resampled percentile interval."""
    keys = np.unique(months)
    sums = np.array([diff[months == k].sum() for k in keys])
    counts = np.array([(months == k).sum() for k in keys])
    draw = rng.integers(0, len(keys), size=(N_RESAMPLES, len(keys)))
    est = sums[draw].sum(axis=1) / counts[draw].sum(axis=1)
    return float(diff.mean()), float(np.percentile(est, 0.5)), float(np.percentile(est, 99.5))


def site_labels() -> dict[int, str]:
    """Series id to D-label, from experiment 155's stored files."""
    out = {}
    for path in sorted(data_module.SRC_DIR.glob("d_scored_D*.parquet")):
        if "spread" in path.name:
            continue
        sid = int(pl.read_parquet(path, columns=["time_series_id"])["time_series_id"][0])
        out[sid] = path.stem.split("_")[-1]
    return out


def load() -> tuple[list[dict[str, np.ndarray]], list[str], list[tuple[np.ndarray, ...]]]:
    """Features per substation (series-id order), labels, and train/valid/scored masks."""
    labels = site_labels()
    ids = sorted(labels)
    feats = []
    for i in ids:
        site = data_module.load_site(time_series_id=i)
        feats.append(data_module.build_features(site=site, error_grid=site.error_grid))
    masks = []
    for f in feats:
        init = f["init"]
        valid = (init >= VALID[0]) & (init < VALID[1])
        purge = (init >= VALID[0] - PURGE) & (init < VALID[1] + PURGE)
        scored = (
            (init >= SCORED_START)
            & (f["lead"] >= data_module.SCORE_LEAD_MIN)
            & (f["lead"] <= data_module.SCORE_LEAD_MAX)
            & ~f["switching"]
        )
        masks.append(((init < TRAIN_END) & ~purge, valid, scored))
    return feats, [labels[i] for i in ids], masks


def stack(*, arrays: list[np.ndarray], masks: list[np.ndarray]) -> np.ndarray:
    """Concatenate the masked rows of every substation."""
    return np.concatenate([a[m] for a, m in zip(arrays, masks, strict=True)])


def run_nets() -> None:
    """Retrain the reference network and each ablation."""
    torch.set_num_threads(4)
    ARMS_DIR.mkdir(parents=True, exist_ok=True)
    feats, _, masks = load()
    x = [model_module.design(f=f) for f in feats]
    sites = [np.full(len(f["target"]), k) for k, f in enumerate(feats)]

    def data(which: int, zero: tuple[int, ...]) -> model_module.Data:
        xs = [m.copy() for m in x]
        for m in xs:
            m[:, list(ERROR_COLUMNS) + list(zero)] = 0.0
        sel = [mk[which] for mk in masks]
        return model_module.Data(
            x=stack(arrays=xs, masks=sel),
            site=stack(arrays=sites, masks=sel),
            y=stack(arrays=[f["target"] for f in feats], masks=sel),
            offsets=stack(arrays=[f["q_offsets"] for f in feats], masks=sel),
        )

    actual = stack(arrays=[f["actual"] for f in feats], masks=[m[2] for m in masks]).astype(
        np.float64
    )
    q50 = stack(arrays=[f["q50"] for f in feats], masks=[m[2] for m in masks]).astype(np.float64)
    for name, zero in ABLATIONS.items():
        if (ARMS_DIR / f"{name}.npy").exists():
            continue
        start = time.time()
        train, valid, scored = data(0, zero), data(1, zero), data(2, zero)
        net = model_module.fit(train=train, valid=valid, n_sites=len(feats), seed=0, device="cpu")
        c, s = model_module.predict(net=net, x=scored.x, site=scored.site, device="cpu")
        q = (q50 + c)[:, None] + np.exp(s)[:, None] * scored.offsets.astype(np.float64)
        np.save(ARMS_DIR / f"{name}.npy", score(q=q, y=actual))
        print(
            f"{name}: mean score {score(q=q, y=actual).mean():.4f} in {time.time() - start:.0f} s",
            flush=True,
        )


def run_xgb() -> None:
    """Fit XGBoost with the pinball loss on the network's inputs."""
    import xgboost as xgb

    ARMS_DIR.mkdir(parents=True, exist_ok=True)
    feats, _, masks = load()

    def matrix(which: int) -> tuple[np.ndarray, np.ndarray]:
        sel = [m[which] for m in masks]
        cols = []
        for k, f in enumerate(feats):
            cols.append(
                np.column_stack(
                    [
                        np.full(len(f["target"]), k),
                        f["lead"],
                        f["hod"],
                        f["dow"],
                        f["mean"],
                        f["era_ens"],
                        f["q_offsets"][:, [1, 3, 7, 9]],
                    ]
                )
            )
        return stack(arrays=cols, masks=sel), stack(arrays=[f["target"] for f in feats], masks=sel)

    (xt, yt), (xv, yv), (xs, _) = matrix(0), matrix(1), matrix(2)
    params = {
        "objective": "reg:quantileerror",
        "quantile_alpha": LEVELS,
        "tree_method": "hist",
        "max_depth": 6,
        "eta": 0.1,
        "min_child_weight": 50,
        "colsample_bytree": 1.0,
        "colsample_bylevel": 1.0,
        "colsample_bynode": 1.0,
        "subsample": 1.0,
        "nthread": 4,
        "seed": 0,
    }
    start = time.time()
    booster = xgb.train(
        params,
        xgb.DMatrix(xt, label=yt),
        num_boost_round=800,
        evals=[(xgb.DMatrix(xv, label=yv), "valid")],
        early_stopping_rounds=30,
        verbose_eval=False,
    )
    print(f"xgboost best round {booster.best_iteration} in {time.time() - start:.0f} s", flush=True)
    pred = np.sort(
        booster.predict(xgb.DMatrix(xs), iteration_range=(0, booster.best_iteration + 1)), axis=1
    )
    actual = stack(arrays=[f["actual"] for f in feats], masks=[m[2] for m in masks]).astype(
        np.float64
    )
    q50 = stack(arrays=[f["q50"] for f in feats], masks=[m[2] for m in masks]).astype(np.float64)
    q = q50[:, None] + pred.astype(np.float64)
    np.save(ARMS_DIR / "xgboost.npy", score(q=q, y=actual))
    np.save(ARMS_DIR / "xgboost_q.npy", q)
    print(f"xgboost: mean score {score(q=q, y=actual).mean():.4f}", flush=True)


def run_report() -> None:
    """Score every arm and print the contrasts."""
    feats, labels, masks = load()
    sel = [m[2] for m in masks]
    site_label = np.concatenate([np.full(m.sum(), labels[k]) for k, m in enumerate(sel)])
    months = np.concatenate(
        [f["valid"][m].astype("datetime64[M]").astype(str) for f, m in zip(feats, sel, strict=True)]
    )
    actual = stack(arrays=[f["actual"] for f in feats], masks=sel).astype(np.float64)
    quant = stack(arrays=[f["quantiles"] for f in feats], masks=sel).astype(np.float64)
    losses = {"arm_d": score(q=quant, y=actual)}
    e200 = (
        data_module.ML_EXPERIMENTS_DATA_DIR
        / "00200-dp-pooled-correction-head-on-kalman-state"
        / "work"
    )
    e209 = (
        data_module.ML_EXPERIMENTS_DATA_DIR
        / "00209-dp-pinball-table-vs-network-head-on-arm-d"
        / "work"
    )
    head_rows = pl.read_parquet(e200 / "stage3_head_rows.parquet")
    assert np.allclose(
        head_rows.filter(pl.col("arm") == "arm_d")["loss"].to_numpy(), losses["arm_d"], atol=1e-3
    )
    losses["stored_blind"] = (
        pl.read_parquet(e200 / "stage3_blind_rows.parquet")
        .filter(pl.col("arm") == "head")["loss"]
        .to_numpy()
    )
    losses["table_T0"] = pl.read_parquet(e209 / "rows.parquet")["T0"].to_numpy()
    for path in sorted(ARMS_DIR.glob("*.npy")):
        if not path.stem.endswith("_q"):
            losses[path.stem] = np.load(path)
    pl.DataFrame({"site": site_label, "month": months, **losses}).write_parquet(
        data_module.WORK_DIR / "rows.parquet"
    )
    rng = np.random.default_rng(0)
    contrasts = [
        ("xgboost", "reference"),
        ("xgboost", "stored_blind"),
        ("xgboost", "table_T0"),
        ("reference", "stored_blind"),
        ("reference", "arm_d"),
        ("xgboost", "arm_d"),
        *[(f"no_{k}", "reference") for k in ("lead", "half_hour", "level", "era")],
    ]
    report: dict[str, object] = {}
    for scope, sites in (("decision", DECISION), ("all6", tuple(labels))):
        mask = np.isin(site_label, sites)
        out = {"scores": {k: float(v[mask].mean()) for k, v in losses.items()}}
        for a, b in contrasts:
            if a not in losses or b not in losses:
                continue
            m, lo, hi = interval(diff=(losses[a] - losses[b])[mask], months=months[mask], rng=rng)
            out[f"{a} - {b}"] = [round(m, 4), round(lo, 4), round(hi, 4)]
            print(f"{scope:8s} {a} - {b}: {m:+.4f} [{lo:+.4f}, {hi:+.4f}]")
        report[scope] = out
    if "xgboost" in losses:
        mask = np.isin(site_label, DECISION)
        report["leave_one_out_xgboost_minus_reference"] = {
            s: round(
                float((losses["xgboost"] - losses["reference"])[mask & (site_label != s)].mean()), 4
            )
            for s in DECISION
        }
        q = np.load(ARMS_DIR / "xgboost_q.npy")
        d = actual[:, None] - q
        report["xgboost_tails"] = {
            "cover95": float((actual <= q[:, 9]).mean()),
            "cover99": float((actual <= q[:, 10]).mean()),
            "pin99": float((2 * np.maximum(0.99 * d[:, 10], -0.01 * d[:, 10])).mean()),
            "pin01": float((2 * np.maximum(0.01 * d[:, 0], -0.99 * d[:, 0])).mean()),
        }
    (data_module.WORK_DIR / "report.json").write_text(json.dumps(report, indent=2))


def main() -> None:
    """Dispatch on `--part`."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--part", choices=["nets", "xgb", "report"], required=True)
    part = parser.parse_args().part
    {"nets": run_nets, "xgb": run_xgb, "report": run_report}[part]()


if __name__ == "__main__":
    main()
