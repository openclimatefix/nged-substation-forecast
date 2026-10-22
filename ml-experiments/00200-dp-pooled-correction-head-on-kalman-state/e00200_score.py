# ruff: noqa: D103,PLR0915
"""Experiment 200, stage 2: the head against arm D on the scored year, 6 stored substations.

Trains on runs before 2025-06-28 (early stopping on 2025-04-28 to 2025-06-23, purged by 15 days),
3 seeds, averages the quantile forecasts, and scores leads 15.5 to 39 h of the runs from 2025-07-02
outside switching events. The head's correction is added to arm D's median and its spread scaled.
Run: `uv run --no-sync python e00200_score.py`.
"""

import json
import sys
from pathlib import Path

import e00200_data as data_module
import e00200_model as model_module
import numpy as np
import polars as pl

SEEDS = (0, 1, 2)
TRAIN_END = np.datetime64("2025-06-28", "us")
VALID = (np.datetime64("2025-04-28", "us"), np.datetime64("2025-06-24", "us"))
SCORED_START = np.datetime64("2025-07-02", "us")
PURGE = np.timedelta64(15, "D")
LEVELS = np.array([0.01, 0.05, 0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90, 0.95, 0.99])
N_RESAMPLES = 20_000


def site_labels() -> dict[int, str]:
    out = {}
    for path in sorted(data_module.SRC_DIR.glob("d_scored_D*.parquet")):
        if "spread" in path.name:
            continue
        out[int(pl.read_parquet(path, columns=["time_series_id"])["time_series_id"][0])] = (
            path.stem.split("_")[-1]
        )
    return out


def score(*, q: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Twice the mean pinball loss over the 11 quantiles, per row."""
    diff = y[:, None] - q
    return 2 * np.maximum(LEVELS * diff, (LEVELS - 1) * diff).mean(axis=1)


def interval(
    *, diff: np.ndarray, months: np.ndarray, level: float, rng: np.random.Generator
) -> tuple[float, float, float]:
    """Mean difference with a month-resampled percentile interval."""
    keys = np.unique(months)
    sums = np.array([diff[months == k].sum() for k in keys])
    counts = np.array([(months == k).sum() for k in keys])
    draw = rng.integers(0, len(keys), size=(N_RESAMPLES, len(keys)))
    est = sums[draw].sum(axis=1) / counts[draw].sum(axis=1)
    tail = (1 - level) / 2 * 100
    return (
        float(diff.mean()),
        float(np.percentile(est, tail)),
        float(np.percentile(est, 100 - tail)),
    )


def main() -> None:
    labels = site_labels()
    ids = sorted(labels)
    sites = [data_module.load_site(time_series_id=i) for i in ids]
    feats = [data_module.build_features(site=s, error_grid=s.error_grid) for s in sites]
    blind = "--blind" in sys.argv
    tag = "3" if data_module.SHARED else "2"
    x = [model_module.design(f=f) for f in feats]
    if blind:
        for matrix in x:
            matrix[:, :18] = 0.0  # every column built from past errors
            matrix[:, -3:-1] = 0.0  # the Kalman level and anomaly
    parts = {"train": [], "valid": []}
    for k, f in enumerate(feats):
        init = f["init"]
        train = (init < TRAIN_END) & ~((init >= VALID[0] - PURGE) & (init < VALID[1] + PURGE))
        # A valid time inside the validation block would leak its error into the training inputs.
        valid = (init >= VALID[0]) & (init < VALID[1])
        for name, m in (("train", train), ("valid", valid)):
            parts[name].append(
                model_module.Data(
                    x=x[k][m], site=np.full(m.sum(), k), y=f["target"][m], offsets=f["q_offsets"][m]
                )
            )
    cat = lambda ps: model_module.Data(  # noqa: E731
        x=np.concatenate([p.x for p in ps]),
        site=np.concatenate([p.site for p in ps]),
        y=np.concatenate([p.y for p in ps]),
        offsets=np.concatenate([p.offsets for p in ps]),
    )
    train, valid = cat(parts["train"]), cat(parts["valid"])
    print(f"train rows {len(train.y)}, validation rows {len(valid.y)}")
    nets = [
        model_module.fit(train=train, valid=valid, n_sites=len(ids), seed=s, device="cuda")
        for s in SEEDS
    ]

    records = []
    for k, f in enumerate(feats):
        scored = (
            (f["init"] >= SCORED_START)
            & (f["lead"] >= data_module.SCORE_LEAD_MIN)
            & (f["lead"] <= data_module.SCORE_LEAD_MAX)
            & ~f["switching"]
        )
        corr = np.zeros(scored.sum())
        scale = np.zeros(scored.sum())
        for net in nets:
            c, s = model_module.predict(
                net=net, x=x[k][scored], site=np.full(scored.sum(), k), device="cuda"
            )
            corr += c / len(nets)
            scale += s / len(nets)
        off = f["q_offsets"][scored]
        head_q = (f["q50"][scored] + corr)[:, None] + np.exp(scale)[:, None] * off
        base_q = f["quantiles"][scored]
        y = f["actual"][scored].astype(np.float64)
        head_q = head_q.astype(np.float64)
        base_q = base_q.astype(np.float64)
        month = f["valid"][scored].astype("datetime64[M]").astype(str)
        for name, q in (("arm_d", base_q), ("head", head_q)):
            records.append(
                pl.DataFrame(
                    {
                        "site": labels[ids[k]],
                        "month": month,
                        "arm": name,
                        "loss": score(q=q, y=y),
                        "abs_err": np.abs(y - q[:, 5]),
                        "cover95": (y <= q[:, 9]).astype(float),
                        "cover99": (y <= q[:, 10]).astype(float),
                        "cover05": (y <= q[:, 1]).astype(float),
                        "cover01": (y <= q[:, 0]).astype(float),
                        "pin99": 2 * np.maximum(0.99 * (y - q[:, 10]), -0.01 * (y - q[:, 10])),
                        "pin01": 2 * np.maximum(0.01 * (y - q[:, 0]), -0.99 * (y - q[:, 0])),
                        "med_minus_mean": q[:, 5] - f["mean"][scored].astype(np.float64),
                    }
                )
            )
    df = pl.concat(records)
    df.write_parquet(
        data_module.WORK_DIR / f"stage{tag}_{'blind' if blind else 'head'}_rows.parquet"
    )
    rng = np.random.default_rng(0)
    report: dict[str, object] = {"train_rows": len(train.y)}
    decision = ["D5", "D10", "D11", "D14", "D16"]
    for scope, sel in (("decision", decision), ("all6", sorted(labels.values()))):
        d = df.filter(pl.col("site").is_in(sel))
        a = d.filter(pl.col("arm") == "arm_d")
        h = d.filter(pl.col("arm") == "head")
        diff = h["loss"].to_numpy() - a["loss"].to_numpy()
        months = a["month"].to_numpy()
        m, lo, hi = interval(diff=diff, months=months, level=0.99, rng=rng)
        report[scope] = {
            "arm_d_score": float(np.mean(a["loss"].to_numpy())),
            "head_score": float(np.mean(h["loss"].to_numpy())),
            "diff": m,
            "lo99": lo,
            "hi99": hi,
            "mae_arm_d": float(np.mean(a["abs_err"].to_numpy())),
            "mae_head": float(np.mean(h["abs_err"].to_numpy())),
            "cover95_d": float(np.mean(a["cover95"].to_numpy())),
            "cover95_h": float(np.mean(h["cover95"].to_numpy())),
            "cover99_d": float(np.mean(a["cover99"].to_numpy())),
            "cover99_h": float(np.mean(h["cover99"].to_numpy())),
            "pin99_d": float(np.mean(a["pin99"].to_numpy())),
            "pin99_h": float(np.mean(h["pin99"].to_numpy())),
            "pin01_d": float(np.mean(a["pin01"].to_numpy())),
            "pin01_h": float(np.mean(h["pin01"].to_numpy())),
        }
    per = {}
    for site in sorted(labels.values()):
        a = df.filter((pl.col("site") == site) & (pl.col("arm") == "arm_d"))["loss"].to_numpy()
        h = df.filter((pl.col("site") == site) & (pl.col("arm") == "head"))["loss"].to_numpy()
        per[site] = float((h - a).mean())
    report["per_site_diff"] = per
    report["median_minus_mean"] = {
        s: float(
            df.filter((pl.col("site") == s) & (pl.col("arm") == "head"))["med_minus_mean"].mean()  # ty: ignore[invalid-argument-type]
        )
        for s in labels.values()
    }
    Path(data_module.WORK_DIR / f"stage{tag}_{'blind' if blind else 'head'}.json").write_text(
        json.dumps(report, indent=2)
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
