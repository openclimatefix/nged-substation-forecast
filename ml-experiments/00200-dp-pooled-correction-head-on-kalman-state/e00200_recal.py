# ruff: noqa: D103,E501,PLR0915,UP034
"""Experiment 200, plain recalibration arm: no network, same rows as the blind head.

Arm D's median is shifted by a ridge regression on one-hot calendar, lead-band, site, and
forecast-level terms (the same inputs the blind head sees), and each quantile offset from the
median is scaled by a factor chosen per (site, lead band, six-hour block) on the training rows. It
trains on the same rows as the head (origins before 2025-06-28 outside the validation block) and is
scored on the same decision rows. Run: `E200_SHARED=1 uv run --no-sync python e00200_recal.py`.
"""

import json
import os

import e00200_data as data_module
import e00200_score as score_module
import numpy as np
import polars as pl
from sklearn.linear_model import Ridge

ENS_ONLY = os.environ.get("E200_ENS_ONLY") == "1"
"""Fit on the ensemble-era training rows only (one year), as the stage 2 head does."""
SUFFIX = "_ensonly" if ENS_ONLY else ""
ALPHAS = (1.0, 10.0, 100.0, 1000.0)
GRID = np.linspace(0.6, 1.6, 21)
LEAD_BANDS = np.array([12.0, 18.0, 24.0, 30.0, 36.0, 42.0, 48.0001])


def codes(*, f: dict[str, np.ndarray], site: int, level_edges: np.ndarray) -> dict[str, np.ndarray]:
    """Integer cell codes for one row set."""
    lead_band = np.clip(np.searchsorted(LEAD_BANDS, f["lead"], side="right") - 1, 0, 5)
    level_bin = np.searchsorted(level_edges, f["mean"])
    n = len(f["lead"])
    return {
        "site_hod": np.full(n, site) * 48 + f["hod"],
        "site_dow": np.full(n, site) * 7 + f["dow"],
        "site_lead": np.full(n, site) * 6 + lead_band,
        "site_level": np.full(n, site) * 12 + level_bin,
        "hod_dow": f["hod"] * 7 + f["dow"],
        "lead_band": lead_band,
        "block": f["hod"] // 8,
        "era": f["era_ens"],
    }


SIZES = {
    "site_hod": 6 * 48,
    "site_dow": 6 * 7,
    "site_lead": 36,
    "site_level": 6 * 12,
    "hod_dow": 336,
}


def one_hot(*, c: dict[str, np.ndarray]) -> np.ndarray:
    """Stack the indicator columns of the additive table."""
    cols = []
    for name, size in SIZES.items():
        m = np.zeros((len(c[name]), size), dtype=np.float32)
        m[np.arange(len(c[name])), c[name]] = 1.0
        cols.append(m)
    cols.append(c["era"][:, None].astype(np.float32))
    return np.concatenate(cols, axis=1)


def main() -> None:
    labels = score_module.site_labels()
    ids = sorted(labels)
    sites = [data_module.load_site(time_series_id=i) for i in ids]
    feats = [data_module.build_features(site=s, error_grid=s.error_grid) for s in sites]
    edges = [np.quantile(f["mean"], np.linspace(0, 1, 13)[1:-1]) for f in feats]
    train_x, train_y, valid_x, valid_y = [], [], [], []
    train_cells = []
    for k, f in enumerate(feats):
        init = f["init"]
        valid = (init >= score_module.VALID[0]) & (init < score_module.VALID[1])
        purge = (init >= score_module.VALID[0] - score_module.PURGE) & (
            init < score_module.VALID[1] + score_module.PURGE
        )
        train = (init < score_module.TRAIN_END) & ~purge
        if ENS_ONLY:
            train &= f["era_ens"] == 1
        c = codes(f=f, site=k, level_edges=edges[k])
        x = one_hot(c=c)
        train_x.append(x[train])
        train_y.append(f["target"][train])
        valid_x.append(x[valid])
        valid_y.append(f["target"][valid])
        train_cells.append((c, train))
    tx, ty = np.concatenate(train_x), np.concatenate(train_y)
    vx, vy = np.concatenate(valid_x), np.concatenate(valid_y)
    best = min(
        ((float(np.abs(vy - Ridge(alpha=a).fit(tx, ty).predict(vx)).mean()), a) for a in ALPHAS)
    )
    print(
        f"train rows {len(ty)}, ridge alpha {best[1]} (validation mean absolute error {best[0]:.3f})"
    )
    ridge = Ridge(alpha=best[1]).fit(tx, ty)

    # Scale factor per (site, lead band, block) from the training rows.
    scale = np.ones((6, 6, 6))
    for k, f in enumerate(feats):
        c, train = train_cells[k]
        centre = ridge.predict(one_hot(c=c)[train])
        y = f["target"][train]
        off = f["q_offsets"][train]
        for lb in range(6):
            for blk in range(6):
                m = (
                    (c["lead_band"][train] == lb)
                    & (c["block"][train] == blk)
                    & (c["era"][train] == 1)
                )
                if m.sum() < 50:
                    continue
                losses = []
                for g in GRID:
                    q = centre[m, None] + g * off[m]
                    d = y[m, None] - q
                    losses.append(
                        np.maximum(score_module.LEVELS * d, (score_module.LEVELS - 1) * d).mean()
                    )
                scale[k, lb, blk] = GRID[int(np.argmin(losses))]

    rows = []
    for k, f in enumerate(feats):
        scored = (
            (f["init"] >= score_module.SCORED_START)
            & (f["lead"] >= data_module.SCORE_LEAD_MIN)
            & (f["lead"] <= data_module.SCORE_LEAD_MAX)
            & ~f["switching"]
        )
        c = codes(f=f, site=k, level_edges=edges[k])
        centre = ridge.predict(one_hot(c=c)[scored])
        g = scale[k][c["lead_band"][scored], c["block"][scored]]
        q = (f["q50"][scored] + centre)[:, None] + g[:, None] * f["q_offsets"][scored]
        y = f["actual"][scored].astype(np.float64)
        rows.append(
            pl.DataFrame(
                {
                    "site": labels[ids[k]],
                    "loss": score_module.score(q=q.astype(np.float64), y=y),
                    "actual": y,
                }
            )
        )
    df = pl.concat(rows)
    df.write_parquet(data_module.WORK_DIR / f"stage{data_module.TAG}_recal{SUFFIX}_rows.parquet")

    head = pl.read_parquet(data_module.WORK_DIR / f"stage{data_module.TAG}_head_rows.parquet")
    blind = pl.read_parquet(data_module.WORK_DIR / f"stage{data_module.TAG}_blind_rows.parquet")
    arms = {
        name: x.filter(pl.col("arm") == a)
        for name, x, a in (
            ("arm_d", head, "arm_d"),
            ("head", head, "head"),
            ("blind", blind, "head"),
        )
    }
    arms["recal"] = df
    rng = np.random.default_rng(0)
    out = {}
    decision = ["D5", "D10", "D11", "D14", "D16"]
    for scope, sel in (("decision", decision), ("all6", sorted(labels.values()))):
        sub = {n: x.filter(pl.col("site").is_in(sel)) for n, x in arms.items()}
        assert len({len(x) for x in sub.values()}) == 1
        months = head.filter((pl.col("arm") == "arm_d") & pl.col("site").is_in(sel))[
            "month"
        ].to_numpy()
        for name, a, b in (
            ("recal - arm D", "recal", "arm_d"),
            ("recal - blind", "recal", "blind"),
            ("head - recal", "head", "recal"),
        ):
            diff = sub[a]["loss"].to_numpy() - sub[b]["loss"].to_numpy()
            m, lo, hi = score_module.interval(diff=diff, months=months, level=0.99, rng=rng)
            out[f"{scope}: {name}"] = [round(m, 4), round(lo, 4), round(hi, 4)]
            print(f"{scope:8s} {name:14s} {m:+.4f} [{lo:+.4f}, {hi:+.4f}]")
        print(scope, "recal score", float(sub["recal"]["loss"].mean()))  # ty: ignore[invalid-argument-type]
    (data_module.WORK_DIR / f"stage{data_module.TAG}_recal{SUFFIX}.json").write_text(
        json.dumps(out, indent=2)
    )


if __name__ == "__main__":
    main()
