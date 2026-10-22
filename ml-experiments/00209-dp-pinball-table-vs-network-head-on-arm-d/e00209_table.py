# ruff: noqa: D103,D107,E501,PLR0915
"""Experiment 209: a pinball-loss table against experiment 200's network head, on arm D's rows.

The table is a sum of embedding terms (no hidden layers) fitted by gradient descent on the pinball
loss over 11 quantiles. Arms: T0 (site x half-hour, site x weekday, site x lead band, site x
forecast level x era, half-hour x weekday, and the spread by site x lead band x six-hour block), T1
(T0 plus a temperature-extremes term), T2 (T0 without the forecast level x era term), T3 (T0 with
the calendar terms fitted separately for each weather era) and T4 (T0 plus forecast level by
six-hour block by era and forecast level by weekday by lead band terms), both post hoc arms.
Run: `uv run --no-sync python e00209_table.py` (CPU, 4 threads).
"""

import json
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from studies.sources import ML_EXPERIMENTS_DATA_DIR

EXPERIMENT_ID: Final[int] = 209
SLUG: Final[str] = "dp-pinball-table-vs-network-head-on-arm-d"
WORK_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"{EXPERIMENT_ID:05d}-{SLUG}" / "work"
FEATURES: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00201-dp-gam-residual-mlp" / "work" / "features"
E200_WORK: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00200-dp-pooled-correction-head-on-kalman-state" / "work"
)
SITES: Final[tuple[str, ...]] = (
    "D9",
    "D11",
    "D10",
    "D14",
    "D5",
    "D16",
)  # by series id, as experiment 200
DECISION: Final[tuple[str, ...]] = ("D5", "D10", "D11", "D14", "D16")
QUANTILES: Final[tuple[str, ...]] = (
    "q01",
    "q05",
    "q10",
    "q25",
    "q40",
    "q50",
    "q60",
    "q75",
    "q90",
    "q95",
    "q99",
)
LEVELS: Final[np.ndarray] = np.array(
    [0.01, 0.05, 0.10, 0.25, 0.40, 0.50, 0.60, 0.75, 0.90, 0.95, 0.99]
)
TRAIN_END: Final[np.datetime64] = np.datetime64("2025-06-28", "us")
VALID: Final[tuple[np.datetime64, np.datetime64]] = (
    np.datetime64("2025-04-28", "us"),
    np.datetime64("2025-06-24", "us"),
)
PURGE: Final[np.timedelta64] = np.timedelta64(15, "D")
SCORED_START: Final[np.datetime64] = np.datetime64("2025-07-02", "us")
LEAD_TRAIN: Final[tuple[float, float]] = (12.0, 48.0)
LEAD_SCORE: Final[tuple[float, float]] = (15.5, 39.0)
LEAD_EDGES: Final[np.ndarray] = np.array([18.0, 24.0, 30.0, 36.0, 42.0])
N_LEVEL: Final[int] = 12
SEEDS: Final[tuple[int, ...]] = (0, 1, 2)
COLD_C: Final[float] = -3.0
HOT_C: Final[float] = 24.0
N_RESAMPLES: Final[int] = 20_000
ERA_SPLIT: Final[tuple[str, ...]] = ("site_hod", "site_dow", "site_lead", "hod_dow")
"""Terms that arm T3 fits separately for each weather era."""
ARMS: Final[tuple[str, ...]] = ("T0", "T1", "T2", "T3", "T4")


def load_site(*, site: str, site_index: int) -> dict[str, np.ndarray]:
    """Rows of one substation: sorted by origin and valid time, one seed, with cell codes."""
    frame = (
        pl.read_parquet(FEATURES / f"{site}.parquet")
        .filter(pl.col("seed") == 0)
        .unique(subset=["power_fcst_init_time", "valid_time"])
        .sort("power_fcst_init_time", "valid_time")
    )
    valid = frame["valid_time"].dt.replace_time_zone(None).to_numpy().astype("datetime64[us]")
    init = (
        frame["power_fcst_init_time"].dt.replace_time_zone(None).to_numpy().astype("datetime64[us]")
    )
    day = valid.astype("datetime64[D]")
    hod = ((valid - day) // np.timedelta64(30, "m")).astype(np.int64)
    dow = ((day.astype(np.int64) + 3) % 7).astype(np.int64)
    lead = frame["lead_hours"].to_numpy()
    mean = frame["mean"].to_numpy().astype(np.float64)
    edges = np.quantile(mean[init < TRAIN_END], np.linspace(0, 1, N_LEVEL + 1)[1:-1])
    quant = frame.select(QUANTILES).to_numpy().astype(np.float64)
    temp = frame["drv_temp_c"].to_numpy().astype(np.float64)
    return {
        "site": np.full(len(frame), site_index),
        "hod": hod,
        "dow": dow,
        "lead_band": np.searchsorted(LEAD_EDGES, lead, side="right"),
        "block": hod // 8,
        "level_bin": np.searchsorted(edges, mean),
        "era": (frame["era"] == "ens").to_numpy().astype(np.int64),
        "cold": np.clip(COLD_C - np.nan_to_num(temp, nan=COLD_C), 0, None) / 10.0,
        "hot": np.clip(np.nan_to_num(temp, nan=HOT_C) - HOT_C, 0, None) / 10.0,
        "y": frame["actual"].to_numpy().astype(np.float64) - quant[:, 5],
        "actual": frame["actual"].to_numpy().astype(np.float64),
        "q50": quant[:, 5],
        "offsets": quant - quant[:, [5]],
        "quant": quant,
        "lead": lead,
        "init": init,
        "valid": valid,
        "switching": frame["switching_mask"].to_numpy(),
    }


class Table(torch.nn.Module):
    """Sum of embedding terms for the median shift and the log-scale; no hidden layers."""

    def __init__(self, *, arm: str) -> None:
        super().__init__()
        self.arm = arm
        sizes = {
            "site_hod": 6 * 48,
            "site_dow": 6 * 7,
            "site_lead": 6 * 6,
            "hod_dow": 48 * 7,
            "site_level_era": 6 * N_LEVEL * 2,
            "era": 2,
        }
        if arm == "T2":
            sizes.pop("site_level_era")
        if arm == "T4":
            sizes["site_level_block_era"] = 6 * N_LEVEL * 6 * 2
            sizes["level_dow_lead"] = N_LEVEL * 7 * 6
        if arm == "T3":
            sizes = {k: v * 2 if k in ERA_SPLIT else v for k, v in sizes.items()}
        self.terms = torch.nn.ModuleDict({k: torch.nn.Embedding(v, 1) for k, v in sizes.items()})
        for emb in self.terms.values():
            torch.nn.init.zeros_(emb.weight)  # ty: ignore[invalid-argument-type]
        self.scale = torch.nn.Embedding(6 * 6 * 6, 1)
        torch.nn.init.zeros_(self.scale.weight)
        self.temperature = torch.nn.Parameter(torch.zeros(6, 2))

    def codes(self, *, b: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Integer cell code of every term."""
        out = {
            "site_hod": b["site"] * 48 + b["hod"],
            "site_dow": b["site"] * 7 + b["dow"],
            "site_lead": b["site"] * 6 + b["lead_band"],
            "hod_dow": b["hod"] * 7 + b["dow"],
            "site_level_era": (b["site"] * N_LEVEL + b["level_bin"]) * 2 + b["era"],
            "era": b["era"],
            "site_level_block_era": ((b["site"] * N_LEVEL + b["level_bin"]) * 6 + b["block"]) * 2
            + b["era"],
            "level_dow_lead": (b["level_bin"] * 7 + b["dow"]) * 6 + b["lead_band"],
        }
        if self.arm == "T3":
            out = {k: (v * 2 + b["era"] if k in ERA_SPLIT else v) for k, v in out.items()}
        return {k: v for k, v in out.items() if k in self.terms}

    def forward(self, *, b: dict[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        """Median shift and log-scale of a batch."""
        shift = sum(self.terms[k](c)[:, 0] for k, c in self.codes(b=b).items())
        if self.arm == "T1":
            shift = (
                shift
                + self.temperature[b["site"], 0] * b["cold"]
                + self.temperature[b["site"], 1] * b["hot"]
            )
        scale_code = (b["site"] * 6 + b["lead_band"]) * 6 + b["block"]
        log_scale = 0.7 * torch.tanh(self.scale(scale_code)[:, 0])
        return shift, log_scale


def tensors(
    *, rows: list[dict[str, np.ndarray]], masks: list[np.ndarray]
) -> dict[str, torch.Tensor]:
    """Concatenate the masked rows of every site into tensors."""
    out: dict[str, torch.Tensor] = {}
    for key in ("site", "hod", "dow", "lead_band", "block", "level_bin", "era"):
        out[key] = torch.from_numpy(
            np.concatenate([r[key][m] for r, m in zip(rows, masks, strict=True)])
        )
    for key in ("cold", "hot", "y"):
        out[key] = torch.from_numpy(
            np.concatenate([r[key][m] for r, m in zip(rows, masks, strict=True)]).astype(np.float32)
        )
    out["offsets"] = torch.from_numpy(
        np.concatenate([r["offsets"][m] for r, m in zip(rows, masks, strict=True)]).astype(
            np.float32
        )
    )
    return out


def pinball(
    *, y: torch.Tensor, shift: torch.Tensor, log_scale: torch.Tensor, offsets: torch.Tensor
) -> torch.Tensor:
    """Mean pinball loss over the 11 quantiles."""
    q = shift[:, None] + torch.exp(log_scale)[:, None] * offsets
    diff = y[:, None] - q
    lv = torch.from_numpy(LEVELS.astype(np.float32))
    return torch.maximum(lv * diff, (lv - 1) * diff).mean()


def fit(
    *, arm: str, train: dict[str, torch.Tensor], valid: dict[str, torch.Tensor], seed: int
) -> Table:
    """Train one table with early stopping on the validation block."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    net = Table(arm=arm)
    opt = torch.optim.Adam(net.parameters(), lr=0.02)
    n = len(train["y"])
    best, stale, state = float("inf"), 0, None
    for _ in range(60):
        net.train()
        order = torch.from_numpy(rng.permutation(n))
        for start in range(0, n, 8192):
            idx = order[start : start + 8192]
            b = {k: v[idx] for k, v in train.items()}
            shift, ls = net(b=b)
            loss = pinball(y=b["y"], shift=shift, log_scale=ls, offsets=b["offsets"])
            opt.zero_grad()
            loss.backward()
            opt.step()
        net.eval()
        with torch.no_grad():
            shift, ls = net(b=valid)
            val = float(pinball(y=valid["y"], shift=shift, log_scale=ls, offsets=valid["offsets"]))
        if val < best - 1e-5:
            best, stale = val, 0
            state = {k: v.clone() for k, v in net.state_dict().items()}
        else:
            stale += 1
            if stale >= 6:
                break
    assert state is not None
    net.load_state_dict(state)
    return net.eval()


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


def main() -> None:
    torch.set_num_threads(4)
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    rows = [load_site(site=s, site_index=i) for i, s in enumerate(SITES)]
    train_masks, valid_masks, score_masks = [], [], []
    for r in rows:
        init = r["init"]
        in_valid = (init >= VALID[0]) & (init < VALID[1])
        purge = (init >= VALID[0] - PURGE) & (init < VALID[1] + PURGE)
        lead_ok = (r["lead"] >= LEAD_TRAIN[0]) & (r["lead"] <= LEAD_TRAIN[1])
        train_masks.append((init < TRAIN_END) & ~purge & lead_ok)
        valid_masks.append(in_valid & lead_ok)
        score_masks.append(
            (init >= SCORED_START)
            & (r["lead"] >= LEAD_SCORE[0])
            & (r["lead"] <= LEAD_SCORE[1])
            & ~r["switching"]
        )
    train = tensors(rows=rows, masks=train_masks)
    valid = tensors(rows=rows, masks=valid_masks)
    scored = tensors(rows=rows, masks=score_masks)
    print(
        f"train rows {len(train['y'])}, validation rows {len(valid['y'])}, scored rows {len(scored['y'])}"
    )

    actual = np.concatenate([r["actual"][m] for r, m in zip(rows, score_masks, strict=True)])
    q50 = np.concatenate([r["q50"][m] for r, m in zip(rows, score_masks, strict=True)])
    quant = np.concatenate([r["quant"][m] for r, m in zip(rows, score_masks, strict=True)])
    site_label = np.concatenate([np.full(m.sum(), SITES[i]) for i, m in enumerate(score_masks)])
    valid_month = np.concatenate(
        [
            r["valid"][m].astype("datetime64[M]").astype(str)
            for r, m in zip(rows, score_masks, strict=True)
        ]
    )

    losses = {"arm_d": score(q=quant, y=actual)}
    tails = {}
    for arm in ARMS:
        shifts, scales = [], []
        for seed in SEEDS:
            net = fit(arm=arm, train=train, valid=valid, seed=seed)
            with torch.no_grad():
                s, ls = net(b=scored)
            shifts.append(s.numpy())
            scales.append(ls.numpy())
            print(f"{arm} seed {seed} fitted")
        shift, ls = np.mean(shifts, axis=0), np.mean(scales, axis=0)
        offsets = scored["offsets"].numpy().astype(np.float64)
        q = (q50 + shift)[:, None] + np.exp(ls)[:, None] * offsets
        losses[arm] = score(q=q, y=actual)
        d = actual[:, None] - q
        tails[arm] = {
            "cover95": float((actual <= q[:, 9]).mean()),
            "cover99": float((actual <= q[:, 10]).mean()),
            "cover05": float((actual <= q[:, 1]).mean()),
            "cover01": float((actual <= q[:, 0]).mean()),
            "pin99": float((2 * np.maximum(0.99 * d[:, 10], -0.01 * d[:, 10])).mean()),
            "pin01": float((2 * np.maximum(0.01 * d[:, 0], -0.99 * d[:, 0])).mean()),
        }
    for name, arm in (("blind", "blind"), ("head", "head")):
        stored = pl.read_parquet(E200_WORK / f"stage3_{name}_rows.parquet")
        if name == "head":
            reference = stored.filter(pl.col("arm") == "arm_d")["loss"].to_numpy()
            assert np.allclose(reference, losses["arm_d"], atol=1e-3), (
                "rows do not align with experiment 200"
            )
        losses[arm] = stored.filter(pl.col("arm") == "head")["loss"].to_numpy()
        assert len(losses[arm]) == len(actual)

    pl.DataFrame({"site": site_label, "month": valid_month, **losses}).write_parquet(
        WORK_DIR / "rows.parquet"
    )
    rng = np.random.default_rng(0)
    contrasts = (
        ("T0 - blind network", "T0", "blind"),
        ("T0 - head", "T0", "head"),
        ("T1 - T0", "T1", "T0"),
        ("T3 - blind network", "T3", "blind"),
        ("T3 - T0", "T3", "T0"),
        ("T4 - blind network", "T4", "blind"),
        ("T4 - T0", "T4", "T0"),
        ("T0 - arm D", "T0", "arm_d"),
        ("T2 - T0", "T2", "T0"),
        ("T1 - blind network", "T1", "blind"),
        ("blind network - arm D", "blind", "arm_d"),
        ("head - arm D", "head", "arm_d"),
    )
    report: dict[str, object] = {"tails": tails}
    for scope, sel in (("decision", DECISION), ("all6", SITES)):
        mask = np.isin(site_label, sel)
        out = {"scores": {k: float(v[mask].mean()) for k, v in losses.items()}}
        for label, a, b in contrasts:
            m, lo, hi = interval(
                diff=(losses[a] - losses[b])[mask], months=valid_month[mask], rng=rng
            )
            out[label] = [round(m, 4), round(lo, 4), round(hi, 4)]
            print(f"{scope:8s} {label:22s} {m:+.4f} [{lo:+.4f}, {hi:+.4f}]")
        report[scope] = out
    mask = np.isin(site_label, DECISION)
    report["leave_one_out_T0_minus_blind"] = {
        s: round(float((losses["T0"] - losses["blind"])[mask & (site_label != s)].mean()), 4)
        for s in DECISION
    }
    report["per_site_T0_minus_arm_d"] = {
        s: round(float((losses["T0"] - losses["arm_d"])[site_label == s].mean()), 4) for s in SITES
    }
    (WORK_DIR / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report["leave_one_out_T0_minus_blind"]), json.dumps(tails["T0"]))


if __name__ == "__main__":
    main()
