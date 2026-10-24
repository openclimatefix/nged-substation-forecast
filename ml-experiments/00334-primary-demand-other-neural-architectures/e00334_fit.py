# ruff: noqa: E501
"""Fit one network per (arm, refit month, seed) and write the test months' calibrated quantiles.

Arms: `m1` and `m2` (mixture-density MLP, rows), `r` (GRU encoder-decoder, runs), `t` (compact
temporal fusion transformer, runs). Every arm trains on the kept half-hours of the 24-month pool,
stops early on the held-out pool months, then gets the as-of calibration of experiment 164: per
temperature class and tail level, multipliers on the offsets from the median, fitted on the held-out
pool months. Output: `work/fits/<arm>_cut<YYYY-MM>_seed<s>.parquet` (calibrated `q01`..`q99` and raw
`raw_q01`..`raw_q99`, in points).

    data/ml-experiments/00334-*/run.sh ml-experiments/00334-*/e00334_fit.py --arms m1 m2 \
        --cuts 2025-07 ... --workers 10
"""

import argparse
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor
from typing import Any, Final, cast

import e00164_common as c164  # ty: ignore[unresolved-import]
import e00164_fit as f164  # ty: ignore[unresolved-import]
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]

import e00334_common as common  # isort: skip
import e00334_data as data_mod  # isort: skip
import e00334_models as models  # isort: skip

MLP_BATCH: Final[int] = 16384
MLP_LR: Final[float] = 5e-4
SEQ_BATCH: Final[int] = 32
WEIGHT_DECAY: Final[float] = 1e-3
MAX_EPOCHS: Final[int] = 40
PATIENCE: Final[int] = 6
INPUT_CLIP_SD: Final[float] = 5.0
T_CONFIGS: Final[dict[str, tuple[int, float]]] = {"a": (16, 1e-3), "b": (16, 3e-4), "c": (32, 1e-3)}
"""Hidden size and learning rate of the three pre-registered TFT tuning configurations."""
R_LR: Final[float] = 2e-3
TUNE_DIR: Final = common.WORK_DIR / "fits_tune"


def columns_for(*, arm: str) -> tuple[int, ...]:
    """Return the input columns an arm reads (M2 gets its explicit solar signal appended separately)."""
    if arm == "m2d":
        return tuple(sorted({*common.M2_COLUMNS, common.COL["dark"], common.COL["elevation"]}))
    return common.M2_COLUMNS if arm == "m2" else common.M1_COLUMNS


class Prepared:
    """Standardised dense tensors of one refit for the runs in `index`."""

    def __init__(
        self,
        *,
        arm: str,
        label: str,
        d: data_mod.Data,
        index: np.ndarray,
        stats: tuple[np.ndarray, np.ndarray] | None,
        capacity: np.ndarray | None,
    ) -> None:
        """Gather `x` (runs, steps, features), the kept mask, targets, and the history."""
        cols = list(columns_for(arm=arm))
        x = np.asarray(d.x[index])[:, :, cols].astype(np.float32)
        if arm in ("m2", "m2d"):
            assert capacity is not None
            pv = (capacity[index] * np.asarray(d.x[index])[:, :, common.COL["unit_pv"]]).astype(
                np.float32
            )
            x = np.concatenate([x, pv[:, :, None]], axis=2)
        self.keep = np.asarray(d.keep[index])
        runs = d.runs[index]
        self.site_id = runs["site_id"].to_numpy()
        self.mu = runs["mu"].to_numpy()[:, None]
        self.sd = runs["sd"].to_numpy()[:, None]
        self.y_points = np.asarray(d.y[index]).astype(np.float64)
        self.z = ((self.y_points - self.mu) / self.sd).astype(np.float32)
        self.temp24 = np.asarray(d.temp24[index])
        self.hist = np.asarray(d.hist[index]).astype(np.float32)
        if stats is None:
            rows = x[self.keep]
            stats = (rows.mean(axis=0), rows.std(axis=0) + 1e-6)
        self.stats = stats
        self.x = np.clip((x - stats[0]) / stats[1], -INPUT_CLIP_SD, INPUT_CLIP_SD).astype(
            np.float32
        )
        self.onehot = np.eye(len(common.SITES), dtype=np.float32)[self.site_id]
        self.origin = runs["power_fcst_init_time"]


def train_mlp(
    *, arm: str, seed: int, train: Prepared, val: Prepared
) -> tuple[models.MixtureMlp, float, int]:
    """Fit the mixture MLP on kept rows; return the best-validation model, its loss, and the epochs."""

    def rows(p: Prepared) -> tuple[torch.Tensor, torch.Tensor]:
        k = p.keep
        x = np.concatenate([p.x[k], np.repeat(p.onehot[:, None, :], k.shape[1], axis=1)[k]], axis=1)
        return torch.tensor(x), torch.tensor(p.z[k])

    x_fit, z_fit = rows(train)
    x_val, z_val = rows(val)
    model = models.MixtureMlp(n_inputs=x_fit.shape[1])
    optimiser = torch.optim.AdamW(model.parameters(), lr=MLP_LR, weight_decay=WEIGHT_DECAY)
    generator = torch.Generator().manual_seed(seed)
    best, best_state, stale, epochs = float("inf"), None, 0, 0
    for epochs in range(1, MAX_EPOCHS + 1):  # noqa: B007
        model.train()
        order = torch.randperm(len(x_fit), generator=generator)
        for begin in range(0, len(order), MLP_BATCH):
            idx = order[begin : begin + MLP_BATCH]
            loss = model.loss(x=x_fit[idx], z=z_fit[idx])
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        model.eval()
        with torch.no_grad():
            score = float(model.loss(x=x_val, z=z_val))
        if score < best:
            best, stale = score, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    return model, best, epochs


def train_seq(
    *, arm: str, seed: int, train: Prepared, val: Prepared, t_config: str
) -> tuple[torch.nn.Module, float, int]:
    """Fit the GRU (r) or the TFT (t) on whole runs on the GPU."""
    device = torch.device("cuda")

    def tensors(p: Prepared) -> dict[str, torch.Tensor]:
        known = (
            p.x
            if arm == "t"
            else np.concatenate(
                [p.x, np.repeat(p.onehot[:, None, :], p.x.shape[1], axis=1)], axis=2
            )
        )
        return {
            "known": torch.tensor(known, device=device),
            "hist": torch.tensor(p.hist, device=device),
            "z": torch.tensor(p.z, device=device),
            "keep": torch.tensor(p.keep, device=device, dtype=torch.float32),
            "static": torch.tensor(p.onehot, device=device),
        }

    fit, held = tensors(train), tensors(val)
    n_known = fit["known"].shape[2]
    if arm == "r":
        model: torch.nn.Module = models.Seq2SeqGru(n_history=fit["hist"].shape[2], n_known=n_known)
        lr = R_LR
    else:
        hidden, lr = T_CONFIGS[t_config]
        model = models.TemporalFusionTransformer(
            n_static=len(common.SITES), n_past=fit["hist"].shape[2], n_known=n_known, d=hidden
        )
    model.to(device)

    def loss_of(b: dict[str, torch.Tensor]) -> torch.Tensor:
        if arm == "r":
            return cast(Any, model).loss(hist=b["hist"], known=b["known"], z=b["z"], keep=b["keep"])
        q = model(static=b["static"], past=b["hist"], known=b["known"])
        return (models.pinball_loss(q=q, z=b["z"]) * b["keep"]).sum() / b["keep"].sum().clamp(
            min=1.0
        )

    optimiser = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=WEIGHT_DECAY)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    n_fit, n_val = len(fit["z"]), len(held["z"])
    best, best_state, stale, epochs = float("inf"), None, 0, 0
    for epochs in range(1, MAX_EPOCHS + 1):  # noqa: B007
        model.train()
        order = torch.randperm(n_fit, generator=generator).to(device)
        for begin in range(0, n_fit, SEQ_BATCH):
            idx = order[begin : begin + SEQ_BATCH]
            loss = loss_of({k: v[idx] for k, v in fit.items()})
            optimiser.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimiser.step()
        model.eval()
        total, weight = 0.0, 0.0
        with torch.no_grad():
            for begin in range(0, n_val, 64):
                batch = {k: v[begin : begin + 64] for k, v in held.items()}
                total += float(loss_of(batch)) * float(batch["keep"].sum())
                weight += float(batch["keep"].sum())
        score = total / weight
        if score < best:
            best, stale = score, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    return model, best, epochs


def predict_z(*, arm: str, model: torch.nn.Module, p: Prepared) -> np.ndarray:
    """Return the 11 standardised quantiles of every (run, step); unkept steps are 0."""
    n_runs, n_steps = p.keep.shape
    out = np.zeros((n_runs, n_steps, len(models.LEVELS)), dtype=np.float64)
    if arm in ("m1", "m2", "m2d"):
        k = p.keep
        x = np.concatenate([p.x[k], np.repeat(p.onehot[:, None, :], n_steps, axis=1)[k]], axis=1)
        with torch.no_grad():
            raw = model(torch.tensor(x))
        out[k] = models.mixture_quantiles(out=raw)
        return out
    device = torch.device("cuda")
    known = (
        p.x
        if arm == "t"
        else np.concatenate([p.x, np.repeat(p.onehot[:, None, :], n_steps, axis=1)], axis=2)
    )
    for begin in range(0, n_runs, 64):
        sl = slice(begin, begin + 64)
        known_b = torch.tensor(known[sl], device=device)
        hist_b = torch.tensor(p.hist[sl], device=device)
        with torch.no_grad():
            if arm == "r":
                raw = model(hist=hist_b, known=known_b)
                q = np.stack([models.mixture_quantiles(out=raw[i]) for i in range(len(raw))])
            else:
                q = (
                    model(
                        static=torch.tensor(p.onehot[sl], device=device), past=hist_b, known=known_b
                    )
                    .float()
                    .cpu()
                    .numpy()
                )
        out[sl] = np.sort(q, axis=2)
    out[~p.keep] = 0.0
    return out


def calibrate(
    *, q_val: np.ndarray, val: Prepared, q_test: np.ndarray, test: Prepared
) -> tuple[np.ndarray, dict]:
    """Apply experiment 164's temperature-class multipliers (fitted on the validation months) to the test quantiles."""

    def points(q: np.ndarray, p: Prepared) -> np.ndarray:
        return q * p.sd[:, :, None] + p.mu[:, :, None]

    q_val_pts = points(q_val, val)[val.keep]
    q_test_pts = points(q_test, test)
    offsets_val = q_val_pts - q_val_pts[:, common.MEDIAN_INDEX : common.MEDIAN_INDEX + 1]
    residual = val.y_points[val.keep] - q_val_pts[:, common.MEDIAN_INDEX]
    classes_val = f164.temperature_class(ens_mean_c=val.temp24[val.keep])
    classes_test = f164.temperature_class(ens_mean_c=test.temp24)
    multipliers: dict[int, dict[float, float]] = {}
    for cls in range(len(c164.TEMPERATURE_CLASS_EDGES_C) + 1):
        chosen = classes_val == cls
        multipliers[cls] = (
            dict.fromkeys(f164.T_LEVELS, 1.0)
            if chosen.sum() < c164.MIN_CLASS_ROWS
            else f164.fit_multipliers(
                residual=residual[chosen], offsets=offsets_val[chosen], levels=f164.T_LEVELS
            )
        )
    median = q_test_pts[:, :, common.MEDIAN_INDEX : common.MEDIAN_INDEX + 1]
    scale = np.ones((*classes_test.shape, len(common.LEVELS)))
    for position, level in enumerate(common.LEVELS.tolist()):
        if level in f164.T_LEVELS:
            table = np.array(
                [multipliers[c][level] for c in range(len(c164.TEMPERATURE_CLASS_EDGES_C) + 1)]
            )
            scale[:, :, position] = table[classes_test]
    calibrated = np.sort(median + (q_test_pts - median) * scale, axis=2)
    return calibrated, {str(c): {str(k): v for k, v in m.items()} for c, m in multipliers.items()}


def run_job(task: tuple[str, str, int, str, bool]) -> str:
    """Fit one (arm, cut, seed) network and write its test-month quantiles."""
    arm, label, seed, t_config, tune = task
    name = f"{arm}-{t_config}" if tune else arm
    target = (TUNE_DIR if tune else common.FIT_DIR) / f"{name}_cut{label}_seed{seed}.parquet"
    if target.exists():
        return f"{name} {label} seed {seed} exists"
    torch.set_num_threads(1)
    torch.manual_seed(seed)
    started = time.monotonic()
    d = data_mod.load()
    split = data_mod.split_for(runs=d.runs, label=label)
    capacity = (
        data_mod.pinned_capacity_points(label=label, runs=d.runs, index=np.arange(d.runs.height))
        if arm in ("m2", "m2d")
        else None
    )
    train = Prepared(arm=arm, label=label, d=d, index=split.train, stats=None, capacity=capacity)
    val = Prepared(arm=arm, label=label, d=d, index=split.val, stats=train.stats, capacity=capacity)
    test = Prepared(
        arm=arm, label=label, d=d, index=split.test, stats=train.stats, capacity=capacity
    )
    if arm in ("m1", "m2", "m2d"):
        model, best, epochs = train_mlp(arm=arm, seed=seed, train=train, val=val)
    else:
        model, best, epochs = train_seq(arm=arm, seed=seed, train=train, val=val, t_config=t_config)
    q_val = predict_z(arm=arm, model=model, p=val)
    q_test = predict_z(arm=arm, model=model, p=test)
    raw_points = q_test * test.sd[:, :, None] + test.mu[:, :, None]
    calibrated, multipliers = calibrate(q_val=q_val, val=val, q_test=q_test, test=test)
    k = test.keep
    r_idx, s_idx = np.nonzero(k)
    frame = (
        pl.DataFrame(
            {
                "site": np.array(common.SITES)[test.site_id[r_idx]],
                "power_fcst_init_time": test.origin.gather(r_idx.tolist()),
                "step": s_idx.astype(np.int16),
            }
        )
        .with_columns(
            valid_time=pl.col("power_fcst_init_time")
            + pl.duration(minutes=30) * (pl.col("step").cast(pl.Int64) + 1),
            lead_hours=(pl.col("step").cast(pl.Float64) + 1.0) / 2.0,
        )
        .drop("step")
    )
    cal = calibrated[k]
    raw = raw_points[k]
    frame = frame.with_columns(
        *[pl.Series(q, cal[:, i].astype(np.float32)) for i, q in enumerate(common.QUANTILES)],
        *[
            pl.Series(f"raw_{q}", raw[:, i].astype(np.float32))
            for i, q in enumerate(common.QUANTILES)
        ],
        n_train=pl.lit(int(train.keep.sum())),
        epochs=pl.lit(epochs),
        best_val=pl.lit(best),
        seconds=pl.lit(time.monotonic() - started),
        multipliers=pl.lit(str(multipliers)),
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_name(f"{target.name}.{os.getpid()}.tmp")
    frame.write_parquet(temporary)
    temporary.rename(target)
    return f"{name} {label} seed {seed}: {epochs} epochs, best_val {best:.4f}, {time.monotonic() - started:.0f} s"


def main() -> None:
    """Fit every requested (arm, cut, seed) with a spawn-based process pool."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cuts", nargs="+", required=True)
    parser.add_argument("--arms", nargs="+", default=list(common.ARMS))
    parser.add_argument("--seeds", nargs="+", type=int, default=list(common.SEEDS))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--t-config", default="a")
    parser.add_argument("--tune", action="store_true", help="write to the tuning folder")
    args = parser.parse_args()
    tasks = [
        (a, c, s, args.t_config, args.tune)
        for c in args.cuts
        for a in args.arms
        for s in args.seeds
    ]
    context = multiprocessing.get_context("spawn")
    with ProcessPoolExecutor(max_workers=args.workers, mp_context=context) as pool:
        for message in pool.map(run_job, tasks):
            print(message, flush=True)


if __name__ == "__main__":
    main()
