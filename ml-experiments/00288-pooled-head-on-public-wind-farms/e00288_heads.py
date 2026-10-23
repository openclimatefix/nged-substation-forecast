# ruff: noqa: D102, D107, PLR0912, PLR0915
"""Train the heads and controls of experiment 00288 and write their scored quantiles.

Every head adds corrections to R0's 11 base quantiles (the sorted sum, clamped to 0 to 1), as
experiment 259's H5c did. Inputs are the 11 base quantiles, the seven member summaries, the lead,
and three static columns (log capacity, turbine rating, hub height); the farm one-hot of experiment
259 is gone, so a head can be applied to a farm it never saw. Training rows have a run start and a
valid time before 2025-04-16 and are not masked; early stopping uses runs from 2025-04-16.

Arms, scored on W1 to W3's 53 Tuesday runs with R0 seeds 0 to 2 and head seed = R0 seed:
P1 (head on W1 to W3), P2 (head on the public farms), P3 (P2 fine-tuned on W1 to W3), X2 (XGBoost
offsets on the public farms), X3 (X2 refitted with W1 to W3's training rows added), BL2 (blind
head on the public farms), T1 (offset table from W1 to W3's own training rows). `fold` scores each
fifth of the public farms with a head trained on the other four fifths. `pool` scores the public
farms' own scored rows with P2 and X2 (trained on every farm).

Output, in `work/heads/`: `<arm>.parquet` (W1 to W3) or `<arm>_pool.parquet` (public farms), one
block of rows per seed, and `validation_loss.json`.

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
      uv run --no-sync python ml-experiments/00288-*/e00288_heads.py [ARM ...]
"""

import json
import sys
from datetime import UTC, datetime
from typing import Final

import e00288_common as common
import e00288_tables as tables
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
import xgboost as xgb
from torch import nn  # ty: ignore[unresolved-import]

OUT_DIR: Final = common.WORK_DIR / "heads"
SEEDS: Final[tuple[int, ...]] = (0, 1, 2)
ARMS: Final[tuple[str, ...]] = ("T1", "P1", "P2", "P3", "X2", "X3", "BL2", "fold")
TRAIN_CUTOFF: Final[datetime] = datetime(2025, 4, 16, tzinfo=UTC)
EPOCHS_W: Final[int] = 60
EPOCHS_POOL: Final[int] = 30
BATCH_W: Final[int] = 512
BATCH_POOL: Final[int] = 2048
LEARNING_RATE: Final[float] = 2e-3
WEIGHT_DECAY: Final[float] = 1e-4
HIDDEN: Final[int] = 64
CORRECTION_SCALE: Final[float] = 0.3
FINE_TUNE_EPOCHS: Final[int] = 3
FINE_TUNE_RATE_FACTOR: Final[float] = 0.1
Q: Final[tuple[str, ...]] = common.LEVEL_COLUMNS
DEVICE: Final = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def features(*, frame: pl.DataFrame, blind: bool = False) -> np.ndarray:
    """Return the head inputs before standardisation, shape (cases, columns)."""
    lead = frame["lead_hours"].to_numpy()[:, None] / 100.0
    static = tables.static_matrix(frame=frame)
    if blind:
        hour = (
            frame["valid_time"].dt.hour().to_numpy()
            + frame["valid_time"].dt.minute().to_numpy() / 60
        )
        day = frame["valid_time"].dt.ordinal_day().to_numpy()
        calendar = np.stack(
            [
                np.sin(2 * np.pi * hour / 24),
                np.cos(2 * np.pi * hour / 24),
                np.sin(2 * np.pi * day / 365.25),
                np.cos(2 * np.pi * day / 365.25),
            ],
            axis=1,
        )
        return np.concatenate([lead, static, calendar], axis=1)
    base = frame.select(Q).to_numpy().astype(np.float64)
    summary = frame.select(tables.SUMMARY_COLUMNS).to_numpy().astype(np.float64)
    return np.concatenate([base, summary, lead, static], axis=1)


def split(*, frame: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return boolean selectors for the training rows and the early-stopping rows."""
    usable = (
        (~frame["masked"]) & frame["actual_norm"].is_finite().fill_null(value=False)
    ).to_numpy()
    init = frame["power_fcst_init_time"]
    train = ((init < TRAIN_CUTOFF) & (frame["valid_time"] < TRAIN_CUTOFF)).to_numpy()
    validation = (init >= TRAIN_CUTOFF).to_numpy()
    return usable & train, usable & validation


def pinball(*, quantiles: torch.Tensor, actual: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
    """Return the mean pinball loss over cases and the 11 levels."""
    error = actual[:, None] - quantiles
    return torch.maximum(levels * error, (levels - 1.0) * error).mean()


class Head(nn.Module):
    """Corrections to the base quantiles: a two-hidden-layer network."""

    def __init__(self, *, n_in: int) -> None:
        super().__init__()
        last = nn.Linear(HIDDEN, 11)
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        self.net = nn.Sequential(
            nn.Linear(n_in, HIDDEN), nn.GELU(), nn.Linear(HIDDEN, HIDDEN), nn.GELU(), last
        )

    def forward(self, x: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
        correction = CORRECTION_SCALE * torch.tanh(self.net(x))
        return torch.clamp(torch.sort(base + correction, dim=1).values, 0.0, 1.0)


class Fitted:
    """A trained head with the standardisation it was trained under."""

    def __init__(self, *, model: Head, mean: np.ndarray, std: np.ndarray, blind: bool) -> None:
        self.model, self.mean, self.std, self.blind = model, mean, std, blind

    def predict(self, *, frame: pl.DataFrame) -> np.ndarray:
        x = torch.tensor(
            (features(frame=frame, blind=self.blind) - self.mean) / self.std,
            dtype=torch.float32,
            device=DEVICE,
        )
        base = torch.tensor(frame.select(Q).to_numpy(), dtype=torch.float32, device=DEVICE)
        with torch.no_grad():
            return self.model(x, base).cpu().numpy()


def tensors(*, frame: pl.DataFrame, rows: np.ndarray, fitted_stats: tuple, blind: bool) -> tuple:
    """Return standardised inputs, base quantiles, and targets of the selected rows on the GPU."""
    mean, std = fitted_stats
    sub = frame.filter(pl.Series(rows))
    x = torch.tensor(
        (features(frame=sub, blind=blind) - mean) / std, dtype=torch.float32, device=DEVICE
    )
    base = torch.tensor(sub.select(Q).to_numpy(), dtype=torch.float32, device=DEVICE)
    actual = torch.tensor(sub["actual_norm"].to_numpy(), dtype=torch.float32, device=DEVICE)
    return x, base, actual


def train_head(
    *,
    train: pl.DataFrame,
    validation: pl.DataFrame,
    seed: int,
    blind: bool = False,
    epochs: int,
    batch: int,
    start_from: Fitted | None = None,
    learning_rate: float = LEARNING_RATE,
    keep_epochs: tuple[int, ...] | None = None,
) -> tuple[Fitted, float, int]:
    """Train a head and return it at its best early-stopping epoch, the loss, and the epoch.

    Args:
        train: Training rows (already filtered).
        validation: Early-stopping rows.
        seed: Seed for initialisation and batch order.
        blind: Use the blind input columns.
        epochs: Number of epochs.
        batch: Batch size.
        start_from: A fitted head to continue from (fine-tuning keeps its standardisation).
        learning_rate: The learning rate.
        keep_epochs: When set, only these 1-based epochs are candidates for the best epoch.

    Returns:
        The fitted head at the best epoch, its validation pinball loss, and that epoch.
    """
    torch.manual_seed(seed)
    if start_from is None:
        x_all = features(frame=train, blind=blind)
        mean, std = x_all.mean(axis=0), np.maximum(x_all.std(axis=0), 1e-6)
        model = Head(n_in=x_all.shape[1]).to(DEVICE)
    else:
        mean, std, blind = start_from.mean, start_from.std, start_from.blind
        model = Head(n_in=len(mean)).to(DEVICE)
        model.load_state_dict(start_from.model.state_dict())
    ones = np.ones(train.height, dtype=bool)
    x_t, base_t, y_t = tensors(frame=train, rows=ones, fitted_stats=(mean, std), blind=blind)
    x_v, base_v, y_v = tensors(
        frame=validation,
        rows=np.ones(validation.height, dtype=bool),
        fitted_stats=(mean, std),
        blind=blind,
    )
    levels = torch.tensor(common.LEVELS, dtype=torch.float32, device=DEVICE)
    optimiser = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=WEIGHT_DECAY)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    best, best_state, best_epoch = float("inf"), None, 0
    for epoch in range(1, epochs + 1):
        order = torch.randperm(x_t.shape[0], generator=generator).to(DEVICE)
        for start in range(0, len(order), batch):
            index = order[start : start + batch]
            loss = pinball(
                quantiles=model(x_t[index], base_t[index]), actual=y_t[index], levels=levels
            )
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
        with torch.no_grad():
            score = float(pinball(quantiles=model(x_v, base_v), actual=y_v, levels=levels))
        if (keep_epochs is None or epoch in keep_epochs) and score < best:
            best, best_epoch = score, epoch
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if best_state is None:
        msg = "no epoch produced a finite validation loss"
        raise RuntimeError(msg)
    model.load_state_dict(best_state)
    return Fitted(model=model, mean=mean, std=std, blind=blind), best, best_epoch


def xgboost_offsets(
    *, train: pl.DataFrame, validation: pl.DataFrame, seed: int
) -> list[xgb.Booster]:
    """Return one XGBoost booster per level, each fitted to the residual of the base quantile."""
    x_t, x_v = features(frame=train), features(frame=validation)
    base_t = train.select(Q).to_numpy().astype(np.float64)
    base_v = validation.select(Q).to_numpy().astype(np.float64)
    boosters = []
    for k, level in enumerate(common.LEVELS):
        parameters = {
            "objective": "reg:quantileerror",
            "quantile_alpha": float(level),
            "eta": 0.05,
            "max_depth": 4,
            "subsample": 0.8,
            "colsample_bytree": 1.0,
            "tree_method": "hist",
            "device": "cuda",
            "seed": seed,
        }
        train_set = xgb.DMatrix(x_t, label=train["actual_norm"].to_numpy() - base_t[:, k])
        validation_set = xgb.DMatrix(x_v, label=validation["actual_norm"].to_numpy() - base_v[:, k])
        boosters.append(
            xgb.train(
                parameters,
                train_set,
                num_boost_round=400,
                evals=[(validation_set, "validation")],
                early_stopping_rounds=30,
                verbose_eval=False,
            )
        )
    return boosters


def apply_offsets(*, boosters: list[xgb.Booster], frame: pl.DataFrame) -> np.ndarray:
    """Return the base quantiles plus each level's XGBoost correction, sorted and clamped."""
    x = xgb.DMatrix(features(frame=frame))
    base = frame.select(Q).to_numpy().astype(np.float64)
    out = np.empty_like(base)
    for k, booster in enumerate(boosters):
        out[:, k] = base[:, k] + booster.predict(x, iteration_range=(0, booster.best_iteration + 1))
    return np.clip(np.sort(out, axis=1), 0.0, 1.0)


def offset_table(*, train: pl.DataFrame, frame: pl.DataFrame) -> np.ndarray:
    """Return T1: the pinball-optimal offset per farm, lead day, and level of the training rows."""
    keys = ["farm", "lead_day"]
    residual = train.select(*keys, *[(pl.col("actual_norm") - pl.col(q)).alias(q) for q in Q])
    offsets = residual.group_by(keys).agg(
        [
            pl.col(q).quantile(level, interpolation="linear")
            for q, level in zip(Q, common.LEVELS, strict=True)
        ]
    )
    joined = frame.select(*keys).join(
        offsets, on=keys, how="left", suffix="_offset", maintain_order="left"
    )
    correction = joined.select([pl.col(f"{q}_offset").fill_null(0.0) for q in Q]).to_numpy()
    return np.clip(np.sort(frame.select(Q).to_numpy() + correction, axis=1), 0.0, 1.0)


def write(*, name: str, frame: pl.DataFrame, by_seed: dict[int, np.ndarray]) -> None:
    """Write an arm's scored quantiles, one block of rows per seed, atomically."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    blocks = [
        frame.select(
            "group",
            "farm",
            "fold",
            "power_fcst_init_time",
            "valid_time",
            "lead_hours",
            "lead_day",
            "actual_norm",
            "masked",
        )
        .hstack(pl.DataFrame(q.astype(np.float32), schema=list(Q)))
        .with_columns(seed=pl.lit(seed, dtype=pl.Int32))
        for seed, q in by_seed.items()
    ]
    path = OUT_DIR / f"{name}.parquet"
    pl.concat(blocks).write_parquet(path.with_suffix(".tmp"))
    path.with_suffix(".tmp").rename(path)


def main() -> None:
    """Train each requested arm for each seed and record the validation losses."""
    wanted = tuple(sys.argv[1:]) or ARMS
    pool_h = tables.pool_table(set_name="hindcast")
    pool_s = tables.pool_table(set_name="scored")
    farms = pl.read_parquet(tables.CLEAN_DIR / "farms.parquet")
    loss_path = OUT_DIR / "validation_loss.json"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    losses = json.loads(loss_path.read_text()) if loss_path.exists() else {}
    pool_train_rows, pool_val_rows = split(frame=pool_h)
    pool_train, pool_val = (pool_h.filter(pl.Series(r)) for r in (pool_train_rows, pool_val_rows))
    w_h = {s: tables.w_table(set_name="hindcast", seed=s, pool_farms=farms) for s in SEEDS}
    w_s = {s: tables.w_table(set_name="scored", seed=s, pool_farms=farms) for s in SEEDS}
    w_split = {s: split(frame=w_h[s]) for s in SEEDS}
    w_train = {s: w_h[s].filter(pl.Series(w_split[s][0])) for s in SEEDS}
    w_val = {s: w_h[s].filter(pl.Series(w_split[s][1])) for s in SEEDS}
    for arm in wanted:
        by_seed: dict[int, np.ndarray] = {}
        pool_by_seed: dict[int, np.ndarray] = {}
        arm_losses: dict[str, object] = {}
        for seed in SEEDS:
            if arm == "T1":
                by_seed[seed] = offset_table(train=w_train[seed], frame=w_s[seed])
            elif arm == "P1":
                fitted, loss, epoch = train_head(
                    train=w_train[seed],
                    validation=w_val[seed],
                    seed=seed,
                    epochs=EPOCHS_W,
                    batch=BATCH_W,
                )
                by_seed[seed] = fitted.predict(frame=w_s[seed])
                arm_losses[str(seed)] = {"loss": loss, "epoch": epoch}
            elif arm in {"P2", "P3", "BL2"}:
                fitted, loss, epoch = train_head(
                    train=pool_train,
                    validation=pool_val,
                    seed=seed,
                    blind=arm == "BL2",
                    epochs=EPOCHS_POOL,
                    batch=BATCH_POOL,
                )
                arm_losses[str(seed)] = {"loss": loss, "epoch": epoch}
                if arm == "P3":
                    tuned, tuned_loss, tuned_epoch = train_head(
                        train=w_train[seed],
                        validation=w_val[seed],
                        seed=seed,
                        epochs=FINE_TUNE_EPOCHS,
                        batch=BATCH_W,
                        start_from=fitted,
                        learning_rate=LEARNING_RATE * FINE_TUNE_RATE_FACTOR,
                        keep_epochs=(1, 2, 3),
                    )
                    by_seed[seed] = tuned.predict(frame=w_s[seed])
                    arm_losses[str(seed)] = {
                        "pool_loss": loss,
                        "loss": tuned_loss,
                        "epoch": tuned_epoch,
                    }
                else:
                    by_seed[seed] = fitted.predict(frame=w_s[seed])
                    if arm == "P2":
                        pool_by_seed[seed] = fitted.predict(frame=pool_s)
            elif arm == "X2":
                boosters = xgboost_offsets(train=pool_train, validation=pool_val, seed=seed)
                by_seed[seed] = apply_offsets(boosters=boosters, frame=w_s[seed])
                pool_by_seed[seed] = apply_offsets(boosters=boosters, frame=pool_s)
            elif arm == "X3":
                boosters = xgboost_offsets(
                    train=pl.concat([pool_train, w_train[seed]]),
                    validation=pl.concat([pool_val, w_val[seed]]),
                    seed=seed,
                )
                by_seed[seed] = apply_offsets(boosters=boosters, frame=w_s[seed])
            elif arm == "fold":
                held = np.zeros((pool_s.height, 11))
                for fold in sorted(farms["fold"].unique().to_list()):
                    in_fold_train = pool_train.filter(pl.col("fold") != fold)
                    in_fold_val = pool_val.filter(pl.col("fold") != fold)
                    fitted, loss, epoch = train_head(
                        train=in_fold_train,
                        validation=in_fold_val,
                        seed=seed,
                        epochs=EPOCHS_POOL,
                        batch=BATCH_POOL,
                    )
                    rows = (pool_s["fold"] == fold).to_numpy()
                    held[rows] = fitted.predict(frame=pool_s.filter(pl.Series(rows)))
                    arm_losses[f"{seed}_fold{fold}"] = {"loss": loss, "epoch": epoch}
                pool_by_seed[seed] = held
            print(f"{arm} seed {seed} done", flush=True)
        if by_seed:
            write(name=arm, frame=w_s[SEEDS[0]], by_seed=by_seed)
        if pool_by_seed:
            write(name=f"{arm}_pool", frame=pool_s, by_seed=pool_by_seed)
        losses[arm] = arm_losses
        loss_path.write_text(json.dumps(losses, indent=1))
        print(f"{arm} written", flush=True)


if __name__ == "__main__":
    main()
