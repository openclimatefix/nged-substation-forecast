# ruff: noqa: D102, D107
"""Train the neural head and its controls for experiment 00259, and write their scored quantiles.

Every head adds corrections to the 11 base quantiles of the physics core (R1) from `e00259_base.py`:
the quantiles are the sorted sum of the base and the correction, clamped to 0 to 1. Training uses
the hindcast cases (not masked, observed) whose run starts and whose valid time falls before
2025-04-16. Early stopping uses the cases of runs initialised from 2025-04-16. Nothing is tuned on
a scored case.

Arms (seeds 0 to 4 for the trained arms): H5c, K5, BL5, H5c-Tue, N5-lin; N5-xgb and T5 have one fit.
Output, in `work/heads/`: `<arm>.parquet` (scored quantiles by seed) and `validation_loss.json`.

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
      uv run --no-sync python ml-experiments/00259-*/e00259_heads.py [ARM ...]
"""

import json
import sys
from datetime import UTC, datetime
from typing import Final

import e00259_common as common
import numpy as np
import polars as pl
import torch
import xgboost as xgb
from torch import nn

BASE_DIR: Final = common.WORK_DIR / "base"
OUT_DIR: Final = common.WORK_DIR / "heads"
SEEDS: Final[tuple[int, ...]] = (0, 1, 2, 3, 4)
ARMS: Final[tuple[str, ...]] = ("H5c", "K5", "BL5", "H5c-Tue", "N5-lin", "N5-xgb", "T5")
TRAIN_CUTOFF: Final[datetime] = datetime(2025, 4, 16, tzinfo=UTC)
"""Training runs start, and training valid times fall, before this. Validation runs start at it."""
EPOCHS: Final[int] = 60
BATCH: Final[int] = 512
LEARNING_RATE: Final[float] = 2e-3
WEIGHT_DECAY: Final[float] = 1e-4
HIDDEN: Final[int] = 64
CORRECTION_SCALE: Final[float] = 0.3
SUMMARY_FEATURES: Final[tuple[str, ...]] = (
    "ws_mean",
    "ws_std",
    "ws_p10",
    "ws_p90",
    "dir_sin",
    "dir_cos",
    "temp_mean",
)
Q: Final[tuple[str, ...]] = common.LEVEL_COLUMNS
DEVICE: Final = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load(*, set_name: str) -> pl.DataFrame:
    """Return the cases of a set with R1's base quantiles, the member summaries, and the mask."""
    summaries = pl.read_parquet(BASE_DIR / f"summaries_{set_name}.parquet")
    base = pl.read_parquet(BASE_DIR / f"R1_{set_name}_seed0.parquet").drop("seed")
    joined = summaries.join(base, on=list(common.KEY_COLUMNS), how="inner", validate="1:1")
    if joined.height != summaries.height:
        msg = f"{set_name}: the summaries and the R1 quantiles cover different cases"
        raise ValueError(msg)
    return joined.sort(*common.KEY_COLUMNS)


def features(*, frame: pl.DataFrame, arm: str) -> np.ndarray:
    """Return the arm's head inputs, before standardisation, as shape (cases, columns)."""
    lead = frame["lead_hours"].to_numpy()[:, None] / 100.0
    farm = np.stack(
        [(frame["site"] == s).to_numpy().astype(np.float64) for s in ("W1", "W2", "W3")], axis=1
    )
    base = frame.select(Q).to_numpy().astype(np.float64)
    if arm == "BL5":
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
        return np.concatenate([lead, farm, calendar], axis=1)
    if arm == "K5":
        return np.concatenate([base, lead, farm], axis=1)
    summary = frame.select(SUMMARY_FEATURES).to_numpy().astype(np.float64)
    return np.concatenate([base, summary, lead, farm], axis=1)


def pinball(*, quantiles: torch.Tensor, actual: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
    """Return the mean pinball loss over cases and the 11 levels."""
    error = actual[:, None] - quantiles
    return torch.maximum(levels * error, (levels - 1.0) * error).mean()


class Head(nn.Module):
    """Corrections to the base quantiles: a two-hidden-layer network or a linear map."""

    def __init__(self, *, n_in: int, linear: bool) -> None:
        super().__init__()
        last = nn.Linear(n_in if linear else HIDDEN, 11)
        nn.init.zeros_(last.weight)
        nn.init.zeros_(last.bias)
        if linear:
            self.net: nn.Module = last
        else:
            self.net = nn.Sequential(
                nn.Linear(n_in, HIDDEN),
                nn.GELU(),
                nn.Linear(HIDDEN, HIDDEN),
                nn.GELU(),
                last,
            )
        self.linear = linear

    def forward(self, x: torch.Tensor, base: torch.Tensor) -> torch.Tensor:
        correction = self.net(x)
        if not self.linear:
            correction = CORRECTION_SCALE * torch.tanh(correction)
        return torch.clamp(torch.sort(base + correction, dim=1).values, 0.0, 1.0)


def split(*, train: pl.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return boolean row selectors for the training cases and the early-stopping cases."""
    usable = (
        (~train["masked"]) & train["actual_norm"].is_finite().fill_null(value=False)
    ).to_numpy()
    init = train["power_fcst_init_time"]
    is_train = ((init < TRAIN_CUTOFF) & (train["valid_time"] < TRAIN_CUTOFF)).to_numpy()
    is_validation = (init >= TRAIN_CUTOFF).to_numpy()
    return usable & is_train, usable & is_validation


def train_head(
    *,
    arm: str,
    seed: int,
    hindcast: pl.DataFrame,
    scored: pl.DataFrame,
    tuesday_only: bool,
) -> tuple[np.ndarray, float]:
    """Return the scored quantiles of one trained head and its best validation loss."""
    torch.manual_seed(seed)
    train_rows, validation_rows = split(train=hindcast)
    if tuesday_only:
        tuesday = (
            (hindcast["power_fcst_init_time"].dt.weekday() == 2)
            & (hindcast["power_fcst_init_time"].dt.hour() >= 0)
        ).to_numpy()
        train_rows &= tuesday
    x_all = features(frame=hindcast, arm=arm)
    mean = x_all[train_rows].mean(axis=0)
    std = np.maximum(x_all[train_rows].std(axis=0), 1e-6)
    standardise = lambda x: torch.tensor((x - mean) / std, dtype=torch.float32, device=DEVICE)  # noqa: E731
    base_h = torch.tensor(hindcast.select(Q).to_numpy(), dtype=torch.float32, device=DEVICE)
    actual_h = torch.tensor(hindcast["actual_norm"].to_numpy(), dtype=torch.float32, device=DEVICE)
    x_h = standardise(x_all)
    levels = torch.tensor(common.LEVELS, dtype=torch.float32, device=DEVICE)
    model = Head(n_in=x_h.shape[1], linear=arm == "N5-lin").to(DEVICE)
    optimiser = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    train_index = torch.tensor(np.flatnonzero(train_rows), device=DEVICE)
    validation_index = torch.tensor(np.flatnonzero(validation_rows), device=DEVICE)
    generator = torch.Generator(device="cpu").manual_seed(seed)
    best, best_state = float("inf"), None
    for _ in range(EPOCHS):
        order = train_index[torch.randperm(len(train_index), generator=generator).to(DEVICE)]
        for start in range(0, len(order), BATCH):
            batch = order[start : start + BATCH]
            loss = pinball(
                quantiles=model(x_h[batch], base_h[batch]), actual=actual_h[batch], levels=levels
            )
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
        with torch.no_grad():
            score = float(
                pinball(
                    quantiles=model(x_h[validation_index], base_h[validation_index]),
                    actual=actual_h[validation_index],
                    levels=levels,
                )
            )
        if score < best:
            best = score
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
    if best_state is None:
        msg = "no epoch produced a finite validation loss"
        raise RuntimeError(msg)
    model.load_state_dict(best_state)
    x_s = standardise(features(frame=scored, arm=arm))
    base_s = torch.tensor(scored.select(Q).to_numpy(), dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        return model(x_s, base_s).cpu().numpy(), best


def xgboost_offsets(
    *, hindcast: pl.DataFrame, scored: pl.DataFrame, seed: int
) -> tuple[np.ndarray, float]:
    """Return N5-xgb, a per-level XGBoost correction, and its validation loss."""
    train_rows, validation_rows = split(train=hindcast)
    x_h = features(frame=hindcast, arm="H5c")
    x_s = features(frame=scored, arm="H5c")
    base_h = hindcast.select(Q).to_numpy().astype(np.float64)
    base_s = scored.select(Q).to_numpy().astype(np.float64)
    actual = hindcast["actual_norm"].to_numpy().astype(np.float64)
    corrected_s = np.empty_like(base_s)
    corrected_v = np.empty((int(validation_rows.sum()), 11))
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
        train_set = xgb.DMatrix(x_h[train_rows], label=actual[train_rows] - base_h[train_rows, k])
        validation_set = xgb.DMatrix(
            x_h[validation_rows], label=actual[validation_rows] - base_h[validation_rows, k]
        )
        booster = xgb.train(
            parameters,
            train_set,
            num_boost_round=400,
            evals=[(validation_set, "validation")],
            early_stopping_rounds=30,
            verbose_eval=False,
        )
        corrected_s[:, k] = base_s[:, k] + booster.predict(
            xgb.DMatrix(x_s), iteration_range=(0, booster.best_iteration + 1)
        )
        corrected_v[:, k] = base_h[validation_rows, k] + booster.predict(
            validation_set, iteration_range=(0, booster.best_iteration + 1)
        )
    corrected_s = np.clip(np.sort(corrected_s, axis=1), 0.0, 1.0)
    corrected_v = np.clip(np.sort(corrected_v, axis=1), 0.0, 1.0)
    loss = pinball(
        quantiles=torch.tensor(corrected_v),
        actual=torch.tensor(actual[validation_rows]),
        levels=torch.tensor(common.LEVELS),
    )
    return corrected_s, float(loss)


def table_offsets(*, hindcast: pl.DataFrame, scored: pl.DataFrame) -> tuple[np.ndarray, float]:
    """Return T5: the pinball-optimal offset per lead day, level, and farm, and validation loss."""
    train_rows, validation_rows = split(train=hindcast)
    keys = ("lead_day", "site")

    def with_day(frame: pl.DataFrame) -> pl.DataFrame:
        return frame.with_columns(
            lead_day=common.lead_day(
                power_fcst_init_time=pl.col("power_fcst_init_time"), valid_time=pl.col("valid_time")
            )
        )

    train = with_day(hindcast.filter(pl.Series(train_rows)))
    residual = train.select(
        *keys,
        *[(pl.col("actual_norm") - pl.col(q)).alias(q) for q in Q],
    )
    offsets = residual.group_by(keys).agg(
        [
            pl.col(q).quantile(level, interpolation="linear")
            for q, level in zip(Q, common.LEVELS, strict=True)
        ]
    )

    def apply(frame: pl.DataFrame) -> np.ndarray:
        joined = with_day(frame).join(
            offsets, on=list(keys), how="left", suffix="_offset", maintain_order="left"
        )
        correction = joined.select([pl.col(f"{q}_offset").fill_null(0.0) for q in Q]).to_numpy()
        return np.clip(np.sort(frame.select(Q).to_numpy() + correction, axis=1), 0.0, 1.0)

    validation = hindcast.filter(pl.Series(validation_rows))
    loss = pinball(
        quantiles=torch.tensor(apply(validation)),
        actual=torch.tensor(validation["actual_norm"].to_numpy()),
        levels=torch.tensor(common.LEVELS),
    )
    return apply(scored), float(loss)


def write(*, arm: str, scored: pl.DataFrame, by_seed: dict[int, np.ndarray]) -> None:
    """Write an arm's scored quantiles, one block of rows per seed, atomically."""
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    frames = [
        scored.select(*common.KEY_COLUMNS, "site", "lead_hours", "actual_norm", "masked")
        .hstack(pl.DataFrame(q.astype(np.float32), schema=list(Q)))
        .with_columns(seed=pl.lit(seed, dtype=pl.Int32))
        for seed, q in by_seed.items()
    ]
    path = OUT_DIR / f"{arm}.parquet"
    pl.concat(frames).write_parquet(path.with_suffix(".tmp"))
    path.with_suffix(".tmp").rename(path)


def main() -> None:
    """Train each requested arm and record the validation losses."""
    wanted = tuple(sys.argv[1:]) or ARMS
    hindcast, scored = load(set_name="hindcast"), load(set_name="scored")
    loss_path = OUT_DIR / "validation_loss.json"
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    losses = json.loads(loss_path.read_text()) if loss_path.exists() else {}
    _, validation_rows = split(train=hindcast)
    levels = torch.tensor(common.LEVELS, dtype=torch.float32)
    losses["R1 on the early-stopping runs"] = float(
        pinball(
            quantiles=torch.tensor(hindcast.select(Q).to_numpy(), dtype=torch.float32)[
                torch.tensor(validation_rows)
            ],
            actual=torch.tensor(hindcast["actual_norm"].to_numpy(), dtype=torch.float32)[
                torch.tensor(validation_rows)
            ],
            levels=levels,
        )
    )
    for arm in wanted:
        by_seed: dict[int, np.ndarray] = {}
        arm_losses: dict[str, float] = {}
        if arm == "T5":
            quantiles, loss = table_offsets(hindcast=hindcast, scored=scored)
            by_seed, arm_losses = {0: quantiles}, {"0": loss}
        elif arm == "N5-xgb":
            quantiles, loss = xgboost_offsets(hindcast=hindcast, scored=scored, seed=0)
            by_seed, arm_losses = {0: quantiles}, {"0": loss}
        else:
            for seed in SEEDS:
                quantiles, loss = train_head(
                    arm="H5c" if arm == "H5c-Tue" else arm,
                    seed=seed,
                    hindcast=hindcast,
                    scored=scored,
                    tuesday_only=arm == "H5c-Tue",
                )
                by_seed[seed], arm_losses[str(seed)] = quantiles, loss
                print(f"{arm} seed {seed} validation pinball {loss:.5f}", flush=True)
        write(arm=arm, scored=scored, by_seed=by_seed)
        losses[arm] = arm_losses
        loss_path.write_text(json.dumps(losses, indent=1))
        print(f"{arm} written", flush=True)


if __name__ == "__main__":
    main()
