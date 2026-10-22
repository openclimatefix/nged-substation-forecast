# ruff: noqa: E501
"""Train the window network for one fold on an injector fitted without that fold's Primary pairs.

    systemd-run --user --scope -p MemoryMax=12G -p OOMPolicy=continue \
        env OMP_NUM_THREADS=1 uv run --no-sync python ml-experiments/00189-*/e00189_train.py FOLD [SEED]

Fold -1 fits the injector on every real event (the model used only for the block-day check).
"""

from __future__ import annotations

import json
import os
import sys
import time
from typing import Final

import numpy as np
import torch
from e00189_common import WORK_DIR, load_panel, noise_units, pair_folds
from e00189_injector import fit_injector
from e00189_net import (
    EVEN_MONTHS,
    REGION,
    Batch,
    WindowNet,
    build_context,
    make_batch,
    n_parameters,
)
from torch.nn import functional

N_TRAIN: Final[int] = 80_000
N_VALID: Final[int] = 6_000
EPOCHS: Final[int] = int(os.environ.get("E189_EPOCHS", "14"))
"""14 for the pre-registered run (seed 0); 40 for the post hoc longer run (seed 1)."""
BATCH: Final[int] = 256
LEARNING_RATE: Final[float] = 2e-3
POSITIVE_SHARE: Final[float] = 0.4


def map_loss(*, logits: torch.Tensor, position: torch.Tensor) -> torch.Tensor:
    """Cross-entropy of the position within `REGION`, over windows that have one."""
    keep = position >= 0
    if not keep.any():
        return logits.sum() * 0.0
    masked = logits[keep].clone()
    masked[:, : REGION[0]] = -1e4
    masked[:, REGION[1] :] = -1e4
    return functional.cross_entropy(masked, position[keep])


def loss_of(*, model: WindowNet, batch: dict[str, torch.Tensor]) -> torch.Tensor:
    """Alarm cross-entropies plus the onset and end position cross-entropies."""
    alarms, maps = model(batch["x"])
    return (
        functional.binary_cross_entropy_with_logits(alarms[:, 0], batch["onset_alarm"])
        + functional.binary_cross_entropy_with_logits(alarms[:, 1], batch["end_alarm"])
        + map_loss(logits=maps[:, 0], position=batch["onset_pos"])
        + map_loss(logits=maps[:, 1], position=batch["end_pos"])
    )


def to_tensors(*, batch: Batch, device: str) -> dict[str, torch.Tensor]:
    """Move a batch to the device."""
    return {
        "x": torch.tensor(batch.x, device=device),
        "onset_alarm": torch.tensor(batch.onset_alarm, device=device),
        "end_alarm": torch.tensor(batch.end_alarm, device=device),
        "onset_pos": torch.tensor(batch.onset_pos, device=device),
        "end_pos": torch.tensor(batch.end_pos, device=device),
    }


def main(*, fold: int, seed: int) -> None:
    """Fit the injector, draw the training windows, train, and save the best-epoch weights."""
    panel = load_panel()
    residual, power = panel["residual"], panel["power"]
    sigma = noise_units(residual=residual)
    ctx = build_context(residual=residual, sigma=sigma)
    folds = pair_folds()
    held_out = frozenset(p for p, f in folds.items() if f == fold)
    fit = fit_injector(held_out_pairs=held_out, sigma=sigma)
    rng = np.random.default_rng(1000 * seed + fold + 7)
    started = time.time()
    train = make_batch(
        n=N_TRAIN,
        fit=fit,
        residual=residual,
        power=power,
        ctx=ctx,
        rng=rng,
        positive_share=POSITIVE_SHARE,
    )
    valid = make_batch(
        n=N_VALID,
        fit=fit,
        residual=residual,
        power=power,
        ctx=ctx,
        rng=rng,
        positive_share=POSITIVE_SHARE,
        allowed_months=EVEN_MONTHS,
    )
    print(f"fold {fold}: windows drawn in {time.time() - started:.0f} s", flush=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(seed)
    model = WindowNet().to(device)
    optimiser = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    steps = EPOCHS * (N_TRAIN // BATCH)
    schedule = torch.optim.lr_scheduler.OneCycleLR(
        optimiser, max_lr=LEARNING_RATE, total_steps=steps
    )
    train_t, valid_t = (
        to_tensors(batch=train, device=device),
        to_tensors(batch=valid, device=device),
    )
    best, best_state, history = float("inf"), None, []
    for epoch in range(EPOCHS):
        model.train()
        order = torch.randperm(N_TRAIN, device=device)
        total = 0.0
        for i in range(N_TRAIN // BATCH):
            idx = order[i * BATCH : (i + 1) * BATCH]
            loss = loss_of(model=model, batch={k: v[idx] for k, v in train_t.items()})
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            schedule.step()
            total += float(loss)
        model.eval()
        with torch.no_grad():
            valid_loss = float(
                np.mean(
                    [
                        float(
                            loss_of(
                                model=model, batch={k: v[j : j + 1000] for k, v in valid_t.items()}
                            )
                        )
                        for j in range(0, N_VALID, 1000)
                    ]
                )
            )
        history.append({"epoch": epoch, "train": total / (N_TRAIN // BATCH), "valid": valid_loss})
        print(history[-1], flush=True)
        if valid_loss < best:
            best, best_state = (
                valid_loss,
                {k: v.cpu().clone() for k, v in model.state_dict().items()},
            )
    directory = WORK_DIR / "models"
    directory.mkdir(parents=True, exist_ok=True)
    torch.save(best_state, directory / f"fold{fold}_seed{seed}.pt")
    (directory / f"fold{fold}_seed{seed}.json").write_text(
        json.dumps(
            {
                "history": history,
                "n_parameters": n_parameters(model=model),
                "n_real_events_in_fit": fit.n_real,
                "held_out_pairs": sorted(held_out),
            },
            indent=2,
        )
    )
    print(f"fold {fold} done in {time.time() - started:.0f} s; best valid {best:.3f}", flush=True)


if __name__ == "__main__":
    main(fold=int(sys.argv[1]), seed=int(sys.argv[2]) if len(sys.argv) > 2 else 0)
