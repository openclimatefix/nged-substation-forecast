# ruff: noqa: D101, D102, D107
"""P6 of experiment 00303: experiment 00259's neural head on the best adapter arm's quantiles.

The base is the post-spread-factor quantiles of the plain arm (P2, P4, or P5) with the lowest
hindcast quantile score; the offset table is not applied, because the head replaces it. The head is
experiment 259's H5c: two hidden layers of 64 units, a zero-initialised last layer, 11 corrections
of 0.3 times tanh added to the base quantiles, sorted and clamped to 0 to 1. Inputs: the 11 base
quantiles, the members' mean, standard deviation, 10th and 90th percentile of the 100 m speed, the
mean direction, the mean temperature, the lead, and the farm. Training cases: not masked, observed,
run and valid time before 2025-04-16; early stopping on runs from 2025-04-16. Three seeds.

    uv run --no-sync python e00303_head.py [BASE_ARM]
"""

import json
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import e00303_common as common
import e00303_ens_final as ens_final
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from torch import nn  # ty: ignore[unresolved-import]

SUMMARY_DIR: Final[Path] = (
    common.ML_EXPERIMENTS_DATA_DIR / "00259-wind-plant-plus-quantile-offset-head" / "work" / "base"
)
FINAL_DIR: Final[Path] = common.WORK_DIR / "final"
HINDCAST_DIR: Final[Path] = common.WORK_DIR / "final_hindcast"
SEEDS: Final[tuple[int, ...]] = (0, 1, 2)
CANDIDATES: Final[tuple[str, ...]] = ("P2", "P4", "P5")
TRAIN_CUTOFF: Final[datetime] = datetime(2025, 4, 16, tzinfo=UTC)
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


class Head(nn.Module):
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


def pinball(*, quantiles: torch.Tensor, actual: torch.Tensor, levels: torch.Tensor) -> torch.Tensor:
    """Return the mean pinball loss over cases and the 11 levels."""
    error = actual[:, None] - quantiles
    return torch.maximum(levels * error, (levels - 1.0) * error).mean()


def features(*, frame: pl.DataFrame, base: np.ndarray) -> np.ndarray:
    """Return the head inputs: base quantiles, member summaries, lead, and farm."""
    lead = frame["lead_hours"].to_numpy()[:, None] / 100.0
    farm = np.stack(
        [(frame["site"] == s).to_numpy().astype(np.float64) for s in ("W1", "W2", "W3")], axis=1
    )
    summary = frame.select(SUMMARY_FEATURES).to_numpy().astype(np.float64)
    return np.concatenate([base.astype(np.float64), summary, lead, farm], axis=1)


def with_summaries(*, cases: pl.DataFrame, set_name: str) -> pl.DataFrame:
    """Join experiment 259's member summaries to `cases` on the case key."""
    summaries = pl.read_parquet(SUMMARY_DIR / f"summaries_{set_name}.parquet").select(
        *common.KEY_COLUMNS, "masked", *SUMMARY_FEATURES
    )
    joined = cases.join(
        summaries, on=list(common.KEY_COLUMNS), how="left", validate="1:1", maintain_order="left"
    )
    if joined["ws_mean"].null_count():
        msg = f"{set_name}: some cases have no member summaries"
        raise ValueError(msg)
    return joined


def train_head(
    *,
    hindcast: pl.DataFrame,
    base_h: np.ndarray,
    scored: pl.DataFrame,
    base_s: np.ndarray,
    seed: int,
) -> tuple[np.ndarray, float]:
    """Return the scored quantiles of one trained head and its best validation loss."""
    torch.manual_seed(seed)
    init, valid = hindcast["power_fcst_init_time"], hindcast["valid_time"]
    usable = (~hindcast["masked"]).to_numpy()
    train_rows = usable & ((init < TRAIN_CUTOFF) & (valid < TRAIN_CUTOFF)).to_numpy()
    validation_rows = usable & (init >= TRAIN_CUTOFF).to_numpy()
    x_all = features(frame=hindcast, base=base_h)
    mean = x_all[train_rows].mean(axis=0)
    std = np.maximum(x_all[train_rows].std(axis=0), 1e-6)

    def standardise(x: np.ndarray) -> torch.Tensor:
        return torch.tensor((x - mean) / std, dtype=torch.float32, device=DEVICE)

    base_t = torch.tensor(base_h, dtype=torch.float32, device=DEVICE)
    actual_t = torch.tensor(hindcast["actual_norm"].to_numpy(), dtype=torch.float32, device=DEVICE)
    x_t = standardise(x_all)
    levels = torch.tensor(common.LEVELS, dtype=torch.float32, device=DEVICE)
    model = Head(n_in=x_t.shape[1]).to(DEVICE)
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
                quantiles=model(x_t[batch], base_t[batch]), actual=actual_t[batch], levels=levels
            )
            optimiser.zero_grad(set_to_none=True)
            loss.backward()
            optimiser.step()
        with torch.no_grad():
            score = float(
                pinball(
                    quantiles=model(x_t[validation_index], base_t[validation_index]),
                    actual=actual_t[validation_index],
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
    x_s = standardise(features(frame=scored, base=base_s))
    base_s_t = torch.tensor(base_s, dtype=torch.float32, device=DEVICE)
    with torch.no_grad():
        return model(x_s, base_s_t).cpu().numpy(), best


def main() -> None:
    """Choose the base arm on the hindcast score, train the heads, and store P6."""
    sites = common.site_table()
    hind = ens_final.hindcast_cases(sites=sites)
    usable = hind["actual_norm"].is_finite().fill_null(value=False).to_numpy()
    hind_usable = hind.filter(pl.Series(usable))
    scores = json.loads((FINAL_DIR / "hindcast_scores.json").read_text())
    pool = {a: scores[a] for a in CANDIDATES if a in scores}
    base_arm = sys.argv[1] if len(sys.argv) > 1 else min(pool, key=lambda a: pool[a])
    print("hindcast scores of the candidates", pool, "-> base", base_arm, flush=True)
    hindcast = with_summaries(cases=hind_usable, set_name="hindcast")
    scored = with_summaries(cases=ens_final.scored_cases(sites=sites), set_name="scored")
    base_hindcast = np.load(HINDCAST_DIR / f"{base_arm}.npy")
    base_scored = np.load(FINAL_DIR / f"{base_arm}.npy")
    outputs, losses = [], {}
    for seed in SEEDS:
        quantiles, loss = train_head(
            hindcast=hindcast,
            base_h=base_hindcast[seed],
            scored=scored,
            base_s=base_scored[seed],
            seed=seed,
        )
        outputs.append(quantiles.astype(np.float32))
        losses[str(seed)] = loss
        print(f"P6 seed {seed} validation pinball {loss:.5f}", flush=True)
    np.save(FINAL_DIR / "P6.npy", np.stack(outputs))
    (FINAL_DIR / "p6_facts.json").write_text(json.dumps({"base_arm": base_arm, "loss": losses}))


if __name__ == "__main__":
    main()
