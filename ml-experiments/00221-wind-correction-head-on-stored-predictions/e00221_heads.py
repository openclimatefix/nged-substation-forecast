# ruff: noqa: E501,ANN001
"""Fit the correction heads (A1 to A4, pooled and leave-one-farm-out) and the table (A5).

For one seed, each scored month's heads are trained on the unmasked cases of all earlier months
(runs initialised before the month, valid times at least 4 hours before the month's first day) and
scored on the unmasked cases of the month's runs. The head emits a median shift `c` and a log-scale
`s`, and the quantiles are `median + c + exp(s) * (q - median)` of the base forecast, clipped to 0
to 1 and sorted. A0 and A4's head-free forecast are written on the same rows.

Output: `work/arms/<arm>_seed<seed>.parquet`. Existing outputs are kept.

    uv run --no-sync python ml-experiments/00221-wind-correction-head-on-stored-predictions/e00221_heads.py --seed 0
"""

import argparse
import time
from datetime import UTC, datetime
from typing import Final

import e00221_common as common
import numpy as np
import polars as pl
import torch  # ty: ignore[unresolved-import]
from e00221_common import KEY_COLUMNS, LEVEL_COLUMNS
from e00221_inputs import MEAN_ONLY_PREFIX, N_SECTORS, OUT_PATH
from torch import nn  # ty: ignore[unresolved-import]

ARMS_DIR: Final = common.WORK_DIR / "arms"
SCORED_MONTHS: Final[tuple[str, ...]] = (
    "2025-10",
    "2025-11",
    "2025-12",
    "2026-01",
    "2026-02",
    "2026-03",
    "2026-04",
    "2026-05",
    "2026-06",
)
PURGE_HOURS: Final[int] = 4
HIDDEN: Final[int] = 64
EPOCHS: Final[int] = 12
BATCH: Final[int] = 512
LEARNING_RATE: Final[float] = 2e-3
WEIGHT_DECAY: Final[float] = 1e-2
DROPOUT: Final[float] = 0.1
SHIFT_SCALE: Final[float] = 0.1
"""The median shift is the network's output times this, in fractions of capacity."""
LOG_SCALE_LIMIT: Final[float] = 0.7
TABLE_SHIFTS: Final[np.ndarray] = np.round(np.arange(-0.05, 0.0501, 0.005), 3)
HEAD_ARMS: Final[dict[str, dict[str, object]]] = {
    "A1": {"base": "bg", "weather": False, "calendar": False, "forecast_inputs": True},
    "A2": {"base": "bg", "weather": True, "calendar": False, "forecast_inputs": True},
    "A3": {"base": "bg", "weather": False, "calendar": True, "forecast_inputs": False},
    "A4": {"base": "mo", "weather": True, "calendar": False, "forecast_inputs": True},
}
LOFO_ARMS: Final[tuple[str, ...]] = ("A1", "A2", "A3")


def prepare(frame: pl.DataFrame) -> pl.DataFrame:
    """Add the run month, the base quantile arrays' columns, and the calendar columns."""
    angle_hour = (
        (pl.col("valid_time").dt.hour() + pl.col("valid_time").dt.minute() / 60.0)
        / 24.0
        * 2
        * np.pi
    )
    angle_year = (pl.col("valid_time").dt.ordinal_day() - 1) / 365.25 * 2 * np.pi
    return frame.with_columns(
        month=pl.col("power_fcst_init_time").dt.strftime("%Y-%m"),
        hour_sin=angle_hour.sin(),
        hour_cos=angle_hour.cos(),
        year_sin=angle_year.sin(),
        year_cos=angle_year.cos(),
        lead_scaled=pl.col("lead_hours") / 100.0,
    )


def base_quantiles(*, frame: pl.DataFrame, base: object) -> np.ndarray:
    """Return the base forecast's nine quantiles (fractions of capacity), float64."""
    prefix = MEAN_ONLY_PREFIX if base == "mo" else ""
    return frame.select([f"{prefix}{c}" for c in LEVEL_COLUMNS]).to_numpy().astype(np.float64)


def design(*, frame: pl.DataFrame, spec: dict[str, object]) -> np.ndarray:
    """Return the continuous input columns of an arm (one-hot columns last, not standardised)."""
    columns: list[np.ndarray] = []
    if spec["forecast_inputs"]:
        quantiles = base_quantiles(frame=frame, base=spec["base"])
        columns += [quantiles[:, i] for i in range(quantiles.shape[1])]
    lead = frame["lead_scaled"].to_numpy()
    columns += [lead, lead**2]
    if spec["calendar"]:
        columns += [frame[c].to_numpy() for c in ("hour_sin", "hour_cos", "year_sin", "year_cos")]
    if spec["weather"]:
        columns += [frame["ens_speed_mean"].to_numpy() / 10.0, frame["ens_speed_std"].to_numpy()]
    continuous = np.column_stack(columns).astype(np.float32)
    if spec["weather"]:
        sector = np.eye(N_SECTORS, dtype=np.float32)[frame["sector"].to_numpy()]
        return np.column_stack([continuous, sector])
    return continuous


class Head(nn.Module):
    """Two hidden layers and a linear skip; the output layers start at zero (the base forecast)."""

    def __init__(self, *, n_inputs: int, hidden: int = HIDDEN) -> None:
        """Build the network and zero its output layers; `hidden=0` gives a linear head."""
        super().__init__()
        self.body: nn.Module
        if hidden == 0:
            self.body = output = nn.Linear(n_inputs, 2)
        else:
            first, second = nn.Linear(n_inputs, hidden), nn.Linear(hidden, hidden)
            output = nn.Linear(hidden, 2)
            self.body = nn.Sequential(
                first, nn.GELU(), nn.Dropout(DROPOUT), second, nn.GELU(), output
            )
        self.skip = nn.Linear(n_inputs, 1)
        for layer in (output, self.skip):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the median shift and the log-scale."""
        out = self.body(x)
        shift = (out[:, 0] + self.skip(x)[:, 0]) * SHIFT_SCALE
        return shift, LOG_SCALE_LIMIT * torch.tanh(out[:, 1])


def apply_head(
    *, shift: torch.Tensor, log_scale: torch.Tensor, quantiles: torch.Tensor
) -> torch.Tensor:
    """Return the unsorted, clipped quantiles the head implies."""
    median = quantiles[:, common.MEDIAN_INDEX : common.MEDIAN_INDEX + 1]
    return torch.clamp(
        median + shift[:, None] + torch.exp(log_scale)[:, None] * (quantiles - median), 0.0, 1.0
    )


def pinball(*, y: torch.Tensor, predicted: torch.Tensor) -> torch.Tensor:
    """Return the mean over cases of the mass-weighted pinball loss of the nine levels."""
    levels = torch.tensor(common.LEVELS, dtype=torch.float32)
    masses = torch.tensor(common.ATOM_MASSES, dtype=torch.float32)
    difference = y[:, None] - predicted
    return (
        (torch.maximum(levels * difference, (levels - 1) * difference) * masses).sum(dim=1).mean()
    )


def fit_predict(
    *,
    spec: dict[str, object],
    train: pl.DataFrame,
    test: pl.DataFrame,
    torch_seed: int,
    epochs: int = EPOCHS,
    hidden: int = HIDDEN,
) -> np.ndarray:
    """Train a head on `train` and return its nine sorted quantiles for `test`."""
    return fit_both(
        spec=spec, train=train, test=test, torch_seed=torch_seed, epochs=epochs, hidden=hidden
    )[0]


def fit_both(
    *,
    spec: dict[str, object],
    train: pl.DataFrame,
    test: pl.DataFrame,
    torch_seed: int,
    epochs: int = EPOCHS,
    hidden: int = HIDDEN,
) -> tuple[np.ndarray, np.ndarray]:
    """Train a head and return its sorted quantiles for `test` and for `train` (in-sample)."""
    x_train = design(frame=train, spec=spec)
    x_test = design(frame=test, spec=spec)
    n_continuous = x_train.shape[1] - (N_SECTORS if spec["weather"] else 0)
    mean = x_train[:, :n_continuous].mean(axis=0)
    std = x_train[:, :n_continuous].std(axis=0) + 1e-6
    for x in (x_train, x_test):
        x[:, :n_continuous] = (x[:, :n_continuous] - mean) / std
    q_train = torch.from_numpy(base_quantiles(frame=train, base=spec["base"]).astype(np.float32))
    q_test = torch.from_numpy(base_quantiles(frame=test, base=spec["base"]).astype(np.float32))
    y_train = torch.from_numpy(train["actual_norm"].to_numpy().astype(np.float32))
    xt = torch.from_numpy(x_train)
    torch.manual_seed(torch_seed)
    rng = np.random.default_rng(torch_seed)
    net = Head(n_inputs=xt.shape[1], hidden=hidden)
    optimiser = torch.optim.AdamW(net.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    for _ in range(epochs):
        net.train()
        order = torch.from_numpy(rng.permutation(len(y_train)))
        for start in range(0, len(order), BATCH):
            batch = order[start : start + BATCH]
            shift, log_scale = net(xt[batch])
            loss = pinball(
                y=y_train[batch],
                predicted=apply_head(shift=shift, log_scale=log_scale, quantiles=q_train[batch]),
            )
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
    net.eval()
    with torch.no_grad():
        shift, log_scale = net(torch.from_numpy(x_test))
        out = apply_head(shift=shift, log_scale=log_scale, quantiles=q_test).numpy()
        shift, log_scale = net(xt)
        in_sample = apply_head(shift=shift, log_scale=log_scale, quantiles=q_train).numpy()
    return np.sort(out, axis=1), np.sort(in_sample, axis=1)


def table_quantiles(*, train: pl.DataFrame, test: pl.DataFrame) -> np.ndarray:
    """Return A5: per lead-day band, a median shift and an offset scale fitted on `train`."""
    out = base_quantiles(frame=test, base="bg")
    for _, low, high in common.LEAD_DAY_BANDS:
        in_train = ((train["lead_day"] >= low) & (train["lead_day"] <= high)).to_numpy()
        in_test = ((test["lead_day"] >= low) & (test["lead_day"] <= high)).to_numpy()
        if not in_test.any():
            continue
        base = base_quantiles(frame=train, base="bg")[in_train]
        y = train["actual_norm"].to_numpy()[in_train]
        median = base[:, common.MEDIAN_INDEX : common.MEDIAN_INDEX + 1]

        def loss(*, shift: float, scale: float, base=base, y=y, median=median) -> float:
            adjusted = np.clip(median + shift + scale * (base - median), 0.0, 1.0)
            return float(
                common.quantile_score(quantiles=np.sort(adjusted, axis=1), actual=y).mean()
            )

        scale = min(common.SPREAD_GRID, key=lambda k: loss(shift=0.0, scale=float(k)))
        shift = min(TABLE_SHIFTS, key=lambda d: loss(shift=float(d), scale=float(scale)))
        scale = min(common.SPREAD_GRID, key=lambda k: loss(shift=float(shift), scale=float(k)))
        test_base = out[in_test]
        test_median = test_base[:, common.MEDIAN_INDEX : common.MEDIAN_INDEX + 1]
        out[in_test] = np.sort(
            np.clip(test_median + shift + scale * (test_base - test_median), 0.0, 1.0), axis=1
        )
    return out


def write_arm(*, name: str, seed: int, pieces: list[pl.DataFrame]) -> None:
    """Write one arm's rows for one seed, atomically."""
    target = ARMS_DIR / f"{name}_seed{seed}.parquet"
    pl.concat(pieces).sort(*KEY_COLUMNS).write_parquet(target.with_suffix(".tmp"))
    target.with_suffix(".tmp").rename(target)


def arm_rows(*, test: pl.DataFrame, quantiles: np.ndarray, seed: int) -> pl.DataFrame:
    """Return the test cases' keys, actual, and the arm's nine quantiles."""
    return (
        test.select(*KEY_COLUMNS, "lead_day", "lead_hours", "actual_norm", "site")
        .with_columns(seed=pl.lit(seed, dtype=pl.Int32))
        .hstack(pl.DataFrame(quantiles.astype(np.float32), schema=list(LEVEL_COLUMNS)))
    )


def main() -> None:
    """Fit all arms for one seed."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--smoke", action="store_true", help="two months, no output")
    args = parser.parse_args()
    seed = args.seed
    torch.set_num_threads(2)
    ARMS_DIR.mkdir(parents=True, exist_ok=True)
    frame = prepare(pl.read_parquet(OUT_PATH).filter((pl.col("seed") == seed) & ~pl.col("masked")))
    pieces: dict[str, list[pl.DataFrame]] = {}

    def add(name: str, rows: pl.DataFrame) -> None:
        pieces.setdefault(name, []).append(rows)

    months = SCORED_MONTHS[:2] if args.smoke else SCORED_MONTHS
    sites = sorted(frame["site"].unique().to_list())
    started = time.monotonic()
    for month_index, month in enumerate(months):
        year, mon = (int(part) for part in month.split("-"))
        start = datetime(year, mon, 1, tzinfo=UTC)
        train = frame.filter(
            (pl.col("power_fcst_init_time") < start)
            & (pl.col("valid_time") < start - np.timedelta64(PURGE_HOURS, "h"))
        )
        test = frame.filter(pl.col("month") == month)
        add("A0", arm_rows(test=test, quantiles=base_quantiles(frame=test, base="bg"), seed=seed))
        add(
            "A4_mean_only",
            arm_rows(test=test, quantiles=base_quantiles(frame=test, base="mo"), seed=seed),
        )
        add("A5", arm_rows(test=test, quantiles=table_quantiles(train=train, test=test), seed=seed))
        torch_seed = seed * 1000 + month_index
        for arm, spec in HEAD_ARMS.items():
            add(
                arm,
                arm_rows(
                    test=test,
                    quantiles=fit_predict(spec=spec, train=train, test=test, torch_seed=torch_seed),
                    seed=seed,
                ),
            )
        for arm in LOFO_ARMS:
            for held_out in sites:
                held_test = test.filter(pl.col("site") == held_out)
                held_train = train.filter(pl.col("site") != held_out)
                add(
                    f"{arm}_lofo",
                    arm_rows(
                        test=held_test,
                        quantiles=fit_predict(
                            spec=HEAD_ARMS[arm],
                            train=held_train,
                            test=held_test,
                            torch_seed=torch_seed,
                        ),
                        seed=seed,
                    ),
                )
        print(
            f"seed {seed} {month}: train {train.height}, test {test.height}, "
            f"{time.monotonic() - started:.0f} s",
            flush=True,
        )
    if args.smoke:
        print("smoke run: no output written", flush=True)
        return
    for name, parts in pieces.items():
        write_arm(name=name, seed=seed, pieces=parts)


if __name__ == "__main__":
    main()
