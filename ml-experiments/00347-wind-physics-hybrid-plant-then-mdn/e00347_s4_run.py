"""Stage 4, step 2 of experiment 00347: train the mixture-density heads on the plants' summaries.

Each head is experiment 00310's three-component Gaussian mixture on the residual from the plant
ensemble's median (the same class experiments 00324 and 00339 used), with 11 quantile outputs read
from the mixture's CDF. Inputs per arm:

- y1, y2, y4: the median, mean, standard deviation, 10th and 90th percentile, and saturation share
  of the plant's 51 member powers, plus the sine and cosine of the lead's hour of day, the lead in
  hundreds of hours, and a one-hot farm. The plant is Stage 2's curve (y1), Stage 3's plant (y2), or
  the pass-2 plant of Stage 3b (y4).
- y3: y2's plant, with only the mean and standard deviation plus the lead terms and the farm.
- base: y2's plant, with only the median plus the lead terms and no farm columns (the unequal
  first base-only control, kept for the record).
- basefarm: y2's plant, with only the median plus the lead terms and the one-hot farm (the
  base-only control with the same farm columns as y1 to y4).

One pooled head per arm and seed is trained on the kept hindcast half-hours of the three farms
(runs and valid times before 2025-04-16), early stopping on the kept half-hours of the later runs.
The replicate trains on runs and valid times before 2024-12-01, stops on December 2024, and scores
the daily runs of 2025-01-01 to 2025-06-16. Leave-one-farm-out trains on the other two farms
without the farm columns. The shuffled-summary placebo predicts the scored cases with the input
columns (the median included, so the centre is shuffled too) taken from another month of the same
farm. The spread-only placebo (`--placebo-spread`) keeps the true median and mean and shuffles the
other four columns.

    PYTHONPATH=<347>:<230>:<303>:<310>:<326>:<336 folders> \
    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00347-*/e00347_s4_run.py [--first-only]
"""

import os
import sys
import time
from datetime import datetime
from typing import Final

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import e00310_fit as f310  # noqa: E402  # isort: skip  # ty: ignore[unresolved-import]
import e00326_common as c326  # noqa: E402  # isort: skip  # ty: ignore[unresolved-import]
import e00347_s4_data as data  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import torch  # noqa: E402  # ty: ignore[unresolved-import]

DEVICE: Final = torch.device(
    "cpu" if os.environ.get("E00347_CPU") else ("cuda" if torch.cuda.is_available() else "cpu")
)
EPOCHS: Final[int] = int(os.environ.get("E00347_EPOCHS", "40"))
BATCH: Final[int] = 2048
LEARNING_RATE: Final[float] = 2e-3
WEIGHT_DECAY: Final[float] = 1e-4
SEEDS: Final[tuple[int, ...]] = (0, 1, 2, 3, 4)
REPLICATE_SEEDS: Final[tuple[int, ...]] = (0, 1, 2)
ARMS: Final[dict[str, tuple[str, str]]] = {
    "y1": ("y1", "full"),
    "y2": ("y2", "full"),
    "y3": ("y2", "minimal"),
    "y4": ("y4", "full"),
    "base": ("y2", "base"),
    "basefarm": ("y2", "basefarm"),
}
"""Arm -> (frozen plant, input set)."""
LOFO_ARMS: Final[tuple[str, ...]] = ("y2", "y4")
PLACEBO_SEED: Final[int] = 347


def save(*, name: str, array: np.ndarray) -> None:
    """Write an array to the parts folder atomically."""
    data.PARTS_DIR.mkdir(parents=True, exist_ok=True)
    path = data.PARTS_DIR / f"{name}.npy"
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as handle:
        np.save(handle, array.astype(np.float32))
    tmp.replace(path)


def build_features(
    *, summary: np.ndarray, frame: pl.DataFrame, kind: str, farm_columns: bool
) -> np.ndarray:
    """Return the head inputs of one arm."""
    hours = frame["lead_hours"].to_numpy().astype(np.float64)
    angle = 2 * np.pi * hours / 24.0
    lead = np.stack([np.sin(angle), np.cos(angle), hours / 100.0], axis=1)
    farm = np.stack(
        [(frame["site"] == f).to_numpy().astype(np.float64) for f in data.FARMS], axis=1
    )
    chosen = {
        "full": summary,
        "minimal": summary[:, [1, 2]],
        "base": summary[:, [0]],
        "basefarm": summary[:, [0]],
    }[kind]
    parts = [chosen.astype(np.float64), lead]
    if farm_columns and kind != "base":
        parts.append(farm)
    return np.concatenate(parts, axis=1)


def train_and_predict(
    *,
    hindcast: pl.DataFrame,
    summary_h: np.ndarray,
    rows: np.ndarray,
    cutoff: datetime,
    valid_limit: datetime | None,
    kind: str,
    farm_columns: bool,
    targets: dict[str, tuple[pl.DataFrame, np.ndarray]],
    seed: int,
) -> dict[str, np.ndarray]:
    """Train one head on the hindcast `rows` and return the 11 quantiles of each target set."""
    torch.manual_seed(seed)
    usable = rows & (~hindcast["masked"]).to_numpy()
    init, valid = hindcast["power_fcst_init_time"], hindcast["valid_time"]
    train = usable & ((init < cutoff) & (valid < cutoff)).to_numpy()
    stop = usable & (init >= cutoff).to_numpy()
    if valid_limit is not None:
        stop &= (valid < valid_limit).to_numpy()
    x_h = build_features(summary=summary_h, frame=hindcast, kind=kind, farm_columns=farm_columns)
    mean = x_h[train].mean(axis=0)
    std = np.maximum(x_h[train].std(axis=0), 1e-6)

    def tensor(x: np.ndarray) -> torch.Tensor:
        return torch.tensor((x - mean) / std, dtype=torch.float32, device=DEVICE)

    mu_t = torch.tensor(summary_h[:, 0], dtype=torch.float32, device=DEVICE)
    y_t = torch.tensor(hindcast["actual_norm"].to_numpy(), dtype=torch.float32, device=DEVICE)
    x_t = tensor(x_h)
    model = f310.MixtureHead(x_h.shape[1]).to(DEVICE)
    optimiser = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    train_i = torch.tensor(np.flatnonzero(train), device=DEVICE)
    stop_i = torch.tensor(np.flatnonzero(stop), device=DEVICE)
    generator = torch.Generator().manual_seed(seed)
    best, best_state = float("inf"), None
    for _ in range(EPOCHS):
        model.train()
        order = train_i[torch.randperm(len(train_i), generator=generator).to(DEVICE)]
        for start in range(0, len(order), BATCH):
            i = order[start : start + BATCH]
            loss = model.nll({"x": x_t[i], "mu": mu_t[i], "actual": y_t[i]})
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        model.eval()
        with torch.no_grad():
            score = float(model.nll({"x": x_t[stop_i], "mu": mu_t[stop_i], "actual": y_t[stop_i]}))
        if score < best:
            best, best_state = score, {k: v.detach().clone() for k, v in model.state_dict().items()}
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    out = {}
    for name, (frame, summary_s) in targets.items():
        xs_t = tensor(
            build_features(summary=summary_s, frame=frame, kind=kind, farm_columns=farm_columns)
        )
        result = np.empty((len(frame), 11), dtype=np.float32)
        for start in range(0, len(frame), 50_000):
            sl = slice(start, start + 50_000)
            batch = {
                "x": xs_t[sl],
                "mu": torch.tensor(summary_s[sl, 0], dtype=torch.float32, device=DEVICE),
            }
            result[sl] = f310.mixture_quantiles(model=model, batch=batch, ceiling=1.0)
        out[name] = np.sort(result, axis=1)
    return out


def shuffled(*, summary: np.ndarray, frame: pl.DataFrame, seed: int) -> np.ndarray:
    """Return `summary` with each farm's rows taken from another month of the same farm."""
    source = np.arange(len(frame))
    sites = frame["site"].to_numpy()
    valid = frame["valid_time"].dt.replace_time_zone(None).to_numpy()
    for farm in data.FARMS:
        rows = np.flatnonzero(sites == farm)
        source[rows] = rows[c326.month_shuffle_index(valid=valid[rows], seed=seed)]
    return summary[source]


def spread_placebo() -> None:
    """Refit y1 to y4 and score the scored cases with only the spread columns shuffled.

    The second placebo keeps the plant's median and mean (so the centre and the first input stay
    true) and takes the standard deviation, 10th and 90th percentile, and saturation share from
    another month of the same farm. It tests whether the spread summaries carry information.
    """
    started = time.time()
    hindcast = data.hindcast_frame()
    scored = data.scored_frame()
    everything = np.ones(len(hindcast), dtype=bool)
    for arm in ("y1", "y2", "y3", "y4"):
        plant_name, kind = ARMS[arm]
        summary_h = data.summaries(arm_plant=plant_name, set_name="hindcast")
        summary_s = data.summaries(arm_plant=plant_name, set_name="scored")
        mixed = summary_s.copy()
        mixed[:, 2:] = shuffled(summary=summary_s, frame=scored, seed=PLACEBO_SEED)[:, 2:]
        for seed in SEEDS:
            if (data.PARTS_DIR / f"{arm}_placebo2_h{seed}.npy").exists():
                continue
            out = train_and_predict(
                hindcast=hindcast,
                summary_h=summary_h,
                rows=everything,
                cutoff=data.MAIN_CUTOFF,
                valid_limit=None,
                kind=kind,
                farm_columns=True,
                targets={"placebo2": (scored, mixed)},
                seed=seed,
            )
            save(name=f"{arm}_placebo2_h{seed}", array=out["placebo2"])
            print(
                f"{arm} spread placebo seed {seed} done {time.time() - started:.0f} s", flush=True
            )
    print("all spread placebos done", flush=True)


def main() -> None:
    """Train every head and store its quantiles."""
    if "--placebo-spread" in sys.argv:
        spread_placebo()
        return
    started = time.time()
    hindcast = data.hindcast_frame()
    scored = data.scored_frame()
    in_rep = (hindcast["power_fcst_init_time"] >= data.REPLICATE_START).to_numpy()
    replicate = hindcast.filter(pl.Series(in_rep))
    cases_336 = pl.read_parquet(data.E336_FINAL / "replicate_cases.parquet")
    assert cases_336.height == replicate.height, (cases_336.height, replicate.height)
    assert (cases_336["masked"].to_numpy() == replicate["masked"].to_numpy()).all()
    site_h, site_s = hindcast["site"].to_numpy(), scored["site"].to_numpy()
    jobs = 0
    first_only = "--first-only" in sys.argv
    for arm, (plant_name, kind) in ARMS.items():
        summary_h = data.summaries(arm_plant=plant_name, set_name="hindcast")
        summary_s = data.summaries(arm_plant=plant_name, set_name="scored")
        assert len(summary_h) == len(hindcast), (len(summary_h), len(hindcast))
        assert len(summary_s) == len(scored), (len(summary_s), len(scored))
        placebo_s = shuffled(summary=summary_s, frame=scored, seed=PLACEBO_SEED)
        everything = np.ones(len(hindcast), dtype=bool)
        for seed in SEEDS:
            if not (data.PARTS_DIR / f"{arm}_h{seed}.npy").exists():
                out = train_and_predict(
                    hindcast=hindcast,
                    summary_h=summary_h,
                    rows=everything,
                    cutoff=data.MAIN_CUTOFF,
                    valid_limit=None,
                    kind=kind,
                    farm_columns=True,
                    targets={"scored": (scored, summary_s), "placebo": (scored, placebo_s)},
                    seed=seed,
                )
                save(name=f"{arm}_h{seed}", array=out["scored"])
                save(name=f"{arm}_placebo_h{seed}", array=out["placebo"])
                jobs += 1
                print(f"{arm} seed {seed} done {time.time() - started:.0f} s", flush=True)
                if first_only:
                    return
        for seed in REPLICATE_SEEDS:
            if not (data.PARTS_DIR / f"{arm}_rep_h{seed}.npy").exists():
                fit_rows = ~in_rep
                out = train_and_predict(
                    hindcast=hindcast.filter(pl.Series(fit_rows)),
                    summary_h=summary_h[fit_rows],
                    rows=np.ones(int(fit_rows.sum()), dtype=bool),
                    cutoff=data.REPLICATE_CUTOFF,
                    valid_limit=data.REPLICATE_START,
                    kind=kind,
                    farm_columns=True,
                    targets={"rep": (replicate, summary_h[in_rep])},
                    seed=seed,
                )
                save(name=f"{arm}_rep_h{seed}", array=out["rep"])
                print(f"{arm} replicate seed {seed} done {time.time() - started:.0f} s", flush=True)
        if arm in LOFO_ARMS:
            for farm in data.FARMS:
                if (data.PARTS_DIR / f"{arm}_lofo_{farm}_h0.npy").exists():
                    continue
                others = site_h != farm
                held = site_s == farm
                out = train_and_predict(
                    hindcast=hindcast.filter(pl.Series(others)),
                    summary_h=summary_h[others],
                    rows=np.ones(int(others.sum()), dtype=bool),
                    cutoff=data.MAIN_CUTOFF,
                    valid_limit=None,
                    kind=kind,
                    farm_columns=False,
                    targets={"lofo": (scored.filter(pl.Series(held)), summary_s[held])},
                    seed=0,
                )
                save(name=f"{arm}_lofo_{farm}_h0", array=out["lofo"])
                print(f"{arm} leave-out {farm} done {time.time() - started:.0f} s", flush=True)
    print("all heads done", flush=True)


if __name__ == "__main__":
    main()
