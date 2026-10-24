# ruff: noqa: E501, D102, D107
"""Experiment 324: train the mixture-density heads on the wind adapter's quantiles and store arms.

Arms stored as `final/<arm>.npy`, shape (3 seeds, scored cases, 11), in the case order of
`e00303_score.ordered_cases`. W0 climatology; W1 experiment 303's P1; W2 and W3 experiment 313's
pure XGBoost and pure mixture network (own history, 51 members); W4 to W7 and the leave-one-farm-out
arms train a three-component mixture head on the hindcast cases (inputs: experiment 303's P2
quantiles, the members' summaries, and lead features); W8 applies experiment 321's offset table.
One spawned process per seed; arms are written atomically so a crash costs one seed.

    PYTHONPATH=ml-experiments/00303-wind-adapter-in-production-setting:ml-experiments/00310-metered-farm-long-lead-scoring:ml-experiments/00321-metered-farm-lower-tail-recalibration-and-tail-head \
    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00324-*/e00324_run.py
"""

import multiprocessing
import os
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "2")

import e00303_common as c303  # noqa: E402  # ty: ignore[unresolved-import]
import e00303_ens_final as ens_final  # noqa: E402  # ty: ignore[unresolved-import]
import e00303_head as head303  # noqa: E402  # ty: ignore[unresolved-import]
import e00303_score as score303  # noqa: E402  # ty: ignore[unresolved-import]
import e00310_fit as f310  # noqa: E402  # ty: ignore[unresolved-import]
import e00321_recal as recal  # noqa: E402  # ty: ignore[unresolved-import]
import e00324_common as common  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

TRAIN_CUTOFF: Final[datetime] = datetime(2025, 4, 16, tzinfo=UTC)
EPOCHS: Final[int] = int(os.environ.get("E00324_EPOCHS", "40"))
BATCH: Final[int] = 2048
LEARNING_RATE: Final[float] = 2e-3
WEIGHT_DECAY: Final[float] = 1e-4
STOP_MARGIN: Final[float] = 0.3
BANDS: Final[tuple[tuple[str, int, int], ...]] = (
    ("d0-2", 0, 2),
    ("d3-6", 3, 6),
    ("d7+", 7, 10_000),
)
SUMMARY: Final[tuple[str, ...]] = head303.SUMMARY_FEATURES
DEVICE: Final = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class EmbeddedMixture(f310.MixtureHead):
    """Mixture head whose last input column is a farm index, mapped to a learned embedding."""

    def __init__(self, n_in: int) -> None:
        super().__init__(n_in + 4)
        self.embedding = nn.Embedding(3, 4)

    def parameters_of(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        return super().parameters_of(torch.cat([x[:, :-1], self.embedding(x[:, -1].long())], 1))


def save(*, path: Path, array: np.ndarray) -> None:
    """Write an array atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as handle:
        np.save(handle, array.astype(np.float32))
    tmp.replace(path)


def build_features(
    *, frame: pl.DataFrame, base: np.ndarray, lead: bool, farm_index: bool
) -> np.ndarray:
    """Return head inputs: base quantiles, member summaries, optional lead terms and farm index."""
    parts = [base.astype(np.float64), frame.select(SUMMARY).to_numpy().astype(np.float64)]
    if lead:
        hours = frame["lead_hours"].to_numpy().astype(np.float64)
        angle = 2 * np.pi * hours / 24.0
        parts += [np.sin(angle)[:, None], np.cos(angle)[:, None], hours[:, None] / 100.0]
    if farm_index:
        parts.append(
            frame["site"].replace_strict(list(common.FARMS), [0, 1, 2]).to_numpy()[:, None]
        )
    return np.concatenate(parts, axis=1)


def train_and_predict(
    *,
    hindcast: pl.DataFrame,
    base_h: np.ndarray,
    rows_h: np.ndarray,
    scored: pl.DataFrame,
    base_s: np.ndarray,
    seed: int,
    lead: bool,
    embed: bool,
) -> np.ndarray:
    """Train one head on the hindcast `rows_h` and return its scored cases' 11 quantiles."""
    torch.manual_seed(seed)
    usable = rows_h & (~hindcast["masked"]).to_numpy()
    init, valid = hindcast["power_fcst_init_time"], hindcast["valid_time"]
    train = usable & ((init < TRAIN_CUTOFF) & (valid < TRAIN_CUTOFF)).to_numpy()
    validate = usable & (init >= TRAIN_CUTOFF).to_numpy()
    x_h = build_features(frame=hindcast, base=base_h, lead=lead, farm_index=embed)
    x_s = build_features(frame=scored, base=base_s, lead=lead, farm_index=embed)
    n_cont = x_h.shape[1] - (1 if embed else 0)
    mean = x_h[train, :n_cont].mean(axis=0)
    std = np.maximum(x_h[train, :n_cont].std(axis=0), 1e-6)

    def tensor(x: np.ndarray) -> torch.Tensor:
        z = x.copy()
        z[:, :n_cont] = (z[:, :n_cont] - mean) / std
        return torch.tensor(z, dtype=torch.float32, device=DEVICE)

    mu_h = base_h[:, c303.MEDIAN_INDEX]
    actual = hindcast["actual_norm"].to_numpy()
    x_t, xs_t = tensor(x_h), tensor(x_s)
    mu_t = torch.tensor(mu_h, dtype=torch.float32, device=DEVICE)
    y_t = torch.tensor(np.nan_to_num(actual), dtype=torch.float32, device=DEVICE)
    model: f310.MixtureHead = (EmbeddedMixture(n_cont) if embed else f310.MixtureHead(n_cont)).to(
        DEVICE
    )
    optimiser = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    train_i = torch.tensor(np.flatnonzero(train), device=DEVICE)
    val_i = torch.tensor(np.flatnonzero(validate), device=DEVICE)
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
            score = float(model.nll({"x": x_t[val_i], "mu": mu_t[val_i], "actual": y_t[val_i]}))
        if score < best:
            best, best_state = score, {k: v.detach().clone() for k, v in model.state_dict().items()}
    assert best_state is not None
    model.load_state_dict(best_state)
    model.eval()
    out = np.empty((len(scored), 11), dtype=np.float32)
    for start in range(0, len(scored), 50_000):
        sl = slice(start, start + 50_000)
        batch = {
            "x": xs_t[sl],
            "mu": torch.tensor(base_s[sl, c303.MEDIAN_INDEX], dtype=torch.float32, device=DEVICE),
        }
        out[sl] = f310.mixture_quantiles(model=model, batch=batch, ceiling=1.0)
    return np.sort(out, axis=1)


def climatology(*, hindcast: pl.DataFrame, scored: pl.DataFrame) -> np.ndarray:
    """Return per farm, month, and hour empirical quantiles of the hindcast year's observed power."""
    keys = ["site", "month", "hour"]

    def add(frame: pl.DataFrame) -> pl.DataFrame:
        return frame.with_columns(
            month=pl.col("valid_time").dt.month(), hour=pl.col("valid_time").dt.hour()
        )

    usable = add(hindcast).filter(~pl.col("masked") & pl.col("actual_norm").is_finite())
    table = usable.group_by(keys).agg(
        [
            pl.col("actual_norm").quantile(level, interpolation="linear").alias(f"q{k}")
            for k, level in enumerate(c303.LEVELS)
        ]
    )
    joined = add(scored).select(keys).join(table, on=keys, how="left", maintain_order="left")
    return np.sort(joined.select([f"q{k}" for k in range(11)]).to_numpy(), axis=1)


def recalibrated(*, cases: pl.DataFrame, quantiles: np.ndarray) -> np.ndarray:
    """Apply experiment 321's rolling-origin offset table to one seed's kept-row quantiles."""
    kept = ~cases["masked"].to_numpy()
    frame = pl.DataFrame(
        {
            "ix": np.flatnonzero(kept),
            "time_series_id": cases["site"].filter(pl.Series(kept)),
            "time": cases["valid_time"].filter(pl.Series(kept)),
            "init_time": cases["power_fcst_init_time"].filter(pl.Series(kept)),
            "lead_hours": cases["lead_hours"].filter(pl.Series(kept)).cast(pl.Float64),
            "actual": cases["actual_norm"].filter(pl.Series(kept)).cast(pl.Float64),
        }
    ).hstack(pl.DataFrame(quantiles[kept].astype(np.float64), schema=recal.QCOLS))
    out = recal.recalibrate(frame=frame, variant="recal", lead_origin_hours=0.0)
    result = quantiles.copy()
    result[out["ix"].to_numpy()] = out.select(recal.QCOLS).to_numpy()
    return result


def seed_worker(seed: int) -> None:
    """Train and store every head arm for one seed."""
    torch.set_num_threads(2)
    started = time.time()
    sites = c303.site_table()
    hind_all = ens_final.hindcast_cases(sites=sites)
    usable = hind_all["actual_norm"].is_finite().fill_null(value=False).to_numpy()
    hindcast = head303.with_summaries(cases=hind_all.filter(pl.Series(usable)), set_name="hindcast")
    scored = head303.with_summaries(cases=ens_final.scored_cases(sites=sites), set_name="scored")
    base_h = np.load(common.E303_WORK / "final_hindcast" / "P2.npy")[seed]
    base_s = np.load(common.E303_WORK / "final" / "P2.npy")[seed]
    assert len(base_h) == len(hindcast), (len(base_h), len(hindcast))
    assert len(base_s) == len(scored), (len(base_s), len(scored))
    site_h, site_s = hindcast["site"].to_numpy(), scored["site"].to_numpy()
    day_h, day_s = hindcast["lead_day"].to_numpy(), scored["lead_day"].to_numpy()
    everything = np.ones(len(hindcast), dtype=bool)

    def path(arm: str) -> Path:
        return common.FINAL_DIR / "parts" / f"{arm}_s{seed}.npy"

    def stitched(*, pieces: list[tuple[np.ndarray, np.ndarray]]) -> np.ndarray:
        out = np.full((len(scored), 11), np.nan, dtype=np.float32)
        for rows, quantiles in pieces:
            out[rows] = quantiles
        return out

    def fit(*, rows_h: np.ndarray, rows_s: np.ndarray, lead: bool, embed: bool) -> tuple:
        sub_h, sub_s = hindcast.filter(pl.Series(rows_h)), scored.filter(pl.Series(rows_s))
        prediction = train_and_predict(
            hindcast=sub_h,
            base_h=base_h[rows_h],
            rows_h=np.ones(len(sub_h), dtype=bool),
            scored=sub_s,
            base_s=base_s[rows_s],
            seed=seed,
            lead=lead,
            embed=embed,
        )
        return rows_s, prediction

    arms: dict[str, list] = {}
    for label, lead in (("W4", False), ("W5", True)):
        arms[label] = [
            fit(rows_h=site_h == f, rows_s=site_s == f, lead=lead, embed=False)
            for f in common.FARMS
        ]
        save(path=path(label), array=stitched(pieces=arms[label]))
        print(f"seed {seed} {label} done {time.time() - started:.0f} s", flush=True)
        if label == "W4":
            cases = score303.ordered_cases()
            actual = cases["actual_norm"].to_numpy()
            kept = ~cases["masked"].to_numpy()
            w4 = np.load(path("W4"))
            p1 = np.load(common.E303_WORK / "final" / "P1.npy")[seed]
            gap = (
                score303.common.quantile_score(
                    quantiles=w4[kept].astype(np.float64), actual=actual[kept]
                )
                - score303.common.quantile_score(
                    quantiles=p1[kept].astype(np.float64), actual=actual[kept]
                )
            ).mean() * 100.0
            print(f"seed {seed} W4 minus W1 on kept rows: {gap:.3f}", flush=True)
            if gap > STOP_MARGIN:
                (common.FINAL_DIR / f"STOPPED_s{seed}").write_text(f"{gap}")
                return
    save(
        path=path("W6"),
        array=stitched(
            pieces=[
                fit(rows_h=everything, rows_s=np.ones(len(scored), bool), lead=True, embed=True)
            ]
        ),
    )
    print(f"seed {seed} W6 done {time.time() - started:.0f} s", flush=True)
    pieces = [
        fit(
            rows_h=(day_h >= lo) & (day_h <= hi),
            rows_s=(day_s >= lo) & (day_s <= hi),
            lead=True,
            embed=True,
        )
        for _, lo, hi in BANDS
    ]
    save(path=path("W7"), array=stitched(pieces=pieces))
    print(f"seed {seed} W7 done {time.time() - started:.0f} s", flush=True)
    pieces = [
        fit(rows_h=site_h != f, rows_s=site_s == f, lead=True, embed=False) for f in common.FARMS
    ]
    save(path=path("W5_LOFO"), array=stitched(pieces=pieces))
    print(f"seed {seed} W5_LOFO done {time.time() - started:.0f} s", flush=True)


def assemble() -> None:
    """Join the per-seed parts, add the loaded and derived arms, and store `final/<arm>.npy`."""
    cases = score303.ordered_cases()
    sites = c303.site_table()
    hind_all = ens_final.hindcast_cases(sites=sites)
    usable = hind_all["actual_norm"].is_finite().fill_null(value=False).to_numpy()
    hindcast = head303.with_summaries(cases=hind_all.filter(pl.Series(usable)), set_name="hindcast")
    out = common.FINAL_DIR
    out.mkdir(parents=True, exist_ok=True)
    for arm in ("W4", "W5", "W6", "W7", "W5_LOFO"):
        files = [out / "parts" / f"{arm}_s{s}.npy" for s in common.SEEDS]
        if all(f.exists() for f in files):
            save(path=out / f"{arm}.npy", array=np.stack([np.load(f) for f in files]))
    clim = climatology(hindcast=hindcast, scored=cases)
    save(path=out / "W0.npy", array=np.stack([clim] * len(common.SEEDS)))
    for arm in ("P1", "P2", "P3"):
        save(path=out / f"{arm}.npy", array=np.load(common.E303_WORK / "final" / f"{arm}.npy"))
    for label, name in (("W2", "pure_xgb"), ("W3", "pure_mdn")):
        stack = [
            np.concatenate(
                [
                    np.load(common.E313_WIND / f"{name}__own_history__{f}__s{seed}__ens.npy")
                    for f in common.FARMS
                ]
            )
            for seed in common.SEEDS
        ]
        array = np.stack(stack)
        assert array.shape[1] == len(cases), (array.shape, len(cases))
        save(path=out / f"{label}.npy", array=array)
    for label, source in (("W8_W1", "P1"), ("W8_W4", "W4")):
        base = np.load(out / f"{source}.npy")
        save(
            path=out / f"{label}.npy",
            array=np.stack([recalibrated(cases=cases, quantiles=q) for q in base]),
        )


def main() -> None:
    """Run one process per seed, then assemble the arms."""
    (common.FINAL_DIR / "parts").mkdir(parents=True, exist_ok=True)
    if "--assemble-only" not in sys.argv:
        context = multiprocessing.get_context("spawn")
        processes = [context.Process(target=seed_worker, args=(s,)) for s in common.SEEDS]
        for p in processes:
            p.start()
        for p in processes:
            p.join()
        if any(p.exitcode for p in processes):
            msg = "a worker failed"
            raise SystemExit(msg)
    assemble()
    print("assembled", flush=True)


if __name__ == "__main__":
    main()
