"""Experiment 339: train the mixture-density heads of arms x1 and x3 and store their quantiles.

Each task is one (base seed, head seed). The main task (head seed = base seed) stores, per farm:
x1 (head on kept hindcast rows, as experiment 324's W4), x3 (head on every hindcast half-hour),
x1 and x3 again for the replicate period, the shuffled-input placebo of x1, and the
leave-one-farm-out x1. Seed-check tasks store x1 only. The head code is experiment 324's
`train_and_predict`, with three changes: the training rows can include the idle rows, the early
stopping cutoff is a parameter, and several scored sets can be predicted by one fitted head.

    PYTHONPATH=<303>:<310>:<321>:<324>:<326>:<336 folders> \
    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00339-*/e00339_run.py
"""

import concurrent.futures
import multiprocessing
import os
import sys
import time
from datetime import datetime
from pathlib import Path

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "2")

import e00303_common as c303  # noqa: E402  # isort: skip  # ty: ignore[unresolved-import]
import e00303_ens_final as ens_final  # noqa: E402  # isort: skip  # ty: ignore[unresolved-import]
import e00303_head as head303  # noqa: E402  # isort: skip  # ty: ignore[unresolved-import]
import e00310_fit as f310  # noqa: E402  # isort: skip  # ty: ignore[unresolved-import]
import e00324_run as run324  # noqa: E402  # isort: skip  # ty: ignore[unresolved-import]
import e00326_common as c326  # noqa: E402  # isort: skip  # ty: ignore[unresolved-import]
import e00339_common as common  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
import torch  # noqa: E402  # ty: ignore[unresolved-import]

DEVICE = run324.DEVICE


def save(*, path: Path, array: np.ndarray) -> None:
    """Write an array atomically."""
    run324.save(path=path, array=array)


def train_and_predict(
    *,
    hindcast: pl.DataFrame,
    base_h: np.ndarray,
    train_on_idle: bool,
    cutoff: datetime,
    valid_limit: datetime | None,
    targets: dict[str, tuple[pl.DataFrame, np.ndarray]],
    seed: int,
) -> dict[str, np.ndarray]:
    """Train one head on the hindcast and return the 11 quantiles of every target set.

    Args:
        hindcast: Hindcast cases of the training farms, with member summaries.
        base_h: Experiment 303's P2 quantiles on `hindcast` for one base seed.
        train_on_idle: Train on the experiment 210 idle rows too (arm x3).
        cutoff: Runs initialised and valid before this train; runs initialised on or after it
            stop the fit early.
        valid_limit: Early-stopping rows must also be valid before this, or None.
        targets: Name -> (case frame, base quantiles) to predict.
        seed: Seeds the head.

    Returns:
        Name -> quantiles, shape (cases, 11), each row sorted.
    """
    torch.manual_seed(seed)
    usable = (
        np.ones(len(hindcast), dtype=bool) if train_on_idle else (~hindcast["masked"]).to_numpy()
    )
    init, valid = hindcast["power_fcst_init_time"], hindcast["valid_time"]
    train = usable & ((init < cutoff) & (valid < cutoff)).to_numpy()
    stop = usable & (init >= cutoff).to_numpy()
    if valid_limit is not None:
        stop &= (valid < valid_limit).to_numpy()
    x_h = run324.build_features(frame=hindcast, base=base_h, lead=False, farm_index=False)
    mean = x_h[train].mean(axis=0)
    std = np.maximum(x_h[train].std(axis=0), 1e-6)

    def tensor(x: np.ndarray) -> torch.Tensor:
        return torch.tensor((x - mean) / std, dtype=torch.float32, device=DEVICE)

    mu_t = torch.tensor(base_h[:, c303.MEDIAN_INDEX], dtype=torch.float32, device=DEVICE)
    y_t = torch.tensor(
        np.nan_to_num(hindcast["actual_norm"].to_numpy()), dtype=torch.float32, device=DEVICE
    )
    x_t = tensor(x_h)
    model = f310.MixtureHead(x_h.shape[1]).to(DEVICE)
    optimiser = torch.optim.AdamW(
        model.parameters(), lr=run324.LEARNING_RATE, weight_decay=run324.WEIGHT_DECAY
    )
    train_i = torch.tensor(np.flatnonzero(train), device=DEVICE)
    stop_i = torch.tensor(np.flatnonzero(stop), device=DEVICE)
    generator = torch.Generator().manual_seed(seed)
    best, best_state = float("inf"), None
    for _ in range(run324.EPOCHS):
        model.train()
        order = train_i[torch.randperm(len(train_i), generator=generator).to(DEVICE)]
        for start in range(0, len(order), run324.BATCH):
            i = order[start : start + run324.BATCH]
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
    for name, (frame, base_s) in targets.items():
        xs_t = tensor(run324.build_features(frame=frame, base=base_s, lead=False, farm_index=False))
        result = np.empty((len(frame), 11), dtype=np.float32)
        for start in range(0, len(frame), 50_000):
            sl = slice(start, start + 50_000)
            batch = {
                "x": xs_t[sl],
                "mu": torch.tensor(
                    base_s[sl, c303.MEDIAN_INDEX], dtype=torch.float32, device=DEVICE
                ),
            }
            result[sl] = f310.mixture_quantiles(model=model, batch=batch, ceiling=1.0)
        out[name] = np.sort(result, axis=1)
    return out


def shuffled(*, frame: pl.DataFrame, seed: int) -> pl.DataFrame:
    """Return `frame` with its member summaries taken from another month of the same farm."""
    source = np.arange(len(frame))
    sites = frame["site"].to_numpy()
    valid = frame["valid_time"].dt.replace_time_zone(None).to_numpy()
    for farm in common.FARMS:
        rows = np.flatnonzero(sites == farm)
        source[rows] = rows[c326.month_shuffle_index(valid=valid[rows], seed=seed)]
    return frame.with_columns(
        [pl.Series(name, frame[name].to_numpy()[source]) for name in head303.SUMMARY_FEATURES]
    )


def task(args: tuple[int, int, bool]) -> None:  # noqa: PLR0915
    """Train and store the arms of one (base seed, head seed) task.

    With `base_only`, the member summaries are set to zero, so the head sees only the 11 base
    quantiles (post hoc control for arm x1); it stores `x1_baseonly` and nothing else.
    """
    base_seed, head_seed, base_only = args
    torch.set_num_threads(2)
    started = time.time()
    main_task = base_seed == head_seed
    sites = c303.site_table()
    hind_all = ens_final.hindcast_cases(sites=sites)
    usable = hind_all["actual_norm"].is_finite().fill_null(value=False).to_numpy()
    hindcast = head303.with_summaries(cases=hind_all.filter(pl.Series(usable)), set_name="hindcast")
    scored = head303.with_summaries(cases=ens_final.scored_cases(sites=sites), set_name="scored")
    if base_only:
        zero = [pl.lit(0.0).alias(name) for name in head303.SUMMARY_FEATURES]
        hindcast, scored = hindcast.with_columns(zero), scored.with_columns(zero)
    base_h = np.load(run324.common.E303_WORK / "final_hindcast" / "P2.npy")[base_seed]
    base_s = np.load(run324.common.E303_WORK / "final" / "P2.npy")[base_seed]
    assert len(base_h) == len(hindcast), (len(base_h), len(hindcast))
    assert len(base_s) == len(scored), (len(base_s), len(scored))
    in_rep = (hindcast["power_fcst_init_time"] >= common.REPLICATE_START).to_numpy()
    site_h, site_s = hindcast["site"].to_numpy(), scored["site"].to_numpy()

    def path(arm: str) -> Path:
        arm = f"{arm}_baseonly" if base_only else arm
        return common.PARTS_DIR / f"{arm}_b{base_seed}_h{head_seed}.npy"

    def stitch(*, pieces: list[tuple[np.ndarray, np.ndarray]], n: int) -> np.ndarray:
        out = np.full((n, 11), np.nan, dtype=np.float32)
        for rows, quantiles in pieces:
            out[rows] = quantiles
        return out

    pieces: dict[str, list[tuple[np.ndarray, np.ndarray]]] = {}
    for farm in common.FARMS:
        fh, fs = site_h == farm, site_s == farm
        frame_h = hindcast.filter(pl.Series(fh))
        frame_s = scored.filter(pl.Series(fs))
        rep_rows = fh & in_rep
        frame_r = hindcast.filter(pl.Series(rep_rows))
        rep_index = np.flatnonzero(in_rep)
        in_farm_rep = np.isin(rep_index, np.flatnonzero(rep_rows))
        targets_main = {"x1": (frame_s, base_s[fs])}
        if main_task and not base_only:
            targets_main["x1_placebo"] = (shuffled(frame=frame_s, seed=base_seed), base_s[fs])
        out = train_and_predict(
            hindcast=frame_h,
            base_h=base_h[fh],
            train_on_idle=False,
            cutoff=common.MAIN_CUTOFF,
            valid_limit=None,
            targets=targets_main,
            seed=head_seed,
        )
        for name, quantiles in out.items():
            pieces.setdefault(name, []).append((fs, quantiles))
        if not main_task or base_only:
            continue
        out = train_and_predict(
            hindcast=frame_h,
            base_h=base_h[fh],
            train_on_idle=True,
            cutoff=common.MAIN_CUTOFF,
            valid_limit=None,
            targets={"x3": (frame_s, base_s[fs])},
            seed=head_seed,
        )
        pieces.setdefault("x3", []).append((fs, out["x3"]))
        for label, idle in (("x1_rep", False), ("x3_rep", True)):
            fit_rows = fh & ~in_rep
            out = train_and_predict(
                hindcast=hindcast.filter(pl.Series(fit_rows)),
                base_h=base_h[fit_rows],
                train_on_idle=idle,
                cutoff=common.REPLICATE_CUTOFF,
                valid_limit=common.REPLICATE_START,
                targets={label: (frame_r, base_h[rep_rows])},
                seed=head_seed,
            )
            pieces.setdefault(label, []).append((in_farm_rep, out[label]))
        others = site_h != farm
        out = train_and_predict(
            hindcast=hindcast.filter(pl.Series(others)),
            base_h=base_h[others],
            train_on_idle=False,
            cutoff=common.MAIN_CUTOFF,
            valid_limit=None,
            targets={"x1_lofo": (frame_s, base_s[fs])},
            seed=head_seed,
        )
        pieces.setdefault("x1_lofo", []).append((fs, out["x1_lofo"]))
        print(
            f"base {base_seed} head {head_seed} {farm} done {time.time() - started:.0f} s",
            flush=True,
        )
    sizes = {"x1_rep": int(in_rep.sum()), "x3_rep": int(in_rep.sum())}
    for arm, parts in pieces.items():
        save(path=path(arm), array=stitch(pieces=parts, n=sizes.get(arm, len(scored))))
    print(f"base {base_seed} head {head_seed} finished {time.time() - started:.0f} s", flush=True)


def main() -> None:
    """Run every task in a pool of spawned processes."""
    common.PARTS_DIR.mkdir(parents=True, exist_ok=True)
    base_only = "--base-only" in sys.argv
    tasks = [(s, s, base_only) for s in common.BASE_SEEDS]
    if not base_only:
        tasks += [(0, h, False) for h in common.EXTRA_HEAD_SEEDS if h != 0]
    workers = int(os.environ.get("E00339_WORKERS", "4"))
    if "--first-only" in sys.argv:
        tasks = tasks[:1]
    context = multiprocessing.get_context("spawn")
    with concurrent.futures.ProcessPoolExecutor(max_workers=workers, mp_context=context) as pool:
        for future in [pool.submit(task, t) for t in tasks]:
            future.result()
    print("all tasks done", flush=True)


if __name__ == "__main__":
    main()
