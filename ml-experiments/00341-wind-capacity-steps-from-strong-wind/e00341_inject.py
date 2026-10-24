# ruff: noqa: E501, PERF401
"""Experiment 00341, step 2: the known-answer test (calibration of the penalty, then the grid).

Stage `calibrate` runs the unmodified series, the controls, and the 25% step over a grid of
change-point penalties `c` and CUSUM limits `h`, picks the penalty by the pre-registered rule, and
writes `work/penalty.json`. Stage `grid` runs every injected step, episode, clip, bias, and control
at that penalty. Both write long tables of (site, cell, start, block length, arm, detector,
result) to `work/`. Workers use the spawn start method and one thread each.

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00341-*/e00341_inject.py calibrate|grid
"""

import json
import multiprocessing as mp
import os
import sys
import time

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "POLARS_MAX_THREADS"):
    os.environ.setdefault(_name, "1")

from typing import Final  # noqa: E402

import e00341_common as common  # noqa: E402
import e00341_lib as lib  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

N_WORKERS: Final[int] = 14
BLOCKS: Final[tuple[int, ...]] = (1, 3, 7, 14, 28, 56)
HMM_BLOCKS: Final[tuple[int, ...]] = (7, 14)
MARGINS: Final[tuple[float, ...]] = (2.0, 0.0)
"""The pre-registered strong-set margin (2 m/s) and the saturation-speed set (experiment 233's)."""
C_GRID: Final[tuple[float, ...]] = (1.0, 2.0, 4.0, 8.0, 16.0)
H_GRID: Final[tuple[float, ...]] = (4.0, 5.0, 6.0, 8.0, 10.0, 12.0)
DURATIONS: Final[tuple[int, ...]] = (3, 7, 14, 30, 60, 90, 365)
SIZES: Final[tuple[str, ...]] = ("t1", "t2", "half", "p10", "p25", "p50")
CLIPS: Final[tuple[float, ...]] = (0.9, 0.75, 0.5)
N_STARTS: Final[int] = 30
SEED: Final[int] = 341
FA_PRIMARY_ARMS: Final[tuple[str, ...]] = ("A_p999@2", "A_p999@0", "B", "C", "D")
FA_PRIMARY_BLOCKS: Final[tuple[int, ...]] = (7, 14, 28)

_STATE: dict = {}


def start_days() -> list[int]:
    """Return the 30 fixed start days, the same for every cell and farm."""
    rng = np.random.default_rng(SEED)
    return sorted(
        rng.integers(common.START_FIRST_DAY, common.START_LAST_DAY + 1, N_STARTS).tolist()
    )


def init_worker() -> None:
    """Load the farms and the prepared sets once per worker."""
    prep_info = json.loads((common.WORK_DIR / "prep.json").read_text())
    farms = common.load_farms()
    _STATE["preps"] = {
        (site, margin): lib.prepare(
            farm=farm,
            margin=margin,
            sigma_v=prep_info[site]["noise"]["sigma_v"],
            sigma_p=prep_info[site]["noise"]["sigma_p"],
        )
        for site, farm in farms.items()
        for margin in MARGINS
    }
    _STATE["cache"] = {}


def effect_list(*, site: str, size: str, duration: int, start: int) -> tuple[tuple, ...]:
    """Return the injected effects of one cell."""
    n = common.N_TURBINES[site]
    fraction = {"t1": 1 / n, "t2": 2 / n, "half": 0.5 / n, "p10": 0.10, "p25": 0.25, "p50": 0.50}
    if size.startswith("clip"):
        return (("clip", float(size[4:]), start, start + duration - 1),)
    return (("loss", fraction[size], start, start + duration - 1),)


def arms_for(*, site: str, effects: tuple, block: int) -> tuple[dict[str, lib.Series], dict]:
    """Return all arm series (A at both margins) and the state-space features, cached by effects."""
    key = (site, effects, block)
    cache = _STATE["cache"]
    if key in cache:
        return cache[key]
    all_arms: dict[str, lib.Series] = {}
    feats: dict = {}
    for margin in MARGINS:
        prep = _STATE["preps"][(site, margin)]
        obs = lib.modified_obs(prep=prep, effects=effects)
        arms, f = lib.block_arms(prep=prep, obs=obs, block=block)
        tag = f"@{int(margin)}"
        for name, series in arms.items():
            if name.startswith("A_"):
                all_arms[name + tag] = series
            elif margin == MARGINS[0]:
                all_arms[name] = series
        if margin == MARGINS[0]:
            feats = f
    if len(cache) > 400:
        cache.clear()
    cache[key] = (all_arms, feats)
    return all_arms, feats


def base_partitions(*, site: str, block: int, c: float) -> dict:
    """Return (series, last) per arm on the unmodified series, cached."""
    key = ("base", site, block, c)
    cache = _STATE["cache"]
    if key not in cache:
        arms, _ = arms_for(site=site, effects=(), block=block)
        cache[key] = {a: (s, lib.partition(series=s, penalty_c=c)[0]) for a, s in arms.items()}
    return cache[key]


def run_task(task: dict) -> list[tuple]:
    """Run one injected cell at one start day and return the result rows."""
    site, cell, effects = task["site"], task["cell"], task["effects"]
    start = task["start"]
    rows: list[tuple] = []
    if task["mode"] == "hmm_only":
        return hmm_rows(task=task)
    for block in BLOCKS:
        arms, _ = arms_for(site=site, effects=effects, block=block)
        base_arms, _ = arms_for(site=site, effects=(), block=block)
        for c in task["cs"]:
            base = base_partitions(site=site, block=block, c=c)
            for arm, series in arms.items():
                if task["mode"] == "step":
                    last = lib.partition(series=series, penalty_c=c)[0]
                    delay = lib.detection_delay(
                        series_inj=series, series_base=base[arm][0], last_inj=last,
                        last_base=base[arm][1], start_day=start, block=block,
                    )  # fmt: skip
                    rows.append((site, cell, start, "pelt", block, arm, c, _enc(delay)))
                elif task["mode"] == "control":
                    last = lib.partition(series=series, penalty_c=c)[0]
                    mine = lib.boundary_positions(series=series, last=last)
                    ref = lib.boundary_positions(series=base[arm][0], last=base[arm][1])
                    new = sum(1 for p in mine if not len(ref) or np.min(np.abs(ref - p)) > 2)
                    cal = [
                        p
                        for p in mine
                        if common.CAL_START_DAY // block <= p <= common.CAL_END_DAY // block
                    ]
                    rows.append((site, cell, start, "new_pelt", block, arm, c, float(new)))
                    rows.append((site, cell, start, "cal_pelt", block, arm, c, float(len(cal))))
                    rows.append((site, cell, start, "all_pelt", block, arm, c, float(len(mine))))
        for h in task["hs"]:
            for arm, series in arms.items():
                if task["mode"] == "step":
                    delay = lib.cusum_delay(
                        series_inj=series,
                        series_base=base_arms[arm],
                        start_day=start,
                        block=block,
                        h=h,
                    )
                    rows.append((site, cell, start, "cusum", block, arm, h, _enc(delay)))
                elif task["mode"] == "control":
                    mine = [i for i, _ in lib.cusum_alarms(series=series, h=h, block=block)]
                    mine_pos = series.pos[mine] if mine else np.array([], dtype=int)
                    ref = lib.cusum_alarms(series=base_arms[arm], h=h, block=block)
                    ref_pos = (
                        base_arms[arm].pos[[i for i, _ in ref]] if ref else np.array([], dtype=int)
                    )
                    new = sum(
                        1 for p in mine_pos if not len(ref_pos) or np.min(np.abs(ref_pos - p)) > 2
                    )
                    cal = [
                        p
                        for p in mine_pos
                        if common.CAL_START_DAY // block <= p <= common.CAL_END_DAY // block
                    ]
                    rows.append((site, cell, start, "new_cusum", block, arm, h, float(new)))
                    rows.append((site, cell, start, "cal_cusum", block, arm, h, float(len(cal))))
    if task.get("hmm"):
        rows.extend(hmm_rows(task=task))
    return rows


def _enc(delay: float | None) -> float:
    """Encode a delay: NaN = start excluded, -1 = not found, otherwise the delay in days."""
    return -1.0 if delay is None else delay


def hmm_rows(*, task: dict) -> list[tuple]:
    """Fit the state-space and lattice-constrained models and report the state found and the delay."""
    site, cell, start, duration = task["site"], task["cell"], task["start"], task["duration"]
    rows: list[tuple] = []
    last_day = start + duration - 1
    prep = _STATE["preps"][(site, MARGINS[0])]
    for block in HMM_BLOCKS:
        _, feats = arms_for(site=site, effects=task["effects"], block=block)
        _, base_feats = arms_for(site=site, effects=(), block=block)
        refs, noise = lib.feature_refs(feats=base_feats, block=block)
        bs, be = start // block, min(last_day // block, len(feats["p999"]) - 1)
        for variant, turbines_only in (("full", False), ("lattice", True)):
            path, names = lib.viterbi(
                feats=feats,
                prep=prep,
                block=block,
                turbines_only=turbines_only,
                refs=refs,
                noise=noise,
            )
            window = path[bs : be + 1]
            window = window[window != 0]
            found = names[int(np.bincount(window).argmax())] if len(window) else "nameplate"
            base_path, _ = lib.viterbi(
                feats=base_feats,
                prep=prep,
                block=block,
                turbines_only=turbines_only,
                refs=refs,
                noise=noise,
            )
            excluded = bool((base_path[max(bs - 1, 0) : bs + 2] != 0).any())
            delay = -1.0
            for t in range(bs, min(bs + lib.HORIZON_DAYS // block + 1, len(path))):
                cut = {k: v[: t + 1] for k, v in feats.items()}
                p_t, _ = lib.viterbi(
                    feats=cut,
                    prep=prep,
                    block=block,
                    turbines_only=turbines_only,
                    refs=refs,
                    noise=noise,
                )
                if (p_t[bs : t + 1] != 0).any():
                    delay = float(max(0, lib.block_end_day(pos=t, block=block) - start))
                    break
            rows.append(
                (
                    site,
                    cell,
                    start,
                    f"hmm_{variant}",
                    block,
                    found,
                    0.0,
                    float("nan") if excluded else delay,
                )
            )
    return rows


def build_tasks(*, stage: str, cs: tuple[float, ...], hs: tuple[float, ...]) -> list[dict]:
    """Return the task list of a stage."""
    tasks: list[dict] = []
    starts = start_days()
    common_args = {"cs": cs, "hs": hs}
    for site in common.SITES:
        controls = {
            "null": (),
            "swing9": (("swing", 0.09),),
            "swing15": (("swing", 0.15),),
            "swing25": (("swing", 0.25),),
            "drift1": (("drift", 0.01),),
        }
        for cell, effects in controls.items():
            if stage == "calibrate" and cell not in {"null", "swing9", "drift1"}:
                continue
            if stage in {"bias", "hmm"}:
                continue
            tasks.append({"site": site, "cell": cell, "effects": effects, "start": 0,
                          "duration": 0, "mode": "control", **common_args})  # fmt: skip
        grid_cells = (
            [("p25", 90)]
            if stage == "calibrate"
            else []
            if stage == "bias"
            else [(s, d) for s in ("t1", "t2", "half", "p25") for d in (3, 7, 14, 30, 60, 90)]
            + [(f"clip{c}", d) for c in CLIPS for d in (3, 7, 14, 30, 60, 90)]
            if stage == "hmm"
            else [(s, d) for s in SIZES for d in DURATIONS]
            + [(f"clip{c}", d) for c in CLIPS for d in DURATIONS]
        )
        for size, duration in grid_cells:
            if site == "W3" and size == "t2":
                continue
            for start in starts:
                effects = effect_list(site=site, size=size, duration=duration, start=start)
                hmm = (
                    stage == "hmm"
                    and duration <= 90
                    and (size.startswith("clip") or size in {"t1", "t2", "half", "p25"})
                    and duration >= 3
                )
                tasks.append({"site": site, "cell": f"{size}_d{duration}", "effects": effects,
                              "start": start, "duration": duration, "mode": "hmm_only" if stage == "hmm" else "step",
                              "hmm": hmm, **common_args})  # fmt: skip
        if stage == "bias":
            for start in starts:
                tasks.append({"site": site, "cell": "bias-5_d30", "effects": (("bias", -0.05, start, start + 29),),
                              "start": start, "duration": 30, "mode": "step", "hmm": False, **common_args})  # fmt: skip
    return tasks


def run_pool(*, tasks: list[dict]) -> pl.DataFrame:
    """Run the tasks in a spawn pool and return the long result table."""
    ctx = mp.get_context("spawn")
    rows: list[tuple] = []
    t0 = time.time()
    with ctx.Pool(N_WORKERS, initializer=init_worker) as pool:
        for i, out in enumerate(pool.imap_unordered(run_task, tasks, chunksize=1)):
            rows.extend(out)
            if i % 100 == 0:
                print(f"{i}/{len(tasks)} tasks, {time.time() - t0:.0f}s", flush=True)
    return pl.DataFrame(
        rows,
        schema=["site", "cell", "start", "detector", "block", "arm", "setting", "value"],
        orient="row",
    )


def main() -> None:
    """Run the requested stage."""
    stage = sys.argv[1]
    if stage == "calibrate":
        tasks = build_tasks(stage="calibrate", cs=C_GRID, hs=H_GRID)
        frame = run_pool(tasks=tasks)
        frame.write_parquet(common.WORK_DIR / "calibrate_rows.parquet")
        print(frame.shape, "rows; run e00341_penalty.py to choose the penalties")
    else:
        settings = json.loads((common.WORK_DIR / "penalty.json").read_text())
        cs = tuple(sorted(set(settings["c"].values())))
        hs = tuple(sorted(set(settings["h"].values())))
        tasks = build_tasks(stage=stage, cs=cs, hs=hs)
        print(len(tasks), "tasks", flush=True)
        frame = run_pool(tasks=tasks)
        group = (
            pl.when(pl.col("arm").str.starts_with("A_")).then(pl.lit("A")).otherwise(pl.col("arm"))
        )
        chosen = pl.concat(
            [
                pl.DataFrame(
                    {"group": list(settings[k]), "keep": list(settings[k].values())}
                ).with_columns(kind=pl.lit(k))
                for k in ("c", "h")
            ]
        )
        frame = (
            frame.with_columns(
                group=group,
                kind=pl.when(pl.col("detector").str.contains("pelt"))
                .then(pl.lit("c"))
                .when(pl.col("detector").str.contains("cusum"))
                .then(pl.lit("h"))
                .otherwise(pl.lit("none")),
            )
            .join(chosen, on=["group", "kind"], how="left")
            .filter((pl.col("kind") == "none") | (pl.col("setting") == pl.col("keep")))
            .drop("group", "kind", "keep")
        )
        frame.write_parquet(common.WORK_DIR / f"{stage}_rows.parquet")
        print(frame.shape)


if __name__ == "__main__":
    main()
