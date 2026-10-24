# ruff: noqa: E501, PLR0912, PLR0915
"""Experiment 00341, step 3: the real series, the lattice test, and the two-farm arm E.

Reads `work/penalty.json` (committed before this script ran), partitions the real block series,
fits the state-space models, tests the whole-turbine lattice, compares change points with
experiment 210's outage runs, builds the capacity-state table, and runs arm E (W2 against W3).
Writes `work/real.json`, `work/arm_e_rows.parquet`, and the private `work/capacity_state.parquet`.

    uv run --no-sync python ml-experiments/00341-*/e00341_real.py
"""

import json
import os
from collections.abc import Mapping
from datetime import timedelta

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import e00341_common as common  # noqa: E402
import e00341_inject as inj  # noqa: E402
import e00341_lib as lib  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from scipy import stats  # noqa: E402

REAL_ARMS: tuple[str, ...] = ("A_p999@2", "A_p999@0", "A_max@0", "B", "C", "D")
REAL_BLOCKS: tuple[int, ...] = (7, 14, 28)
E_BLOCKS: tuple[int, ...] = (14, 28)
LATTICE_TOLERANCE_TURBINES: float = 0.1


def day_to_date(day: int) -> str:
    """Return an ISO date for a day index."""
    return (common.DAY0 + timedelta(days=int(day))).isoformat()


def segments(*, series: lib.Series, last: np.ndarray, block: int) -> list[dict]:
    """Return the segments of the full-series partition with dates and levels."""
    starts = lib.backtrack(last=last, t=len(series.y))
    ends = [*starts[1:], len(series.y)]
    out = []
    for a, b in zip(starts, ends, strict=True):
        level = float(np.average(series.y[a:b], weights=series.omega[a:b]))
        out.append(
            {
                "first_day": day_to_date(int(series.pos[a]) * block),
                "last_day": day_to_date(int(series.pos[b - 1] + 1) * block - 1),
                "n_blocks": b - a,
                "level": level,
            }
        )
    return out


def lattice_test(*, steps_turbines: np.ndarray, n_candidates: Mapping[str, float]) -> dict:
    """Return the share of steps within a tenth of a whole turbine, per candidate turbine count.

    `steps_turbines` holds each step as a share of the farm's level, times the REPD turbine count
    `n_ref`. For a candidate count `n`, the step in turbines is that share times `n`. The null is a
    continuous step size, for which the distance to the nearest whole number is uniform on 0 to
    0.5, so the chance of lying within 0.1 is 0.2 (exact binomial test, one-sided).
    """
    out: dict = {}
    for name, n in n_candidates.items():
        u = np.abs(steps_turbines) * n
        dist = np.abs(u - np.round(u))
        within = int((dist <= LATTICE_TOLERANCE_TURBINES).sum())
        k = len(u)
        out[name] = {
            "n": k,
            "within_tenth": within,
            "p_value": float(stats.binomtest(within, k, 0.2, alternative="greater").pvalue)
            if k
            else None,
            "distances": [round(float(x), 3) for x in dist],
        }
    return out


def main() -> None:
    """Run the real-series analysis."""
    penalty = json.loads((common.WORK_DIR / "penalty.json").read_text())
    inj.init_worker()
    farms = common.load_farms()
    outage = pl.read_parquet(common.E210_DIR / "outage_runs.parquet")
    result: dict = {"penalty": penalty}
    state_frames: list[pl.DataFrame] = []
    for site in common.SITES:
        prep0 = inj._STATE["preps"][(site, 0.0)]
        info: dict = {"segments": {}, "hmm": {}, "outage_runs": []}
        for block in REAL_BLOCKS:
            arms, feats = inj.arms_for(site=site, effects=(), block=block)
            for arm in REAL_ARMS:
                group = "A" if arm.startswith("A_") else arm
                series = arms[arm]
                if len(series.y) < 12:
                    info["segments"][f"{arm}_B{block}"] = {"n_observed_blocks": len(series.y)}
                    continue
                last, _ = lib.partition(series=series, penalty_c=penalty["c"][group])
                info["segments"][f"{arm}_B{block}"] = {
                    "n_observed_blocks": len(series.y),
                    "segments": segments(series=series, last=last, block=block),
                }
        for block in (7, 14):
            _, feats = inj.arms_for(site=site, effects=(), block=block)
            refs, noise = lib.feature_refs(feats=feats, block=block)
            for variant, turbines_only in (("full", False), ("lattice", True)):
                path, names = lib.viterbi(feats=feats, prep=prep0, block=block,
                                          turbines_only=turbines_only, refs=refs, noise=noise)  # fmt: skip
                runs = []
                i = 0
                while i < len(path):
                    j = i
                    while j + 1 < len(path) and path[j + 1] == path[i]:
                        j += 1
                    if path[i] != 0:
                        runs.append({"state": names[path[i]], "first_day": day_to_date(i * block),
                                     "last_day": day_to_date((j + 1) * block - 1)})  # fmt: skip
                    i = j + 1
                info["hmm"][f"{variant}_B{block}"] = runs
        site_outage = outage.filter(pl.col("site") == site)
        info["outage_runs"] = [
            {
                "first": r["start"].date().isoformat(),
                "last": r["end"].date().isoformat(),
                "half_hours": r["n_half_hours"],
            }
            for r in site_outage.iter_rows(named=True)
        ]
        # Lattice test: step sizes of arms D (ratio, B=14) and A at the saturation set (B=28)
        steps_by_arm: dict = {}
        for arm, block in (("D", 14), ("A_p999@0", 28)):
            arms, _ = inj.arms_for(site=site, effects=(), block=block)
            series = arms[arm]
            if len(series.y) < 12:
                continue
            group = "A" if arm.startswith("A_") else arm
            last, _ = lib.partition(series=series, penalty_c=penalty["c"][group])
            starts = lib.backtrack(last=last, t=len(series.y))
            ends = [*starts[1:], len(series.y)]
            levels = [
                float(np.average(series.y[a:b], weights=series.omega[a:b]))
                for a, b in zip(starts, ends, strict=True)
            ]
            steps = np.array(
                [(levels[i] - levels[i - 1]) / levels[i - 1] for i in range(1, len(levels))]
            )
            n_ref = farms[site].n_turbines
            candidates = {
                f"N={n_ref}": n_ref,
                f"N={n_ref - 1}": n_ref - 1,
                f"N={n_ref + 1}": n_ref + 1,
            }
            if site == "W2":
                candidates["N=14"] = 14
            steps_by_arm[f"{arm}_B{block}"] = {
                "steps_share": [round(float(s), 3) for s in steps],
                "lattice": lattice_test(steps_turbines=steps, n_candidates=candidates)
                if len(steps)
                else {},
            }
        info["lattice"] = steps_by_arm
        # Capacity-state table from arm D at 14 days
        arms, _ = inj.arms_for(site=site, effects=(), block=14)
        series = arms["D"]
        last, _ = lib.partition(series=series, penalty_c=penalty["c"]["D"])
        starts = lib.backtrack(last=last, t=len(series.y))
        ends = [*starts[1:], len(series.y)]
        levels = np.array(
            [
                np.average(series.y[a:b], weights=series.omega[a:b])
                for a, b in zip(starts, ends, strict=True)
            ]
        )
        mode_level = float(np.quantile(series.y, 0.75))
        rows = []
        for a, b, level in zip(starts, ends, levels, strict=True):
            first = int(series.pos[a]) * 14
            last_day = (int(series.pos[b - 1]) + 1) * 14 - 1
            share = max(0.0, 1.0 - level / mode_level)
            state = "full" if share < 0.1 else "reduced"
            rows.append((site, first, last_day, state, round(share, 3)))
        seg = pl.DataFrame(
            rows, schema=["site", "first_day", "last_day", "state", "reduced_share"], orient="row"
        )
        info["capacity_state_segments"] = [
            {
                "first_day": day_to_date(r[1]),
                "last_day": day_to_date(r[2]),
                "state": r[3],
                "reduced_share": r[4],
            }
            for r in rows
        ]
        state_frames.append(seg)
        # Gap to registered capacity: yearly 99th percentile of observed power over capacity
        farm = farms[site]
        year = (farm.day // 365.25).astype(int)
        info["p99_by_year_of_capacity"] = {
            str(2019 + y): round(float(np.quantile(farm.obs[year == y] * farm.scale, 0.99)), 3)
            for y in np.unique(year)
        }
        half_year = (farm.day // 182.6).astype(int)
        info["p99_by_half_year_of_capacity"] = {
            f"{2019 + (h * 182.6 + 243) / 365.25:.1f}": round(
                float(np.quantile(farm.obs[half_year == h] * farm.scale, 0.99)), 3
            )
            for h in np.unique(half_year)
        }
        result[site] = info
    pl.concat(state_frames).write_parquet(common.WORK_DIR / "capacity_state_segments.parquet")
    result["arm_e_real"] = arm_e_real(penalty=penalty)
    (common.WORK_DIR / "real.json").write_text(json.dumps(result, indent=1))
    arm_e_known_answer(penalty=penalty)


def diff_series(*, a: lib.Series, b: lib.Series) -> lib.Series:
    """Return the difference of two block series on their common blocks, with combined weights."""
    common_pos, ia, ib = np.intersect1d(a.pos, b.pos, return_indices=True)
    omega = 1.0 / (1.0 / a.omega[ia] + 1.0 / b.omega[ib])
    return lib.Series(common_pos, a.y[ia] - b.y[ib], omega)


def arm_e_real(*, penalty: dict) -> dict:
    """Return the change points of the W2 minus W3 log-ratio on the real series."""
    out: dict = {}
    for block in E_BLOCKS:
        arms2, _ = inj.arms_for(site="W2", effects=(), block=block)
        arms3, _ = inj.arms_for(site="W3", effects=(), block=block)
        for arm in ("B", "C"):
            diff = diff_series(a=arms2[arm], b=arms3[arm])
            if len(diff.y) < 12:
                out[f"{arm}_B{block}"] = {"n_observed_blocks": len(diff.y)}
                continue
            last, _ = lib.partition(series=diff, penalty_c=penalty["c"][arm])
            out[f"{arm}_B{block}"] = {
                "n_observed_blocks": len(diff.y),
                "segments": segments(series=diff, last=last, block=block),
            }
    return out


def arm_e_known_answer(*, penalty: dict) -> None:
    """Inject steps into W2 or W3 and detect them on the W2 minus W3 series (arm E)."""
    rows: list[tuple] = []
    starts = inj.start_days()
    cells: list[tuple[str, str, tuple]] = [("null", "W2", ())]
    for target in ("W2", "W3"):
        n = common.N_TURBINES[target]
        for name, frac in (("p10", 0.10), ("p25", 0.25), ("p50", 0.50), ("t1", 1 / n)):
            cells.append((f"{name}_d90", target, ("loss", frac)))
        cells.append(("p25_d14", target, ("loss", 0.25)))
    for block in E_BLOCKS:
        base = {s: inj.arms_for(site=s, effects=(), block=block)[0] for s in ("W2", "W3")}
        for arm in ("B", "C"):
            c = penalty["c"][arm]
            base_diff = diff_series(a=base["W2"][arm], b=base["W3"][arm])
            base_last, _ = lib.partition(series=base_diff, penalty_c=c)
            for cell, target, spec in cells:
                for start in starts if spec else [0]:
                    dur = int(cell.split("_d")[1]) if "_d" in cell else 0
                    eff = {"W2": (), "W3": ()}
                    if spec:
                        eff[target] = ((spec[0], spec[1], start, start + dur - 1),)
                    s2 = inj.arms_for(site="W2", effects=eff["W2"], block=block)[0][arm]
                    s3 = inj.arms_for(site="W3", effects=eff["W3"], block=block)[0][arm]
                    diff = diff_series(a=s2, b=s3)
                    last, _ = lib.partition(series=diff, penalty_c=c)
                    if spec:
                        # W2 losing capacity lowers the difference, W3 losing capacity raises it
                        direction = -1 if target == "W2" else 1
                        start_block = start // block
                        acc = lib.accepted_by_prefix(series=diff, last=last, start_block=start_block,
                                                     block=block, direction=direction)  # fmt: skip
                        acc_base = lib.accepted_by_prefix(series=base_diff, last=base_last, start_block=start_block,
                                                          block=block, direction=direction)  # fmt: skip
                        f_i = next((t for t, ok in acc.items() if ok), None)
                        f_b = next((t for t, ok in acc_base.items() if ok), None)
                        if f_b is not None and (f_i is None or f_b <= f_i):
                            value = float("nan")
                        elif f_i is None:
                            value = -1.0
                        else:
                            value = float(
                                max(
                                    0,
                                    lib.block_end_day(pos=int(diff.pos[f_i - 1]), block=block)
                                    - start,
                                )
                            )
                    else:
                        mine = lib.boundary_positions(series=diff, last=last)
                        cal = [
                            p
                            for p in mine
                            if common.CAL_START_DAY // block <= p <= common.CAL_END_DAY // block
                        ]
                        value = float(len(cal))
                    rows.append((cell, target, start, block, arm, value))
    pl.DataFrame(
        rows, schema=["cell", "target", "start", "block", "arm", "value"], orient="row"
    ).write_parquet(common.WORK_DIR / "arm_e_rows.parquet")


if __name__ == "__main__":
    main()
