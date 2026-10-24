# ruff: noqa: E501
r"""Stage C: rebuild Primary demand research default 3 with each variant's responses.

For each variant (`v0r`, the rebuilt fit without a response, then `f1` and `f1s`), the chain's mean
and all 11 quantiles are the variant's rebuilt chain of stage B, and the 51 members' responses
are the variant's.
Every later stage is then rerun exactly as in the earlier experiments, with each factor refitted on
completed history:

- **0.5 to 168 hours** (experiment 202's rolling replay): the offset factor `f` and spread factor
  `s` per lead band, fitted on the trailing 365 days of runs completed before the month's first
  origin, give `r2`. Experiment 220's median correction `delta`, recovered as experiment 251's
  `arm1` minus `a_s`, is added afterwards.
- **168 to 240 hours** (experiments 319, 320, and 322): the same mixture `x1`, then the heuristic
  blend with weights fitted on the chain's quantiles, used on daylight half-hours only (`d1`).

The replicate period is fitted on the tuning months and applied to the replicate months. The gate
is that `v0r` reproduces experiment 322's stored default.

    PYTHONPATH=<folders of experiments 184, 202, 220, 251, 317, 319, 320, 322, 325> \
    OMP_NUM_THREADS=1 systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00325-*/e00323_replay.py [--workers 6]
"""

import argparse
import json
import multiprocessing as mp
from datetime import timedelta

import e00202_arms as arms202  # ty: ignore[unresolved-import]
import e00202_common as c202  # ty: ignore[unresolved-import]
import e00202_replay as replay202  # ty: ignore[unresolved-import]
import e00319_arms as a319  # ty: ignore[unresolved-import]
import e00320_arms as a320  # ty: ignore[unresolved-import]
import e00325_common as common
import numpy as np
import polars as pl

QUANTILES = a319.common.QUANTILES
VARIANTS = common.VARIANTS
DELTA_OF = {"v0r": "v0", "f1": "f1", "f1s": "f1s"}
STATE: dict = {}


def member_store() -> tuple[pl.DataFrame, dict[str, np.ndarray]]:
    """Return the stored members' keys (with a row position) and arrays, all Primaries and periods."""
    keys, arrays = (
        [],
        {
            name: []
            for name in (
                "n_members",
                *[f"delta_{name}" for name in ("v0", *common.RESPONSE_VARIANTS)],
            )
        },
    )
    start = 0
    for site in a319.common.SCORED_SITES:
        for period in common.PERIODS:
            with np.load(common.MEMBERS_DIR / f"{site}_{period}.npz") as stored:
                n = len(stored["run"])
                keys.append(
                    pl.DataFrame(
                        {
                            "site": site,
                            "run": stored["run"].astype(np.int32),
                            "valid_time": pl.Series(stored["valid_time"]).dt.replace_time_zone(
                                "UTC"
                            ),
                            "pos": np.arange(start, start + n),
                        }
                    )
                )
                for name, store in arrays.items():
                    store.append(stored[name])
                start += n
    return pl.concat(keys), {name: np.concatenate(store) for name, store in arrays.items()}


def positions(*, frame: pl.DataFrame, keys: pl.DataFrame) -> np.ndarray:
    """Return each frame row's position in the member store, raising if a row has none."""
    joined = frame.select(pl.col("site"), pl.col("run").cast(pl.Int32), pl.col("valid_time")).join(
        keys.with_columns(pl.col("run").cast(pl.Int32)),
        on=["site", "run", "valid_time"],
        how="left",
        validate="m:1",
    )
    if joined["pos"].null_count():
        msg = f"{joined['pos'].null_count()} rows have no stored member response"
        raise ValueError(msg)
    return joined["pos"].to_numpy()


def variant_inputs(
    *, frame: pl.DataFrame, pos: np.ndarray, store: dict[str, np.ndarray], variant: str
) -> tuple[pl.DataFrame, np.ndarray]:
    """Return the frame with the variant's rebuilt chain on `mean` and the quantiles, and its responses."""
    delta = store[f"delta_{DELTA_OF[variant]}"][pos].astype(np.float64)
    chain = pl.concat(
        [
            pl.read_parquet(
                common.CHAIN_DIR / variant / f"{site}.parquet",
                columns=["run", "valid_time", "mean", *QUANTILES],
            ).with_columns(site=pl.lit(site))
            for site in a319.common.SCORED_SITES
        ]
    ).with_columns(pl.col("run").cast(pl.Int32))
    joined = frame.select("site", pl.col("run").cast(pl.Int32), "valid_time").join(
        chain, on=["site", "run", "valid_time"], how="left", validate="m:1"
    )
    if joined["mean"].null_count():
        msg = f"{joined['mean'].null_count()} rows have no rebuilt chain row"
        raise ValueError(msg)
    STATE.setdefault("chain_gate", {})[f"{variant}_{frame.height}_rows"] = {
        c: float(np.abs(frame[c].to_numpy().astype(np.float64) - joined[c].to_numpy()).max())
        for c in ("mean", *QUANTILES)
    }
    return (
        frame.with_columns(
            *[
                pl.Series(c, joined[c].to_numpy()).cast(pl.Float32).alias(c)
                for c in ("mean", *QUANTILES)
            ]
        ),
        delta,
    )


# ---- 0.5 to 168 hours ------------------------------------------------------------------------


def short_month(task: tuple[str, str]) -> tuple[str, str, np.ndarray, np.ndarray]:
    """Return `(variant, month, rows, r2 quantiles)` of one scored month, factors on completed history."""
    variant, month = task
    s = STATE["short"][variant]
    frame, delta, origin = s["frame"], s["delta"], STATE["short_origin"]
    rows_m = np.flatnonzero(STATE["short_scored"] & (STATE["short_month"] == month))
    first = origin[rows_m].min()
    horizon = np.timedelta64(int(timedelta(hours=arms202.data.HORIZON_HOURS).total_seconds()), "s")
    completed = (origin + horizon <= first) & (
        origin >= first - np.timedelta64(c202.WINDOW_DAYS, "D")
    )
    fit_rows = np.flatnonzero(completed)
    stats = replay202.stats_float32(delta=delta)
    factors = replay202.fit_on(frame=frame, stats=stats, rows=fit_rows, spread=True)
    q = arms202.apply_arm(
        frame=frame[rows_m], make_points=replay202.maker(stats, rows_m), factors=factors
    )
    return variant, month, rows_m, q


def short_replicate(variant: str) -> tuple[str, np.ndarray, np.ndarray]:
    """Return `(variant, rows, r2 quantiles)` of the replicate months, factors fitted on the tuning months."""
    s = STATE["short"][variant]
    frame, delta = s["frame"], s["delta"]
    period = frame["period"].to_numpy()
    usable = arms202.data.usable_rows(frame=frame)
    fit_rows = np.flatnonzero((period == "tuning") & usable)
    rows = np.flatnonzero(period == "replicate")
    stats = replay202.stats_float32(delta=delta)
    factors = replay202.fit_on(frame=frame, stats=stats, rows=fit_rows, spread=True)
    q = arms202.apply_arm(
        frame=frame[rows], make_points=replay202.maker(stats, rows), factors=factors
    )
    return variant, rows, q


# ---- 168 to 240 hours ------------------------------------------------------------------------


def long_arms(
    *, frame: pl.DataFrame, delta: np.ndarray, fit_rows: np.ndarray, rows: np.ndarray
) -> dict[str, np.ndarray]:
    """Return `x1`, the blend, and `d1` (the blend on daylight half-hours only) at `rows`."""
    factors = a319.fit_factors(frame=frame, delta=delta, rows=fit_rows)
    weights = a319.fit_weights(frame=frame, rows=fit_rows)
    x1 = a319.apply_x1(frame=frame, delta=delta, rows=rows, factors=factors)
    blended = a320.apply_blend(
        base=x1,
        heuristic=frame["heuristic_median"].to_numpy()[rows],
        w=a320._table_w(frame=frame, rows=rows, weights=weights),
    )
    day = (frame["kind"].to_numpy()[rows] == "daylight")[:, None]
    return {"x1": x1, "r319": blended, "d1": np.where(day, blended, x1)}


def long_month(task: tuple[str, str]) -> tuple[str, str, np.ndarray, dict[str, np.ndarray]]:
    """Return one scored month's long-lead arms for a variant."""
    variant, month = task
    s = STATE["long"][variant]
    frame, delta, origin = s["frame"], s["delta"], STATE["long_origin"]
    rows_m = np.flatnonzero(STATE["long_scored"] & (STATE["long_month"] == month))
    first = origin[rows_m].min()
    horizon = np.timedelta64(int(timedelta(hours=a319.common.HORIZON_HOURS).total_seconds()), "s")
    completed = (
        (origin + horizon <= first)
        & (origin >= first - np.timedelta64(a319.common.WINDOW_DAYS, "D"))
        & STATE["long_usable"]
    )
    return (
        variant,
        month,
        rows_m,
        long_arms(frame=frame, delta=delta, fit_rows=np.flatnonzero(completed), rows=rows_m),
    )


def long_replicate(variant: str) -> tuple[str, np.ndarray, dict[str, np.ndarray]]:
    """Return the replicate months' long-lead arms for a variant, factors fitted on the tuning months."""
    s = STATE["long"][variant]
    frame, delta = s["frame"], s["delta"]
    period = frame["period"].to_numpy()
    usable = STATE["long_usable"]
    fit_rows = np.flatnonzero((period == "tuning") & usable)
    rows = np.flatnonzero((period == "replicate") & usable)
    return variant, rows, long_arms(frame=frame, delta=delta, fit_rows=fit_rows, rows=rows)


def run_task(task: tuple):  # noqa: ANN201
    """Dispatch one task in a worker; return its kind and payload."""
    kind, *rest = task
    if kind == "short_month":
        return kind, short_month((rest[0], rest[1]))
    if kind == "long_month":
        return kind, long_month((rest[0], rest[1]))
    if kind == "short_rep":
        return kind, short_replicate(rest[0])
    return kind, long_replicate(rest[0])


def build_state(variants: tuple[str, ...] = VARIANTS) -> None:
    """Load the short and long frames and every variant's inputs (the pool forks after this)."""
    keys, store = member_store()
    # 0.5 to 168 hours
    frame, delta0, n_fit_end = replay202.load_all()
    dates = c202.run_dates()
    origin_by_run = np.array(
        [np.datetime64(c202.origin_of(run=d).replace(tzinfo=None), "us") for d in dates]
    )
    frame = frame.with_columns(
        power_fcst_init_time=pl.Series(origin_by_run[frame["run"].to_numpy()]).dt.replace_time_zone(
            "UTC"
        )
    )
    pos = positions(frame=frame, keys=keys)
    STATE["short"] = {}
    for variant in variants:
        frame_v, delta_v = variant_inputs(frame=frame, pos=pos, store=store, variant=variant)
        STATE["short"][variant] = {"frame": frame_v, "delta": delta_v}
    STATE["gates"] = {}
    STATE["gates"]["chain_vs_frame_max_abs"] = {
        k: max(v.values()) for k, v in STATE["chain_gate"].items()
    }
    if "v0r" in variants:
        STATE["gates"]["short_v0_delta_vs_stored_184"] = float(
            np.nanmax(np.abs(STATE["short"]["v0r"]["delta"] - delta0))
        )
    STATE["short_origin"] = replay202.run_origin_array(frame=frame)
    STATE["short_scored"] = np.arange(frame.height) >= n_fit_end
    STATE["short_month"] = frame["month"].to_numpy()
    STATE["short_frame"] = frame
    # 168 to 240 hours
    long_frame, long_delta0 = a319.load_rows()
    pos = positions(frame=long_frame, keys=keys)
    STATE["long"] = {}
    for variant in variants:
        frame_v, delta_v = variant_inputs(frame=long_frame, pos=pos, store=store, variant=variant)
        STATE["long"][variant] = {"frame": frame_v, "delta": delta_v}
    STATE["gates"]["chain_vs_frame_max_abs"] = {
        k: max(v.values()) for k, v in STATE["chain_gate"].items()
    }
    if "v0r" in variants:
        STATE["gates"]["long_v0_delta_vs_stored_319"] = float(
            np.nanmax(np.abs(STATE["long"]["v0r"]["delta"] - long_delta0))
        )
    STATE["long_origin"] = long_frame["power_fcst_init_time"].dt.replace_time_zone(None).to_numpy()
    STATE["long_scored"] = (long_frame["period"] == "scored").to_numpy()
    STATE["long_month"] = long_frame["power_fcst_init_time"].dt.strftime("%Y-%m").to_numpy()
    STATE["long_usable"] = a319.usable(frame=long_frame)
    STATE["long_frame"] = long_frame


def write_outputs(*, results: list) -> None:
    """Assemble the tasks' outputs into per-variant parquet files."""
    short_frame, long_frame = STATE["short_frame"], STATE["long_frame"]
    active = tuple(STATE["short"])
    short_out = {
        v: {
            "scored": np.full((short_frame.height, 11), np.nan),
            "replicate": np.full((short_frame.height, 11), np.nan),
        }
        for v in active
    }
    names = ("x1", "r319", "d1")
    long_out = {v: {n: np.full((long_frame.height, 11), np.nan) for n in names} for v in active}
    for kind, payload in results:
        if kind == "short_month":
            variant, _, rows, q = payload
            short_out[variant]["scored"][rows] = q
        elif kind == "short_rep":
            variant, rows, q = payload
            short_out[variant]["replicate"][rows] = q
        elif kind == "long_month":
            variant, _, rows, arms = payload
            for n in names:
                long_out[variant][n][rows] = arms[n]
        else:
            variant, rows, arms = payload
            for n in names:
                long_out[variant][n][rows] = arms[n]
    common.ARMS_DIR.mkdir(parents=True, exist_ok=True)
    keys_short = short_frame.select(
        "site", "run", "power_fcst_init_time", "valid_time", "lead_hours", "period", "kind"
    )
    for v in active:
        merged = np.where(
            np.isfinite(short_out[v]["scored"]), short_out[v]["scored"], short_out[v]["replicate"]
        )
        keep = np.isfinite(merged).all(axis=1)
        keys_short.filter(pl.Series(keep)).with_columns(
            *[
                pl.Series(f"r2_{q}", merged[keep, k].astype(np.float32))
                for k, q in enumerate(QUANTILES)
            ]
        ).write_parquet(common.ARMS_DIR / f"short_{v}.parquet")
        keys_long = long_frame.select(
            "site", "run", "power_fcst_init_time", "valid_time", "lead_hours", "period", "kind"
        )
        arrays = long_out[v]
        keep = np.isfinite(arrays["d1"]).all(axis=1)
        keys_long.filter(pl.Series(keep)).with_columns(
            *[
                pl.Series(f"{n}_{q}", arrays[n][keep, k].astype(np.float32))
                for n in names
                for k, q in enumerate(QUANTILES)
            ]
        ).write_parquet(common.ARMS_DIR / f"long_{v}.parquet")


def run_variant(*, variant: str, workers: int) -> None:
    """Run one variant's tasks in a spawn pool, each worker loading only that variant."""
    if (common.ARMS_DIR / f"long_{variant}.parquet").exists():
        print(variant, "exists", flush=True)
        return
    build_state((variant,))
    print("gates", variant, json.dumps(STATE["gates"]), flush=True)
    s_months = sorted(set(STATE["short_month"][STATE["short_scored"]]))
    l_months = sorted(set(STATE["long_month"][STATE["long_scored"]]))
    tasks: list[tuple] = [("short_month", variant, m) for m in s_months] + [
        ("long_month", variant, m) for m in l_months
    ]
    tasks += [("short_rep", variant), ("long_rep", variant)]
    with mp.get_context("spawn").Pool(
        workers, initializer=build_state, initargs=((variant,),)
    ) as pool:
        results = []
        for out in pool.imap_unordered(run_task, tasks, chunksize=1):
            results.append(out)
            print(variant, len(results), "of", len(tasks), flush=True)
    write_outputs(results=results)
    common.write_json_atomic(
        path=common.REPORT_DIR / f"gates_replay_{variant}.json", content=STATE["gates"]
    )


def main() -> None:
    """Run every variant in turn."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()
    for variant in VARIANTS:
        run_variant(variant=variant, workers=args.workers)


if __name__ == "__main__":
    main()
