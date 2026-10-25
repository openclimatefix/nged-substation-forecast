"""Experiment 00346: refit the offset table of the wind default on recent runs, month by month.

Reads the mixtures of `e00346_mix.py`, applies experiment 33's spread factor, and builds the
scored quantiles of each arm. Every table has one offset per farm, lead day, and quantile level
(experiment 259's table). The table used for a scored run is fitted on the first day of the run's
month from the spread-adjusted quantiles of earlier daily runs whose valid time is at least the
telemetry delay (3.5 hours) before that day:

- `default`: the table fitted once on the hindcast (valid times before 2025-06-17).
- `x1`: monthly refit, expanding window from 2024-04. `x2`: monthly, trailing 6 months.
  `x3`: monthly, trailing 3 months. `refq`: quarterly refit, expanding window (descriptive).
- `plc`: the placebo, each scored run gets the x2 table of a month drawn at random before its own.
- `defaultG`: the gate arm, the spread factor fitted on the hindcast and the 53 scored Tuesday
  runs only, as experiment 340 did.

Outputs, in `work/final/`: `<arm>.npy` (3 seeds, scored cases, 11) and `<arm>_oot.npy` for the
Tuesday runs of 2026-07 to 2026-10, `frames.parquet` and `frames_oot.parquet`,
`tables_seed0.parquet`
(every fitted offset), `leakage.json`, `known_answer.json`, `reproduction.json`.

    uv run --no-sync python e00346_final.py   (PYTHONPATH: see e00346_common)
"""

import json
import os
from datetime import UTC, datetime, timedelta

os.environ.setdefault("POLARS_MAX_THREADS", "8")
for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import e00303_ens_final as ens_final  # noqa: E402  # isort: skip  # ty: ignore[unresolved-import]
import e00303_score as score303  # noqa: E402  # isort: skip  # ty: ignore[unresolved-import]
import e00346_common as common  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402

c303 = common.c303
c336 = common.c336
ARMS_MONTHLY = {"x1": None, "x2": 6, "x3": 3}
"""Arm -> trailing window in months (None: expanding from the first run)."""
FIRST_TABLE_MONTH = (2024, 10)
LAST_TABLE_MONTH = (2026, 10)
HINDCAST_LAST_INIT = datetime(2025, 6, 17, tzinfo=UTC)
MASK_END = datetime(2026, 9, 13, 18, tzinfo=UTC)
GATE_SCORE = 11.352
DELAY = timedelta(minutes=common.TELEMETRY_DELAY_MINUTES)


def add_months(*, year: int, month: int, delta: int) -> tuple[int, int]:
    """Return (year, month) shifted by `delta` months."""
    index = year * 12 + (month - 1) + delta
    return index // 12, index % 12 + 1


def month_start(*, ym: tuple[int, int]) -> datetime:
    """Return 00:00 UTC on the first day of the month."""
    return datetime(ym[0], ym[1], 1, tzinfo=UTC)


def months_between(*, first: tuple[int, int], last: tuple[int, int]) -> list[tuple[int, int]]:
    """Return every (year, month) from `first` to `last` inclusive."""
    out, current = [], first
    while current <= last:
        out.append(current)
        current = add_months(year=current[0], month=current[1], delta=1)
    return out


def label(*, ym: tuple[int, int]) -> str:
    """Return the month as `YYYY-MM`."""
    return f"{ym[0]}-{ym[1]:02d}"


def load_cases() -> tuple[pl.DataFrame, dict[int, np.ndarray]]:
    """Return every daily case with an observed fraction, and each seed's mixtures for them."""
    sites = c303.site_table().select("time_series_id", "site")
    keys = pl.concat(
        [pl.read_parquet(common.MIX_DIR / f"keys_{farm}.parquet") for farm in common.FARMS]
    ).join(sites, on="time_series_id", validate="m:1")
    mix = {
        seed: np.concatenate(
            [np.load(common.MIX_DIR / f"default_s{seed}_{farm}.npy") for farm in common.FARMS]
        )
        for seed in common.SEEDS
    }
    c303.TRAIN_END = datetime(2027, 1, 1, tzinfo=UTC)  # observed power beyond the hindcast
    actual = c303.observed_training_power(sites=c303.site_table()).rename({"target": "actual_norm"})
    frame = (
        keys.join(actual, on=["time_series_id", "valid_time"], how="left", validate="m:1")
        .join(
            c336.idle_mask_frame(), on=["time_series_id", "valid_time"], how="left", validate="m:1"
        )
        .with_columns(
            masked=pl.col("masked").fill_null(value=False),
            lead_hours=(pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_minutes()
            / 60.0,
            month=pl.col("power_fcst_init_time").dt.strftime("%Y-%m"),
            valid_us=pl.col("valid_time").dt.epoch("us"),
            row=pl.int_range(pl.len()),
        )
        .with_columns(
            lead_day=c303.lead_day(
                power_fcst_init_time=pl.col("power_fcst_init_time"),
                valid_time=pl.col("valid_time"),
            ).clip(0, common.MAX_LEAD_DAY)
        )
    )
    usable = frame["actual_norm"].is_finite().fill_null(value=False).to_numpy()
    frame = frame.filter(pl.Series(usable)).with_columns(row=pl.int_range(pl.len()))
    return frame, {seed: values[usable] for seed, values in mix.items()}


def fit_table(
    *, cases: pl.DataFrame, adjusted: np.ndarray, select: np.ndarray
) -> tuple[np.ndarray, list[tuple[str, int]]]:
    """Return experiment 259's offset table fitted on the cases in `select`."""
    index = np.flatnonzero(select)
    return ens_final.offsets_table(hindcast=cases[index], quantiles=adjusted[index])


def window_select(
    *, cases: pl.DataFrame, ym: tuple[int, int], months: int | None
) -> tuple[np.ndarray, datetime]:
    """Return the cases a table fitted on the first day of month `ym` may use, and the cutoff."""
    start = month_start(ym=ym)
    cutoff = start - DELAY
    valid = cases["valid_time"]
    select = (valid <= cutoff).to_numpy()
    if months is not None:
        lowest = month_start(ym=add_months(year=ym[0], month=ym[1], delta=-months))
        select &= (valid >= lowest).to_numpy()
    return select, cutoff


def main() -> None:  # noqa: PLR0912, PLR0915
    """Build every arm's scored quantiles and the controls."""
    common.FINAL_DIR.mkdir(parents=True, exist_ok=True)
    cases, mix = load_cases()
    print(f"{cases.height:,} usable daily cases", flush=True)
    # The scored rows: experiment 33's Tuesday cases, in experiment 303's order.
    ordered = score303.ordered_cases().select(*c303.KEY_COLUMNS).with_row_index("order")
    scored_rows = ordered.join(
        cases.select(*c303.KEY_COLUMNS, "row"), on=list(c303.KEY_COLUMNS), how="left"
    ).sort("order")["row"]
    if scored_rows.null_count():
        msg = f"{scored_rows.null_count()} scored cases are missing from the daily cases"
        raise ValueError(msg)
    scored_idx = scored_rows.to_numpy()
    tuesday = (cases["power_fcst_init_time"].dt.weekday() == 2).to_numpy()
    oot_idx = np.flatnonzero(
        tuesday & (cases["power_fcst_init_time"] >= common.OUT_OF_TIME_FIRST_RUN).to_numpy()
    )
    frames = {"": scored_idx, "_oot": oot_idx}
    for suffix, index in frames.items():
        cases[index].drop("valid_us", "row").write_parquet(
            common.FINAL_DIR / f"frames{suffix}.parquet"
        )
    in_hindcast = (cases["power_fcst_init_time"] < HINDCAST_LAST_INIT).to_numpy()
    gate_rows = in_hindcast.copy()
    gate_rows[scored_idx] = True
    all_months = months_between(first=FIRST_TABLE_MONTH, last=LAST_TABLE_MONTH)
    scored_months = [m for m in all_months if label(ym=m) >= common.SCORED_FIRST_MONTH]
    rng = np.random.default_rng(common.EXPERIMENT_ID)
    placebo_pick: dict[str, tuple[int, int]] = {}
    for ym in scored_months:
        earlier = [m for m in all_months if m < ym]
        placebo_pick[label(ym=ym)] = earlier[int(rng.integers(0, len(earlier)))]

    out: dict[str, dict[str, list[np.ndarray]]] = {}
    leakage: list[dict] = []
    tables: list[pl.DataFrame] = []
    reproduction: dict = {}

    def put(*, arm: str, suffix: str, values: np.ndarray) -> None:
        out.setdefault(arm, {"": [], "_oot": []})[suffix].append(values.astype(np.float32))

    for seed in common.SEEDS:
        print(f"seed {seed}: spread factor", flush=True)
        adjusted = c303.apply_spread(cases=cases, mixture=mix[seed])
        gate_cases = cases.filter(pl.Series(gate_rows))
        gate_adj = np.full_like(adjusted, np.nan)
        gate_adj[gate_rows] = c303.apply_spread(cases=gate_cases, mixture=mix[seed][gate_rows])
        print(f"seed {seed}: tables", flush=True)
        default_table = fit_table(cases=cases, adjusted=adjusted, select=in_hindcast)
        gate_table = fit_table(cases=cases, adjusted=gate_adj, select=in_hindcast)
        fitted: dict[tuple[str, str], tuple[np.ndarray, list]] = {}
        for ym in all_months:
            for arm, window in ARMS_MONTHLY.items():
                select, cutoff = window_select(cases=cases, ym=ym, months=window)
                if select.sum() == 0:
                    continue
                latest = cases["valid_time"].filter(pl.Series(select)).max()
                if not isinstance(latest, datetime):
                    msg = f"no valid times selected for {arm} {label(ym=ym)}"
                    raise TypeError(msg)
                leakage.append(
                    {
                        "seed": seed,
                        "arm": arm,
                        "month": label(ym=ym),
                        "rows": int(select.sum()),
                        "latest_valid_used": latest.isoformat(),
                        "month_start": month_start(ym=ym).isoformat(),
                    }
                )
                if latest > cutoff:
                    msg = f"leak: {arm} {label(ym=ym)} uses valid time {latest}"
                    raise ValueError(msg)
                fitted[(arm, label(ym=ym))] = fit_table(
                    cases=cases, adjusted=adjusted, select=select
                )
            quarter_month = add_months(year=ym[0], month=ym[1], delta=-((ym[1] - 1) % 3))
            select, _ = window_select(cases=cases, ym=quarter_month, months=None)
            if select.sum():
                key = ("refq", label(ym=quarter_month))
                if key not in fitted:
                    fitted[key] = fit_table(cases=cases, adjusted=adjusted, select=select)
                fitted[("refq", label(ym=ym))] = fitted[key]
        if seed == common.SEEDS[0]:
            for (arm, month), (table, keys) in fitted.items():
                if arm == "refq":
                    continue
                tables.append(
                    pl.DataFrame(
                        {
                            "arm": arm,
                            "month": month,
                            "site": [k[0] for k in keys for _ in c303.LEVEL_COLUMNS],
                            "lead_day": [k[1] for k in keys for _ in c303.LEVEL_COLUMNS],
                            "level": [lv for _ in keys for lv in c303.LEVEL_COLUMNS],
                            "offset": table.reshape(-1),
                        }
                    )
                )
        print(f"seed {seed}: applying", flush=True)
        for suffix, index in frames.items():
            sub = cases[index]
            quantiles = adjusted[index]
            gate_quantiles = gate_adj[index]
            months = sub["month"].to_numpy()
            result: dict[str, np.ndarray] = {
                arm: np.zeros_like(quantiles, dtype=np.float32)
                for arm in ("default", "x1", "x2", "x3", "refq", "plc")
            }
            if suffix == "":
                result["defaultG"] = np.zeros_like(quantiles, dtype=np.float32)
            for month in np.unique(months):
                rows = np.flatnonzero(months == month)
                piece, q = sub[rows], quantiles[rows]
                result["default"][rows] = ens_final.apply_offsets(
                    cases=piece, quantiles=q, table=default_table[0], keys=default_table[1]
                )
                if suffix == "":
                    result["defaultG"][rows] = ens_final.apply_offsets(
                        cases=piece,
                        quantiles=gate_quantiles[rows],
                        table=gate_table[0],
                        keys=gate_table[1],
                    )
                for arm in ("x1", "x2", "x3", "refq"):
                    table, keys = fitted[(arm, month)]
                    result[arm][rows] = ens_final.apply_offsets(
                        cases=piece, quantiles=q, table=table, keys=keys
                    )
                if suffix == "":
                    # The placebo gives each run, not each month, a randomly drawn earlier month.
                    inits = piece["power_fcst_init_time"].to_numpy()
                    run_picks = {}
                    for init in np.unique(inits):
                        pool = [m for m in all_months if m < tuple(map(int, month.split("-")))]
                        run_picks[init] = pool[int(rng.integers(0, len(pool)))]
                    for init, pick in run_picks.items():
                        sel = inits == init
                        table, keys = fitted[("x2", label(ym=pick))]
                        result["plc"][rows[sel]] = ens_final.apply_offsets(
                            cases=piece.filter(pl.Series(sel)),
                            quantiles=q[sel],
                            table=table,
                            keys=keys,
                        )
                else:
                    result["plc"][rows] = result["x2"][rows]
            for arm, values in result.items():
                put(arm=arm, suffix=suffix, values=values)
    for arm, by_suffix in out.items():
        for suffix, values in by_suffix.items():
            if values:
                np.save(common.FINAL_DIR / f"{arm}{suffix}.npy", np.stack(values))
    pl.concat(tables).write_parquet(common.FINAL_DIR / "tables_seed0.parquet")
    (common.FINAL_DIR / "leakage.json").write_text(json.dumps(leakage, indent=1))
    scored = cases[scored_idx]
    actual = scored["actual_norm"].to_numpy().astype(np.float64)
    kept = ~scored["masked"].to_numpy()
    for arm in ("defaultG", "default"):
        qs = score303.row_metrics(quantiles=np.stack(out[arm][""]), actual=actual)["qs"]
        reproduction[arm] = float(qs[:, kept].mean())
    reproduction["gate_target"] = GATE_SCORE
    reproduction["difference_defaultG"] = reproduction["defaultG"] - GATE_SCORE
    (common.FINAL_DIR / "reproduction.json").write_text(json.dumps(reproduction, indent=1))
    print(json.dumps(reproduction, indent=1), flush=True)


if __name__ == "__main__":
    main()
