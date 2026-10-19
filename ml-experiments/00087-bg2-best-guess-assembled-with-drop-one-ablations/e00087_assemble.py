"""Assemble every arm of experiment 00087 from the members' stored predictions.

Reads the refit predictions that `e00087_fit_demand.py` and `e00087_fit_generators.py` wrote and
builds each arm of `e00087_arms.py`. Every arm is written to `work/arms/<name>.parquet` (run keys,
`seed`, `lead_hours`, `actual` and the 11 quantiles `q01` to `q99` as fractions of capacity, plus
`site`, `group`, and `capacity_mw` so a scorer can convert to MW) and, for the 292 runs of
`fold_a_daily_gap_v1`, to the prediction store through `studies.experiment_store.write_predictions`.
The arms cover all 365 daily runs; the 14-day correction of a run reads the runs before it.

Sources, each copied here (an experiment may not import another): the regime reader and the arm
tables of experiment 00048's `e00048_assemble.py` (`regime_rows`, `attach_actual`,
`stored_quantiles`); the monthly-weight blend B1 of experiment 00049's `e00049_stage0.py`
(`weight_grid`, `rolling_blend`); and the stored-baseline reader of experiment 00013's
`e00013_baselines.py`.

**Primary members.** W (weekly refits), M (monthly), AN (monthly with weekly analogues), PT (pooled
by metered units, monthly), and ST (frozen, with the 14-day correction). Every member has the
spread factor 1.15 about the median. The correction lives only in ST, and in ablation A4 for every
member. At the five offset series W, M, AN, and ST use their offset twins (`_o`, vintage capacity)
in the best guess; PT never has an offset.

**The blend.** The five members' quantiles are averaged level by level with one weight vector per
origin month, chosen on a grid of quarters by the quantile score on earlier cases only: a case
informs month `m` only when its valid time is before the first instant of `m`. The first month
(July 2025) has no history and gets equal weights. The weights are chosen on all 365 runs.

**Generators.** The best guess is the weekly forecast-trained model with no spread factor (the
frozen N1 of experiment 00033 had none). The weather-trained reference gets experiment 00033's
month-by-month fitted spread factor (`e00087_gen_common.apply_spread_factor`). The design's
ablation "spread off" is therefore moot for the forecast-trained model; the ablation recorded here
is "spread factor on", which applies the fitted factor to the weekly forecast-trained mixture, the
factor being fitted on that mixture's own earlier cases.

**References.** `ref_climatology`, `ref_manual_heuristic`, and `ref_xgboost_baseline` are
recomputed at 11 levels as the quantiles of the stored members' forecasts (51 members for
climatology and the XGBoost baseline, 13 for the manual heuristic), because experiment 00048's
stored arms hold nine levels. The `xgboost_baseline` rows exist for the whole validation year: the
local Delta table `data/power_forecasts` holds fold `mid_2025_to_mid_2026` for 380 daily runs
(power forecast initialisation times 2025-06-16 09:00 to 2026-06-30 09:00, which contains the 365
runs here), 31 series, and 51 members, so no extra run of the cross-validation asset is needed. The
table is a local store written by delta-rs (partitioned by `experiment_name` and `fold_id`); the
table itself does not say which job wrote it, so whoever owns experiments 13 and 33 must confirm
that it came from the local Dagster cross-validation assets and not from S3. Its band comes from
the spread of 51 point forecasts and is far too narrow, so its tails are noise. `ref_perfect`
holds the actual as every quantile. The NGED `capacity_forecast` arm needs the private node table
and is built in `e00087_curtailment.py`.

    uv run python ml-experiments/00087-*/e00087_assemble.py [--parts demand generators references
        perfect] [--smoke] [--no-store]
    uv run python ml-experiments/00087-*/e00087_assemble.py --self-test
"""

import argparse
import itertools
import shutil
import tempfile
import time
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final, Literal

import e00087_common as common
import e00087_gen_common as gen
import e00087_offset as offsets
import e00087_regimes as regimes
import e00087_stack as stack
import numpy as np
import polars as pl

from studies import experiment_store

OffsetType = Literal["none", "vintage", "held"]
"""`none`: no base margin. `vintage`: the offset with the capacity vintage of each run. `held`: the
offset with the capacity held at its 2024/25 value (ablation A6)."""

KEY_COLUMNS: Final[tuple[str, ...]] = (*common.KEYS, "seed")
ARM_COLUMNS: Final[tuple[str, ...]] = (
    *KEY_COLUMNS,
    "lead_hours",
    "actual",
    *common.LEVEL_COLUMNS,
    "site",
    "group",
    "capacity_mw",
)
OFFSET_SUFFIX: Final[dict[str, str]] = {"none": "", "vintage": "_o", "held": "_oheld"}
FOLD_ID: Final[str] = "mid_2025_to_mid_2026"
STORED_BASELINES: Final[dict[str, str]] = {
    "ref_climatology": "climatology",
    "ref_manual_heuristic": "manual_heuristic",
    "ref_xgboost_baseline": "xgboost_baseline",
}
"""Arm name to the `experiment_name` of the `power_forecasts` Delta table."""
BASELINE_CHUNK_RUNS: Final[int] = 10
WEIGHT_GRID_STEP: Final[int] = 4
"""Blend weights are multiples of 1/4."""
SCORING_CHUNK_ROWS: Final[int] = 1_000_000
REPEAT_DIR: Final[Path] = common.WORK_DIR / "anchor_preds_repeat"
"""Where the noise-floor repeat fit of the weekly model writes (same layout as `anchor_preds`)."""
OPTIONAL_ARMS: Final[frozenset[str]] = frozenset({"p_a6_capacity_held", "g_a_wind_margin"})
"""Arms a strict run may skip when their fits are absent; every other missing file is an error."""
SELF_TEST_RUNS: Final[int] = 45
SELF_TEST_STEPS: Final[int] = 96


@dataclass(frozen=True)
class MemberSpec:
    """Where one blend member's refit predictions live.

    Attributes:
        config: The folder under `anchor_preds`.
        regime: The retraining regime (a key of `regimes.REGIMES`).
        offsettable: Whether the member has offset twins at the offset series.
    """

    config: str
    regime: str
    offsettable: bool


MEMBERS: Final[dict[str, MemberSpec]] = {
    "W": MemberSpec("n1h", "weekly", offsettable=True),
    "M": MemberSpec("n1h", "monthly", offsettable=True),
    "AN": MemberSpec("an", "monthly", offsettable=True),
    "PT": MemberSpec("pt", "monthly", offsettable=False),
    "ST": MemberSpec("frozen", "frozen", offsettable=True),
}
BLEND_FIVE: Final[tuple[str, ...]] = ("W", "M", "AN", "PT", "ST")
BLEND_THREE: Final[tuple[str, ...]] = ("W", "AN", "ST")
BLEND_MONTHLY_IN_WEEKLY_SLOT: Final[tuple[str, ...]] = ("M", "M", "AN", "PT", "ST")


@dataclass(frozen=True)
class Context:
    """Everything an arm builder reads.

    Attributes:
        series: `common.series_table()`.
        sites: `gen.site_table()`.
        arrays: Each demand site's observed power over capacity on the dense grid.
        power_of: Each series' (demand and generator) observed power over capacity by identifier.
        labels: Each series identifier's site label and group (`Primary`, `BSP`, `GSP`, `solar`, or
            `wind`).
        capacity: Each series' effective capacity in MW.
        origins: The `power_fcst_init_time` of the runs to cover.
        demand_root: The folder of demand refit predictions.
        generator_root: The folder of generator refit predictions.
        arms_dir: Where arm tables are written.
        results_dir: Where the blend weights are written.
        strict: Raise if a refit file or an assigned run is missing.
        store: Also write the 292 gap runs to the prediction store.
    """

    series: pl.DataFrame
    sites: pl.DataFrame
    arrays: dict[str, common.SeriesArrays]
    power_of: dict[int, np.ndarray]
    labels: dict[int, tuple[str, str]]
    capacity: dict[int, float]
    origins: list[datetime]
    demand_root: Path
    generator_root: Path
    arms_dir: Path
    results_dir: Path
    strict: bool
    store: bool

    def sites_of(self, *, groups: Sequence[str]) -> list[str]:
        """Return the demand site labels whose substation type is in `groups`."""
        return self.series.filter(pl.col("substation_type").is_in(list(groups)))["site"].to_list()


def make_context(
    *, smoke: bool, store: bool, origins: list[datetime] | None = None, root: Path | None = None
) -> Context:
    """Build the context: series tables, observed power, and the folders to read and write.

    Args:
        smoke: Read the smoke folders, tolerate missing files, write apart, and do not store.
        store: Write the gap runs to the prediction store (ignored for a smoke run).
        origins: The runs to cover, or `None` for the 365 daily runs.
        root: A folder that replaces `work/` for every input and output (the self-test).

    Returns:
        The context.
    """
    series = common.series_table()
    sites = gen.site_table()
    arrays = common.load_power_arrays(series=series)
    gen_power = gen.full_power(sites=sites)
    power_of = {
        **{int(r["time_series_id"]): arrays[r["site"]].power for r in series.iter_rows(named=True)},
        **{int(k): v for k, v in gen_power.items()},
    }
    labels = {
        **{
            int(r["time_series_id"]): (r["site"], r["substation_type"])
            for r in series.iter_rows(named=True)
        },
        **{
            int(r["time_series_id"]): (r["site"], r["technology"])
            for r in sites.iter_rows(named=True)
        },
    }
    capacity = {
        **{
            int(r["time_series_id"]): float(r["effective_capacity_mw"])
            for r in series.iter_rows(named=True)
        },
        **{
            int(r["time_series_id"]): float(r["effective_capacity_mw"])
            for r in sites.iter_rows(named=True)
        },
    }
    base = root if root is not None else common.WORK_DIR
    suffix = "_smoke" if smoke else ""
    return Context(
        series=series,
        sites=sites,
        arrays=arrays,
        power_of=power_of,
        labels=labels,
        capacity=capacity,
        origins=origins if origins is not None else common.all_origins(),
        demand_root=base / f"anchor_preds{suffix}",
        generator_root=base / f"generator_preds{suffix}",
        arms_dir=base / f"arms{suffix}",
        results_dir=base / f"results{suffix}",
        strict=not smoke,
        store=store and not smoke and root is None,
    )


# --- reading refit predictions -----------------------------------------------------------------


def job_path(*, root: Path, config: str, site: str, refit: datetime, seed: int = 0) -> Path:
    """Return the file that one refit's predictions for one series go to, as the fit scripts do."""
    return root / config / f"{site}_s{seed}_{refit:%Y%m%dT%H%M}.parquet"


def regime_rows(
    *,
    root: Path,
    config: str,
    regime: str,
    site: str,
    origins: list[datetime],
    strict: bool,
    any_seed: bool = False,
) -> pl.DataFrame:
    """Return one series' quantiles for a configuration and regime, from the chosen refits.

    Copied from experiment 00048's `regime_rows`.

    Args:
        root: The folder of refit predictions.
        config: The configuration folder under `root`.
        regime: A key of `regimes.REGIMES`.
        site: The series label.
        origins: The `power_fcst_init_time` of the runs to cover.
        strict: Raise if a refit file or a run assigned to it is missing.
        any_seed: Accept the file of any fitting seed (the noise-floor repeat), and set `seed` to 0.

    Returns:
        The quantiles keyed by series, run, valid time, and seed.

    Raises:
        ValueError: If `strict` and a refit file lacks a run assigned to it.
        FileNotFoundError: If no refit file of the series exists for the runs.
    """
    pieces = []
    for refit, chosen in regimes.served(name=regime, origins=origins).items():
        path = job_path(root=root, config=config, site=site, refit=refit)
        if any_seed:
            found = sorted((root / config).glob(f"{site}_s*_{refit:%Y%m%dT%H%M}.parquet"))
            path = found[0] if found else path
        if not path.exists():
            if strict:
                msg = f"missing refit file {path}"
                raise FileNotFoundError(msg)
            continue
        frame = pl.read_parquet(path).filter(pl.col("power_fcst_init_time").is_in(chosen))
        if strict and frame["power_fcst_init_time"].n_unique() != len(chosen):
            msg = f"{config} {site} refit {refit}: the file lacks some assigned runs"
            raise ValueError(msg)
        pieces.append(frame)
    if not pieces:
        msg = f"no refit file of {config} {site} serves the runs"
        raise FileNotFoundError(msg)
    table = pl.concat(pieces)
    return table.with_columns(seed=pl.lit(0).cast(pl.Int8)) if any_seed else table


def raw_member(
    *, ctx: Context, member: str, offset: OffsetType, sites: Sequence[str]
) -> pl.DataFrame:
    """Return a member's raw quantiles for some series, offset twins at the offset series.

    Args:
        ctx: The context.
        member: A key of `MEMBERS`.
        offset: Which offset twin the offset series use. A member with no offset twins, and a
            series that is not an offset series, use the plain configuration.
        sites: The series labels to read.

    Returns:
        The member's quantiles for all the context's runs, before the spread factor.

    Raises:
        FileNotFoundError: If nothing is found (or, in a strict run, if any file is missing).
    """
    spec = MEMBERS[member]
    pieces = []
    for site in sites:
        twin = spec.offsettable and offset != "none" and site in offsets.OFFSET_SITES
        config = spec.config + (OFFSET_SUFFIX[offset] if twin else "")
        try:
            pieces.append(
                regime_rows(
                    root=ctx.demand_root,
                    config=config,
                    regime=spec.regime,
                    site=site,
                    origins=ctx.origins,
                    strict=ctx.strict,
                )
            )
        except FileNotFoundError:
            if ctx.strict:
                raise
    if not pieces:
        msg = f"no predictions of member {member} ({offset}) under {ctx.demand_root}"
        raise FileNotFoundError(msg)
    return pl.concat(pieces)


def finish_member(*, ctx: Context, raw: pl.DataFrame, correct: bool, spread: bool) -> pl.DataFrame:
    """Apply the 14-day correction and the spread factor 1.15 to a member's raw quantiles.

    Args:
        ctx: The context.
        raw: `raw_member`'s table.
        correct: Add the 14-day correction (the stack, and ablation A4).
        spread: Widen the quantiles about the median by 1.15.

    Returns:
        The keys, `seed`, and the 11 sorted quantile columns.
    """
    if not correct:
        return stack.stack(frame=raw, correct=False, spread=spread)
    corrected = stack.with_corrections(
        base=raw, series=ctx.series, arrays=ctx.arrays, window_days=stack.CORRECTION_DAYS
    )
    return stack.stack(frame=corrected, correct=True, spread=spread)


# --- arm tables --------------------------------------------------------------------------------


def actual_array(*, frame: pl.DataFrame, power_of: Mapping[int, np.ndarray]) -> np.ndarray:
    """Return the observed power over capacity at each row's valid time, NaN where unobserved."""
    ids = frame["time_series_id"].to_numpy()
    index = np.asarray(common.grid_index(times=frame["valid_time"]))
    out = np.full(frame.height, np.nan, dtype=np.float32)
    for series_id, power in power_of.items():
        mask = ids == series_id
        if mask.any():
            out[mask] = common.take(values=power, index=index[mask])
    return out


def finalise(*, ctx: Context, frame: pl.DataFrame) -> pl.DataFrame:
    """Add `lead_hours`, `actual`, `site`, `group`, and `capacity_mw` and order the columns.

    Args:
        ctx: The context.
        frame: Keys, `seed`, and the 11 quantile columns.

    Returns:
        The arm table in the order of `ARM_COLUMNS`, sorted by the keys.
    """
    ids = sorted({int(i) for i in frame["time_series_id"].unique().to_list()})
    mapping = pl.DataFrame(
        {
            "time_series_id": pl.Series(ids, dtype=frame.schema["time_series_id"]),
            "site": [ctx.labels[i][0] for i in ids],
            "group": [ctx.labels[i][1] for i in ids],
            "capacity_mw": pl.Series([ctx.capacity[i] for i in ids], dtype=pl.Float32),
        }
    )
    return (
        frame.join(mapping, on="time_series_id", how="left", validate="m:1", maintain_order="left")
        .with_columns(
            lead_hours=(
                (pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_minutes() / 60.0
            ).cast(pl.Float32),
            actual=pl.Series(actual_array(frame=frame, power_of=ctx.power_of)),
        )
        .select(ARM_COLUMNS)
        .sort(*KEY_COLUMNS)
    )


def write_arm(*, ctx: Context, name: str, frame: pl.DataFrame) -> None:
    """Write an arm to `work/arms` and, for the gap runs before the holdout, to the store.

    Args:
        ctx: The context.
        name: The arm's name (a key of `arms.BY_NAME`).
        frame: The finalised arm table.
    """
    common.atomic_write(frame=frame, target=ctx.arms_dir / f"{name}.parquet")
    if not ctx.store:
        return
    gap = common.origins_of_subset(subset="gap")
    rows = frame.filter(
        pl.col("power_fcst_init_time").is_in(gap) & (pl.col("valid_time") < common.HOLDOUT)
    )
    try:
        experiment_store.write_predictions(
            experiment_id=common.EXPERIMENT_ID,
            name=name,
            frame=rows,
            kind="quantiles",
            row_set_id=common.ROW_SET_ID,
        )
    except FileExistsError:
        print(f"{name}: already in the prediction store", flush=True)


def emit(*, ctx: Context, name: str, table: pl.DataFrame, started: float) -> None:
    """Finalise and write one arm and print a progress line; an empty arm is skipped."""
    if table.is_empty():
        print(f"{name}: no cases shared by the members, skipped", flush=True)
        return
    frame = finalise(ctx=ctx, frame=table)
    if frame.select(KEY_COLUMNS).is_duplicated().any():
        msg = f"arm {name} holds two rows with the same series, run, valid time, and seed"
        raise ValueError(msg)
    write_arm(ctx=ctx, name=name, frame=frame)
    print(f"{name}: {frame.height:,} rows, {time.monotonic() - started:.0f} s", flush=True)


# --- the monthly-weight blend ------------------------------------------------------------------


def weight_grid(*, members: int, step: int = WEIGHT_GRID_STEP) -> np.ndarray:
    """Return every weight vector of multiples of `1/step` that sums to 1, shape (vectors, members).

    Copied from experiment 00049's `weight_grid`.
    """
    rows = [v for v in itertools.product(range(step + 1), repeat=members) if sum(v) == step]
    return np.array(rows, dtype=np.float64) / step


def blend_quantiles(*, members: Sequence[np.ndarray], weights: np.ndarray) -> np.ndarray:
    """Return the weighted mean of the members' quantile columns.

    Each member's quantiles rise along the levels, so the mean does too and needs no sorting.

    Args:
        members: One array per member, shape (cases, 11).
        weights: Either one weight per member, shape (members,), or one vector per case, shape
            (cases, members).

    Returns:
        Shape (cases, 11), float64.
    """
    out = np.zeros(members[0].shape, dtype=np.float64)
    for slot, quantiles in enumerate(members):
        weight = weights[slot] if weights.ndim == 1 else weights[:, [slot]]
        if np.ndim(weight) == 0 and weight == 0.0:
            continue
        out += weight * quantiles
    return out


def case_scores(
    *, members: Sequence[np.ndarray], weights: np.ndarray, actual: np.ndarray
) -> np.ndarray:
    """Return the quantile score per case, in fractions of capacity."""
    scores = np.empty(actual.size, dtype=np.float64)
    for start in range(0, actual.size, SCORING_CHUNK_ROWS):
        part = slice(start, start + SCORING_CHUNK_ROWS)
        blended = blend_quantiles(members=[m[part] for m in members], weights=weights)
        scores[part] = common.pinball_quantile_score(
            quantiles=blended, actual=actual[part].astype(np.float64)
        )
    return scores


def rolling_blend(
    *,
    keys: pl.DataFrame,
    actual: np.ndarray,
    members: Sequence[np.ndarray],
    names: Sequence[str],
) -> tuple[np.ndarray, pl.DataFrame]:
    """Blend the members with one weight vector per origin month, chosen on earlier cases only.

    Adapted from experiment 00049's `rolling_blend`. For each origin month `m` the vector is the
    one on the quarter grid with the lowest mean quantile score over the cases whose valid time
    is before the first instant of `m` (so every case's observation is known before any run of
    `m` starts). A month with no such case gets equal weights. Ties go to the vector nearest to
    equal weights. A case with a missing observation, or a valid time on or after the holdout,
    never informs a vector.

    Args:
        keys: The cases' `power_fcst_init_time` and `valid_time`, one row per case.
        actual: The observed power over capacity per case (NaN if unobserved).
        members: One array per member, shape (cases, 11), float32 or float64.
        names: The members' names, for the weight table.

    Returns:
        The blended quantiles, shape (cases, 11) float32, and a table with one row per origin month:
        `month`, `history_cases`, and a `w_<slot>_<name>` column per member.
    """
    grid = weight_grid(members=len(members))
    equal = np.full(len(members), 1.0 / len(members))
    nearest = np.argsort(np.abs(grid - equal).sum(axis=1), kind="stable")
    origin_month = keys["power_fcst_init_time"].dt.strftime("%Y-%m").to_numpy()
    valid_month = keys["valid_time"].dt.strftime("%Y-%m").to_numpy()
    usable = np.isfinite(actual) & (keys["valid_time"] < common.HOLDOUT).to_numpy()
    valid_labels, valid_index = np.unique(valid_month[usable], return_inverse=True)
    counts = np.bincount(valid_index, minlength=len(valid_labels)).astype(np.float64)
    sums = np.zeros((len(grid), len(valid_labels)))
    used_members = [m[usable] for m in members]
    used_actual = actual[usable]
    for g, vector in enumerate(grid):
        scores = case_scores(members=used_members, weights=vector, actual=used_actual)
        sums[g] = np.bincount(valid_index, weights=scores, minlength=len(valid_labels))
    weights = np.tile(equal, (len(actual), 1))
    rows = []
    for month in sorted(set(origin_month.tolist())):
        history = valid_labels < month
        count = float(counts[history].sum())
        if count > 0:
            mean = sums[:, history].sum(axis=1) / count
            chosen = grid[nearest[np.argsort(mean[nearest], kind="stable")[0]]]
        else:
            chosen = equal
        weights[origin_month == month] = chosen
        rows.append(
            {
                "month": month,
                "history_cases": int(count),
                **{
                    f"w_{i}_{n}": float(w)
                    for i, (n, w) in enumerate(zip(names, chosen, strict=True))
                },
            }
        )
    blended = blend_quantiles(members=members, weights=weights)
    return blended.astype(np.float32), pl.DataFrame(rows)


def align(
    *, tables: Mapping[str, pl.DataFrame], strict: bool
) -> tuple[pl.DataFrame, dict[str, np.ndarray]]:
    """Put several members' tables on the same sorted cases.

    Args:
        tables: Each member's keys, `seed`, and 11 quantile columns.
        strict: Raise if the tables do not hold the same cases.

    Returns:
        The shared keys and each member's quantiles as a float32 array in the keys' order.

    Raises:
        ValueError: If `strict` and the members' cases differ.
    """
    shared: pl.DataFrame | None = None
    for table in tables.values():
        these = table.select(KEY_COLUMNS)
        shared = these if shared is None else shared.join(these, on=list(KEY_COLUMNS), how="inner")
    if shared is None:
        msg = "no member tables to align"
        raise ValueError(msg)
    shared = shared.sort(*KEY_COLUMNS)
    arrays = {}
    for name, table in tables.items():
        if strict and table.height != shared.height:
            msg = f"member {name} holds {table.height:,} cases, the shared set {shared.height:,}"
            raise ValueError(msg)
        arrays[name] = (
            shared.join(
                table, on=list(KEY_COLUMNS), how="left", validate="1:1", maintain_order="left"
            )
            .select(common.LEVEL_COLUMNS)
            .to_numpy()
            .astype(np.float32)
        )
    return shared, arrays


def blend_arm(
    *,
    ctx: Context,
    slots: Sequence[str],
    offset: OffsetType,
    correct_all: bool = False,
    spread: bool = True,
    weights_name: str,
) -> pl.DataFrame:
    """Return a Primary blend: the members' tables, then the monthly-weight blend of them.

    Args:
        ctx: The context.
        slots: The blend's members in order (a name may repeat, as ablation A3 does).
        offset: The offset twin the offset series use.
        correct_all: Add the 14-day correction to every member (ablation A4); otherwise only `ST`.
        spread: Apply the spread factor 1.15 to each member.
        weights_name: The name the weight table is written under.

    Returns:
        The blend's keys, `seed`, and 11 quantile columns.
    """
    sites = ctx.sites_of(groups=("Primary",))
    tables: dict[str, pl.DataFrame] = {}
    for member in dict.fromkeys(slots):
        raw = raw_member(ctx=ctx, member=member, offset=offset, sites=sites)
        tables[member] = finish_member(
            ctx=ctx, raw=raw, correct=correct_all or member == "ST", spread=spread
        )
        del raw
    keys, by_member = align(tables=tables, strict=ctx.strict)
    del tables
    actual = actual_array(frame=keys, power_of=ctx.power_of)
    blended, table = rolling_blend(
        keys=keys,
        actual=actual,
        members=[by_member[m] for m in slots],
        names=list(slots),
    )
    ctx.results_dir.mkdir(parents=True, exist_ok=True)
    table.write_parquet(ctx.results_dir / f"blend_weights_{weights_name}.parquet")
    with pl.Config(tbl_rows=24, tbl_cols=12):
        print(f"blend weights {weights_name}:\n{table}", flush=True)
    return keys.hstack(pl.DataFrame(blended, schema=list(common.LEVEL_COLUMNS)))


# --- the demand arms ---------------------------------------------------------------------------


def single_member_arm(
    *, ctx: Context, member: str, groups: Sequence[str], offset: OffsetType, correct: bool
) -> pl.DataFrame:
    """Return one member's table for the series of some groups, with the spread factor."""
    raw = raw_member(ctx=ctx, member=member, offset=offset, sites=ctx.sites_of(groups=groups))
    return finish_member(ctx=ctx, raw=raw, correct=correct, spread=True)


def build_primary(*, ctx: Context, only: Collection[str] | None = None) -> None:
    """Write the Primary staircase S1 to S6 and the ablations A2 to A7 (A1 is S5).

    Args:
        ctx: The context.
        only: Build only these arms, or every arm if `None` (an early reproduction check needs
            only the weekly arm).
    """
    primary = ("Primary",)
    plan: dict[str, Callable[[], pl.DataFrame]] = {
        "p_s1_frozen": lambda: single_member_arm(
            ctx=ctx, member="ST", groups=primary, offset="none", correct=False
        ),
        "p_s2_stack": lambda: single_member_arm(
            ctx=ctx, member="ST", groups=primary, offset="none", correct=True
        ),
        "p_s3_monthly": lambda: single_member_arm(
            ctx=ctx, member="M", groups=primary, offset="none", correct=False
        ),
        "p_s4_weekly": lambda: single_member_arm(
            ctx=ctx, member="W", groups=primary, offset="none", correct=False
        ),
        "p_s5_blend_no_offset": lambda: blend_arm(
            ctx=ctx, slots=BLEND_FIVE, offset="none", weights_name="s5"
        ),
        "p_s6_bg2": lambda: blend_arm(
            ctx=ctx, slots=BLEND_FIVE, offset="vintage", weights_name="s6"
        ),
        "p_a2_no_blend": lambda: single_member_arm(
            ctx=ctx, member="W", groups=primary, offset="vintage", correct=False
        ),
        "p_a3_monthly_in_weekly_slot": lambda: blend_arm(
            ctx=ctx, slots=BLEND_MONTHLY_IN_WEEKLY_SLOT, offset="vintage", weights_name="a3"
        ),
        "p_a4_correction_all": lambda: blend_arm(
            ctx=ctx, slots=BLEND_FIVE, offset="vintage", correct_all=True, weights_name="a4"
        ),
        "p_a5_spread_off": lambda: blend_arm(
            ctx=ctx, slots=BLEND_FIVE, offset="vintage", spread=False, weights_name="a5"
        ),
        "p_a6_capacity_held": lambda: blend_arm(
            ctx=ctx, slots=BLEND_FIVE, offset="held", weights_name="a6"
        ),
        "p_a7_three_members": lambda: blend_arm(
            ctx=ctx, slots=BLEND_THREE, offset="vintage", weights_name="a7"
        ),
    }
    for name, build in plan.items():
        if only is not None and name not in only:
            continue
        started = time.monotonic()
        try:
            table = build()
        except FileNotFoundError as error:
            if ctx.strict and name not in OPTIONAL_ARMS:
                raise
            print(f"{name}: skipped, {error}", flush=True)
            continue
        emit(ctx=ctx, name=name, table=table, started=started)
    if only is None:
        build_noise_repeat(ctx=ctx)


def build_noise_repeat(*, ctx: Context) -> None:
    """Write the repeat fit of the monthly model, if `work/anchor_preds_repeat` holds one."""
    if not (REPEAT_DIR / "n1h").exists() or ctx.demand_root.name != "anchor_preds":
        print("p_noise_repeat_monthly: no repeat fit found, skipped", flush=True)
        return
    started = time.monotonic()
    pieces = []
    for site in ctx.sites_of(groups=("Primary",)):
        try:
            pieces.append(
                regime_rows(
                    root=REPEAT_DIR,
                    config="n1h",
                    regime="monthly",
                    site=site,
                    origins=ctx.origins,
                    strict=False,
                    any_seed=True,
                )
            )
        except FileNotFoundError:
            continue
    if not pieces:
        print("p_noise_repeat_monthly: no repeat fit found, skipped", flush=True)
        return
    table = stack.stack(frame=pl.concat(pieces), correct=False, spread=True)
    emit(ctx=ctx, name="p_noise_repeat_monthly", table=table, started=started)


def build_bspgsp(*, ctx: Context, only: Collection[str] | None = None) -> None:
    """Write the BSP and GSP staircase and ablations.

    Args:
        ctx: The context.
        only: Build only these arms, or every arm if `None`.
    """
    groups = ("BSP", "GSP")
    plan: dict[str, Callable[[], pl.DataFrame]] = {
        "b_frozen": lambda: single_member_arm(
            ctx=ctx, member="ST", groups=groups, offset="none", correct=False
        ),
        "b_stack": lambda: single_member_arm(
            ctx=ctx, member="ST", groups=groups, offset="none", correct=True
        ),
        "b_monthly": lambda: single_member_arm(
            ctx=ctx, member="M", groups=groups, offset="none", correct=False
        ),
        "b_weekly": lambda: single_member_arm(
            ctx=ctx, member="W", groups=groups, offset="none", correct=False
        ),
        "b_a_correction": lambda: single_member_arm(
            ctx=ctx, member="W", groups=groups, offset="none", correct=True
        ),
    }
    for name, build in plan.items():
        if only is not None and name not in only:
            continue
        started = time.monotonic()
        try:
            table = build()
        except FileNotFoundError as error:
            if ctx.strict and name not in OPTIONAL_ARMS:
                raise
            print(f"{name}: skipped, {error}", flush=True)
            continue
        emit(ctx=ctx, name=name, table=table, started=started)


# --- the generator arms ------------------------------------------------------------------------


def generator_mixtures(*, ctx: Context, arm: str) -> pl.DataFrame | None:
    """Return every refit file of a generator arm, as written (no spread factor), or `None`."""
    files = sorted((ctx.generator_root / arm).glob("*.parquet"))
    if not files:
        return None
    table = pl.concat([pl.read_parquet(path) for path in files]).filter(
        pl.col("power_fcst_init_time").is_in(ctx.origins)
    )
    runs = table.group_by("time_series_id").agg(runs=pl.col("power_fcst_init_time").n_unique())
    if ctx.strict and (runs["runs"] != len(ctx.origins)).any():
        msg = f"generator arm {arm}: some series lack runs; runs per series {runs.rows()}"
        raise ValueError(msg)
    return table


def with_spread_factor(*, ctx: Context, table: pl.DataFrame) -> pl.DataFrame:
    """Apply experiment 00033's month-by-month fitted spread factor to a generator mixture table.

    Args:
        ctx: The context.
        table: Keys, `seed`, and the 11 quantile columns of generator series, no spread factor.

    Returns:
        The table with the scaled quantile columns, one row per case with an observation.
    """
    actual = actual_array(frame=table, power_of=ctx.power_of)
    technology = pl.Series(
        [ctx.labels[int(i)][1] for i in table["time_series_id"].to_numpy()], dtype=pl.String
    )
    cases = table.with_columns(
        actual_norm=pl.Series(actual).cast(pl.Float64),
        technology=technology,
        lead_hours=(
            (pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_minutes() / 60.0
        ),
    ).filter(pl.col("actual_norm").is_finite() & (pl.col("valid_time") < common.HOLDOUT))
    cases = cases.with_columns(
        band=pl.Series(gen.band_names(lead_hours=cases["lead_hours"].to_numpy()))
    )
    pieces = []
    for name in ("solar", "wind"):
        part = cases.filter(pl.col("technology") == name)
        if part.is_empty():
            continue
        scaled = gen.apply_spread_factor(
            cases=part,
            mixture=part.select(common.LEVEL_COLUMNS).to_numpy().astype(np.float64),
            technology=name,
        )
        pieces.append(
            part.drop(common.LEVEL_COLUMNS).hstack(
                pl.DataFrame(scaled.astype(np.float32), schema=list(common.LEVEL_COLUMNS))
            )
        )
    return pl.concat(pieces).select(*KEY_COLUMNS, *common.LEVEL_COLUMNS)


def build_generators(*, ctx: Context) -> None:
    """Write the generator staircase, ablations, and the wind add-one arm."""
    started = time.monotonic()
    raw = {
        arm: generator_mixtures(ctx=ctx, arm=arm)
        for arm in ("era5_frozen_mixture", "ens_frozen", "ens_weekly", "ens_weekly_margin")
    }
    plain = {
        arm: table.select(*KEY_COLUMNS, *common.LEVEL_COLUMNS)
        for arm, table in raw.items()
        if table is not None
    }
    plan: dict[str, tuple[str, bool]] = {
        "g_prev_best_era5": ("era5_frozen_mixture", True),
        "g_ens_frozen": ("ens_frozen", False),
        "g_ens_weekly": ("ens_weekly", False),
        "g_a_spread_on": ("ens_weekly", True),
        "g_a_wind_margin": ("ens_weekly_margin", False),
    }
    for name, (source, spread) in plan.items():
        if source not in plain:
            if ctx.strict and name not in OPTIONAL_ARMS:
                msg = f"{name}: no {source} predictions under {ctx.generator_root}"
                raise FileNotFoundError(msg)
            print(f"{name}: skipped, no {source} predictions", flush=True)
            continue
        table = with_spread_factor(ctx=ctx, table=plain[source]) if spread else plain[source]
        emit(ctx=ctx, name=name, table=table, started=started)
        started = time.monotonic()


# --- the references ----------------------------------------------------------------------------


def stored_quantiles(
    *, experiment_name: str, capacity: pl.DataFrame, origins: list[datetime]
) -> pl.DataFrame:
    """Return one stored baseline's capacity-normalised 11 quantiles of its members' forecasts.

    Copied from experiment 00048's `stored_quantiles` and widened to 11 levels. The members are
    sorted within each case and the quantile is the linear interpolation of the sorted members, as
    `numpy.quantile` and Polars' `quantile(..., "linear")` both define it.

    Args:
        experiment_name: The `experiment_name` of the `power_forecasts` Delta table.
        capacity: `time_series_id` and `capacity` (effective capacity in MW) of the series.
        origins: The runs' `power_fcst_init_time`.

    Returns:
        The quantiles divided by the series' capacity, keyed by series, run, and valid time.

    Raises:
        ValueError: If a case holds duplicate member rows.
    """
    frame = (
        pl.scan_delta(str(common.POWER_FORECASTS_DIR))
        .filter(
            (pl.col("experiment_name") == experiment_name)
            & (pl.col("fold_id") == FOLD_ID)
            & pl.col("power_fcst_init_time").is_in(origins)
            & pl.col("time_series_id").is_in(capacity["time_series_id"].to_list())
        )
        .select(
            "time_series_id",
            "power_fcst_init_time",
            "valid_time",
            "ensemble_member",
            pl.col("power_fcst").cast(pl.Float64),
        )
        .collect(engine="streaming")
        .sort("time_series_id", "power_fcst_init_time", "valid_time", "ensemble_member")
    )
    if frame.is_empty():
        return pl.DataFrame()
    case_keys = ["time_series_id", "power_fcst_init_time", "valid_time"]
    members = int(frame["ensemble_member"].n_unique())
    values = frame["power_fcst"].to_numpy()
    first = frame.gather_every(members).select(case_keys)
    last = frame.slice(members - 1).gather_every(members).select(case_keys)
    if frame.height % members == 0 and first.equals(last):
        cube = values.reshape(-1, members)
        quantiles = np.quantile(cube, common.LEVELS, axis=1).T
        table = first.hstack(pl.DataFrame(quantiles, schema=list(common.LEVEL_COLUMNS)))
    else:
        table = frame.group_by(case_keys).agg(
            *[
                pl.col("power_fcst").quantile(float(level), "linear").alias(name)
                for level, name in zip(common.LEVELS, common.LEVEL_COLUMNS, strict=True)
            ],
            members=pl.col("ensemble_member").n_unique(),
            stored_rows=pl.len(),
        )
        if table.filter(pl.col("members") != pl.col("stored_rows")).height:
            msg = f"{experiment_name}: a case holds duplicate member rows"
            raise ValueError(msg)
        table = table.drop("members", "stored_rows")
    return (
        table.cast({"time_series_id": pl.Int32})
        .join(capacity, on="time_series_id", how="left", validate="m:1")
        .with_columns(*[pl.col(name) / pl.col("capacity") for name in common.LEVEL_COLUMNS])
        .drop("capacity")
    )


def build_references(*, ctx: Context, limit_series: int | None, limit_runs: int | None) -> None:
    """Write the three stored baselines at 11 levels for the 29 series.

    Args:
        ctx: The context.
        limit_series: Keep only this many series (the smoke test), or `None`.
        limit_runs: Keep only this many runs (the smoke test), or `None`.
    """
    capacity = pl.DataFrame(
        {
            "time_series_id": pl.Series(list(ctx.capacity), dtype=pl.Int32),
            "capacity": list(ctx.capacity.values()),
        }
    )
    if limit_series is not None:
        capacity = capacity.head(limit_series)
    origins = ctx.origins if limit_runs is None else ctx.origins[:limit_runs]
    for name, experiment_name in STORED_BASELINES.items():
        started = time.monotonic()
        pieces = []
        for start in range(0, len(origins), BASELINE_CHUNK_RUNS):
            chunk = stored_quantiles(
                experiment_name=experiment_name,
                capacity=capacity,
                origins=origins[start : start + BASELINE_CHUNK_RUNS],
            )
            if not chunk.is_empty():
                pieces.append(chunk)
        if not pieces:
            print(f"{name}: no stored forecasts for these runs, skipped", flush=True)
            continue
        table = pl.concat(pieces).with_columns(seed=pl.lit(common.SEED).cast(pl.Int8))
        runs = int(table["power_fcst_init_time"].n_unique())
        print(f"{name}: {runs} of {len(origins)} runs found", flush=True)
        emit(ctx=ctx, name=name, table=table.cast({"time_series_id": pl.Int32}), started=started)


def build_perfect(*, ctx: Context) -> None:
    """Write `ref_perfect`: the actual as every quantile, on the cases of the best-guess arms."""
    pieces = []
    for name in ("p_s6_bg2", "b_weekly", "g_ens_weekly"):
        path = ctx.arms_dir / f"{name}.parquet"
        if path.exists():
            pieces.append(pl.read_parquet(path))
    if not pieces:
        print("ref_perfect: no best-guess arm found, skipped", flush=True)
        return
    started = time.monotonic()
    frame = pl.concat(pieces).filter(pl.col("actual").is_finite())
    perfect = frame.with_columns(*[pl.col("actual").alias(q) for q in common.LEVEL_COLUMNS])
    write_arm(ctx=ctx, name="ref_perfect", frame=perfect.select(ARM_COLUMNS))
    print(f"ref_perfect: {perfect.height:,} rows, {time.monotonic() - started:.0f} s", flush=True)


# --- self-test ---------------------------------------------------------------------------------


def _check_blend_is_causal() -> None:
    """Check that a month's blend weights ignore that month's cases, and that July is equal.

    Three members and 40 days in two months. Member 0 is exact in July and wrong in August, so the
    August weights must favour it (they see July only). Changing the August actuals must leave the
    August weights unchanged and may change nothing before them.
    """
    rng = np.random.default_rng(3)
    days = 40
    init = [
        datetime(2025, 7, 1, 9, tzinfo=UTC) + timedelta(days=d)
        for d in range(days)
        for _ in range(4)
    ]
    valid = [t + (k % 4 + 1) * common.STEP for k, t in enumerate(init)]
    keys = pl.DataFrame(
        {"power_fcst_init_time": init, "valid_time": valid},
        schema={
            "power_fcst_init_time": pl.Datetime("us", "UTC"),
            "valid_time": pl.Datetime("us", "UTC"),
        },
    )
    actual = rng.uniform(0.2, 0.8, size=days * 4).astype(np.float32)
    august = keys["valid_time"].dt.strftime("%Y-%m").to_numpy() >= "2025-08"
    spread = np.linspace(-0.1, 0.1, 11, dtype=np.float32)
    exact = actual[:, None] + spread
    bad = actual[:, None] + 0.3 + spread
    junk = actual[:, None] + 0.2 + spread
    members = [np.where(august[:, None], bad, exact), junk.astype(np.float32), bad]
    _, table = rolling_blend(keys=keys, actual=actual, members=members, names=["a", "b", "c"])
    rows = {r["month"]: r for r in table.iter_rows(named=True)}
    equal = {f"w_{i}_{n}": 1 / 3 for i, n in enumerate("abc")}
    assert all(abs(rows["2025-07"][k] - v) < 1e-9 for k, v in equal.items()), rows["2025-07"]
    assert rows["2025-08"]["w_0_a"] == 1.0, rows["2025-08"]
    shifted = actual.copy()
    shifted[august] = 5.0
    _, again = rolling_blend(keys=keys, actual=shifted, members=members, names=["a", "b", "c"])
    august_rows = {r["month"]: r for r in again.iter_rows(named=True)}
    assert august_rows["2025-08"] == rows["2025-08"], "August weights read August observations"
    blended, _ = rolling_blend(keys=keys, actual=actual, members=[members[0]] * 3, names=["a"] * 3)
    assert np.allclose(blended, members[0], atol=1e-6)
    print("blend check passed: July equal weights, August weights ignore August observations")


def _write_fixture_fits(*, ctx: Context) -> None:
    """Write fake refit predictions for every config the demand builders read.

    Each series' median is its observed power plus a configuration-dependent bias, with a fixed
    spread; the horizon is `SELF_TEST_STEPS` half-hours per run. Offset twins exist for the offset
    series only, as the fit scripts write them.
    """
    configs = {
        "n1h": ("weekly", "monthly"),
        "an": ("monthly",),
        "pt": ("monthly",),
        "frozen": ("frozen",),
    }
    biases = {"n1h": 0.02, "an": 0.03, "pt": 0.04, "frozen": 0.05}
    levels = np.linspace(-0.08, 0.08, 11)
    for config, config_regimes in configs.items():
        for suffix in ("", "_o", "_oheld"):
            if suffix and config == "pt":
                continue
            for site in ctx.sites_of(groups=("Primary", "BSP", "GSP")):
                if suffix and site not in offsets.OFFSET_SITES:
                    continue
                row = ctx.series.filter(pl.col("site") == site).row(0, named=True)
                assigned: dict[datetime, set[datetime]] = {}
                for regime in config_regimes:
                    for refit, runs in regimes.served(name=regime, origins=ctx.origins).items():
                        assigned.setdefault(refit, set()).update(runs)
                for refit, runs in assigned.items():
                    frames = []
                    for origin in sorted(runs):
                        valid = [
                            origin + (step + 1) * common.STEP for step in range(SELF_TEST_STEPS)
                        ]
                        keys = pl.DataFrame(
                            {
                                "time_series_id": [row["time_series_id"]] * SELF_TEST_STEPS,
                                "power_fcst_init_time": [origin] * SELF_TEST_STEPS,
                                "valid_time": valid,
                            },
                            schema={
                                "time_series_id": pl.Int32,
                                "power_fcst_init_time": pl.Datetime("us", "UTC"),
                                "valid_time": pl.Datetime("us", "UTC"),
                            },
                        )
                        truth = common.take(
                            values=ctx.arrays[site].power,
                            index=np.asarray(common.grid_index(times=keys["valid_time"])),
                        )
                        centre = np.nan_to_num(truth, nan=0.3) + biases[config] + 0.01 * len(suffix)
                        quantiles = centre[:, None] + levels
                        frames.append(
                            keys.hstack(
                                pl.DataFrame(
                                    quantiles.astype(np.float32), schema=list(common.LEVEL_COLUMNS)
                                )
                            ).with_columns(seed=pl.lit(0).cast(pl.Int8))
                        )
                    common.atomic_write(
                        frame=pl.concat(frames),
                        target=job_path(
                            root=ctx.demand_root, config=config + suffix, site=site, refit=refit
                        ),
                    )


def _write_fixture_generators(*, ctx: Context) -> None:
    """Write fake generator mixtures: the weekly arms by refit date, the others as `frozen`."""
    levels = np.linspace(-0.1, 0.1, 11)
    arms_of = {
        "era5_frozen_mixture": ("frozen", 0.03),
        "ens_frozen": ("frozen", 0.02),
        "ens_weekly": ("weekly", 0.01),
        "ens_weekly_margin": ("weekly", 0.005),
    }
    for arm, (regime, bias) in arms_of.items():
        for row in ctx.sites.iter_rows(named=True):
            if arm == "ens_weekly_margin" and row["technology"] != "wind":
                continue
            power = ctx.power_of[int(row["time_series_id"])]
            if regime == "frozen":
                assigned = {None: list(ctx.origins)}
            else:
                assigned = dict(regimes.served(name="weekly", origins=ctx.origins))
            for refit, runs in assigned.items():
                frames = []
                for origin in runs:
                    valid = [origin + (k + 1) * common.STEP for k in range(SELF_TEST_STEPS)]
                    keys = pl.DataFrame(
                        {
                            "time_series_id": [row["time_series_id"]] * SELF_TEST_STEPS,
                            "power_fcst_init_time": [origin] * SELF_TEST_STEPS,
                            "valid_time": valid,
                        },
                        schema={
                            "time_series_id": pl.Int32,
                            "power_fcst_init_time": pl.Datetime("us", "UTC"),
                            "valid_time": pl.Datetime("us", "UTC"),
                        },
                    )
                    truth = common.take(
                        values=power, index=np.asarray(common.grid_index(times=keys["valid_time"]))
                    )
                    centre = np.clip(np.nan_to_num(truth, nan=0.3) + bias, 0.0, 1.0)
                    frames.append(
                        keys.hstack(
                            pl.DataFrame(
                                (centre[:, None] + levels).astype(np.float32),
                                schema=list(common.LEVEL_COLUMNS),
                            )
                        ).with_columns(seed=pl.lit(0).cast(pl.Int8))
                    )
                label = "frozen" if refit is None else f"{refit:%Y%m%d}"
                common.atomic_write(
                    frame=pl.concat(frames),
                    target=ctx.generator_root / arm / f"{row['site']}_{label}.parquet",
                )


def self_test() -> None:
    """Check the blend's causality, then build every demand arm from fake refit files."""
    _check_blend_is_causal()
    origins = common.all_origins()[:SELF_TEST_RUNS]
    folder = Path(tempfile.mkdtemp(prefix="selftest_", dir=common.WORK_DIR.parent))
    try:
        ctx = make_context(smoke=False, store=False, origins=origins, root=folder)
        _write_fixture_fits(ctx=ctx)
        build_primary(ctx=ctx)
        build_bspgsp(ctx=ctx)
        _write_fixture_generators(ctx=ctx)
        build_generators(ctx=ctx)
        generator_names = [
            "g_prev_best_era5",
            "g_ens_frozen",
            "g_ens_weekly",
            "g_a_spread_on",
            "g_a_wind_margin",
        ]
        for name in generator_names:
            table = pl.read_parquet(ctx.arms_dir / f"{name}.parquet")
            assert table.select(KEY_COLUMNS).is_duplicated().sum() == 0, name
            assert table["group"].unique().sort().to_list() == (
                ["wind"] if name == "g_a_wind_margin" else ["solar", "wind"]
            ), name
        build_perfect(ctx=ctx)
        perfect = pl.read_parquet(ctx.arms_dir / "ref_perfect.parquet")
        assert (perfect["q01"] == perfect["actual"]).all()
        assert (perfect["q99"] == perfect["actual"]).all()
        names = [
            "p_s1_frozen",
            "p_s2_stack",
            "p_s3_monthly",
            "p_s4_weekly",
            "p_s5_blend_no_offset",
            "p_s6_bg2",
            "p_a2_no_blend",
            "p_a3_monthly_in_weekly_slot",
            "p_a4_correction_all",
            "p_a5_spread_off",
            "p_a6_capacity_held",
            "p_a7_three_members",
            "b_frozen",
            "b_stack",
            "b_monthly",
            "b_weekly",
            "b_a_correction",
        ]
        tables = {n: pl.read_parquet(ctx.arms_dir / f"{n}.parquet") for n in names}
        for name, table in tables.items():
            quantiles = table.select(common.LEVEL_COLUMNS).to_numpy()
            assert np.isfinite(quantiles).all(), name
            assert (np.diff(quantiles, axis=1) >= -1e-6).all(), f"{name} quantiles cross"
            assert table.select(KEY_COLUMNS).is_duplicated().sum() == 0, name
        # A2 is the weekly model at the non-offset series, exactly as S4 holds it.
        keys = list(KEY_COLUMNS)
        non_offset = pl.col("site").is_in(list(offsets.OFFSET_SITES)).not_()
        a2 = tables["p_a2_no_blend"].filter(non_offset).sort(keys)
        s4 = tables["p_s4_weekly"].filter(non_offset).sort(keys)
        assert np.array_equal(
            a2.select(common.LEVEL_COLUMNS).to_numpy(), s4.select(common.LEVEL_COLUMNS).to_numpy()
        )
        # The spread factor widens S4 about the median by exactly 1.15.
        raw = pl.read_parquet(
            job_path(
                root=ctx.demand_root,
                config="n1h",
                site="D2",
                refit=regimes.WEEKLY[0],
            )
        ).filter(pl.col("power_fcst_init_time") == origins[0])
        widened = tables["p_s4_weekly"].filter(
            (pl.col("site") == "D2") & (pl.col("power_fcst_init_time") == origins[0])
        )
        assert np.allclose(
            (widened["q95"] - widened["q05"]).to_numpy()[:SELF_TEST_STEPS],
            1.15 * (raw["q95"] - raw["q05"]).to_numpy()[:SELF_TEST_STEPS],
            atol=1e-5,
        )
        # S2 is the corrected stack of the frozen file; S1 has no correction.
        assert not np.allclose(
            tables["p_s1_frozen"].sort(keys).select("q50").to_numpy(),
            tables["p_s2_stack"].sort(keys).select("q50").to_numpy(),
        )
        # A blend never leaves the range of its members.
        weights = pl.read_parquet(ctx.results_dir / "blend_weights_s6.parquet")
        first = weights.row(0, named=True)
        assert all(abs(first[c] - 0.2) < 1e-9 for c in first if c.startswith("w_")), first
        print(weights)
        print(
            f"self-test passed: {len(names) + len(generator_names) + 1} arms built from fake files"
        )
    finally:
        shutil.rmtree(folder, ignore_errors=True)


def main() -> None:
    """Build and write the requested parts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--parts",
        nargs="+",
        choices=("demand", "generators", "references", "perfect"),
        default=["demand", "generators", "references", "perfect"],
    )
    parser.add_argument(
        "--arms", nargs="+", help="build only these demand arms (for an early reproduction check)"
    )
    parser.add_argument("--smoke", action="store_true", help="read smoke folders, write apart")
    parser.add_argument("--no-store", action="store_true", help="do not write the prediction store")
    parser.add_argument("--self-test", action="store_true", help="build arms from fake refit files")
    args = parser.parse_args()
    if args.self_test:
        self_test()
        return
    ctx = make_context(smoke=args.smoke, store=not args.no_store)
    if "demand" in args.parts:
        build_primary(ctx=ctx, only=args.arms)
        build_bspgsp(ctx=ctx, only=args.arms)
    if "generators" in args.parts:
        build_generators(ctx=ctx)
    if "references" in args.parts:
        build_references(
            ctx=ctx,
            limit_series=3 if args.smoke else None,
            limit_runs=2 if args.smoke else None,
        )
    if "perfect" in args.parts:
        build_perfect(ctx=ctx)


if __name__ == "__main__":
    main()
