"""Assemble every new arm of experiment 00102 (SIMP1) from experiment 00087's stored members.

Reads the BG2 members' refit predictions in experiment 00087's work folder (read only) and this
experiment's no-lag refits, and writes each arm to `work/arms/<name>.parquet` (run keys, `seed`,
`lead_hours`, `actual`, the 11 quantiles `q01` to `q99` as fractions of capacity, and `site`,
`group`, `capacity_mw`) and, for the 292 runs of `fold_a_daily_gap_v1`, to the prediction store.
Helpers (`regime_rows`, the monthly-weight blend, `finalise`, `write_arm`) are copied from
experiment 00087's `e00087_assemble.py`. The arms cover all 365 daily runs, because the monthly
blend weights read the cases of earlier months and the 73 runs of `fold_a_every5d_v1` choose the
best pair and triple.

**Primary arms.**

- `p_m_<member>` (lower case): one blend member alone, with the offset twin at the 5 offset
  series (not for PT),
  the spread factor 1.15, and the 14-day correction for ST only, exactly as BG2 builds its members.
  The members are W (weekly), M (monthly), AN (analogue), PT (pooled by type), and ST (frozen with
  correction).
- `p_b_<members>` (lower case): the monthly-weight blend of two or three members, for every
  subset (BG2's blend
  weights are chosen the same way, on earlier cases only). `p_b_bg2_rebuilt` is BG2 rebuilt and is
  compared with the stored `p_s6_bg2`.
- `p_b_bg2_st_uncorrected`: BG2's blend with ST left without the 14-day correction.
- `p_nl_monthly`, `p_nl_weekly`, `p_nl_weekly_nospread`: the single N1 without the three power lags
  (`e00102_fit_demand.py`), no offset, no correction.

**Supply point arms.** `b_nl_monthly` and `b_nl_weekly`: the no-lag N1 for the 2 BSP and 2 GSP, with
the spread factor, like `b_monthly` and `b_weekly`.

**Farm arm.** `g_nl_monthly`: the no-lag ENS-trained N1, monthly refits, no spread factor.

    uv run python ml-experiments/00102-*/e00102_assemble.py [--parts members blends nolag farms]
        [--no-store]
"""

import argparse
import itertools
import time
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Final

import e00102_common as common
import e00102_gen_common as gen
import e00102_regimes as regimes
import e00102_stack as stack
import numpy as np
import polars as pl

from studies import experiment_store

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
OFFSET_SITES: Final[tuple[str, ...]] = ("D9", "D10", "D11", "D14", "D16")
"""The 5 Primary series with a clean embedded-solar capacity (experiment 00087's offset series)."""
WEIGHT_GRID_STEP: Final[int] = 4
"""Blend weights are multiples of 1/4."""
SCORING_CHUNK_ROWS: Final[int] = 1_000_000
MEMBER_NAMES: Final[tuple[str, ...]] = ("W", "M", "AN", "PT", "ST")
BG2_ARMS_DIR: Final[Path] = common.BG2_WORK_DIR / "arms"
BG2_DEMAND_ROOT: Final[Path] = common.BG2_WORK_DIR / "anchor_preds"


@dataclass(frozen=True)
class MemberSpec:
    """Where one blend member's refit predictions live in experiment 00087's work folder.

    Attributes:
        config: The folder under `anchor_preds`.
        regime: The retraining regime (a key of `regimes.REGIMES`).
        offsettable: Whether the member has offset twins (`<config>_o`) at the offset series.
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


@dataclass(frozen=True)
class Context:
    """Everything an arm builder reads.

    Attributes:
        series: `common.series_table()`.
        sites: `gen.site_table()`.
        arrays: Each demand site's observed power over capacity on the dense grid.
        power_of: Each series' observed power over capacity by identifier.
        labels: Each series identifier's site label and group.
        capacity: Each series' effective capacity in MW.
        origins: The `power_fcst_init_time` of the runs to cover.
        own_demand_root: This experiment's demand refit predictions.
        own_generator_root: This experiment's generator refit predictions.
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
    own_demand_root: Path
    own_generator_root: Path
    arms_dir: Path
    results_dir: Path
    strict: bool
    store: bool

    def sites_of(self, *, groups: Sequence[str]) -> list[str]:
        """Return the demand site labels whose substation type is in `groups`."""
        return self.series.filter(pl.col("substation_type").is_in(list(groups)))["site"].to_list()


def make_context(*, store: bool) -> Context:
    """Build the context: series tables, observed power, and the folders to read and write."""
    series = common.series_table()
    sites = gen.site_table()
    arrays = common.load_power_arrays(series=series)
    gen_power = gen.full_power(sites=sites)
    rows = [(r, r["substation_type"]) for r in series.iter_rows(named=True)] + [
        (r, r["technology"]) for r in sites.iter_rows(named=True)
    ]
    return Context(
        series=series,
        sites=sites,
        arrays=arrays,
        power_of={
            **{
                int(r["time_series_id"]): arrays[r["site"]].power
                for r in series.iter_rows(named=True)
            },
            **{int(k): v for k, v in gen_power.items()},
        },
        labels={int(r["time_series_id"]): (r["site"], group) for r, group in rows},
        capacity={int(r["time_series_id"]): float(r["effective_capacity_mw"]) for r, _ in rows},
        origins=common.all_origins(),
        own_demand_root=common.ANCHOR_PREDS_DIR,
        own_generator_root=common.GENERATOR_PREDS_DIR,
        arms_dir=common.WORK_DIR / "arms",
        results_dir=common.WORK_DIR / "results",
        strict=True,
        store=store,
    )


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


# --- the demand members ------------------------------------------------------------------------


def member_table(*, ctx: Context, member: str, correct: bool | None = None) -> pl.DataFrame:
    """Return one BG2 blend member's quantiles at the 16 Primaries, as BG2 builds the member.

    The member uses the offset twin at the 5 offset series when it has one, the spread factor 1.15,
    and the 14-day correction for ST only (`correct=None`); `correct` overrides that.

    Args:
        ctx: The context.
        member: A key of `MEMBERS`.
        correct: Whether to add the 14-day correction, or `None` for BG2's choice.

    Returns:
        The member's keys, `seed`, and 11 quantile columns for every run.
    """
    spec = MEMBERS[member]
    pieces = []
    for site in ctx.sites_of(groups=("Primary",)):
        twin = spec.offsettable and site in OFFSET_SITES
        pieces.append(
            regime_rows(
                root=BG2_DEMAND_ROOT,
                config=spec.config + ("_o" if twin else ""),
                regime=spec.regime,
                site=site,
                origins=ctx.origins,
                strict=ctx.strict,
            )
        )
    return finish_member(
        ctx=ctx,
        raw=pl.concat(pieces),
        correct=(member == "ST") if correct is None else correct,
        spread=True,
    )


def subset_label(*, members: Sequence[str]) -> str:
    """Return the arm-name suffix of a subset of members, such as `W_AN_ST`."""
    return "_".join(members)


def build_members_and_blends(*, ctx: Context) -> None:
    """Write each member alone, every 2- and 3-member blend, BG2 rebuilt, and BG2 uncorrected."""
    started = time.monotonic()
    tables = {member: member_table(ctx=ctx, member=member) for member in MEMBER_NAMES}
    tables["ST_uncorrected"] = member_table(ctx=ctx, member="ST", correct=False)
    for member in MEMBER_NAMES:
        emit(ctx=ctx, name=f"p_m_{member.lower()}", table=tables[member], started=started)
    keys, by_member = align(tables=tables, strict=ctx.strict)
    del tables
    actual = actual_array(frame=keys, power_of=ctx.power_of)
    ctx.results_dir.mkdir(parents=True, exist_ok=True)
    subsets: list[tuple[str, ...]] = [
        *itertools.combinations(MEMBER_NAMES, 2),
        *itertools.combinations(MEMBER_NAMES, 3),
        MEMBER_NAMES,
        ("W", "M", "AN", "PT", "ST_uncorrected"),
    ]
    for subset in subsets:
        started = time.monotonic()
        blended, weights = rolling_blend(
            keys=keys,
            actual=actual,
            members=[by_member[m] for m in subset],
            names=list(subset),
        )
        label = subset_label(members=subset)
        weights.write_parquet(ctx.results_dir / f"blend_weights_{label}.parquet")
        name = (
            "p_b_bg2_st_uncorrected"
            if subset[-1] == "ST_uncorrected"
            else "p_b_bg2_rebuilt"
            if subset == MEMBER_NAMES
            else f"p_b_{label.lower()}"
        )
        emit(
            ctx=ctx,
            name=name,
            table=keys.hstack(pl.DataFrame(blended, schema=list(common.LEVEL_COLUMNS))),
            started=started,
        )


# --- the no-lag arms ---------------------------------------------------------------------------


def nolag_demand_table(
    *, ctx: Context, groups: Sequence[str], regime: str, spread: bool
) -> pl.DataFrame:
    """Return the no-lag N1's quantiles for the series of some groups, no offset, no correction."""
    raw = pl.concat(
        [
            regime_rows(
                root=ctx.own_demand_root,
                config="n1h_nolag",
                regime=regime,
                site=site,
                origins=ctx.origins,
                strict=ctx.strict,
            )
            for site in ctx.sites_of(groups=groups)
        ]
    )
    return stack.stack(frame=raw, correct=False, spread=spread)


def build_nolag_demand(*, ctx: Context) -> None:
    """Write the no-lag Primary and supply point arms."""
    plan = {
        "p_nl_monthly": (("Primary",), "monthly", True),
        "p_nl_weekly": (("Primary",), "weekly", True),
        "p_nl_weekly_nospread": (("Primary",), "weekly", False),
        "b_nl_monthly": (("BSP", "GSP"), "monthly", True),
        "b_nl_weekly": (("BSP", "GSP"), "weekly", True),
    }
    for name, (groups, regime, spread) in plan.items():
        started = time.monotonic()
        try:
            table = nolag_demand_table(ctx=ctx, groups=groups, regime=regime, spread=spread)
        except FileNotFoundError as error:
            print(f"{name}: skipped, {error}", flush=True)
            continue
        emit(ctx=ctx, name=name, table=table, started=started)


def build_nolag_farms(*, ctx: Context) -> None:
    """Write the no-lag farm arm from the refit files, which carry no spread factor."""
    started = time.monotonic()
    files = sorted((ctx.own_generator_root / "ens_monthly_nolag").glob("*.parquet"))
    if not files:
        print("g_nl_monthly: skipped, no refit files", flush=True)
        return
    table = pl.concat([pl.read_parquet(path) for path in files]).filter(
        pl.col("power_fcst_init_time").is_in(ctx.origins)
    )
    runs = table.group_by("time_series_id").agg(runs=pl.col("power_fcst_init_time").n_unique())
    if (runs["runs"] != len(ctx.origins)).any():
        msg = f"g_nl_monthly: some series lack runs; runs per series {runs.rows()}"
        raise ValueError(msg)
    emit(
        ctx=ctx,
        name="g_nl_monthly",
        table=table.select(*KEY_COLUMNS, *common.LEVEL_COLUMNS),
        started=started,
    )


D18_CUT_ARMS: Final[dict[str, tuple[str, str, str, str]]] = {
    "p_nl_weekly_d18cut": ("p_nl_weekly", "n1h_nolag_cut", "weekly", "own"),
    "p_s4_weekly_d18cut": ("p_s4_weekly", "n1h_lag_cut", "weekly", "bg2"),
    "p_nl_monthly_d18cut": ("p_nl_monthly", "n1h_nolag_cut", "monthly", "own"),
    "p_s3_monthly_d18cut": ("p_s3_monthly", "n1h_lag_cut", "monthly", "bg2"),
}
"""Each D18-cut arm: the arm whose D18 rows it replaces, the refit folder of the replacement, the
regime, and whether the base arm is in this experiment's arms folder or experiment 00087's."""


def build_d18_cut(*, ctx: Context) -> None:
    """Write the Primary arms whose D18 rows come from refits that cut D18's commissioning months.

    Each arm is an existing Primary arm with the rows of D18 replaced by the rows of the refit
    trained from 2025-03-01 onwards, with the spread factor and no correction, as the base arm has.
    """
    for name, (base, config, regime, where) in D18_CUT_ARMS.items():
        started = time.monotonic()
        base_path = (ctx.arms_dir if where == "own" else BG2_ARMS_DIR) / f"{base}.parquet"
        try:
            raw = regime_rows(
                root=ctx.own_demand_root,
                config=config,
                regime=regime,
                site="D18",
                origins=ctx.origins,
                strict=ctx.strict,
            )
        except FileNotFoundError as error:
            print(f"{name}: skipped, {error}", flush=True)
            continue
        replacement = finalise(ctx=ctx, frame=stack.stack(frame=raw, correct=False, spread=True))
        table = pl.concat(
            [
                pl.read_parquet(base_path).filter(pl.col("site") != "D18"),
                replacement.select(pl.read_parquet_schema(base_path).keys()),
            ]
        ).sort(*KEY_COLUMNS)
        emit(
            ctx=ctx,
            name=name,
            table=table.select(*KEY_COLUMNS, *common.LEVEL_COLUMNS),
            started=started,
        )


PARTS: Final[tuple[str, ...]] = ("members", "nolag", "farms", "d18cut")


def main() -> None:
    """Build the requested parts."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parts", nargs="+", choices=PARTS, default=list(PARTS))
    parser.add_argument("--no-store", action="store_true", help="skip the prediction store")
    args = parser.parse_args()
    ctx = make_context(store=not args.no_store)
    builders = {
        "members": build_members_and_blends,
        "nolag": build_nolag_demand,
        "farms": build_nolag_farms,
        "d18cut": build_d18_cut,
    }
    for part in args.parts:
        builders[part](ctx=ctx)


if __name__ == "__main__":
    main()
