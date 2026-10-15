"""Blend stored members' quantiles with rolling-origin lead-band weights and write the arms.

Reads the stored arm tables of experiment 33 (read only) and writes, under
`data/ml-experiments/00035-.../work/`, each blend arm's quantiles and the chosen weight tables.
The weights for runs initialised in month m use only cases whose run was initialised before month
m and whose valid time is before the first day of month m.

    uv run python ml-experiments/00035-lead-weighted-quantile-blend-of-stored-models/e00035_blend.py
"""

import itertools
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR, study_dir_for

SLUG: Final[str] = "lead-weighted-quantile-blend-of-stored-models"
WORK_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"00035-{SLUG}" / "work"
SOURCE_DIR: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00033-best-guess-architecture-vs-production" / "work"
)
HOLDOUT: Final[datetime] = datetime(2026, 7, 1, tzinfo=UTC)
LEVELS: Final[np.ndarray] = np.array([0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95])
LEVEL_COLUMNS: Final[tuple[str, ...]] = tuple(f"q{round(100 * x):02d}" for x in LEVELS)
ATOM_MASSES: Final[np.ndarray] = np.diff(
    np.concatenate([[0.0], (LEVELS[:-1] + LEVELS[1:]) / 2, [1.0]])
)
MEDIAN: Final[int] = 4
SPREAD_FACTOR: Final[float] = 1.15
LEAD_BANDS: Final[tuple[tuple[str, float, float], ...]] = (
    ("0-24 h", 0.0, 24.0),
    ("1-2 d", 24.0, 48.0),
    ("2-6 d", 48.0, 144.0),
    ("6-10 d", 144.0, 240.0),
    ("10+ d", 240.0, 10_000.0),
)
GRID_STEP: Final[int] = 4
"""Weights are multiples of 1/4."""
CASE_KEYS: Final[tuple[str, ...]] = ("time_series_id", "power_fcst_init_time", "valid_time")
KEYS: Final[tuple[str, ...]] = (*CASE_KEYS, "seed")
CORRECTION_DAYS: Final[int] = 14
MIN_CORRECTION_DAYS: Final[int] = 3
NEWEST_AGE: Final[int] = 7
"""Half-hours between the origin and the newest usable observation (3.5 hours, as 00013)."""
STEPS_PER_DAY: Final[int] = 48

# Each technology: group label, arm folder, list of member (name, stored file) pairs.
DEMAND_ARMS: Final = SOURCE_DIR / "arms"
GENERATOR_ARMS: Final = SOURCE_DIR / "generator_tables"
TECHNOLOGIES: Final[dict[str, dict[str, Any]]] = {
    "Primary": {
        "kind": "demand",
        "members": {"monthly": "monthly", "a7_cs": "a7_cs", "n1_cs": "n1_cs"},
    },
    "solar": {
        "kind": "generator",
        "members": {"n1": "n1", "best_guess": "bg_base_spread"},
    },
    "wind": {
        "kind": "generator",
        "members": {"n1": "n1", "best_guess": "bg_full"},
    },
    "BSP+GSP": {"kind": "demand", "members": {"n1": "n1", "a7": "a7"}},
}
D3_MEMBERS: Final[dict[str, str]] = {"monthly_cs": "monthly_cs", "a7_cs": "a7_cs", "n1_cs": "n1_cs"}
D2_MEMBERS: Final[dict[str, str]] = {"monthly": "monthly", "a7": "a7", "n1": "n1"}


def weight_grid(*, members: int) -> np.ndarray:
    """Return every weight vector on the grid that sums to 1, shape (vectors, members)."""
    rows = [
        v for v in itertools.product(range(GRID_STEP + 1), repeat=members) if sum(v) == GRID_STEP
    ]
    return np.array(rows, dtype=np.float64) / GRID_STEP


def quantile_score(*, quantiles: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """Return twice the mass-weighted mean pinball loss of the nine levels, per case."""
    difference = actual[:, None] - quantiles
    pinball = np.where(difference >= 0, LEVELS * difference, (LEVELS - 1) * difference)
    return 2.0 * (pinball * ATOM_MASSES).sum(axis=1)


def vincent(*, member_quantiles: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
    """Return the weighted average of each quantile level, sorted along the levels.

    Args:
        member_quantiles: One (cases, 9) array per member.
        weights: One weight per member, or shape (cases, members) for per-case weights.
    """
    stacked = np.stack(member_quantiles, axis=0).astype(np.float64)
    if weights.ndim == 1:
        blend = np.tensordot(weights, stacked, axes=1)
    else:
        blend = np.einsum("cm,mcl->cl", weights, stacked)
    return np.sort(blend, axis=1)


def _read(*, kind: str, file: str, technology: str) -> pl.DataFrame:
    """Read one stored arm table with the key columns, quantiles, observation, and lead."""
    path = (DEMAND_ARMS if kind == "demand" else GENERATOR_ARMS) / f"{file}.parquet"
    frame = pl.read_parquet(path)
    if kind == "generator":
        frame = frame.filter(pl.col("technology") == technology).rename({"actual_norm": "actual"})
    return frame.select(*KEYS, *LEVEL_COLUMNS, "actual").with_columns(pl.col("seed").cast(pl.Int8))


def load_members(
    *, kind: str, members: dict[str, str], groups: pl.DataFrame | None, group: str
) -> tuple[pl.DataFrame, dict[str, np.ndarray]]:
    """Return the shared cases and each member's (cases, 9) quantile array.

    Cases are those every member holds with finite quantiles and a finite observation, with a valid
    time before the held-out period, and that are in experiment 33's shared case set for the group.
    For demand, `group` keeps one substation type (or "BSP+GSP"). Rows are sorted by the keys so
    all members line up.
    """
    tables = {name: _read(kind=kind, file=file, technology=group) for name, file in members.items()}
    names = list(tables)
    shared: pl.DataFrame | None = None
    for name in names:
        t = tables[name]
        if groups is not None:
            t = t.join(groups, on="time_series_id", how="inner", validate="m:1").filter(
                pl.col("group").is_in(group.split("+"))
            )
        t = t.filter(
            pl.all_horizontal(pl.col(*LEVEL_COLUMNS).is_finite())
            & pl.col("actual").is_finite()
            & (pl.col("valid_time") < HOLDOUT)
        )
        tables[name] = t
        keys = t.select(*KEYS)
        shared = keys if shared is None else shared.join(keys, on=list(KEYS), how="inner")
    assert shared is not None
    shared_keys = pl.concat(
        [
            pl.read_parquet(SOURCE_DIR / "metrics" / g / "shared_keys.parquet")
            for g in group.split("+")
        ]
    ).unique()
    shared = shared.join(shared_keys, on=list(CASE_KEYS), how="semi").unique().sort(*KEYS)
    base = shared.join(
        tables[names[0]].select(*KEYS, "actual"), on=list(KEYS), how="left", validate="1:1"
    ).sort(*KEYS)
    joined = {
        name: shared.join(tables[name], on=list(KEYS), how="left", validate="1:1").sort(*KEYS)
        for name in names
    }
    for name in names:
        if not np.array_equal(joined[name]["actual"].to_numpy(), base["actual"].to_numpy()):
            msg = f"member {name} holds a different observation from {names[0]} on some case"
            raise ValueError(msg)
    arrays = {name: joined[name].select(LEVEL_COLUMNS).to_numpy() for name in names}
    base = base.with_columns(
        lead_hours=(pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_minutes()
        / 60.0,
        month=pl.col("power_fcst_init_time").dt.strftime("%Y-%m"),
    )
    band = np.full(base.height, -1)
    lead = base["lead_hours"].to_numpy()
    for i, (_, low, high) in enumerate(LEAD_BANDS):
        band[(lead >= low) & (lead < high)] = i
    if (band < 0).any():
        msg = "a case has a lead outside every band"
        raise ValueError(msg)
    return base.with_columns(band=pl.Series(band)), arrays


def choose_weights(
    *,
    cases: pl.DataFrame,
    arrays: dict[str, np.ndarray],
    by_band: bool,
) -> pl.DataFrame:
    """Return the rolling-origin weight vector for each scored month and lead band.

    A month's weights come from cases whose run was initialised in an earlier month and whose
    valid time is before the month's first day. A month with no such cases gets equal weights and
    `history_cases` of 0. Ties go to the vector nearest equal weights.
    """
    names = list(arrays)
    grid = weight_grid(members=len(names))
    equal = np.full(len(names), 1.0 / len(names))
    nearest_equal = np.argsort(np.abs(grid - equal).sum(axis=1), kind="stable")
    member_q = [arrays[n] for n in names]
    actual = cases["actual"].to_numpy().astype(np.float64)
    scores = np.stack(
        [
            quantile_score(quantiles=vincent(member_quantiles=member_q, weights=w), actual=actual)
            for w in grid
        ],
        axis=1,
    )
    months = sorted(cases["month"].unique().to_list())
    month_col = cases["month"].to_numpy()
    valid = cases["valid_time"].to_numpy()
    band = cases["band"].to_numpy()
    rows = []
    for month in months:
        start = np.datetime64(
            datetime.strptime(month, "%Y-%m").replace(tzinfo=UTC).replace(tzinfo=None), "us"
        )
        history = (month_col < month) & (valid.astype("datetime64[us]") < start)
        for b in range(len(LEAD_BANDS)) if by_band else [-1]:
            mask = history & ((band == b) if by_band else True)
            count = int(mask.sum())
            if count == 0:
                chosen = equal
            else:
                mean = scores[mask].mean(axis=0)
                order = nearest_equal[np.argsort(mean[nearest_equal], kind="stable")]
                chosen = grid[order[0]]
            rows.append(
                {
                    "month": month,
                    "band": b,
                    "history_cases": count,
                    **{f"w_{n}": float(w) for n, w in zip(names, chosen, strict=True)},
                }
            )
    return pl.DataFrame(rows)


def apply_weights(
    *, cases: pl.DataFrame, arrays: dict[str, np.ndarray], table: pl.DataFrame, by_band: bool
) -> np.ndarray:
    """Return the blend of the members under a weight table, per case."""
    names = list(arrays)
    keyed = cases.select("month", "band").with_columns(
        band=pl.col("band") if by_band else pl.lit(-1)
    )
    joined = keyed.join(table, on=["month", "band"], how="left", validate="m:1")
    weights = joined.select([f"w_{n}" for n in names]).to_numpy()
    if np.isnan(weights).any():
        msg = "a case has no weight row"
        raise ValueError(msg)
    return vincent(member_quantiles=[arrays[n] for n in names], weights=weights)


def correction_table(
    *, frame: pl.DataFrame, power: np.ndarray, grid_start: datetime
) -> pl.DataFrame:
    """Return each run's 14-day additive correction for one series and seed.

    The mean of (observed minus median forecast) over the 14 days ending at the newest usable
    observation, using for each past half-hour the median of the latest earlier run that covers
    it. Zero with fewer than 3 days of history. Mirrors experiment 19's computation.

    Args:
        frame: One series and seed: `power_fcst_init_time`, `valid_time`, `median`.
        power: Observed power over capacity on the half-hourly grid from `grid_start`.
        grid_start: Timestamp of the first grid cell.
    """

    def index(times: pl.Series) -> np.ndarray:
        return (
            (times.dt.epoch("us") - int(grid_start.timestamp() * 1e6)) // 1_800_000_000
        ).to_numpy()

    runs = frame["power_fcst_init_time"].unique().sort()
    origins = index(runs)
    median = np.full(power.shape, np.nan)
    for position, origin in enumerate(runs):
        part = frame.filter(pl.col("power_fcst_init_time") == origin)
        idx = index(part["valid_time"])
        end = origins[position + 1] if position + 1 < len(origins) else np.inf
        keep = (idx >= origins[position]) & (idx < end)
        median[idx[keep]] = part["median"].to_numpy()[keep]
    window_size = CORRECTION_DAYS * STEPS_PER_DAY
    rows = []
    for origin_time, origin in zip(runs, origins, strict=True):
        newest = origin - NEWEST_AGE
        window = np.arange(newest - window_size + 1, newest + 1)
        used = ~np.isnan(median[window]) & ~np.isnan(power[window])
        count = int(used.sum())
        residual = power[window][used].astype(np.float64) - median[window][used]
        rows.append(
            {
                "power_fcst_init_time": origin_time,
                "correction": float(residual.mean())
                if count >= MIN_CORRECTION_DAYS * STEPS_PER_DAY
                else 0.0,
            }
        )
    return pl.DataFrame(rows)


def observed_power() -> tuple[dict[int, np.ndarray], datetime]:
    """Return each Primary-or-other demand series' observed power over capacity on the grid."""
    demand = study_dir_for(study="stage1_pilot") / "demand" / "inputs"
    series = pl.read_parquet(demand / "series.parquet")
    observed = pl.read_parquet(
        demand / "observed.parquet", columns=["time_series_id", "valid_time", "power"]
    )
    start = datetime(2019, 9, 1, tzinfo=UTC)
    end = datetime(2026, 7, 31, 23, 30, tzinfo=UTC)
    grid = pl.DataFrame(
        {
            "valid_time": pl.datetime_range(
                start, end, interval="30m", time_unit="us", time_zone="UTC", eager=True
            )
        }
    )
    out: dict[int, np.ndarray] = {}
    for row in series.iter_rows(named=True):
        joined = grid.join(
            observed.filter(pl.col("time_series_id") == row["time_series_id"]).select(
                "valid_time", pl.col("power") / row["effective_capacity_mw"]
            ),
            on="valid_time",
            how="left",
            validate="1:1",
        ).sort("valid_time")
        out[int(row["time_series_id"])] = joined["power"].to_numpy().astype(np.float32)
    return out, start


def correct_and_spread(*, cases: pl.DataFrame, blend: np.ndarray) -> np.ndarray:
    """Apply the 14-day correction (from the blend's own history) and the spread factor."""
    power, start = observed_power()
    frame = cases.select(
        "time_series_id", "seed", "power_fcst_init_time", "valid_time"
    ).with_columns(median=pl.Series(blend[:, MEDIAN]), row=pl.int_range(pl.len()))
    corrections = np.zeros(cases.height)
    for (series_id, _seed), part in frame.group_by("time_series_id", "seed", maintain_order=True):
        table = correction_table(frame=part, power=power[int(series_id)], grid_start=start)
        joined = part.join(table, on="power_fcst_init_time", how="left", validate="m:1")
        corrections[joined["row"].to_numpy()] = joined["correction"].to_numpy()
    shifted = blend + corrections[:, None]
    median = shifted[:, [MEDIAN]]
    return np.sort(median + SPREAD_FACTOR * (shifted - median), axis=1)


def write_arm(*, folder: Path, name: str, cases: pl.DataFrame, quantiles: np.ndarray) -> None:
    """Write one arm's quantiles beside the case keys, observation, band, and lead."""
    folder.mkdir(parents=True, exist_ok=True)
    cases.select(*KEYS, "actual", "lead_hours", "band", "month").with_columns(
        pl.DataFrame(quantiles.astype(np.float32), schema=list(LEVEL_COLUMNS), orient="row")
    ).write_parquet(folder / f"{name}.parquet")


def run_technology(*, label: str, spec: dict[str, Any], groups: pl.DataFrame) -> None:
    """Choose weights and write every arm for one technology."""
    kind = str(spec["kind"])
    members = dict(spec["members"])
    cases, arrays = load_members(
        kind=kind, members=members, groups=groups if kind == "demand" else None, group=label
    )
    folder = WORK_DIR / label.replace("+", "_")
    names = list(arrays)
    member_q = [arrays[n] for n in names]
    for name in names:
        write_arm(folder=folder, name=f"member_{name}", cases=cases, quantiles=arrays[name])
    equal = np.full(len(names), 1.0 / len(names))
    write_arm(
        folder=folder,
        name="equal",
        cases=cases,
        quantiles=vincent(member_quantiles=member_q, weights=equal),
    )
    for tag, by_band in (("rolling", True), ("global", False)):
        table = choose_weights(cases=cases, arrays=arrays, by_band=by_band)
        table.write_parquet(folder / f"weights_{tag}.parquet")
        write_arm(
            folder=folder,
            name=tag,
            cases=cases,
            quantiles=apply_weights(cases=cases, arrays=arrays, table=table, by_band=by_band),
        )
    # In-sample bound: weights per band chosen on all cases, the same for every month.
    grid = weight_grid(members=len(names))
    actual = cases["actual"].to_numpy().astype(np.float64)
    band = cases["band"].to_numpy()
    in_sample = np.zeros((cases.height, len(names)))
    for b in range(len(LEAD_BANDS)):
        mask = band == b
        best = min(
            grid,
            key=lambda w, m=mask: quantile_score(
                quantiles=vincent(member_quantiles=[q[m] for q in member_q], weights=w),
                actual=actual[m],
            ).mean(),
        )
        in_sample[mask] = best
    write_arm(
        folder=folder,
        name="in_sample",
        cases=cases,
        quantiles=vincent(member_quantiles=member_q, weights=in_sample),
    )
    print(f"{label}: {cases.height} cases, members {names}")


def run_primary_placement(*, groups: pl.DataFrame) -> None:
    """Write arms D3 (correction inside members) and D2 (correction after the blend)."""
    folder = WORK_DIR / "Primary"
    d3_cases, d3_arrays = load_members(
        kind="demand", members=D3_MEMBERS, groups=groups, group="Primary"
    )
    table = choose_weights(cases=d3_cases, arrays=d3_arrays, by_band=True)
    table.write_parquet(folder / "weights_d3.parquet")
    write_arm(
        folder=folder,
        name="d3",
        cases=d3_cases,
        quantiles=apply_weights(cases=d3_cases, arrays=d3_arrays, table=table, by_band=True),
    )
    d2_cases, d2_arrays = load_members(
        kind="demand", members=D2_MEMBERS, groups=groups, group="Primary"
    )
    if d2_cases.height != d3_cases.height or not d2_cases.select(*KEYS).equals(
        d3_cases.select(*KEYS)
    ):
        msg = "D2 and D3 do not hold the same cases"
        raise ValueError(msg)
    renamed = table.rename(
        {f"w_{a}": f"w_{b}" for a, b in zip(D3_MEMBERS, D2_MEMBERS, strict=True)}
    )
    blend = apply_weights(cases=d2_cases, arrays=d2_arrays, table=renamed, by_band=True)
    write_arm(
        folder=folder,
        name="d2",
        cases=d2_cases,
        quantiles=correct_and_spread(cases=d2_cases, blend=blend),
    )


def main() -> None:
    """Write every technology's arms."""
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    metadata = pl.read_parquet(
        study_dir_for(study="quick_wins_fixed_pipeline_1156")
        / "frames"
        / "combined"
        / "fold_a"
        / "metadata.parquet",
        columns=["time_series_id", "substation_type"],
    ).select("time_series_id", group=pl.col("substation_type").cast(pl.String))
    for label, spec in TECHNOLOGIES.items():
        run_technology(label=label, spec=spec, groups=metadata)
    run_primary_placement(groups=metadata)


if __name__ == "__main__":
    main()
