"""Stage 0 of experiment 00049: re-tune the spread factor and re-run the blend, on stored arms.

Reads the arm tables that experiments 00033 and 00034 stored (read only) for the 16 Primary
substations, seed 0, and writes the tables the write-up quotes under
`data/ml-experiments/00049-.../work/stage0/`. Two analyses:

- D0: for the monthly-retrained, weekly-retrained, and P-roll arms without the 14-day correction,
  apply spread factors about the median and choose the factor for each scored month on earlier
  months only (the rule of experiment 00035). Contrast: the rolling-origin factor minus the fixed
  1.15.
- B1: blend the quantiles of five members with one weight vector per scored month (grid of
  quarters, chosen on earlier months only). Contrast: the blend minus the best single member.

    uv run python ml-experiments/00049-vp1-screening-reference-under-p-roll-demand/e00049_stage0.py
"""

import itertools
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import numpy as np
import polars as pl
from studies.sources import ML_EXPERIMENTS_DATA_DIR

from studies import bootstrap, experiment_store

SLUG: Final[str] = "vp1-screening-reference-under-p-roll-demand"
WORK_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / f"00049-{SLUG}" / "work" / "stage0"
ARMS_DIR_00034: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR / "00034-retraining-cadence-and-cheap-proxies-demand" / "work" / "arms"
)
REFERENCE_DIR_00034: Final[Path] = ARMS_DIR_00034.parent / "reference"
SHARED_KEYS_00033: Final[Path] = (
    ML_EXPERIMENTS_DATA_DIR
    / "00033-best-guess-architecture-vs-production"
    / "work"
    / "metrics"
    / "Primary"
    / "shared_keys.parquet"
)
HOLDOUT: Final[datetime] = datetime(2026, 7, 1, tzinfo=UTC)
LEVELS: Final[np.ndarray] = np.array([0.05, 0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 0.9, 0.95])
LEVEL_COLUMNS: Final[tuple[str, ...]] = tuple(f"q{round(100 * x):02d}" for x in LEVELS)
ATOM_MASSES: Final[np.ndarray] = np.diff(
    np.concatenate([[0.0], (LEVELS[:-1] + LEVELS[1:]) / 2, [1.0]])
)
MEDIAN: Final[int] = 4
CASE_KEYS: Final[tuple[str, ...]] = ("time_series_id", "power_fcst_init_time", "valid_time")
KEYS: Final[tuple[str, ...]] = (*CASE_KEYS, "seed")
CAPACITY_POINTS: Final[float] = 100.0
FIXED_FACTOR: Final[float] = 1.15
FACTORS: Final[np.ndarray] = np.round(np.arange(1.0, 1.3001, 0.025), 3)
"""The spread-factor grid, 1.000 to 1.300."""
SPREAD_ARMS: Final[tuple[str, ...]] = ("monthly", "weekly", "roll12")
"""Raw (no correction, no spread) arms of experiment 00034 that D0 re-tunes."""
BLEND_MEMBERS: Final[dict[str, tuple[Path, str]]] = {
    "monthly": (ARMS_DIR_00034, "monthly_s"),
    "weekly": (ARMS_DIR_00034, "weekly_s"),
    "a7_monthly": (ARMS_DIR_00034, "a7_monthly_s"),
    "type_id_monthly": (ARMS_DIR_00034, "type_id_monthly_s"),
    "stack": (REFERENCE_DIR_00034, "n1_cs"),
}
OTHER_ARMS: Final[dict[str, tuple[Path, str]]] = {
    "roll12_s": (ARMS_DIR_00034, "roll12_s"),
    "type_id_roll12_s": (ARMS_DIR_00034, "type_id_roll12_s"),
    "manual_heuristic": (REFERENCE_DIR_00034, "manual_heuristic"),
    "climatology": (REFERENCE_DIR_00034, "climatology"),
}
GRID_STEP: Final[int] = 4
"""Blend weights are multiples of 1/4."""
LEAD_BANDS: Final[tuple[tuple[str, float, float], ...]] = (
    ("0-24 h", 0.0, 24.0),
    ("1-2 d", 24.0, 48.0),
    ("2-6 d", 48.0, 144.0),
    ("6-10 d", 144.0, 240.0),
    ("10+ d", 240.0, 10_000.0),
)
DECISIVE_COUNT: Final[int] = 6
DECISIVE_LEVEL: Final[float] = 100.0 * (1.0 - 0.05 / DECISIVE_COUNT)
N_RESAMPLES_DECISIVE: Final[int] = 20_000
N_RESAMPLES: Final[int] = 5_000
BLEND_THRESHOLD: Final[float] = 0.05


def quantile_score(*, quantiles: np.ndarray, actual: np.ndarray) -> np.ndarray:
    """Return twice the mass-weighted mean pinball loss of the nine levels, per case."""
    difference = actual[:, None] - quantiles
    pinball = np.where(difference >= 0, LEVELS * difference, (LEVELS - 1) * difference)
    return 2.0 * (pinball * ATOM_MASSES).sum(axis=1)


def spread(*, quantiles: np.ndarray, factor: float | np.ndarray) -> np.ndarray:
    """Scale each case's quantiles about its median by `factor` and sort the levels."""
    median = quantiles[:, [MEDIAN]]
    scale = np.asarray(factor).reshape(-1, 1) if np.ndim(factor) else factor
    return np.sort(median + scale * (quantiles - median), axis=1)


def read_arm(*, folder: Path, name: str, shared: pl.DataFrame) -> pl.DataFrame:
    """Return an arm's seed-0 cases that experiment 00033's Primary shared set holds."""
    frame = pl.read_parquet(folder / f"{name}.parquet").filter(
        (pl.col("seed") == 0) & (pl.col("valid_time") < HOLDOUT)
    )
    return (
        frame.select(*KEYS, *LEVEL_COLUMNS, "actual")
        .join(shared, on=list(CASE_KEYS), how="semi")
        .filter(pl.all_horizontal(pl.col(*LEVEL_COLUMNS, "actual").is_finite()))
        .sort(*KEYS)
    )


def load_all() -> tuple[pl.DataFrame, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Return the shared cases, each raw arm's quantiles, and each scored arm's quantiles.

    Every table is restricted to the cases that all of them hold, so scores are comparable.
    """
    shared = pl.read_parquet(SHARED_KEYS_00033)
    raw_tables = {
        name: read_arm(folder=ARMS_DIR_00034, name=name, shared=shared) for name in SPREAD_ARMS
    }
    scored_tables = {
        name: read_arm(folder=folder, name=file, shared=shared)
        for name, (folder, file) in {**BLEND_MEMBERS, **OTHER_ARMS}.items()
    }
    keys = None
    for table in (*raw_tables.values(), *scored_tables.values()):
        these = table.select(*KEYS)
        keys = these if keys is None else keys.join(these, on=list(KEYS), how="inner")
    assert keys is not None
    keys = keys.sort(*KEYS)

    def restrict(table: pl.DataFrame) -> pl.DataFrame:
        return keys.join(table, on=list(KEYS), how="left", validate="1:1").sort(*KEYS)

    raw_tables = {n: restrict(t) for n, t in raw_tables.items()}
    scored_tables = {n: restrict(t) for n, t in scored_tables.items()}
    first = next(iter(raw_tables.values()))
    for name, table in {**raw_tables, **scored_tables}.items():
        if not np.array_equal(table["actual"].to_numpy(), first["actual"].to_numpy()):
            msg = f"arm {name} holds a different observation on some case"
            raise ValueError(msg)
    cases = first.select(*KEYS, "actual").with_columns(
        lead_hours=(pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_minutes()
        / 60.0,
        month=pl.col("power_fcst_init_time").dt.strftime("%Y-%m"),
    )
    band = np.full(cases.height, -1)
    lead = cases["lead_hours"].to_numpy()
    for i, (_, low, high) in enumerate(LEAD_BANDS):
        band[(lead >= low) & (lead < high)] = i
    cases = cases.with_columns(band=pl.Series(band))
    raw = {n: t.select(LEVEL_COLUMNS).to_numpy().astype(np.float64) for n, t in raw_tables.items()}
    scored = {
        n: t.select(LEVEL_COLUMNS).to_numpy().astype(np.float64) for n, t in scored_tables.items()
    }
    return cases, raw, scored


def history_mask(*, cases: pl.DataFrame, month: str) -> np.ndarray:
    """Return the cases a scored month may learn from: earlier runs with earlier valid times."""
    start = np.datetime64(
        datetime.strptime(month, "%Y-%m").replace(tzinfo=UTC).replace(tzinfo=None), "us"
    )
    valid = cases["valid_time"].to_numpy().astype("datetime64[us]")
    return (cases["month"].to_numpy() < month) & (valid < start)


def rolling_factors(*, cases: pl.DataFrame, curve: np.ndarray) -> dict[str, float]:
    """Return the factor chosen for each scored month from `curve` (cases by factors of scores).

    A month with no history uses `FIXED_FACTOR`. Ties go to the factor nearest `FIXED_FACTOR`.
    """
    nearest = np.argsort(np.abs(FACTORS - FIXED_FACTOR), kind="stable")
    chosen = {}
    for month in sorted(cases["month"].unique().to_list()):
        mask = history_mask(cases=cases, month=month)
        if not mask.any():
            chosen[month] = FIXED_FACTOR
            continue
        mean = curve[mask].mean(axis=0)
        chosen[month] = float(FACTORS[nearest[np.argsort(mean[nearest], kind="stable")[0]]])
    return chosen


def weight_grid(*, members: int) -> np.ndarray:
    """Return every weight vector on the grid that sums to 1, shape (vectors, members)."""
    rows = [
        v for v in itertools.product(range(GRID_STEP + 1), repeat=members) if sum(v) == GRID_STEP
    ]
    return np.array(rows, dtype=np.float64) / GRID_STEP


def blend(*, members: list[np.ndarray], weights: np.ndarray) -> np.ndarray:
    """Return the weighted mean of each quantile level, sorted; weights are per member or case."""
    stacked = np.stack(members, axis=0)
    if weights.ndim == 1:
        mixed = np.tensordot(weights, stacked, axes=1)
    else:
        mixed = np.einsum("cm,mcl->cl", weights, stacked)
    return np.sort(mixed, axis=1)


def rolling_blend(
    *, cases: pl.DataFrame, members: dict[str, np.ndarray], by_band: bool
) -> tuple[np.ndarray, pl.DataFrame]:
    """Return the blend with weights chosen on earlier months, and the weight table.

    A month with no history gets equal weights. Ties go to the vector nearest equal weights.
    """
    names = list(members)
    grid = weight_grid(members=len(names))
    equal = np.full(len(names), 1.0 / len(names))
    nearest = np.argsort(np.abs(grid - equal).sum(axis=1), kind="stable")
    actual = cases["actual"].to_numpy().astype(np.float64)
    arrays = [members[n] for n in names]
    scores = np.stack(
        [quantile_score(quantiles=blend(members=arrays, weights=w), actual=actual) for w in grid],
        axis=1,
    )
    month_col = cases["month"].to_numpy()
    band_col = cases["band"].to_numpy()
    case_weights = np.tile(equal, (cases.height, 1))
    rows = []
    for month in sorted(cases["month"].unique().to_list()):
        history = history_mask(cases=cases, month=month)
        for b in range(len(LEAD_BANDS)) if by_band else [-1]:
            use = history & ((band_col == b) if by_band else True)
            target = (month_col == month) & ((band_col == b) if by_band else True)
            count = int(use.sum())
            if count:
                mean = scores[use].mean(axis=0)
                chosen = grid[nearest[np.argsort(mean[nearest], kind="stable")[0]]]
            else:
                chosen = equal
            case_weights[target] = chosen
            rows.append(
                {
                    "month": month,
                    "band": b,
                    "history_cases": count,
                    **{f"w_{n}": float(w) for n, w in zip(names, chosen, strict=True)},
                }
            )
    return blend(members=arrays, weights=case_weights), pl.DataFrame(rows)


def per_case_loss(*, cases: pl.DataFrame, quantiles: np.ndarray) -> pl.DataFrame:
    """Return the keys with the quantile score and the absolute error of the median, in points."""
    actual = cases["actual"].to_numpy().astype(np.float64)
    return cases.select(*KEYS, "band").with_columns(
        qs=pl.Series(CAPACITY_POINTS * quantile_score(quantiles=quantiles, actual=actual)),
        mae=pl.Series(CAPACITY_POINTS * np.abs(quantiles[:, MEDIAN] - actual)),
    )


def contrast(
    *, arms: dict[str, pl.DataFrame], treatment: str, reference: str, level: float, n: int
) -> dict[str, Any]:
    """Return treatment minus reference in quantile score, with its interval."""
    losses = experiment_store.losses_for_bootstrap(
        arm_losses={name: arms[name].select(*KEYS, "qs") for name in (treatment, reference)}
    )
    differences, _ = bootstrap.paired_differences(
        losses=losses, treatment=treatment, reference=reference, metric="qs"
    )
    lower, upper = bootstrap.bootstrap_difference_at_level(
        losses=losses,
        treatment=treatment,
        reference=reference,
        metric="qs",
        level=level,
        n_resamples=n,
    )
    return {
        "treatment": treatment,
        "reference": reference,
        "level": level,
        "difference": float(differences.mean()),
        "lower": lower,
        "upper": upper,
        "cases": int(differences.size),
    }


def blend_outcome(*, row: dict[str, Any]) -> str:
    """Return B1's outcome under the plan: port, small gain, harm, or null."""
    lower, upper, estimate = float(row["lower"]), float(row["upper"]), float(row["difference"])
    if upper < 0:
        return "port" if estimate <= -BLEND_THRESHOLD else "small gain"
    if lower > 0:
        return "harm"
    return "null"


def main() -> None:
    """Run D0 and B1 and write and print every table."""
    WORK_DIR.mkdir(parents=True, exist_ok=True)
    cases, raw, scored = load_all()
    actual = cases["actual"].to_numpy().astype(np.float64)
    print(f"cases: {cases.height:,}, series: {cases['time_series_id'].n_unique()}")
    arms: dict[str, pl.DataFrame] = {}
    curve_rows, factor_rows, d0_rows = [], [], []
    for name in SPREAD_ARMS:
        curve = np.stack(
            [
                quantile_score(quantiles=spread(quantiles=raw[name], factor=f), actual=actual)
                for f in FACTORS
            ],
            axis=1,
        )
        for f, column in zip(FACTORS, curve.T, strict=True):
            curve_rows.append(
                {"arm": name, "factor": float(f), "qs": CAPACITY_POINTS * float(column.mean())}
            )
        chosen = rolling_factors(cases=cases, curve=curve)
        for month, factor in chosen.items():
            factor_rows.append({"arm": name, "month": month, "factor": factor})
        per_case = np.array([chosen[m] for m in cases["month"].to_list()])
        arms[f"{name}_fixed"] = per_case_loss(
            cases=cases, quantiles=spread(quantiles=raw[name], factor=FIXED_FACTOR)
        )
        arms[f"{name}_rolling"] = per_case_loss(
            cases=cases, quantiles=spread(quantiles=raw[name], factor=per_case)
        )
        arms[f"{name}_nospread"] = per_case_loss(cases=cases, quantiles=raw[name])
        d0_rows.append(
            {
                "family": "D0" if name == "monthly" else "descriptive",
                **contrast(
                    arms=arms,
                    treatment=f"{name}_rolling",
                    reference=f"{name}_fixed",
                    level=DECISIVE_LEVEL if name == "monthly" else 95.0,
                    n=N_RESAMPLES_DECISIVE if name == "monthly" else N_RESAMPLES,
                ),
            }
        )
    for name, quantiles in scored.items():
        arms[name] = per_case_loss(cases=cases, quantiles=quantiles)
    members = {n: scored[n] for n in BLEND_MEMBERS}
    blend_rows = []
    for label, chosen_members, by_band in (
        ("blend_five", members, False),
        ("blend_five_bands", members, True),
        (
            "blend_three",
            {n: members[n] for n in ("monthly", "a7_monthly", "stack")},
            False,
        ),
    ):
        quantiles, table = rolling_blend(cases=cases, members=chosen_members, by_band=by_band)
        arms[label] = per_case_loss(cases=cases, quantiles=quantiles)
        table.write_parquet(WORK_DIR / f"weights_{label}.parquet")
    equal_quantiles = blend(
        members=list(members.values()), weights=np.full(len(members), 1 / len(members))
    )
    arms["blend_five_equal"] = per_case_loss(cases=cases, quantiles=equal_quantiles)
    member_scores = {n: float(arms[n]["qs"].to_numpy().mean()) for n in BLEND_MEMBERS}
    best = min(member_scores, key=lambda n: member_scores[n])
    three_best = min(("monthly", "a7_monthly", "stack"), key=lambda n: member_scores[n])
    plan = [
        ("B1", "blend_five", best, DECISIVE_LEVEL, N_RESAMPLES_DECISIVE),
        ("descriptive", "blend_five_bands", best, 95.0, N_RESAMPLES),
        ("descriptive", "blend_five_equal", best, 95.0, N_RESAMPLES),
        ("descriptive", "blend_three", three_best, 95.0, N_RESAMPLES),
        ("descriptive", "blend_five", "blend_three", 95.0, N_RESAMPLES),
        ("descriptive", "weekly", "monthly", 95.0, N_RESAMPLES),
        ("descriptive", "monthly", "stack", 95.0, N_RESAMPLES),
    ]
    for family, treatment, reference, level, n in plan:
        blend_rows.append(
            {
                "family": family,
                **contrast(arms=arms, treatment=treatment, reference=reference, level=level, n=n),
            }
        )
    absolute = []
    for name, frame in arms.items():
        interval = bootstrap.bootstrap_absolute(
            losses=experiment_store.losses_for_bootstrap(
                arm_losses={name: frame.select(*KEYS, "qs")}
            ),
            arm=name,
            metric="qs",
        )
        absolute.append(
            {
                "arm": name,
                "qs": interval["value"],
                "lower_95": interval["lower_95"],
                "upper_95": interval["upper_95"],
                "mae": float(frame["mae"].to_numpy().mean()),
            }
        )
    tables = {
        "curve": pl.DataFrame(curve_rows),
        "rolling_factors": pl.DataFrame(factor_rows),
        "contrasts": pl.DataFrame([*d0_rows, *blend_rows]),
        "absolute": pl.DataFrame(absolute),
    }
    for name, table in tables.items():
        table.write_parquet(WORK_DIR / f"{name}.parquet")
    best_in_sample = (
        tables["curve"].sort("qs").group_by("arm", maintain_order=True).first().sort("arm")
    )
    d0 = next(r for r in d0_rows if r["family"] == "D0")
    print("best single blend member:", best, member_scores)
    print("B1 outcome:", blend_outcome(row=next(r for r in blend_rows if r["family"] == "B1")))
    print("D0 estimate and interval:", d0["difference"], d0["lower"], d0["upper"])
    with pl.Config(tbl_rows=200, tbl_cols=30, tbl_width_chars=250):
        print(best_in_sample)
        print(tables["curve"].pivot(on="arm", index="factor", values="qs"))
        print(tables["rolling_factors"].pivot(on="arm", index="month", values="factor"))
        print(tables["absolute"].sort("qs"))
        print(tables["contrasts"])


if __name__ == "__main__":
    main()
