# ruff: noqa: E501
"""Re-score experiments 224, 243, 248, 257, and 260 with and without sorted quantiles.

Each stored row has 11 quantiles; the second score sorts them ascending.

For every arm the script counts how often a row has crossing quantiles (a lower level above a higher
one), then scores the stored rows as they are ("original") and with the quantiles sorted ascending
in each row ("sorted"). The scoring copies the scorers of the five experiments: rows are experiment
224's common daylight half-hours at 12 to 36 hour leads, the quantile score is twice the mean
pinball loss over the 11 levels in points of capacity, losses are averaged over seeds per row, farms
have equal weight, and intervals resample whole calendar months 20,000 times with each experiment's
original random seed. The original column therefore reproduces the published numbers.

    uv run --no-sync python ml-experiments/00273-*/e00273_rescore.py
"""

import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
from studies.ens_members import MIXTURE_LEVELS
from studies.sources import ML_EXPERIMENTS_DATA_DIR
from studies.tail_calibration import level_column

SLUG: Final[str] = "00273-metered-farm-sorted-quantile-rescore"
DATA_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / SLUG
E224_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00224-metered-farm-dayahead-two-number"
E243_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00243-metered-farm-architecture-comparison"
E248_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00248-metered-farm-architecture-ukv-inputs"
E257_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00257-metered-farm-fair-plant-arm"
E260_DIR: Final[Path] = ML_EXPERIMENTS_DATA_DIR / "00260-metered-farm-quantile-combination"
FARMS: Final[tuple[str, ...]] = ("A", "B", "C", "D", "E", "F")
LEVELS: Final[np.ndarray] = np.array(MIXTURE_LEVELS)
QCOLS: Final[list[str]] = [level_column(level=lv) for lv in MIXTURE_LEVELS]
DRAWS: Final[int] = 20_000
FLOORS: Final[tuple[str, ...]] = ("floor_heuristic", "floor_climatology")
METRICS: Final[tuple[str, ...]] = ("score", "ae_median", "pin01", "pin99", "cover01", "cover99")
BOOT_METRICS: Final[tuple[str, ...]] = ("score", "ae_median", "pin01", "pin99")
VERSIONS: Final[tuple[str, ...]] = ("original", "sorted")
KEYS: Final[list[str]] = ["time_series_id", "time"]
COLUMNS: Final[list[str]] = ["time_series_id", "time", "arm", "seed", "actual", *QCOLS]
PAIRS: Final[list[str]] = [f"{a}>{b}" for a, b in itertools.pairwise(QCOLS)]


@dataclass(frozen=True)
class Contrast:
    """One headline contrast: `arm` against `reference`, as a percentage change of each metric."""

    name: str
    arm: str
    reference: str
    metrics: tuple[str, ...] = ("score",)


@dataclass(frozen=True)
class Unit:
    """A set of rows scored together: one experiment, one setting or fold."""

    experiment: str
    unit: str
    seed: int
    rows: pl.DataFrame
    contrasts: list[Contrast] = field(default_factory=list)


def read_prediction_file(*, path: Path) -> pl.DataFrame:
    """Return one stored prediction file with the common columns in float64."""
    return (
        pl.read_parquet(path)
        .select(COLUMNS)
        .with_columns(pl.col(QCOLS).cast(pl.Float64), pl.col("actual").cast(pl.Float64))
    )


def read_folder(*, folder: Path, patterns: tuple[str, ...] = ("*.parquet",)) -> pl.DataFrame:
    """Return every stored prediction file in a folder that matches a pattern."""
    files = sorted(f for p in patterns for f in folder.glob(p))
    return pl.concat([read_prediction_file(path=f) for f in files])


def e224_rows(*, fold: str) -> tuple[pl.DataFrame, pl.DataFrame]:
    """Return experiment 224's band-0 rows (as `COLUMNS`) that every arm holds, and its common-row keys."""
    frame = pl.read_parquet(E224_DIR / "predictions" / f"{fold}.parquet").filter(
        pl.col("band") == 0
    )
    ok = frame.filter(pl.all_horizontal([pl.col(c).is_finite() for c in QCOLS]))
    keys = (
        ok.group_by("time_series_id", "valid_time")
        .agg(n=pl.col("arm").n_unique())
        .filter(pl.col("n") == frame["arm"].n_unique())
        .select("time_series_id", "valid_time")
    )
    rows = (
        frame.join(keys, on=["time_series_id", "valid_time"])
        .rename({"valid_time": "time"})
        .select(COLUMNS)
    )
    return rows, keys.rename({"valid_time": "time"})


def on_keys(*, frame: pl.DataFrame, keys: pl.DataFrame) -> pl.DataFrame:
    """Keep the rows of a frame that sit on the common-row keys."""
    return frame.join(keys, on=KEYS, how="inner")


def with_month(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Add the calendar month the bootstrap resamples."""
    return frame.with_columns(month=pl.col("time").dt.strftime("%Y-%m"))


def build_units() -> list[Unit]:
    """Return every unit scored: the five experiments' settings, their rows, and headline contrasts."""
    units: list[Unit] = []
    rows224, keys = e224_rows(fold="scored_year")
    floors = rows224.filter(pl.col("arm").is_in(FLOORS))
    units.append(
        Unit(
            "224",
            "scored_year",
            224,
            with_month(frame=rows224),
            [
                Contrast("C0 members minus ensemble mean", "a4_members", "a2_mean"),
                Contrast("C1 two-number minus ensemble mean", "a3_two_number", "a2_mean"),
                Contrast("C2 two-number minus members", "a3_two_number", "a4_members"),
                Contrast(
                    "C3 two-number minus XGBoost",
                    "a3_two_number",
                    "a5_xgboost",
                    ("score", "ae_median", "pin01", "pin99"),
                ),
                Contrast(
                    "C8 two-number minus manual heuristic", "a3_two_number", "floor_heuristic"
                ),
            ],
        )
    )
    for setting in ("own_history", "zero_history"):
        e243 = on_keys(frame=read_folder(folder=E243_DIR / "predictions" / setting), keys=keys)
        units.append(
            Unit(
                "243",
                setting,
                243,
                with_month(frame=pl.concat([e243, floors])),
                [
                    Contrast("H5 minus plant", "h5_quantile", "plant"),
                    Contrast(
                        "H5 minus XGBoost",
                        "h5_quantile",
                        "xgboost",
                        ("score", "ae_median", "pin01", "pin99"),
                    ),
                    Contrast("Neural network minus plant", "nn_pure", "plant"),
                    Contrast(
                        "Neural network minus XGBoost", "nn_pure", "xgboost", ("score", "pin99")
                    ),
                ],
            )
        )
        p1 = (
            on_keys(
                frame=rows224.filter(pl.col("arm") == "a3_two_number"),
                keys=keys,
            )
            .group_by(KEYS)
            .agg(pl.col(QCOLS).mean(), pl.col("actual").mean())
            .with_columns(arm=pl.lit("p1"), seed=pl.lit(0, dtype=pl.Int32))
            .select(COLUMNS)
        )
        own = on_keys(frame=read_folder(folder=E257_DIR / "predictions" / setting), keys=keys)
        pool = [own, e243.filter(pl.col("arm").is_in(["plant", "xgboost", "h5_quantile"])), floors]
        if setting == "own_history":
            pool.append(p1)
        units.append(
            Unit(
                "257",
                setting,
                257,
                with_month(frame=pl.concat(pool)),
                [
                    Contrast("C1 P2 minus H5", "p2", "h5_quantile", ("score", "pin01", "pin99")),
                    Contrast("C3 P2 minus P1", "p2", "p1"),
                    Contrast("C5 P2 minus P2 without ratio", "p2", "p2_no_ratio"),
                ]
                if setting == "own_history"
                else [Contrast("C4 P2 minus H5", "p2", "h5_quantile")],
            )
        )
        own260 = on_keys(frame=read_folder(folder=E260_DIR / "predictions" / setting), keys=keys)
        ref260 = e243.filter(pl.col("arm").is_in(["plant", "xgboost", "h5_quantile", "nn_pure"]))
        units.append(
            Unit(
                "260",
                setting,
                260,
                with_month(frame=pl.concat([own260, ref260, floors])),
                [
                    Contrast("C1 C2 minus H5", "c2", "h5_quantile"),
                    Contrast("M1 C2 minus H5 with seeds averaged", "c2", "h5_mean"),
                    Contrast("C2 C2 minus neural network at 0.99", "c2", "nn_pure", ("pin99",)),
                    Contrast("C3 C3 minus C2", "c3", "c2", ("score", "pin01", "pin99")),
                    Contrast("X3 C2 minus XGBoost", "c2", "xgboost"),
                ]
                if setting == "own_history"
                else [Contrast("C4 C2 minus H5", "c2", "h5_quantile")],
            )
        )
    both: list[pl.DataFrame] = []
    for fold, unit_name in (("scored_year", "scored"), ("replicate", "replicate")):
        rows, fold_keys = e224_rows(fold=fold)
        fold_floors = rows.filter(pl.col("arm").is_in(FLOORS))
        e248 = on_keys(
            frame=read_folder(folder=E248_DIR / "predictions" / unit_name), keys=fold_keys
        )
        fold_floors = on_keys(frame=fold_floors, keys=e248.select(KEYS).unique())
        fold_rows = with_month(frame=pl.concat([e248, fold_floors]))
        both.append(fold_rows)
        units.append(
            Unit(
                "248",
                unit_name,
                248,
                fold_rows,
                [Contrast("H5 with UKV minus H5", "h5_ukv", "h5_quantile")],
            )
        )
    units.append(
        Unit(
            "248",
            "both_folds",
            248,
            pl.concat(both),
            [
                Contrast("H5 with UKV minus H5", "h5_ukv", "h5_quantile"),
                Contrast("H2 with UKV minus H2", "h2_ukv", "h2_residual"),
                Contrast("XGBoost with UKV minus XGBoost", "xgboost_ukv", "xgboost"),
                Contrast("H5 with UKV minus placebo", "h5_ukv", "h5_ukv_placebo"),
                Contrast("H5 with UKV minus XGBoost with UKV", "h5_ukv", "xgboost_ukv"),
                Contrast("H5 with UKV minus plant", "h5_ukv", "plant"),
            ],
        )
    )
    return units


def crossing_table(*, unit: Unit) -> pl.DataFrame:
    """Return, per arm, the share of stored rows with any crossing and the share crossing at each adjacent pair."""
    q = unit.rows.select(QCOLS).to_numpy()
    assert np.isfinite(q).all(), "A NaN quantile would count as no crossing"
    crossed = q[:, :-1] > q[:, 1:]
    size = np.clip(q[:, :-1] - q[:, 1:], 0.0, None).sum(axis=1)
    table = unit.rows.select("arm").with_columns(
        any_crossing=pl.Series(crossed.any(axis=1).astype(float)),
        any_tie=pl.Series((q[:, :-1] == q[:, 1:]).any(axis=1).astype(float)),
        crossing_size=pl.Series(100.0 * size),
        **{p: pl.Series(crossed[:, i].astype(float)) for i, p in enumerate(PAIRS)},
    )
    return (
        table.group_by("arm")
        .agg(
            n_rows=pl.len(),
            any_crossing=pl.col("any_crossing").mean(),
            any_tie=pl.col("any_tie").mean(),
            mean_crossing_size=pl.col("crossing_size").mean(),
            **{p: pl.col(p).mean() for p in PAIRS},
        )
        .with_columns(experiment=pl.lit(unit.experiment), unit=pl.lit(unit.unit))
        .sort("arm")
    )


def row_losses(*, frame: pl.DataFrame, version: str) -> pl.DataFrame:
    """Return per-row losses (points of capacity) of the stored or the sorted quantiles."""
    q = frame.select(QCOLS).to_numpy()
    if version == "sorted":
        q = np.sort(q, axis=1)
    actual = frame["actual"].to_numpy()
    error = actual[:, None] - q
    pin = np.maximum(LEVELS * error, (LEVELS - 1.0) * error)
    return frame.select("time_series_id", "month", "time", "arm").with_columns(
        score=pl.Series(200.0 * pin.mean(axis=1)),
        ae_median=pl.Series(100.0 * np.abs(error[:, 5])),
        pin01=pl.Series(100.0 * pin[:, 0]),
        pin99=pl.Series(100.0 * pin[:, 10]),
        cover01=pl.Series((actual <= q[:, 0]).astype(float)),
        cover99=pl.Series((actual <= q[:, 10]).astype(float)),
    )


def seed_averaged(*, unit: Unit, version: str) -> pl.DataFrame:
    """Return per-(arm, farm, time) losses averaged over seeds, like the original scorers."""
    losses = row_losses(frame=unit.rows, version=version)
    return losses.group_by("arm", "time_series_id", "time", "month").agg(
        pl.col(m).mean() for m in METRICS
    )


def cell_sums(
    *, rows: pl.DataFrame, value: str
) -> tuple[np.ndarray, np.ndarray, list[str], list[str]]:
    """Return sums[arm, farm, month], counts[farm, month], and the arm and month labels."""
    arms = sorted(rows["arm"].unique().to_list())
    months = sorted(rows["month"].unique().to_list())
    sums = np.zeros((len(arms), len(FARMS), len(months)))
    counts = np.zeros((len(FARMS), len(months)))
    grouped = rows.group_by("arm", "time_series_id", "month").agg(
        total=pl.col(value).sum(), n=pl.len()
    )
    arm_index = {a: i for i, a in enumerate(arms)}
    month_index = {m: i for i, m in enumerate(months)}
    for r in grouped.iter_rows(named=True):
        a, f, m = (
            arm_index[r["arm"]],
            FARMS.index(r["time_series_id"]),
            month_index[r["month"]],
        )
        sums[a, f, m], counts[f, m] = r["total"], r["n"]
    return sums, counts, arms, months


def boot(
    *, sums: np.ndarray, counts: np.ndarray, weights: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return the farm-averaged point estimate and the bootstrap draws, both per arm."""
    farm_n = counts.sum(axis=1)
    point = np.nanmean(sums.sum(axis=2) / np.where(farm_n > 0, farm_n, np.nan)[None, :], axis=1)
    denominator = weights @ counts.T
    draws = np.empty((sums.shape[0], weights.shape[0]))
    with np.errstate(invalid="ignore", divide="ignore"):
        for a in range(sums.shape[0]):
            draws[a] = np.nanmean(
                (weights @ sums[a].T) / np.where(denominator > 0, denominator, np.nan), axis=1
            )
    return point, draws


def interval(values: np.ndarray) -> tuple[float, float]:
    """Return the 2.5th and 97.5th percentiles."""
    return float(np.nanpercentile(values, 2.5)), float(np.nanpercentile(values, 97.5))


def score_unit(
    *, unit: Unit
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    """Return the arm score rows, the contrast rows, and the sorted-minus-original rows of a unit."""
    arm_rows: list[dict[str, object]] = []
    contrast_rows: list[dict[str, object]] = []
    gain_rows: list[dict[str, object]] = []
    base = {"experiment": unit.experiment, "unit": unit.unit}
    averaged = {v: seed_averaged(unit=unit, version=v) for v in VERSIONS}
    months = sorted(averaged["original"]["month"].unique().to_list())
    rng = np.random.default_rng(unit.seed)
    weights = rng.multinomial(len(months), np.ones(len(months)) / len(months), size=DRAWS).astype(
        float
    )
    result: dict[tuple[str, str], tuple[np.ndarray, np.ndarray, dict[str, int]]] = {}
    for version in VERSIONS:
        for metric in METRICS:
            sums, counts, arms, _ = cell_sums(rows=averaged[version], value=metric)
            point, draws = boot(sums=sums, counts=counts, weights=weights)
            index = {a: i for i, a in enumerate(arms)}
            result[version, metric] = (point, draws, index)
            for arm in arms:
                lo, hi = interval(draws[index[arm]])
                arm_rows.append(
                    {
                        **base,
                        "version": version,
                        "metric": metric,
                        "arm": arm,
                        "value": float(point[index[arm]]),
                        "lo": lo,
                        "hi": hi,
                    }
                )
    for metric in BOOT_METRICS:
        point_o, draws_o, index = result["original", metric]
        point_s, draws_s, _ = result["sorted", metric]
        for arm, i in index.items():
            change = 100.0 * (draws_s[i] - draws_o[i]) / draws_o[i]
            lo, hi = interval(change)
            gain_rows.append(
                {
                    **base,
                    "metric": metric,
                    "arm": arm,
                    "percent_change": float(100.0 * (point_s[i] - point_o[i]) / point_o[i]),
                    "lo": lo,
                    "hi": hi,
                }
            )
    for contrast in unit.contrasts:
        for metric in contrast.metrics:
            for version in VERSIONS:
                point, draws, index = result[version, metric]
                i, j = index[contrast.arm], index[contrast.reference]
                change = 100.0 * (draws[i] - draws[j]) / draws[j]
                lo, hi = interval(change)
                contrast_rows.append(
                    {
                        **base,
                        "contrast": contrast.name,
                        "metric": metric,
                        "version": version,
                        "arm": contrast.arm,
                        "reference": contrast.reference,
                        "percent_change": float(100.0 * (point[i] - point[j]) / point[j]),
                        "lo": lo,
                        "hi": hi,
                    }
                )
    if unit.experiment == "224":
        for version in VERSIONS:
            point, draws, index = result[version, "score"]
            ref = index["a2_mean"]
            c0 = 100.0 * (draws[index["a4_members"]] - draws[ref]) / draws[ref]
            c1 = 100.0 * (draws[index["a3_two_number"]] - draws[ref]) / draws[ref]
            share = c1 / np.where(np.abs(c0) > 1e-9, c0, np.nan)
            c0p = point[index["a4_members"]] - point[ref]
            c1p = point[index["a3_two_number"]] - point[ref]
            lo, hi = interval(share)
            contrast_rows.append(
                {
                    **base,
                    "contrast": "retained share C1/C0",
                    "metric": "score",
                    "version": version,
                    "arm": "a3_two_number",
                    "reference": "a4_members",
                    "percent_change": float(c1p / c0p),
                    "lo": lo,
                    "hi": hi,
                }
            )
    return arm_rows, contrast_rows, gain_rows


def main() -> None:
    """Write the crossing, arm score, contrast, and sorted-minus-original tables and print them."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    crossing, arm_scores, contrasts, gains = [], [], [], []
    for unit in build_units():
        print(f"scoring {unit.experiment} {unit.unit}: {unit.rows.height} rows")
        if unit.unit != "both_folds":  # its rows are the two fold units' rows again
            crossing.append(crossing_table(unit=unit))
        arm_rows, contrast_rows, gain_rows = score_unit(unit=unit)
        arm_scores += arm_rows
        contrasts += contrast_rows
        gains += gain_rows
    tables = {
        "crossing": pl.concat(crossing),
        "arm_scores": pl.DataFrame(arm_scores),
        "contrasts": pl.DataFrame(contrasts),
        "sorting_gain": pl.DataFrame(gains),
    }
    for name, table in tables.items():
        table.write_csv(DATA_DIR / f"{name}.csv")
    with pl.Config(tbl_rows=400, tbl_cols=30, fmt_str_lengths=50):
        print(
            tables["crossing"]
            .select("experiment", "unit", "arm", "n_rows", "any_crossing", "mean_crossing_size")
            .sort("experiment", "unit", "arm")
        )
        print(
            tables["sorting_gain"]
            .filter(pl.col("metric") == "score")
            .sort("experiment", "unit", "arm")
        )
        print(tables["contrasts"].filter(pl.col("metric") == "score"))


if __name__ == "__main__":
    main()
