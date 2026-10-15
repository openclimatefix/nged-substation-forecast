"""Fit experiment 00026's stage 1a quantile models and build each arm's pooled mixture.

For each seed, series, and arm this fits a 9-level multi-quantile XGBoost model (experiment 00020's
recipe: ERA5-trained, with the power-curve base margin for wind) on half-hourly valid times before
2025-06-17, with the arm's effective-capacity columns added. The columns come from the series'
daily feature table (`e00026_features.py`): a training row reads the row for its cutoff day (5 days
before the UTC day of an origin drawn as its valid time minus a lead copied from the validation
frames), and a validation case reads the row for its `power_fcst_init_time`. Arms A6 and A7 read a
table shuffled across years within each calendar month. The fitted model is applied to the 51 ENS
members of the 53 validation runs and each case's members are pooled into 9 mixture quantiles.

Output under `data/ml-experiments/00026-.../work/` (each skipped if it exists, so a crash costs one
seed): `tables/<series>.npy` and `mixture/<arm>_<technology>_seed<s>.parquet`.

    uv run python ml-experiments/00026-*/e00026_fit.py [--smoke]
"""

import argparse
import time
from pathlib import Path
from typing import Final

import e00026_features as features
import numpy as np
import polars as pl
import xgboost as xgb
from e00026_common import (
    ANALYSIS_PATH,
    ARMS,
    ENS_VALIDATION_DIR,
    ERA5_START,
    KEY_COLUMNS,
    LEVEL_COLUMNS,
    MEMBERS,
    N_ROUNDS,
    SEEDS,
    SOLAR_FEATURES,
    TRAIN_END,
    WIND_FEATURES,
    WORK_DIR,
    mixture_quantiles,
    observed_training_power,
    power_curve,
    power_grids,
    read_levels,
    site_table,
)
from e00026_common import XGBOOST_PARAMETERS as PARAMETERS

TABLE_DIR: Final[Path] = WORK_DIR / "tables"
MIXTURE_DIR: Final[Path] = WORK_DIR / "mixture"
CHECK_CUTOFF_DAYS: Final[tuple[int, ...]] = (1_500, 2_200)
"""The cutoff days of the no-lookahead check: mid-2023, and inside the validation year."""
CAP_COLUMNS: Final[tuple[str, ...]] = tuple(f"cap_{i}" for i in range(5))


def _matrix(*, frame: pl.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
    return frame.select([pl.col(c).cast(pl.Float32) for c in columns]).to_numpy()


def training_rows(*, series: int, analysis: pl.DataFrame, observed: pl.DataFrame) -> pl.DataFrame:
    """Return one series' ERA5 rows with their target."""
    rows = (
        analysis.filter(pl.col("time_series_id") == series)
        .join(
            observed.filter(pl.col("time_series_id") == series),
            on=["time_series_id", "valid_time"],
            how="inner",
            validate="1:1",
        )
        .filter((pl.col("valid_time") >= ERA5_START) & (pl.col("valid_time") < TRAIN_END))
    )
    return rows.sort("valid_time")


class FittedQuantiles:
    """One series' fitted quantile model, with the wind power curve when a base margin is used."""

    def __init__(
        self,
        *,
        rows: pl.DataFrame,
        columns: tuple[str, ...],
        seed: int,
        rounds: int,
        use_margin: bool,
    ) -> None:
        """Fit the model; `use_margin` makes the power curve the base margin of all 9 levels."""
        if rows.height == 0:
            msg = "no training rows"
            raise ValueError(msg)
        self.columns = columns
        self.curve: tuple[np.ndarray, np.ndarray] | None = None
        features = _matrix(frame=rows, columns=columns)
        target = rows["target"].to_numpy()
        margin = None
        if use_margin:
            self.curve = power_curve(
                speed=rows["wind_speed_100m"].to_numpy(), power_fraction=target
            )
            margin = self._margin(speed=rows["wind_speed_100m"].to_numpy())
        matrix = xgb.DMatrix(features, label=target, base_margin=margin)
        self.booster = xgb.train({**PARAMETERS, "seed": seed}, matrix, num_boost_round=rounds)
        if margin is not None:
            plain = self.booster.predict(xgb.DMatrix(features[:2000]))
            with_margin = self.booster.predict(
                xgb.DMatrix(features[:2000], base_margin=margin[:2000])
            )
            if np.allclose(plain, with_margin):
                msg = "the base margin did not change the predictions"
                raise ValueError(msg)

    def _margin(self, *, speed: np.ndarray) -> np.ndarray:
        if self.curve is None:
            msg = "no power curve"
            raise ValueError(msg)
        centres, curve = self.curve
        return np.tile(np.interp(speed, centres, curve)[:, None], (1, len(LEVEL_COLUMNS)))

    def predict(self, *, frame: pl.DataFrame) -> np.ndarray:
        """Return the 9 sorted quantile columns for every row (a fraction of capacity)."""
        margin = (
            None if self.curve is None else self._margin(speed=frame["wind_speed_100m"].to_numpy())
        )
        matrix = xgb.DMatrix(_matrix(frame=frame, columns=self.columns), base_margin=margin)
        return np.sort(self.booster.predict(matrix), axis=1)


def pooled_mixture(*, part: pl.DataFrame, member_quantiles: np.ndarray) -> pl.DataFrame:
    """Return the keys and 9 mixture quantiles of every case in `part`.

    Args:
        part: One run's rows for one series, with `ensemble_member` and the key columns.
        member_quantiles: Shape (rows, 9), the member quantiles of each row of `part`.

    Returns:
        One row per case.

    Raises:
        ValueError: If a case lacks a member.
    """
    ordered = (
        part.select(*KEY_COLUMNS, "ensemble_member")
        .with_columns(row=pl.int_range(pl.len()))
        .sort(*KEY_COLUMNS, "ensemble_member")
    )
    if ordered.height % MEMBERS:
        msg = f"a case lacks some of the {MEMBERS} members"
        raise ValueError(msg)
    matrix = member_quantiles[ordered["row"].to_numpy()].reshape(-1, MEMBERS, len(LEVEL_COLUMNS))
    keys = ordered.gather_every(MEMBERS).select(KEY_COLUMNS)
    if (ordered["ensemble_member"].to_numpy().reshape(-1, MEMBERS) != np.arange(MEMBERS)).any():
        msg = "the members of a case are not 0 to 50"
        raise ValueError(msg)
    mixture = read_levels(fine=mixture_quantiles(member_quantiles=matrix.astype(np.float32)))
    return pl.concat([keys, pl.DataFrame(mixture, schema=list(LEVEL_COLUMNS))], how="horizontal")


def feature_tables(*, sites: pl.DataFrame, analysis: pl.DataFrame) -> dict[int, np.ndarray]:
    """Return (or build and cache) each series' daily feature table, after the lookahead check."""
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    grids = None
    out = {}
    for row in sites.iter_rows(named=True):
        series, technology = int(row["time_series_id"]), row["technology"]
        path = TABLE_DIR / f"{series}.npy"
        if path.exists():
            out[series] = np.load(path)
            continue
        if grids is None:
            grids = power_grids(sites=sites)
        other, zenith = features.series_grids(
            analysis=analysis, series=series, technology=technology
        )
        power = grids[series][: features.N_STEPS]
        for cutoff in CHECK_CUTOFF_DAYS:
            features.check_no_lookahead(
                technology=technology, power=power, other=other, zenith=zenith, cutoff=cutoff
            )
        out[series] = features.daily_table(
            technology=technology, power=power, other=other, zenith=zenith
        )
        np.save(path, out[series])
        print(f"series {series}: feature table built and lookahead check passed", flush=True)
    return out


def arm_columns(*, arm: str, table: np.ndarray, shuffled: np.ndarray) -> tuple[np.ndarray, int]:
    """Return the table an arm reads and the number of capacity columns it uses."""
    chosen = features.ARM_FEATURES[arm]
    source = shuffled if arm in ("a6", "a7") else table
    return source[:, list(chosen)], len(chosen)


def with_capacity(
    *, frame: pl.DataFrame, source: np.ndarray, origins: np.ndarray
) -> tuple[pl.DataFrame, tuple[str, ...]]:
    """Return `frame` with the capacity columns of the origins' cutoff days, and their names."""
    values = features.cutoff_rows(table=source, origins=origins)
    names = CAP_COLUMNS[: values.shape[1]]
    columns = [pl.Series(name, values[:, i]) for i, name in enumerate(names)]
    return frame.with_columns(columns), names


def validation_leads(*, files: list[Path]) -> np.ndarray:
    """Return the distinct leads in hours over all the validation frames."""
    frame = pl.concat(
        [
            pl.read_parquet(path, columns=["valid_time", "power_fcst_init_time"]).select(
                lead=(pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_minutes()
                / 60.0
            )
            for path in files
        ]
    )
    return np.sort(frame["lead"].unique().to_numpy())


def fit_seed(
    *,
    seed: int,
    sites: pl.DataFrame,
    analysis: pl.DataFrame,
    observed: pl.DataFrame,
    tables: dict[int, np.ndarray],
    rounds: int,
    limit_runs: int | None,
) -> None:
    """Fit every arm for one seed and write its mixtures."""
    technologies = dict(zip(sites["time_series_id"], sites["technology"], strict=True))
    targets: dict[tuple[str, str], Path] = {
        (arm, technology): MIXTURE_DIR / f"{arm}_{technology}_seed{seed}.parquet"
        for arm in ARMS
        for technology in ("solar", "wind")
    }
    if limit_runs is None and all(p.exists() for p in targets.values()):
        return
    started = time.monotonic()
    files = sorted(ENS_VALIDATION_DIR.glob("init=*.parquet"))[:limit_runs]
    leads = validation_leads(files=files)
    last_source_day = int(
        features.day_index(
            times=np.array([TRAIN_END.replace(tzinfo=None)], dtype="datetime64[us]")
        )[0]
    )
    shuffled = {
        series: features.shuffled_table(
            table=table,
            seed=1_000 * seed + series,
            last_source_day=last_source_day,
        )
        for series, table in tables.items()
    }
    models: dict[tuple[int, str], tuple[FittedQuantiles, tuple[str, ...]]] = {}
    for series, technology in technologies.items():
        base_columns = SOLAR_FEATURES if technology == "solar" else WIND_FEATURES
        rows = training_rows(series=series, analysis=analysis, observed=observed)
        origins = features.training_origins(
            valid=rows["valid_time"].to_numpy(), leads_hours=leads, seed=seed * 100 + series
        )
        for arm in ARMS:
            source, _ = arm_columns(arm=arm, table=tables[series], shuffled=shuffled[series])
            augmented, names = with_capacity(frame=rows, source=source, origins=origins)
            models[series, arm] = (
                FittedQuantiles(
                    rows=augmented,
                    columns=(*base_columns, *names),
                    seed=seed,
                    rounds=rounds,
                    use_margin=technology == "wind",
                ),
                names,
            )
    print(f"seed {seed}: fits in {time.monotonic() - started:.0f} s", flush=True)
    pieces: dict[tuple[str, str], list[pl.DataFrame]] = {key: [] for key in targets}
    for path in files:
        frame = pl.read_parquet(path)
        for (series_key,), part in frame.partition_by("time_series_id", as_dict=True).items():
            series = int(series_key)
            technology = technologies[series]
            origins = part["power_fcst_init_time"].to_numpy()
            for arm in ARMS:
                source, _ = arm_columns(arm=arm, table=tables[series], shuffled=shuffled[series])
                inputs, _ = with_capacity(frame=part, source=source, origins=origins)
                model, _ = models[series, arm]
                pieces[arm, technology].append(
                    pooled_mixture(part=part, member_quantiles=model.predict(frame=inputs))
                )
    if limit_runs is not None:
        print("smoke run: no output written", flush=True)
        return
    MIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    for key, frames in pieces.items():
        target = targets[key]
        pl.concat(frames).sort(*KEY_COLUMNS).write_parquet(target.with_suffix(".tmp"))
        target.with_suffix(".tmp").rename(target)
    print(f"seed {seed} done in {time.monotonic() - started:.0f} s", flush=True)


def main() -> None:
    """Build the feature tables and fit every seed, skipping seeds already written."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="20 rounds, 2 runs, no output")
    smoke = parser.parse_args().smoke
    sites = site_table()
    analysis = pl.read_parquet(ANALYSIS_PATH)
    observed = observed_training_power(sites=sites)
    tables = feature_tables(sites=sites, analysis=analysis)
    for seed in SEEDS[:1] if smoke else SEEDS:
        fit_seed(
            seed=seed,
            sites=sites,
            analysis=analysis,
            observed=observed,
            tables=tables,
            rounds=20 if smoke else N_ROUNDS,
            limit_runs=2 if smoke else None,
        )


if __name__ == "__main__":
    main()
