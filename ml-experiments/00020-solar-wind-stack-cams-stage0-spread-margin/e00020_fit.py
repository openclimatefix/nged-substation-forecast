"""Fit experiment 00020's stage 1a quantile models and build each variant's pooled mixture.

For each seed and series this fits a 9-level multi-quantile XGBoost model on the half-hourly valid
times before 2025-06-17 for each training variant of `e00020_common.VARIANTS`, applies it to the 51
ENS members of the 53 validation runs, and pools each case's members into a mixture of 9 quantiles.
Stage 0 (an empirical quantile mapping of ENS shortwave onto CAMS or ERA5 irradiance, fitted on the
ENS control member's pairs before 2025-06-17) is applied to the inputs of the `cams_s0` and
`base_s0` variants. The spread factor is a separate step (`e00020_arms.py`).

Output under `data/ml-experiments/00020-.../work/` (each skipped if it exists, so a crash costs one
seed): `stage0/<target>.npz` and `stage0/pairs.parquet`, and `mixture/<variant>_seed<s>.parquet`
with the key columns and the 9 mixture quantiles of every case.

    uv run python ml-experiments/00020-*/e00020_fit.py [--smoke]
"""

import argparse
import time
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
import xgboost as xgb
from e00020_common import (
    ANALYSIS_PATH,
    ENS_TRAIN_PATH,
    ENS_VALIDATION_DIR,
    ERA5_START,
    KEY_COLUMNS,
    LEAD_BAND_EDGES,
    LEVEL_COLUMNS,
    MAX_ZENITH_DEGREES,
    MEMBERS,
    MIN_PAIRS_PER_BIN,
    N_QUANTILE_POINTS,
    N_ROUNDS,
    N_ZENITH_BINS,
    SEEDS,
    SOLAR_FEATURES,
    TRAIN_END,
    VARIANTS,
    WIND_FEATURES,
    WORK_DIR,
    TechnologyType,
    VariantType,
    mixture_quantiles,
    observed_training_power,
    power_curve,
    read_levels,
    site_table,
)
from e00020_common import XGBOOST_PARAMETERS as PARAMETERS

STAGE0_DIR: Final[Path] = WORK_DIR / "stage0"
MIXTURE_DIR: Final[Path] = WORK_DIR / "mixture"
TARGETS: Final[tuple[str, ...]] = ("cams", "era5")
TARGET_COLUMN: Final[dict[str, str]] = {"cams": "shortwave_cams", "era5": "shortwave"}
N_BANDS: Final[int] = len(LEAD_BAND_EDGES) - 1
COS_ZENITH_DAYLIGHT_EDGE: Final[float] = float(np.cos(np.radians(MAX_ZENITH_DEGREES)))
COS_ZENITH_EDGES: Final[np.ndarray] = np.linspace(COS_ZENITH_DAYLIGHT_EDGE, 1.0, N_ZENITH_BINS + 1)
STAGE0_VARIANT_TARGET: Final[dict[str, str]] = {"cams_s0": "cams", "base_s0": "era5"}
"""The irradiance each stage 0 variant maps ENS shortwave onto: the one its stage 1a trained on."""


def _band(*, lead_hours: np.ndarray) -> np.ndarray:
    """Return each lead's band index, 0 for up to 1 day and 4 for 11 to 15 days."""
    return np.clip(np.searchsorted(LEAD_BAND_EDGES, lead_hours, side="right") - 1, 0, N_BANDS - 1)


def _zenith_bin(*, cos_zenith: np.ndarray) -> np.ndarray:
    """Return each cosine's bin index, or -1 where the sun is below the daylight limit."""
    index = np.searchsorted(COS_ZENITH_EDGES, cos_zenith, side="right") - 1
    index = np.minimum(index, N_ZENITH_BINS - 1)
    return np.where(cos_zenith < COS_ZENITH_DAYLIGHT_EDGE, -1, index)


def fit_mapping(
    *, ens: np.ndarray, target: np.ndarray, lead_hours: np.ndarray, cos_zenith: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit one quantile mapping per lead band and zenith bin.

    Args:
        ens: Paired ENS shortwave.
        target: Paired target irradiance.
        lead_hours: Lead of each pair.
        cos_zenith: Cosine of the solar zenith at each pair's valid time.

    Returns:
        Arrays of shape (bands, zenith bins, quantile points) holding the ENS quantiles and the
        target quantiles, and the pair count of each bin. A bin with too few pairs takes the
        mapping of the nearest zenith bin in the same band that has enough.

    Raises:
        ValueError: If a lead band has no bin with enough pairs.
    """
    points = np.linspace(0.0, 1.0, N_QUANTILE_POINTS)
    band, zbin = _band(lead_hours=lead_hours), _zenith_bin(cos_zenith=cos_zenith)
    xs = np.full((N_BANDS, N_ZENITH_BINS, points.size), np.nan)
    ys = np.full_like(xs, np.nan)
    counts = np.zeros((N_BANDS, N_ZENITH_BINS), dtype=np.int64)
    for b in range(N_BANDS):
        for z in range(N_ZENITH_BINS):
            keep = (band == b) & (zbin == z)
            counts[b, z] = keep.sum()
            if counts[b, z] >= MIN_PAIRS_PER_BIN:
                xs[b, z] = np.quantile(ens[keep], points)
                ys[b, z] = np.quantile(target[keep], points)
    for b in range(N_BANDS):
        enough = np.flatnonzero(counts[b] >= MIN_PAIRS_PER_BIN)
        if enough.size == 0:
            msg = f"lead band {b} has no zenith bin with {MIN_PAIRS_PER_BIN} pairs"
            raise ValueError(msg)
        for z in range(N_ZENITH_BINS):
            if counts[b, z] < MIN_PAIRS_PER_BIN:
                nearest = enough[np.abs(enough - z).argmin()]
                xs[b, z], ys[b, z] = xs[b, nearest], ys[b, nearest]
    return xs, ys, counts


def apply_mapping(
    *,
    mapping: tuple[np.ndarray, np.ndarray],
    ens: np.ndarray,
    lead_hours: np.ndarray,
    cos_zenith: np.ndarray,
) -> np.ndarray:
    """Map each ENS value through its bin's piecewise-linear quantile mapping.

    Night values pass through, and values beyond a bin's fitted range take the end point.
    """
    xs, ys = mapping
    band, zbin = _band(lead_hours=lead_hours), _zenith_bin(cos_zenith=cos_zenith)
    out = ens.astype(np.float64).copy()
    for b in range(N_BANDS):
        for z in range(N_ZENITH_BINS):
            keep = (band == b) & (zbin == z)
            if keep.any():
                out[keep] = np.interp(ens[keep], xs[b, z], ys[b, z])
    return out.astype(np.float32)


def stage0_mappings(
    *, ens_train: pl.DataFrame, analysis: pl.DataFrame
) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Fit (or load) the ENS-to-CAMS and ENS-to-ERA5 mappings on pairs before `TRAIN_END`."""
    STAGE0_DIR.mkdir(parents=True, exist_ok=True)
    if all((STAGE0_DIR / f"{t}.npz").exists() for t in TARGETS):
        loaded = {t: np.load(STAGE0_DIR / f"{t}.npz") for t in TARGETS}
        return {t: (loaded[t]["xs"], loaded[t]["ys"]) for t in TARGETS}
    pairs = (
        ens_train.filter(pl.col("valid_time") < TRAIN_END)
        .select("time_series_id", "valid_time", "lead_hours", "solar_zenith", ens="shortwave")
        .join(
            analysis.select("time_series_id", "valid_time", "shortwave", "shortwave_cams"),
            on=["time_series_id", "valid_time"],
            how="inner",
            validate="m:1",
        )
        .with_columns(cos_zenith=pl.col("solar_zenith").radians().cos())
        .filter(pl.all_horizontal(pl.col("ens", "shortwave", "shortwave_cams").is_finite()))
    )
    if pairs.select(pl.col("valid_time").max()).item() >= TRAIN_END:
        msg = "a stage 0 pair is at or after the end of training"
        raise ValueError(msg)
    mappings = {}
    count_rows = []
    for target in TARGETS:
        xs, ys, counts = fit_mapping(
            ens=pairs["ens"].to_numpy(),
            target=pairs[TARGET_COLUMN[target]].to_numpy(),
            lead_hours=pairs["lead_hours"].to_numpy(),
            cos_zenith=pairs["cos_zenith"].to_numpy(),
        )
        np.savez(STAGE0_DIR / f"{target}.npz", xs=xs, ys=ys, counts=counts)
        mappings[target] = (xs, ys)
        count_rows += [
            {"target": target, "band": b, "zenith_bin": z, "pairs": int(counts[b, z])}
            for b in range(N_BANDS)
            for z in range(N_ZENITH_BINS)
        ]
    pl.DataFrame(count_rows).write_parquet(STAGE0_DIR / "pairs.parquet")
    print(f"stage 0 fitted on {pairs.height} pairs", flush=True)
    return mappings


def _matrix(*, frame: pl.DataFrame, columns: tuple[str, ...]) -> np.ndarray:
    return frame.select([pl.col(c).cast(pl.Float32) for c in columns]).to_numpy()


def training_rows(
    *, series: int, analysis: pl.DataFrame, observed: pl.DataFrame, cams: bool
) -> pl.DataFrame:
    """Return one series' ERA5 rows with their target; CAMS irradiance replaces the shortwave."""
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
    if cams:
        rows = rows.with_columns(shortwave=pl.col("shortwave_cams")).drop_nulls("shortwave")
    return rows


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


def variant_inputs(
    *,
    variant: VariantType,
    frame: pl.DataFrame,
    mappings: dict[str, tuple[np.ndarray, np.ndarray]],
) -> pl.DataFrame:
    """Return the frame with the shortwave a variant reads: mapped for the stage 0 variants."""
    target = STAGE0_VARIANT_TARGET.get(variant)
    if target is None:
        return frame
    mapped = apply_mapping(
        mapping=mappings[target],
        ens=frame["shortwave"].to_numpy(),
        lead_hours=frame["lead_hours"].to_numpy(),
        cos_zenith=np.cos(np.radians(frame["solar_zenith"].to_numpy())),
    )
    return frame.with_columns(shortwave=pl.Series(mapped))


def fit_seed(
    *,
    seed: int,
    sites: pl.DataFrame,
    analysis: pl.DataFrame,
    observed: pl.DataFrame,
    mappings: dict[str, tuple[np.ndarray, np.ndarray]],
    rounds: int,
    limit_runs: int | None,
) -> None:
    """Fit every variant for one seed and write its mixtures."""
    targets: dict[tuple[TechnologyType, VariantType], Path] = {
        (technology, variant): MIXTURE_DIR / f"{variant}_{technology}_seed{seed}.parquet"
        for technology, variants in VARIANTS.items()
        for variant in variants
    }
    if limit_runs is None and all(p.exists() for p in targets.values()):
        return
    started = time.monotonic()
    models: dict[tuple[int, str], FittedQuantiles] = {}
    for row in sites.iter_rows(named=True):
        series, technology = int(row["time_series_id"]), row["technology"]
        columns = SOLAR_FEATURES if technology == "solar" else WIND_FEATURES
        for fit_name in ("base", "cams") if technology == "solar" else ("base", "margin"):
            rows = training_rows(
                series=series, analysis=analysis, observed=observed, cams=fit_name == "cams"
            )
            models[series, fit_name] = FittedQuantiles(
                rows=rows,
                columns=columns,
                seed=seed,
                rounds=rounds,
                use_margin=fit_name == "margin",
            )
    print(f"seed {seed}: fits in {time.monotonic() - started:.0f} s", flush=True)
    model_of: dict[VariantType, str] = {
        "base": "base",
        "cams": "cams",
        "cams_s0": "cams",
        "base_s0": "base",
        "margin": "margin",
    }
    pieces: dict[tuple[TechnologyType, VariantType], list[pl.DataFrame]] = {
        key: [] for key in targets
    }
    files = sorted(ENS_VALIDATION_DIR.glob("init=*.parquet"))[:limit_runs]
    technology_of = dict(zip(sites["time_series_id"], sites["technology"], strict=True))
    for path in files:
        frame = pl.read_parquet(path).with_columns(
            lead_hours=(pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_minutes()
            / 60.0
        )
        for (series_key,), part in frame.partition_by("time_series_id", as_dict=True).items():
            series = int(series_key)
            technology = technology_of[series]
            for variant in VARIANTS[technology]:
                inputs = variant_inputs(variant=variant, frame=part, mappings=mappings)
                member_quantiles = models[series, model_of[variant]].predict(frame=inputs)
                pieces[technology, variant].append(
                    pooled_mixture(part=part, member_quantiles=member_quantiles)
                )
    if limit_runs is not None:
        print("smoke run: no output written", flush=True)
        return
    MIXTURE_DIR.mkdir(parents=True, exist_ok=True)
    for (technology, variant), frames in pieces.items():
        target = targets[technology, variant]
        pl.concat(frames).sort(*KEY_COLUMNS).write_parquet(target.with_suffix(".tmp"))
        target.with_suffix(".tmp").rename(target)
    print(f"seed {seed} done in {time.monotonic() - started:.0f} s", flush=True)


def main() -> None:
    """Fit stage 0 and every seed, skipping seeds already written."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true", help="20 rounds, 2 runs, no output")
    smoke = parser.parse_args().smoke
    sites = site_table()
    analysis = pl.read_parquet(ANALYSIS_PATH)
    observed = observed_training_power(sites=sites)
    solar_ids = sites.filter(pl.col("technology") == "solar")["time_series_id"].to_list()
    ens_train = (
        pl.read_parquet(ENS_TRAIN_PATH)
        .filter(pl.col("time_series_id").is_in(solar_ids))
        .with_columns(
            lead_hours=(pl.col("valid_time") - pl.col("power_fcst_init_time")).dt.total_minutes()
            / 60.0
        )
    )
    mappings = stage0_mappings(ens_train=ens_train, analysis=analysis)
    for seed in SEEDS[:1] if smoke else SEEDS:
        fit_seed(
            seed=seed,
            sites=sites,
            analysis=analysis,
            observed=observed,
            mappings=mappings,
            rounds=20 if smoke else N_ROUNDS,
            limit_runs=2 if smoke else None,
        )


if __name__ == "__main__":
    main()
