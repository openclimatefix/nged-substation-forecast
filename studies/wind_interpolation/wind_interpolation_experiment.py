"""Compare three u/v interpolators with four-feature GPU wind-power forecasts.

Run with `uv run python studies/wind_interpolation/wind_interpolation_experiment.py`.
The committed README defines the planned contrasts. Outputs are anonymised and kept under
`data/studies/per_study/wind_interpolation/`. Production processing is unchanged.
"""

import argparse
import hashlib
import inspect
import itertools
import json
import logging
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Final, Literal, cast

import h3
import numpy as np
import polars as pl
import scipy
import xgboost as xgb
from contracts.weather_schemas import ECMWF_ENS_LEAD_TIME_HOURS
from deltalake import DeltaTable
from scipy.interpolate import Akima1DInterpolator, PchipInterpolator
from studies.bootstrap import (
    bootstrap_difference,
    bootstrap_difference_at_level,
)
from studies.cross_validation import (
    PRIMARY_HYPER_PARAMETERS,
    SEEDS,
    SENSITIVITY_HYPER_PARAMETERS,
    HyperParameters,
    booster_parameters,
)
from studies.power import CLEANED_POWER_DELTA_URI, scan_power
from studies.pv_dataset import wind_sites
from studies.sources import REPO_DATA_DIR, study_dir_for

_LOG: Final[logging.Logger] = logging.getLogger(__name__)
MethodType = Literal["linear", "pchip", "makima"]
METHODS: Final[tuple[MethodType, ...]] = ("linear", "pchip", "makima")
COMPONENTS: Final[tuple[str, ...]] = ("wind_u_10m", "wind_v_10m", "wind_u_100m", "wind_v_100m")
FEATURES: Final[tuple[str, ...]] = (
    "wind_speed_10m",
    "wind_direction_10m",
    "wind_speed_100m",
    "wind_direction_100m",
)
KEYS: Final[tuple[str, ...]] = ("site", "init_time", "time", "band")
LEADS: Final[np.ndarray] = np.array(ECMWF_ENS_LEAD_TIME_HOURS, dtype=float)
TARGETS: Final[np.ndarray] = np.concatenate((np.arange(24, 48, 0.5), np.arange(168, 192, 0.5)))
TRAIN_START: Final[datetime] = datetime(2024, 4, 1, tzinfo=UTC)
TRAIN_STOP: Final[datetime] = datetime(2025, 6, 24, tzinfo=UTC)
VAL_START: Final[datetime] = datetime(2025, 7, 1, tzinfo=UTC)
VAL_STOP: Final[datetime] = datetime(2026, 7, 1, tzinfo=UTC)
OUTPUT: Final[Path] = study_dir_for(study="wind_interpolation")
SETTINGS: Final[dict[str, HyperParameters]] = {
    "primary": PRIMARY_HYPER_PARAMETERS,
    "sensitivity": SENSITIVITY_HYPER_PARAMETERS,
}


def interpolate(
    *, values: np.ndarray, x: np.ndarray, targets: np.ndarray, method: MethodType
) -> np.ndarray:
    """Evaluate rows of instantaneous components inside their native anchor range."""
    if targets.min() < x[0] or targets.max() > x[-1]:
        raise ValueError("No extrapolation is allowed")
    if method == "pchip":
        return PchipInterpolator(x, values, axis=1, extrapolate=False)(targets)
    if method == "makima":
        return Akima1DInterpolator(x, values, axis=1, method="makima", extrapolate=False)(targets)
    right = np.clip(np.searchsorted(x, targets, side="right"), 1, len(x) - 1)
    left = right - 1
    weight = (targets - x[left]) / (x[right] - x[left])
    return values[:, left] * (1 - weight) + values[:, right] * weight


def _checks() -> dict[str, float]:
    """Check affine-vector recovery and native anchors independently of fitting."""
    errors = {}
    x = np.array([0, 3, 6, 9, 15], dtype=float)
    target = np.arange(0, 15.5, 0.5)
    values = np.stack((2 * x - 4, 7 - x))
    for method in METHODS:
        actual = interpolate(values=values, x=x, targets=target, method=method)
        error = float(np.max(np.abs(actual - np.stack((2 * target - 4, 7 - target)))))
        assert error < 1e-10, (method, error)
        np.testing.assert_allclose(
            interpolate(values=values, x=x, targets=x, method=method), values
        )
        errors[method] = error
    return errors


def _build(*, output: Path) -> None:
    """Pin the archive and materialise matched, anonymised half-hourly inputs."""
    if (output / "provenance.json").exists():
        raise FileExistsError("Use the saved inputs or choose a fresh output directory")
    output.mkdir(parents=True, exist_ok=True)
    sites = wind_sites()
    assert sorted(sites["site"].to_list()) == ["W1", "W2", "W3"]
    cells = [
        h3.str_to_int(h3.latlng_to_cell(lat=lat, lng=lon, res=5))
        for lat, lon in sites.select("latitude", "longitude").iter_rows()
    ]
    lookup = sites.select("site").with_columns(h3_index=pl.Series(cells, dtype=pl.Int64))
    nwp_path = REPO_DATA_DIR / "NWP"
    nwp_version = DeltaTable(str(nwp_path)).version()
    native = (
        pl.scan_delta(str(nwp_path), version=nwp_version)
        .filter(
            (pl.col("ensemble_member") == 0)
            & pl.col("h3_index").is_in(cells)
            & (pl.col("init_time") >= TRAIN_START)
            & (pl.col("init_time") < VAL_STOP)
            & (pl.col("init_time").dt.hour() == 0)
        )
        .select("h3_index", "init_time", "valid_time", *COMPONENTS)
        .join(lookup.lazy(), on="h3_index")
        .with_columns(lead=(pl.col("valid_time") - pl.col("init_time")).dt.total_minutes() / 60)
        .select("site", "init_time", "valid_time", "lead", *COMPONENTS)
        .collect()
        .sort("site", "init_time", "lead")
    )
    assert native.select("site", "init_time", "lead").unique().height == native.height
    complete = (
        native.group_by("site", "init_time")
        .agg(pl.len().alias("steps"))
        .filter(pl.col("steps") == len(LEADS))
    )
    total_runs = native.select("site", "init_time").unique().height
    native = native.join(complete.select("site", "init_time"), on=["site", "init_time"]).sort(
        "site", "init_time", "lead"
    )
    matrices = {name: native[name].to_numpy().reshape(-1, len(LEADS)) for name in COMPONENTS}
    assert np.all(native["lead"].to_numpy().reshape(-1, len(LEADS)) == LEADS)
    assert all(np.isfinite(values).all() for values in matrices.values())
    native.write_parquet(output / "native.parquet")
    power_version = DeltaTable(CLEANED_POWER_DELTA_URI).version()
    power_scan = pl.LazyFrame._from_pyldf(scan_power()._ldf)
    power = (
        power_scan.filter((pl.col("time") >= TRAIN_START) & (pl.col("time") < VAL_STOP))
        .join(sites.lazy().select("time_series_id", "site"), on="time_series_id")
        .select("site", "time", "power")
        .collect()
        .sort("site", "time")
    )
    assert DeltaTable(CLEANED_POWER_DELTA_URI).version() == power_version, (
        "Power changed during extraction"
    )
    assert power.select("site", "time").unique().height == power.height
    scales = (
        power.filter(pl.col("time") < TRAIN_STOP).group_by("site").agg(scale=pl.col("power").max())
    )
    assert scales.height == 3
    assert scales.select((pl.col("scale") > 0).all()).item()
    power = (
        power.join(scales, on="site")
        .with_columns(actual=pl.col("power") / pl.col("scale"))
        .drop("power", "scale")
    )
    power.write_parquet(output / "power.parquet")
    run_keys = native.select("site", "init_time").unique(maintain_order=True)
    base = (
        pl.DataFrame(
            {
                "site": np.repeat(run_keys["site"].to_numpy(), len(TARGETS)),
                "init_time": np.repeat(run_keys["init_time"].to_numpy(), len(TARGETS)),
                "lead": np.tile(TARGETS, run_keys.height),
                "band": np.tile(["day1"] * 48 + ["day7"] * 48, run_keys.height),
            }
        )
        .with_columns(pl.col("init_time").dt.replace_time_zone("UTC"))
        .with_columns(time=pl.col("init_time") + pl.duration(minutes=pl.col("lead") * 60))
        .filter((pl.col("time") >= TRAIN_START) & (pl.col("time") < VAL_STOP))
    )
    assert base.select(*KEYS).unique().height == base.height
    common = (
        base.join(power, on=["site", "time"], how="inner")
        .filter(pl.col("actual").is_finite())
        .sort(*KEYS)
    )
    for method in METHODS:
        arrays = {
            name: interpolate(values=matrix, x=LEADS, targets=TARGETS, method=method)
            for name, matrix in matrices.items()
        }
        columns = {}
        for height in ("10m", "100m"):
            u, v = arrays[f"wind_u_{height}"], arrays[f"wind_v_{height}"]
            columns[f"wind_speed_{height}"] = np.hypot(u, v).flatten().astype(np.float32)
            columns[f"wind_direction_{height}"] = (
                ((np.degrees(np.arctan2(u, v)) + 180) % 360).flatten().astype(np.float32)
            )
        all_rows = pl.DataFrame(
            {
                "site": np.repeat(run_keys["site"].to_numpy(), len(TARGETS)),
                "init_time": np.repeat(run_keys["init_time"].to_numpy(), len(TARGETS)),
                "lead": np.tile(TARGETS, run_keys.height),
                **columns,
            }
        ).with_columns(pl.col("init_time").dt.replace_time_zone("UTC"))
        frame = common.join(all_rows, on=["site", "init_time", "lead"], validate="1:1").sort(*KEYS)
        assert frame.select(pl.all_horizontal(pl.col(FEATURES).is_finite()).all()).item()
        assert frame.height == common.height
        frame.write_parquet(output / f"inputs-{method}.parquet")
    _withheld_check(matrices=matrices, output=output)
    provenance = {
        "nwp_path": str(nwp_path),
        "nwp_version": nwp_version,
        "power_version": power_version,
        "device": "cuda",
        "xgboost": xgb.__version__,
        "scipy": scipy.__version__,
        "features": FEATURES,
        "settings": SETTINGS,
        "seeds": SEEDS,
        "affine_checks": _checks(),
        "excluded_incomplete_site_runs": total_runs - complete.height,
        "matched_rows": common.height,
        "missing_target_rows": base.height - common.height,
        "training_stop_exclusive": TRAIN_STOP.isoformat(),
        "validation_start": VAL_START.isoformat(),
        "validation_stop_exclusive": VAL_STOP.isoformat(),
        "normalisation": "maximum cleaned pre-embargo training power per site",
        "code_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    }
    (output / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    _LOG.info(
        "Built %d matched rows, excluding %d incomplete site-runs",
        common.height,
        total_runs - complete.height,
    )


def _withheld_check(*, matrices: dict[str, np.ndarray], output: Path) -> None:
    """Save vector reconstruction error at withheld three-hour native samples."""
    hidden = []
    coarse = LEADS <= 144
    kept = coarse & (LEADS % 6 == 0)
    withheld = coarse & (LEADS % 6 == 3)
    for method in METHODS:
        squared = np.zeros((len(next(iter(matrices.values()))), withheld.sum()))
        for values in matrices.values():
            predicted = interpolate(
                values=values[:, kept], x=LEADS[kept], targets=LEADS[withheld], method=method
            )
            squared += (predicted - values[:, withheld]) ** 2 / 2
        hidden.append(
            {
                "method": method,
                "mean_vector_error_m_s": float(np.sqrt(squared).mean()),
                "samples": int(squared.size),
            }
        )
    pl.DataFrame(hidden).write_csv(output / "withheld-native.csv")


def _fit(*, output: Path) -> None:
    """Fit fixed GPU models on chronological training rows and save validation forecasts."""
    (output / "fits").mkdir(exist_ok=True)
    identity = _fit_identity(output=output)
    identity_path = output / "fit-identity.json"
    if identity_path.exists():
        assert json.loads(identity_path.read_text()) == identity, (
            "Fit inputs/code/settings changed; use a fresh output directory"
        )
    else:
        identity_path.write_text(json.dumps(identity, indent=2) + "\n")
    for method in METHODS:
        frame = pl.read_parquet(output / f"inputs-{method}.parquet").sort(*KEYS)
        for (site, band), rows in frame.partition_by("site", "band", as_dict=True).items():
            train = rows.filter(pl.col("time") < TRAIN_STOP)
            test = rows.filter(pl.col("time") >= VAL_START)
            assert train.height > 1000
            assert test.height > 1000
            assert train.select(pl.col("time").max() < pl.lit(test["init_time"].min())).item()
            assert test.select("site", "time").unique().height == test.height
            for setting, parameters in SETTINGS.items():
                for seed in SEEDS:
                    path = output / "fits" / f"{method}-{site}-{band}-{setting}-{seed}.parquet"
                    if path.exists() and path.with_suffix(".json").exists():
                        continue
                    dtrain = xgb.DMatrix(
                        train.select(FEATURES).to_numpy(),
                        label=train["actual"].to_numpy(),
                        feature_names=list(FEATURES),
                    )
                    dtest = xgb.DMatrix(
                        test.select(FEATURES).to_numpy(), feature_names=list(FEATURES)
                    )
                    params = booster_parameters(
                        hyper_parameters=parameters, seed=seed, device="cuda"
                    )
                    params.update(objective="reg:squarederror", colsample_bytree=1.0)
                    booster = xgb.train(
                        params=params, dtrain=dtrain, num_boost_round=parameters["num_boost_round"]
                    )
                    assert json.loads(booster.save_config())["learner"]["generic_param"][
                        "device"
                    ].startswith("cuda")
                    prediction = np.maximum(booster.predict(dtest), 0)
                    result = (
                        test.select(*KEYS, "actual")
                        .with_columns(
                            prediction=pl.Series(prediction),
                            arm=pl.lit(method),
                            setting=pl.lit(setting),
                            seed=pl.lit(seed),
                            month=pl.col("time").dt.strftime("%Y-%m"),
                        )
                        .with_columns(mae=100 * (pl.col("prediction") - pl.col("actual")).abs())
                    )
                    result.write_parquet(path)
                    gain = booster.get_score(importance_type="total_gain")
                    gains = {name: cast(float, gain.get(name, 0.0)) for name in FEATURES}
                    total = sum(gains.values())
                    path.with_suffix(".json").write_text(
                        json.dumps(
                            {
                                "training_rows": train.height,
                                "validation_rows": test.height,
                                "gain_shares": {
                                    name: value / total for name, value in gains.items()
                                },
                            },
                            indent=2,
                        )
                        + "\n"
                    )
                    _LOG.info("%s: MAE %.4f", path.stem, result["mae"].mean())


def _fit_identity(*, output: Path) -> dict[str, str]:
    """Fingerprint the inputs and fitting choices before reusing saved predictions."""
    configuration = json.dumps(
        {
            "settings": SETTINGS,
            "features": FEATURES,
            "seeds": SEEDS,
            "device": "cuda",
            "train_stop": TRAIN_STOP.isoformat(),
            "val_start": VAL_START.isoformat(),
            "fit_source": inspect.getsource(_fit),
        },
        sort_keys=True,
    )
    return {
        "configuration_sha256": hashlib.sha256(configuration.encode()).hexdigest(),
        **{
            method: hashlib.sha256((output / f"inputs-{method}.parquet").read_bytes()).hexdigest()
            for method in METHODS
        },
    }


def _report(*, output: Path) -> None:
    """Print every published score and paired interval from saved forecasts."""
    assert json.loads((output / "fit-identity.json").read_text()) == _fit_identity(output=output)
    expected_fits = {
        f"{method}-{site}-{band}-{setting}-{seed}"
        for method, site, band, setting, seed in itertools.product(
            METHODS, ("W1", "W2", "W3"), ("day1", "day7"), SETTINGS, SEEDS
        )
    }
    assert {path.stem for path in (output / "fits").glob("*.parquet")} == expected_fits
    assert all((output / "fits" / f"{name}.json").exists() for name in expected_fits)
    losses = pl.read_parquet(output / "fits" / "*.parquet")
    assert losses.select(*KEYS, "arm", "setting", "seed").unique().height == losses.height
    summaries = (
        losses.group_by("setting", "band", "arm")
        .agg(mae=pl.col("mae").mean(), rows=pl.len())
        .sort("setting", "band", "arm")
    )
    summaries.write_csv(output / "absolute.csv")
    losses.group_by("setting", "band", "site", "arm").agg(mae=pl.col("mae").mean()).sort(
        "setting", "band", "site", "arm"
    ).write_csv(output / "per-site.csv")
    contrasts = []
    for (setting, band), rows in losses.partition_by("setting", "band", as_dict=True).items():
        expected = rows.filter(pl.col("arm") == "linear").select(*KEYS, "seed").sort(*KEYS, "seed")
        for method in ("pchip", "makima"):
            actual = rows.filter(pl.col("arm") == method).select(*KEYS, "seed").sort(*KEYS, "seed")
            assert actual.equals(expected)
            interval = bootstrap_difference(
                losses=rows, treatment=method, reference="linear", metric="mae"
            )
            low, high = bootstrap_difference_at_level(
                losses=rows, treatment=method, reference="linear", metric="mae", level=98.75
            )
            contrasts.append(
                {
                    "setting": setting,
                    "band": band,
                    "method": method,
                    **interval,
                    "lower_98_75": low,
                    "upper_98_75": high,
                }
            )
    contrasts = pl.DataFrame(contrasts).sort("setting", "band", "method")
    contrasts.write_csv(output / "contrasts.csv")
    lines = [
        "# Wind interpolation results",
        "",
        "MAE unit: percentage points of each farm's maximum cleaned training power.",
        "",
        "## Absolute errors",
        "",
        summaries.write_csv(),
        "## Planned paired contrasts",
        "",
        "Negative differences favour the smooth method. Intervals resample months and one seed.",
        "",
        contrasts.write_csv(),
        "## Withheld native-vector diagnostic",
        "",
        (output / "withheld-native.csv").read_text(),
        "## Provenance",
        "",
        (output / "provenance.json").read_text(),
    ]
    (output / "report.md").write_text("\n".join(lines))
    _LOG.info("Report: %s", output / "report.md")


def main() -> None:
    """Build, fit, and report the planned comparison, or resume one stage."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=("all", "build", "fit", "report"), default="all")
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if args.stage in ("all", "build"):
        _build(output=args.output)
    if args.stage in ("all", "fit"):
        _fit(output=args.output)
    if args.stage in ("all", "report"):
        _report(output=args.output)


if __name__ == "__main__":
    main()
