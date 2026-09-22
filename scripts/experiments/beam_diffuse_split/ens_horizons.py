"""Score ECMWF ENS against ERA5 at a run of forecast horizons, and four ways of using its members.

One-off throwaway script for the experiment in
<https://github.com/openclimatefix/nged-substation-forecast/issues/784>.

**Every other source in this experiment estimates an hour that has already happened; ENS forecasts
one that has not.** That makes an ENS figure incomparable with a CAMS figure and comparable with
another ENS figure, so this script reports ENS against ERA5 on shared rows at each horizon, and
ENS against itself across horizons, rather than adding a row to the four-source table.

**Scoring happens on ENS's own 3-hourly stamps, with the hourly sources aggregated up to them.**
ENS radiation is period-ending over the preceding forecast step, and every other source here is
period-ending over one hour. Interpolating ENS down to an hour it never resolved would charge it
for an interpolation error that is not its own, so the hourly sources are averaged over the three
hours ending at each ENS stamp instead, and a window missing any of its three hours is dropped.

**Four ways of turning 51 members into one power forecast, because the obvious one is biased.**
The power curve bends at clipping and at the export cap, so the expected power is not the power of
the expected irradiance: `mean(f(x))` and `f(mean(x))` differ, and only the first estimates what
the meter will read. `median(f(x))` is reported beside them because mean absolute error is
minimised by the median rather than the mean, and `control` is reported because the production
analysis-proxy selection keeps the control member and drops the rest.

Run it with `uv run --no-project --with polars --with numpy --with xgboost --with scipy --with
pvlib --with xarray --with netcdf4 --with pandas --with deltalake python
scripts/experiments/beam_diffuse_split/ens_horizons.py`.
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Final

import numpy as np
import polars as pl
import xgboost as xgb
from run_experiment import (
    N_FOLDS,
    PRIMARY_HYPER_PARAMETERS,
    SHARED_FEATURES,
    _add_time_features,
    _assign_folds,
    _booster_parameters,
    dataset_path_for,
)
from sources import REPO_DATA_DIR

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ENS_PATH: Final[Path] = REPO_DATA_DIR / "ENS" / "beam_diffuse_ens.parquet"
"""Where `fetch_ens_point.py` wrote the per-meter, per-member, per-horizon frame."""

OUTPUT_DIR: Final[Path] = REPO_DATA_DIR / "ERA5" / "beam_diffuse_ens_horizons"
"""Where this script writes its per-row losses and its report."""

WINDOW_HOURS: Final[int] = 3
"""How many hourly rows make up one ENS stamp's period.

ENS steps every 3 hours out to lead 144 and every 6 beyond it. The long horizons are therefore
scored on a 3-hour window inside a 6-hour step rather than the whole step, which keeps one window
definition across every horizon at the cost of covering half of each long-lead step.
"""

BOOST_ROUNDS: Final[int] = 400
"""How many trees each fit grows, matching the primary setting of the main experiment."""

VARIANTS: Final[tuple[str, ...]] = (
    "control",
    "mean_of_irradiance",
    "mean_of_power",
    "median_of_power",
)
"""The four ways of reducing 51 members to one forecast, in the order the table prints them.

`control` takes the control member alone, which is what the production analysis-proxy selection
does. `mean_of_irradiance` averages the members' irradiance and predicts once, which is the cheap
route and the biased one. `mean_of_power` and `median_of_power` predict once per member and reduce
the 51 power values, which is the unbiased route and the one that also yields a distribution.
"""


def _three_hourly(*, hourly: pl.DataFrame) -> pl.DataFrame:
    """Average the cleaned hourly frame onto the stamps ENS resolves.

    A window is kept only where all three of its hours survived the hourly cleaning, so a window is
    never a partial average masquerading as a whole one.

    Args:
        hourly: The cleaned hourly dataset, carrying `site`, `time`, and the columns to average.

    Returns:
        One row per site and 3-hourly stamp.
    """
    # Round each period-ending hour up to the period-ending 3-hourly stamp whose window contains
    # it. Working on the whole timestamp rather than the hour of day is what makes the midnight
    # boundary come out right: 22:00 and 23:00 belong to the next day's 00:00 stamp, and 00:00
    # belongs to its own.
    floored = pl.col("time").dt.truncate(f"{WINDOW_HOURS}h")
    stamp_time = (
        pl.when(floored == pl.col("time"))
        .then(floored)
        .otherwise(floored + pl.duration(hours=WINDOW_HOURS))
    )
    return (
        hourly.with_columns(valid_time=stamp_time)
        .group_by("site", "valid_time")
        .agg(
            power_mw=pl.col("power_mw").mean(),
            era5_ghi_w_m2=pl.col("ghi_w_m2").mean(),
            temp_c=pl.col("temp_c").mean(),
            solar_zenith_deg=pl.col("solar_zenith_deg").mean(),
            solar_azimuth_deg=pl.col("solar_azimuth_deg").mean(),
            extraterrestrial_horizontal_w_m2=pl.col("extraterrestrial_horizontal_w_m2").mean(),
            hours=pl.len(),
        )
        .filter(pl.col("hours") == WINDOW_HOURS)
        .drop("hours")
        .sort("site", "valid_time")
    )


def _fit_and_predict(
    *, train: pl.DataFrame, test_inputs: list[pl.DataFrame], features: list[str], seed: int
) -> list[np.ndarray]:
    """Fit one booster on the training rows and predict each supplied test frame.

    Fitting once and predicting many times is what makes the per-member route cheap: the 51
    members differ only in the irradiance column, so they are 51 predictions from one model rather
    than 51 models.

    Args:
        train: The training rows.
        test_inputs: One frame per prediction wanted, each carrying `features`.
        features: The feature columns.
        seed: The booster seed.

    Returns:
        One prediction array per entry in `test_inputs`.
    """
    booster = xgb.train(
        _booster_parameters(hyper_parameters=PRIMARY_HYPER_PARAMETERS, seed=seed),
        xgb.DMatrix(train.select(features).to_numpy(), label=train["power_mw"].to_numpy()),
        num_boost_round=BOOST_ROUNDS,
    )
    return [
        booster.predict(xgb.DMatrix(frame.select(features).to_numpy())) for frame in test_inputs
    ]


def _paired_month_bootstrap(
    *, losses: pl.DataFrame, treatment: str, reference: str, draws: int = 1000
) -> dict[str, float]:
    """Interval the difference between two variants by resampling whole months.

    Written here rather than reused from `run_experiment` because that module's bootstrap expects
    the main experiment's frame — arms, seeds and a per-row pairing this script does not produce.

    Both variants are scored on the same stamps, so the difference is taken per month before
    resampling: the weather both saw cancels, leaving only where they disagree. Months are the
    resampling unit because errors within a day are not independent.

    Args:
        losses: Rows carrying `month`, `variant`, and `absolute_error_mw`.
        treatment: The variant whose score is being tested.
        reference: The variant it is tested against.
        draws: How many resamples to take.

    Returns:
        The mean difference and its 95% interval, in MW.
    """
    per_month = (
        losses.filter(pl.col("variant").is_in([treatment, reference]))
        .group_by("month", "variant")
        .agg(mean_error=pl.col("absolute_error_mw").mean(), rows=pl.len())
        .pivot(on="variant", index="month", values=["mean_error", "rows"])
    )
    treatment_column = f"mean_error_{treatment}"
    reference_column = f"mean_error_{reference}"
    paired = per_month.drop_nulls([treatment_column, reference_column])
    difference = (paired[treatment_column] - paired[reference_column]).to_numpy()
    weight = paired[f"rows_{treatment}"].to_numpy().astype(float)

    generator = np.random.default_rng(0)
    picks = generator.integers(0, len(difference), size=(draws, len(difference)))
    resampled = np.array([float(np.average(difference[row], weights=weight[row])) for row in picks])
    lower, upper = np.percentile(resampled, (2.5, 97.5))
    return {
        "difference": float(np.average(difference, weights=weight)),
        "lower_95": float(lower),
        "upper_95": float(upper),
        "months": float(len(difference)),
    }


def _scored(*, rows: pl.DataFrame, horizon: str, seed: int) -> pl.DataFrame:
    """Score every variant and the ERA5 baseline on one horizon's rows.

    Args:
        rows: One horizon's joined frame, carrying the members' irradiance columns.
        horizon: The horizon label to stamp on the output.
        seed: The booster seed.

    Returns:
        One row per site, stamp, and variant, carrying the absolute error.
    """
    members = sorted(int(name.split("_")[-1]) for name in rows.columns if name.startswith("ens_m_"))
    control = "ens_m_0"
    mean_column = "ens_mean"
    outputs: list[pl.DataFrame] = []

    for site in sorted(rows["site"].unique().to_list()):
        at_site = rows.filter(pl.col("site") == site)
        for fold in range(N_FOLDS):
            train = at_site.filter(pl.col("fold") != fold)
            test = at_site.filter(pl.col("fold") == fold)
            if train.is_empty() or test.is_empty():
                continue

            named = {
                "control": control,
                "mean_of_irradiance": mean_column,
                "era5": "era5_ghi_w_m2",
            }
            for variant, column in named.items():
                features = [*SHARED_FEATURES, "ghi_w_m2"]
                fitted = _fit_and_predict(
                    train=train.with_columns(ghi_w_m2=pl.col(column)),
                    test_inputs=[test.with_columns(ghi_w_m2=pl.col(column))],
                    features=features,
                    seed=seed,
                )[0]
                outputs.append(
                    test.select("site", "month", "fold", time=pl.col("valid_time")).with_columns(
                        horizon=pl.lit(horizon),
                        variant=pl.lit(variant),
                        seed=pl.lit(seed, dtype=pl.Int32),
                        absolute_error_mw=pl.Series(np.abs(test["power_mw"].to_numpy() - fitted)),
                    )
                )

            # One model, 51 member inputs: the unbiased route, reduced two ways.
            per_member = _fit_and_predict(
                train=train.with_columns(ghi_w_m2=pl.col(control)),
                test_inputs=[
                    test.with_columns(ghi_w_m2=pl.col(f"ens_m_{member}")) for member in members
                ],
                features=[*SHARED_FEATURES, "ghi_w_m2"],
                seed=seed,
            )
            stacked = np.vstack(per_member)
            for variant, reduced in (
                ("mean_of_power", stacked.mean(axis=0)),
                ("median_of_power", np.median(stacked, axis=0)),
            ):
                outputs.append(
                    test.select("site", "month", "fold", time=pl.col("valid_time")).with_columns(
                        horizon=pl.lit(horizon),
                        variant=pl.lit(variant),
                        seed=pl.lit(seed, dtype=pl.Int32),
                        absolute_error_mw=pl.Series(np.abs(test["power_mw"].to_numpy() - reduced)),
                    )
                )
    return pl.concat(outputs)


def _joined(*, three_hourly: pl.DataFrame, ens: pl.DataFrame, horizon: str) -> pl.DataFrame:
    """Join one horizon's ENS members onto the 3-hourly power frame.

    Args:
        three_hourly: The aggregated power and ERA5 frame.
        ens: The full ENS frame.
        horizon: Which horizon to select.

    Returns:
        One row per site and stamp, with one column per member and their mean.
    """
    wide = (
        ens.filter(pl.col("horizon") == horizon)
        .select("site", "valid_time", "ensemble_member", "ghi_w_m2")
        .pivot(on="ensemble_member", index=["site", "valid_time"], values="ghi_w_m2")
    )
    members = [name for name in wide.columns if name not in ("site", "valid_time")]
    wide = wide.rename({name: f"ens_m_{name}" for name in members}).with_columns(
        ens_mean=pl.mean_horizontal([f"ens_m_{name}" for name in members])
    )
    joined = three_hourly.join(wide, on=["site", "valid_time"], how="inner").drop_nulls()
    return _assign_folds(
        dataset=_add_time_features(dataset=joined.with_columns(time=pl.col("valid_time")))
    )


def main() -> int:
    """Score every horizon and variant, and write the report.

    Returns:
        The process exit status.

    Raises:
        FileNotFoundError: If the ENS download or the ERA5 dataset is missing.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--alignment", default="piecewise")
    parser.add_argument("--seed", type=int, default=0)
    arguments = parser.parse_args()

    era5_path = dataset_path_for(source="open-meteo", alignment=arguments.alignment)
    for path in (ENS_PATH, era5_path):
        if not path.exists():
            msg = f"{path} missing; run its fetcher and build_dataset first"
            raise FileNotFoundError(msg)

    three_hourly = _three_hourly(hourly=pl.read_parquet(era5_path))
    ens = pl.read_parquet(ENS_PATH)
    logger.info("3-hourly power rows: %s", f"{three_hourly.height:,}")

    losses: list[pl.DataFrame] = []
    for horizon in ens["horizon"].unique(maintain_order=True).to_list():
        rows = _joined(three_hourly=three_hourly, ens=ens, horizon=horizon)
        logger.info("%s: %s scored stamps", horizon, f"{rows.height:,}")
        losses.append(_scored(rows=rows, horizon=horizon, seed=arguments.seed))
    every = pl.concat(losses)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    every.write_parquet(OUTPUT_DIR / "per_row_losses.parquet")

    lines = ["### ENS against ERA5, by horizon and by how the members are used", ""]
    lines.append("| Horizon | " + " | ".join(VARIANTS) + " | ERA5 | best ENS − ERA5 |")
    lines.append("|---" * (len(VARIANTS) + 3) + "|")
    for horizon in ens["horizon"].unique(maintain_order=True).to_list():
        at = every.filter(pl.col("horizon") == horizon)
        cells, scores = [], {}
        for variant in VARIANTS:
            value = at.filter(pl.col("variant") == variant)["absolute_error_mw"].mean()
            scores[variant] = value
            cells.append(f"{value:.4f}")
        era5 = at.filter(pl.col("variant") == "era5")["absolute_error_mw"].mean()
        best = min(scores, key=lambda name: scores[name])
        lines.append(
            f"| {horizon} | " + " | ".join(cells) + f" | {era5:.4f} | "
            f"{scores[best] - era5:+.4f} ({best}) |"
        )
    lines += ["", "MAE in MW. Lower is better. ERA5 is scored on the same rows.", ""]

    lines += [
        "| Horizon | contrast | ΔMAE (MW) | 95% interval | excludes zero? |",
        "|---|---|---|---|---|",
    ]
    contrasts = (
        ("mean_of_irradiance", "era5"),
        ("mean_of_power", "era5"),
        ("mean_of_power", "mean_of_irradiance"),
        ("median_of_power", "mean_of_power"),
        ("mean_of_power", "control"),
    )
    for horizon in ens["horizon"].unique(maintain_order=True).to_list():
        at = every.filter(pl.col("horizon") == horizon)
        for treatment, reference in contrasts:
            interval = _paired_month_bootstrap(losses=at, treatment=treatment, reference=reference)
            excludes = interval["lower_95"] > 0.0 or interval["upper_95"] < 0.0
            lines.append(
                f"| {horizon} | {treatment} − {reference} | {interval['difference']:+.5f} | "
                f"[{interval['lower_95']:+.5f}, {interval['upper_95']:+.5f}] | "
                f"{'**yes**' if excludes else 'no'} |"
            )
    report = "\n".join(lines) + "\n"
    (OUTPUT_DIR / "report.md").write_text(report)
    sys.stdout.write(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
