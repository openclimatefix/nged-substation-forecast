"""Run one XGBoost fit per (arm, site) concurrently, and the features every arm shares.

Written for the experiment in
<https://github.com/openclimatefix/nged-substation-forecast/issues/784>. Every later study that fits
arms reuses `run_all`, `Job` and `add_time_features`, so a change to how the fits run changes every
study at once. Scripts in `studies/beam_diffuse_split/`, `studies/nwp_forecast_comparison/`,
`studies/open_meteo_ensemble_means/`, and `studies/past_weather/` import it.
"""

import concurrent.futures
import logging
from pathlib import Path
from typing import Final

import polars as pl

from studies.cross_validation import DeviceType, HyperParameters, out_of_fold_losses
from studies.fractions_skill_score import MONTH_FORMAT
from studies.sources import STUDY_INPUTS_DIR

_LOG: Final[logging.Logger] = logging.getLogger(__name__)

MAX_CONCURRENT_FITS: Final[int] = 8
"""How many (arm, site) fits to run at once, each on `THREADS_PER_FIT` cores."""


SHARED_FEATURES: Final[tuple[str, ...]] = (
    "solar_zenith_deg",
    "solar_azimuth_deg",
    "extraterrestrial_horizontal_w_m2",
    "temp_c",
    "hour_of_day",
    "day_of_year",
)
"""Features every arm gets in the beam/diffuse experiment.

Solar geometry and season are in here deliberately. A fixed-tilt array's sensitivity to the
beam/diffuse split is partly a function of sun position, which a tree can absorb from these, so
giving every arm the geometry makes the global-irradiance-only arm as strong as it can be. That
makes any advantage arm C shows a lower bound on what a transposition model would extract.
"""


def dataset_path_for(*, source: str) -> Path:
    """Return the frame `build_dataset.py` wrote for one ERA5 source."""
    return STUDY_INPUTS_DIR / f"beam_diffuse_dataset_{source}.parquet"


def add_time_features(*, dataset: pl.DataFrame) -> pl.DataFrame:
    """Add the calendar features and the month label the block bootstrap resamples on."""
    return dataset.with_columns(
        hour_of_day=pl.col("time").dt.hour(),
        day_of_year=pl.col("time").dt.ordinal_day(),
        month=pl.col("time").dt.strftime(MONTH_FORMAT),
    )


Job = tuple[str, str, str, tuple[str, ...], HyperParameters, bool]
"""One (arm, setting name, target, features, settings, whether to score quantiles) to fit."""


def run_all(
    *,
    dataset: pl.DataFrame,
    jobs: list[Job],
    max_workers: int = MAX_CONCURRENT_FITS,
    device: DeviceType = "cpu",
) -> pl.DataFrame:
    """Run every (arm, site) job concurrently and concatenate the losses.

    XGBoost releases the interpreter lock while it trains, so threads give real parallelism here
    without the cost of shipping a copy of the frame to a subprocess.

    Args:
        dataset: The full frame, already carrying `fold`, `month`, `cap_mw` and `constrained`.
        jobs: The fits to run.
        max_workers: How many (arm, site) fits run at once, each on `THREADS_PER_FIT` cores.
            Lower it to share the machine with another run.
        device: XGBoost's device for every fit, `"cpu"` or `"cuda"`.

    Returns:
        Every job's losses, stacked, labelled with the arm, the setting and the target.
    """
    sites = sorted(dataset["site"].unique().to_list())
    outputs: list[pl.DataFrame] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {}
        for arm, setting_name, target, features, hyper_parameters, with_quantiles in jobs:
            for site in sites:
                future = pool.submit(
                    out_of_fold_losses,
                    site_rows=dataset.filter(pl.col("site") == site),
                    features=features,
                    target=target,
                    hyper_parameters=hyper_parameters,
                    with_quantiles=with_quantiles,
                    device=device,
                )
                futures[future] = (setting_name, arm, target, site)
        for done, future in enumerate(concurrent.futures.as_completed(futures), start=1):
            setting_name, arm, target, site = futures[future]
            outputs.append(
                future.result().with_columns(
                    arm=pl.lit(arm), setting=pl.lit(setting_name), target=pl.lit(target)
                )
            )
            _LOG.info("%d/%d done: %s / %s / site %s", done, len(futures), setting_name, arm, site)
    return pl.concat(outputs)
