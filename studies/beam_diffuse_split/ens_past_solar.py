"""Score ECMWF ENS's shortest-available-lead forecast in the past-solar weather-products study.

One-off throwaway script for the addition to
<https://github.com/openclimatefix/nged-substation-forecast/issues/810>, extending
`weather_products.py`'s comparison with ECMWF ENS, the live service's own forecast product.

**This section scores a genuine forecast, not a near-zero-lead description of the past.** Every
other product on the page reads a short lead: UKV's archive holds the analysis, ICON-D2 and ICON-EU
1 to 3 hours, and ERA5's radiation 1 to 12 hours. ENS's shortest available lead in the download used
here, the `T+3` band, is leads 3 to 21 hours from each day's 00 UTC run, so the run behind any hour
this section scores is always the *previous* day's, not the current day's.

**Data.** `data/studies/weather/ENS/beam_diffuse_ens.parquet`
(`data/studies/weather/ENS/README.md`), filtered to `horizon == "T+3"`: seven 3-hour radiation and
temperature steps per generator per run, leads 3, 6, ..., 21, all 51 members. ENS's coverage starts
2024-04-01, well inside the main row set's December 2022 start, so this section's row set is shorter
and later than the rest of the page, the same shape of caveat the "four extra Open-Meteo models" and
ICON-DREAM-EU sections carry. ENS publishes no direct-beam field, so this section, like the page's
other global-only products, carries a global-irradiance arm only.

**Upsampling to hourly, reusing the study's own tested machinery.** The `T+3` band already gives one
3-hourly-to-hourly step per calendar hour, not a sparse sample, so it is rebuilt to a genuine hourly
series by the clear-sky-index reconstruction `ens_forecast_horizons.py` picked as the best technique
for ENS's radiation (`COMBINATIONS["solar"]["clear_sky"]` there): `ens_forecast_horizons.Steps`,
`ens_forecast_horizons._clear_sky_arrays`, `studies.resample.clear_sky_index_resample`, and
`studies.resample.interpolate_linear` are reused unchanged; only the code that arranges this file's
own seven fixed leads into a `Steps` object is new, because `ens_forecast_horizons.py`'s own
`band_steps` assumes the wider, day-numbered grid its own extract holds, which this file does not.
Every hour from 3 to 21 is scored except the last, whose target midpoint (20.5) falls just past the
last step midpoint (19.5) and is therefore held flat at that step's clear-sky index, exactly as
`interpolate_linear` and `clear_sky_index_resample` hold every input flat beyond its own steps.

**Arms, refit on this section's own shorter row set.** `ens_mean_t3` is an XGBoost model shown the
mean of the 51 members' hourly global irradiance and temperature. `era5_global` and `cams_global`
(the page's ERA5 and CAMS arms) are refit here because the row set is shorter than the published
panel's — the `study` skill's shared-rows rule. Every arm carries the same eight feature columns
(the shared geometry and calendar features, `era_code`, and either the arm's own global irradiance
plus temperature, or the ensemble's), and every fit uses `colsample_bytree=1` (XGBoost's default,
never overridden here), so no arm wins on column count alone.

**The two planned contrasts, named before any result existed:**

- `ens_mean_t3 − era5_global`: ENS's own mean-of-members forecast against ERA5, the other
  reanalysis-adjacent product a reader might reach for first.
- `ens_mean_t3 − cams_global`: ENS against CAMS, the best product on the main leaderboard, to show
  how far a genuine forecast trails the best available description of the same hours.

**Exploratory, labelled so in the report:** the two planned contrasts per generator, and
`ens_mean_t3` against `ens_control_t3` (the control member alone, through its own XGBoost model),
which shows what averaging the 50 perturbed members over the control member is worth. This section
carries no member-by-member arm and no year-by-year panel: the 51-way member fit day-1 the horizon
study runs costs 51 times a normal fit, out of proportion to this section's two contrasts, and ENS's
own coverage here is under 2.5 years, too short for the "too few months" year-by-year rule to add
much.

**This is solar only.** ENS has no wind data downloaded yet; a wind addition is a separate, future
study once that data lands.

Run it with `uv run python studies/beam_diffuse_split/ens_past_solar.py`, after
`weather_products.py` has built its datasets (`build_dataset.py` for `open-meteo` and `cams
--min-cams-reliability 0 --suffix _allhours`). `--report-only` rebuilds `report.md` from the saved
`losses.parquet` alone, still checking the saved fingerprint against what this code would now fit.
`refuse_to_overwrite` means a re-run first moves `losses.parquet`, `losses.fingerprint` and
`report.md` to a `superseded/` subfolder.
"""

import argparse
import hashlib
import logging
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Final, cast

import numpy as np
import polars as pl
from blend_products import SOLAR, _solar_frame
from build_dataset import _pv_sites
from ens_forecast_horizons import (
    Steps,
    _clear_sky_arrays,
    _long,
    _prefixed,
    ens_columns,
    reduce_members,
    shared_features,
)
from fetch_ens_forecast_horizons import ENSEMBLE_SIZE
from run_experiment import MAX_CONCURRENT_FITS, Job, run_all
from sources import STUDY_DATA_DIR, WEATHER_DATA_DIR
from studies.baselines import hourly_clear_sky
from studies.bootstrap import bootstrap_absolute
from studies.cross_validation import PRIMARY_HYPER_PARAMETERS, SEEDS
from studies.guards import refuse_to_overwrite
from studies.resample import (
    DEFAULT_DAYLIGHT_FLOOR_W_M2,
    clear_sky_index_resample,
    interpolate_linear,
)
from weather_products import (
    CONTRAST_HEADER,
    METRIC,
    PERCENTAGE_POINTS,
    _contrast_line,
    _mae,
    geometry_lines,
    with_eras,
)

_LOG: Final[logging.Logger] = logging.getLogger("ens_past_solar")

T3_PATH: Final[Path] = WEATHER_DATA_DIR / "ENS" / "beam_diffuse_ens.parquet"
"""The download this section reads, filtered to `horizon == "T+3"`."""

OUTPUT_DIR: Final[Path] = STUDY_DATA_DIR / "past_weather_v2" / "ens_past_solar"
"""Where this section writes `losses.parquet`, `losses.fingerprint`, `report.md`, and a
`superseded/` folder for re-runs."""

ENS_START: Final[datetime] = datetime(2024, 4, 1, tzinfo=UTC)
"""ENS's own coverage start (`data/studies/weather/ENS/README.md`), which this section's row set
never reaches before."""

STEP_LEADS: Final[tuple[int, ...]] = (3, 6, 9, 12, 15, 18, 21)
"""The `T+3` band's seven leads, in hours, each a 3-hour mean ending at the lead."""

STEP_WIDTH_HOURS: Final[int] = 3
"""Every `T+3` step's width."""

TARGET_LEADS: Final[np.ndarray] = np.arange(STEP_LEADS[0], STEP_LEADS[-1] + 1, dtype=np.float64)
"""Every hour this section scores, labelled by its end: leads 3 to 21, 19 hourly values."""

MEAN_ARM: Final[str] = "ens_mean_t3"
"""The planned arm: an XGBoost model shown the mean of the 51 members' hourly fields."""

CONTROL_ARM: Final[str] = "ens_control_t3"
"""The exploratory arm: an XGBoost model shown the control member's own hourly fields."""

DECIDING_CONTRASTS: Final[tuple[tuple[str, str], ...]] = (
    (MEAN_ARM, "era5_global"),
    (MEAN_ARM, "cams_global"),
)
"""The two contrasts named before any result existed: ENS against ERA5, and ENS against CAMS."""

EXPLORATORY_CONTRASTS: Final[tuple[tuple[str, str], ...]] = ((MEAN_ARM, CONTROL_ARM),)
"""What averaging the 50 perturbed members over the control member alone is worth."""


def _t3_members() -> pl.DataFrame:
    """Read the download and keep the `T+3` band only.

    Returns:
        One row per generator, run, valid time and member, at leads 3 to 21.
    """
    return pl.read_parquet(T3_PATH).filter(pl.col("horizon") == "T+3")


def _t3_steps(*, members: pl.DataFrame) -> Steps:
    """Arrange the `T+3` band's members as arrays over its seven fixed steps.

    Mirrors `ens_forecast_horizons.band_steps`'s pivot-and-complete-ensemble logic, but over this
    file's own fixed leads (`STEP_LEADS`) rather than a day-numbered window, because
    `beam_diffuse_ens.parquet` holds the `T+3` band alone, not the continuous grid `band_steps`
    assumes. A (site, run) is dropped whole where any member lacks any of the seven leads, so every
    run kept holds all 51 members, in member order.

    Args:
        members: `_t3_members`'s output.

    Returns:
        The arrays.

    Raises:
        ValueError: If a kept run does not hold every member.
    """
    columns = ("ghi_w_m2", "temp_c")
    wide = members.pivot(
        on="lead_hours",
        index=["site", "init_time", "ensemble_member"],
        values=list(columns),
        sort_columns=True,
    )
    names = {column: [f"{column}_{lead}" for lead in STEP_LEADS] for column in columns}
    complete = wide.drop_nulls([name for group in names.values() for name in group])
    runs = (
        complete.group_by("site", "init_time")
        .len()
        .filter(pl.col("len") == ENSEMBLE_SIZE)
        .select("site", "init_time")
    )
    kept = complete.join(runs, on=["site", "init_time"]).sort(
        "site", "init_time", "ensemble_member"
    )
    expected_members = np.tile(np.arange(ENSEMBLE_SIZE), kept.height // ENSEMBLE_SIZE)
    if not np.array_equal(kept["ensemble_member"].to_numpy(), expected_members):
        msg = f"a kept run does not hold members 0 to {ENSEMBLE_SIZE - 1} in order"
        raise ValueError(msg)
    values = {column: kept.select(names[column]).to_numpy() for column in columns}
    return Steps(
        keys=kept.select("site", "init_time", "ensemble_member"),
        leads=np.array(STEP_LEADS, dtype=np.float64),
        widths=np.full(len(STEP_LEADS), STEP_WIDTH_HOURS, dtype=np.float64),
        values=values,
    )


def _t3_upsampled(*, steps: Steps, clear_sky: pl.DataFrame) -> dict[str, np.ndarray]:
    """Upsample every member's `T+3` steps to hourly: `clear_sky` for radiation, `linear` for temp.

    The technique `ens_forecast_horizons.py`'s pre-registered choosing rule picked as best for
    solar's radiation (`COMBINATIONS["solar"]["clear_sky"]`).

    Args:
        steps: The `T+3` band's steps.
        clear_sky: `hourly_clear_sky`'s table, wide enough for every run's leads 1 to 21.

    Returns:
        `ghi` and `temp`, each shape (n_series, 19), matching `TARGET_LEADS`.
    """
    x = steps.leads
    step_clear_sky, target_clear_sky = _clear_sky_arrays(
        steps=steps, targets=TARGET_LEADS, clear_sky=clear_sky
    )
    midpoints = x - steps.widths / 2.0
    morning = np.mod(midpoints, 24.0) < 12.0
    ghi = clear_sky_index_resample(
        values=steps.values["ghi_w_m2"],
        step_clear_sky=step_clear_sky,
        step_midpoints=midpoints,
        morning=morning,
        target_clear_sky=target_clear_sky,
        target_midpoints=TARGET_LEADS - 0.5,
        daylight_floor_w_m2=DEFAULT_DAYLIGHT_FLOOR_W_M2,
    )
    temp = interpolate_linear(values=steps.values["temp_c"], x=x, targets=TARGET_LEADS - 0.5)
    return {"ghi": ghi, "temp": temp}


def build_rows() -> pl.DataFrame:
    """Build this section's row set.

    `weather_products.py`'s common rows, ENS's own window, and every hour the `T+3` band supplies.

    Returns:
        One row per (site, time), with every arm's feature columns, `power_mw`, `cap_mw`,
        `constrained`, `era`, `era_code` and `fold` recomputed on this restricted row set.
    """
    sites = _pv_sites()
    base = _solar_frame().filter(pl.col("time") >= ENS_START)
    used_sites = sorted(base["site"].unique().to_list())
    members = _t3_members().filter(pl.col("site").is_in(used_sites))
    steps = _t3_steps(members=members)
    clear_sky = hourly_clear_sky(
        sites=sites.filter(pl.col("site").is_in(used_sites)),
        first=cast("datetime", steps.keys["init_time"].min()) + timedelta(hours=1),
        last=cast("datetime", steps.keys["init_time"].max()) + timedelta(hours=STEP_LEADS[-1]),
    )
    upsampled = _t3_upsampled(steps=steps, clear_sky=clear_sky)
    hourly = _long(steps=steps, targets=TARGET_LEADS, values=upsampled)
    mean_frame = _prefixed(
        frame=reduce_members(hourly=hourly, domain="solar", way="mean"),
        arm=MEAN_ARM,
        domain="solar",
    )
    control_frame = _prefixed(
        frame=reduce_members(hourly=hourly, domain="solar", way="control"),
        arm=CONTROL_ARM,
        domain="solar",
    )
    joined = (
        base.join(mean_frame, on=["site", "time"], how="inner")
        .join(control_frame, on=["site", "time"], how="inner")
        .drop("era", "era_code", "fold")
        .sort("site", "time")
    )
    _LOG.info("%d rows in this section's own row set", joined.height)
    return with_eras(frame=joined)


def jobs() -> list[Job]:
    """Return every arm's job: the two ENS arms, and ERA5 and CAMS refit on this row set.

    Returns:
        One job per arm, every arm shown eight feature columns.
    """
    ens_features = (*shared_features(domain=SOLAR), *ens_columns(arm=MEAN_ARM, domain="solar"))
    control_features = (
        *shared_features(domain=SOLAR),
        *ens_columns(arm=CONTROL_ARM, domain="solar"),
    )
    era5_features = (*SOLAR.shared_features, *SOLAR.columns("era5"))
    cams_features = (*SOLAR.shared_features, *SOLAR.columns("cams"))
    return [
        (MEAN_ARM, "pooled", "power_mw", ens_features, PRIMARY_HYPER_PARAMETERS, False),
        (CONTROL_ARM, "pooled", "power_mw", control_features, PRIMARY_HYPER_PARAMETERS, False),
        ("era5_global", "pooled", "power_mw", era5_features, PRIMARY_HYPER_PARAMETERS, False),
        ("cams_global", "pooled", "power_mw", cams_features, PRIMARY_HYPER_PARAMETERS, False),
    ]


def _fingerprint(*, frame: pl.DataFrame, job_list: list[Job]) -> str:
    """Return a hash covering every row's values, every job's columns, and the seeds.

    `--report-only` refuses to reuse a saved `losses.parquet` when this does not match, so a code
    change to the row set, a feature, a column, a seed, or a hyperparameter setting cannot silently
    mix its fits with a previous run's.

    Args:
        frame: The row set every job is fitted on.
        job_list: Every job this run means to fit.

    Returns:
        A hex digest.
    """
    ordered = frame.select(sorted(frame.columns)).sort("site", "time")
    row_hashes = ordered.hash_rows(seed=0).to_list()
    payload = repr(
        (
            row_hashes,
            [
                (arm, setting, target, tuple(features), tuple(sorted(hyper_parameters.items())))
                for arm, setting, target, features, hyper_parameters, _ in job_list
            ],
            SEEDS,
        )
    )
    return hashlib.sha256(payload.encode()).hexdigest()


def _arm_columns_lines(*, job_list: list[Job]) -> list[str]:
    """Render every fitted arm's feature columns, once per arm, as markdown.

    Args:
        job_list: Every job `jobs()` returns.

    Returns:
        Markdown lines.
    """
    seen: dict[str, tuple[str, ...]] = {}
    for arm, _, _, columns, _, _ in job_list:
        seen.setdefault(arm, columns)
    lines = ["#### Every arm's feature columns", ""]
    lines += [
        f"- `{arm}` ({len(columns)} columns): {', '.join(f'`{c}`' for c in columns)}"
        for arm, columns in seen.items()
    ]
    return lines


def _report(
    *, frame: pl.DataFrame, losses: pl.DataFrame, sites: pl.DataFrame, job_list: list[Job]
) -> str:
    """Assemble the markdown report.

    Args:
        frame: This section's own row set.
        losses: Every arm's losses.
        sites: The solar roster, for the geometry lines.
        job_list: Every job `jobs()` returns, for the feature-column section.

    Returns:
        The report.
    """
    site_labels = sorted(frame["site"].unique().to_list())
    arms = (MEAN_ARM, CONTROL_ARM, "era5_global", "cams_global")
    lines = [
        (
            f"### ECMWF ENS's `T+3` band on {frame.height:,} common site-hours of solar "
            f"({frame['time'].min():%Y-%m-%d} to {frame['time'].max():%Y-%m-%d})"
        ),
        "",
        "| Arm | All sites | 95% interval |",
        "|---|---|---|",
    ]
    for arm in arms:
        interval = bootstrap_absolute(losses=losses, arm=arm, metric=METRIC)
        lower, upper = (interval[key] * PERCENTAGE_POINTS for key in ("lower_95", "upper_95"))
        lines.append(f"| {arm} | {_mae(losses=losses, arm=arm):.3f} | [{lower:.3f}, {upper:.3f}] |")
    lines += [
        "",
        (
            "Mean absolute error as a percentage of each site's P99 output. The interval is a 95% "
            "bound from resampling whole months and a fitting seed."
        ),
        "",
        *_arm_columns_lines(job_list=job_list),
        "",
        "#### Deciding contrasts, named before the run",
        "",
        *CONTRAST_HEADER,
    ]
    for treatment, reference in DECIDING_CONTRASTS:
        lines.append(
            _contrast_line(losses=losses, treatment=treatment, reference=reference, label="all")
        )
    lines += [
        "",
        "#### The same two contrasts, per generator (exploratory)",
        "",
        *CONTRAST_HEADER,
    ]
    for treatment, reference in DECIDING_CONTRASTS:
        lines += [
            _contrast_line(
                losses=losses.filter(pl.col("site") == site),
                treatment=treatment,
                reference=reference,
                label=f"site {site}",
            )
            for site in site_labels
        ]
    lines += ["", "#### What averaging the members is worth (exploratory)", "", *CONTRAST_HEADER]
    lines += [
        _contrast_line(losses=losses, treatment=t, reference=r, label="all")
        for t, r in EXPLORATORY_CONTRASTS
    ]
    lines.append("")
    lines += geometry_lines(sites=sites, noun="solar farms")
    return "\n".join(lines) + "\n"


def main() -> int:
    """Build the row set, fit every arm, and write the report."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Fit nothing; rebuild report.md from the saved losses.parquet alone.",
    )
    arguments = parser.parse_args()

    sites = _pv_sites()
    frame = build_rows()
    _LOG.info("%d rows, %s to %s", frame.height, frame["time"].min(), frame["time"].max())

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUTPUT_DIR / "losses.parquet"
    fingerprint_path = OUTPUT_DIR / "losses.fingerprint"
    report_path = OUTPUT_DIR / "report.md"

    all_jobs = jobs()
    fingerprint = _fingerprint(frame=frame, job_list=all_jobs)

    if arguments.report_only:
        saved_fingerprint = (
            fingerprint_path.read_text().strip() if fingerprint_path.exists() else None
        )
        if saved_fingerprint != fingerprint:
            msg = (
                f"--report-only: {path} was fitted on a different row set, column set, seed set, "
                "or hyperparameter setting than this code now produces; re-run without "
                "--report-only"
            )
            raise ValueError(msg)
        losses = pl.read_parquet(path)
    else:
        refuse_to_overwrite(paths=[path, fingerprint_path, report_path])
        losses = run_all(dataset=frame, jobs=all_jobs, max_workers=MAX_CONCURRENT_FITS)
        losses.write_parquet(path)
        fingerprint_path.write_text(fingerprint)

    report = _report(frame=frame, losses=losses, sites=sites, job_list=all_jobs)
    report_path.write_text(report)
    sys.stdout.write(report)
    return 0


if __name__ == "__main__":
    sys.exit(main())
