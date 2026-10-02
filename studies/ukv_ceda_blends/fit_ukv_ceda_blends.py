"""Fit ENS's mean with and without UKV-CEDA at lead days 1 to 4, and write the losses and report.

One-off throwaway script for
<https://github.com/openclimatefix/nged-substation-forecast/issues/1016>. It answers whether the
03 UTC run of UKV-CEDA lowers the power-forecast error of the ECMWF ENS mean at lead days 1 to 4.
For solar and for wind separately, at each lead day `N`, it fits four arms out of fold on the GPU,
each an XGBoost model per generator:

- `blend_ukv_ceda_dayN_pad`: ENS's mean padded to the blend's column count with exact copies of its
  own columns, the reference.
- `blend_ukv_ceda_dayN`: ENS's mean plus UKV-CEDA's columns.
- `blend_ukv_ceda_dayN_control` and `blend_ukv_ceda_dayN_control_b`: ENS's mean plus UKV-CEDA's
  columns shuffled within generator, year-month, and hour of day, under two shuffle seeds.

Two contrasts are planned before any fit. **P1** is the blend minus the padded ENS arm. **P2** is
the blend minus each control. The blend lowers the error at a technology and lead day only if the
upper 95% bound of P1 and of both P2 contrasts is below zero at both hyperparameter settings. The
rows, eras, folds, settings, seeds, metric, and paired month-resampled intervals are those of
`nwp_forecast_comparison` and `fit_aifs`, which this script imports and does not change. Each lead
day has its own rows, so contrasts are made within a lead day only.

**Rows.** `nwp_forecast_comparison.candidate_rows` (2024-12 onwards, 2026-01 dropped) joined with
ENS's day-4 mean and the UKV-CEDA inputs `build_ukv_ceda_inputs.py` wrote, kept where the target,
ENS's columns at that day, and UKV-CEDA's columns at that day are all present. The folds are the
shared design (`assign_folds_with_eras`); a stage whose rows leave a calendar month uncovered takes
the first rotation `search_fold_offsets` returns, before any fit.

**Outputs**, all under `--output-dir` (`data/studies/ukv_ceda_blends`) and written once:
`<domain>_day<N>_planned_losses.parquet`, `_predictions.parquet`, and a `.json` stamp for each
technology and lead day, `wind_day1_cpu_losses.parquet` (the GPU-CPU noise floor), `report.md`, and
`intervals.parquet`. Every output carries only the anonymised `site` label.

**Modes.** `--dry-run` builds every frame, checks every arm's columns, and lists the fits, writing
nothing. `--check` fits one arm at one wind site twice on the GPU, stops unless the two fingerprints
agree, prints a time estimate, and compares ENS's mean alone with its padded copy on the wind day-1
rows. A stage whose losses exist is not refitted. `--only-missing` fits the (arm, setting) pairs no
saved file holds into a new `_added_<k>` file. `--report-only` writes the report from the saved
losses to a new `--report-name`, fitting nothing. Run `uptime` and `nvidia-smi` before a fit, and
start only below a load average of about 24.

Run it with `uv run python studies/ukv_ceda_blends/fit_ukv_ceda_blends.py --dry-run`.
"""

import argparse
import json
import logging
import re
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Final, NamedTuple, TypedDict

import numpy as np
import polars as pl

_STUDIES_DIR: Final[Path] = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_STUDIES_DIR / "ukv_ceda_blends"))
sys.path.insert(0, str(_STUDIES_DIR / "nwp_forecast_comparison"))
sys.path.insert(0, str(_STUDIES_DIR / "beam_diffuse_split"))
sys.path.insert(0, str(_STUDIES_DIR / "weather_downloads"))

import build_ukv_ceda_inputs as build  # noqa: E402
import fit_aifs  # noqa: E402
from fit_extra_leads import error_text, interval_text  # noqa: E402
from nwp_forecast_comparison import (  # noqa: E402
    METRIC,
    NWP_ERA_FOLD_OFFSETS,
    NWP_ERA_START_MONTHS,
    PERCENTAGE_POINTS,
    SETTINGS,
    TARGET,
    DomainType,
    assign_folds_with_eras,
    coverage_table,
    difference,
    leaderboard,
    predictions_from_losses,
)
from paths import REPO_DATA_DIR  # noqa: E402
from studies.bootstrap import (  # noqa: E402
    MIN_MONTHS_FOR_INTERVAL,
    NO_DETECTABLE_DIFFERENCE,
    BootstrapInterval,
    bootstrap_difference_at_level,
    combine_setting_verdicts,
    paired_differences,
)
from studies.cross_validation import (  # noqa: E402
    calendar_month_coverage,
    cut_eras,
    out_of_fold_losses,
    search_fold_offsets,
    uncovered_months,
)
from studies.guards import check_no_missing, refuse_to_overwrite  # noqa: E402
from studies.ifs_single_runs import served_init_time  # noqa: E402

_LOG: Final[logging.Logger] = logging.getLogger(__name__)

PRIMARY: Final[str] = fit_aifs.PRIMARY
SENSITIVITY: Final[str] = fit_aifs.SENSITIVITY

PRODUCT: Final[str] = "ukv_ceda"
"""The product's key in `fit_aifs.BLEND_AIFS_PREFIXES`."""

DAYS: Final[tuple[int, ...]] = build.LEAD_DAYS

PLANNED_GROUP: Final[str] = "planned"
CPU_GROUP: Final[str] = "cpu"
CPU_SITE: Final[str] = "W1"
"""The wind site whose day-1 blend is refitted on the CPU to measure the GPU-CPU difference."""

MAX_WORKERS: Final[int] = 2
"""The load rule: at most two (arm, site) fits at once, each on `THREADS_PER_FIT` threads."""

N_P1_INTERVALS: Final[int] = 2 * len(DAYS)
"""The planned P1 intervals per setting: four lead days at two technologies."""

BONFERRONI_LEVEL: Final[float] = 100.0 - 5.0 / N_P1_INTERVALS
"""The coverage, in percent, of the wider P1 interval: 95% corrected across `N_P1_INTERVALS`."""

SECOND_SEED_VARIANT: Final[str] = "_b"
BEFORE_UPGRADE_ERAS: Final[tuple[int, ...]] = (0, 1)
AFTER_UPGRADE_ERAS: Final[tuple[int, ...]] = (2,)

ROW_KEYS: Final[tuple[str, ...]] = ("site", "time", "fold")

SPECS: Final[Mapping[str, str]] = {
    "p1": "blend minus padded ENS",
    "p2": "blend minus control",
    "p2b": "blend minus second-seed control",
}


class Stage(NamedTuple):
    """One frame to build and fit: a technology and a lead day."""

    domain: DomainType
    day: int


class IntervalRecord(TypedDict):
    """One interval the report prints, for `intervals.parquet`."""

    domain: str
    day: int
    setting: str
    contrast: str
    scope: str
    level: float
    difference: float
    lower: float
    upper: float
    n_rows: int
    n_months: int


# --- Arms and jobs --------------------------------------------------------------------------------


def arm_name(*, day: int, role: fit_aifs.BlendRoleType) -> str:
    """Return one of a lead day's blend arms, such as `blend_ukv_ceda_day2_control`."""
    return fit_aifs.blend_arm_name(product=PRODUCT, day=day, role=role)


def stage_arms(*, day: int) -> tuple[str, str, str, str]:
    """Return a lead day's arms: the padded ENS reference, the blend, and its two controls."""
    return (
        arm_name(day=day, role="_pad"),
        arm_name(day=day, role=""),
        arm_name(day=day, role="_control"),
        arm_name(day=day, role="_control_b"),
    )


def planned_jobs(*, day: int) -> list[fit_aifs.Job]:
    """Return every (arm, setting) fit of a lead day: each arm at both settings."""
    return [(arm, setting) for arm in stage_arms(day=day) for setting in SETTINGS]


def stages() -> list[Stage]:
    """Return every stage, solar before wind, each lead day in order."""
    return [Stage(domain=domain, day=day) for domain in fit_aifs.DOMAINS for day in DAYS]


def stem(*, stage: Stage, group: str) -> str:
    """Return the file stem of a stage's group, such as `wind_day1_planned`."""
    return f"{stage.domain}_day{stage.day}_{group}"


def group_files(*, output_dir: Path, stage: Stage) -> list[Path]:
    """Return every saved losses file of a stage's fits at the two settings, oldest group first.

    Args:
        output_dir: The write-once folder.
        stage: The stage.

    Returns:
        The `planned` file and every `added_<k>` file that exists. The CPU refit is not included.
    """
    prefix = f"{stage.domain}_day{stage.day}_"
    found = [
        path
        for path in output_dir.glob(f"{prefix}*_losses.parquet")
        if re.fullmatch(
            rf"{PLANNED_GROUP}|added_\d+",
            path.name.removeprefix(prefix).removesuffix("_losses.parquet"),
        )
    ]
    return sorted(
        found, key=lambda path: (path.name != f"{prefix}{PLANNED_GROUP}_losses.parquet", path.name)
    )


def saved_pairs(*, output_dir: Path, stage: Stage) -> set[fit_aifs.Job]:
    """Return every (arm, setting) pair that a saved losses file of the stage holds."""
    pairs: set[fit_aifs.Job] = set()
    for path in group_files(output_dir=output_dir, stage=stage):
        pairs |= set(pl.read_parquet(path).select("arm", "setting").unique().iter_rows())
    return pairs


def next_group(*, output_dir: Path, stage: Stage) -> str:
    """Return the group the next fit of a stage writes: `planned`, then `added_1`, `added_2`."""
    existing = group_files(output_dir=output_dir, stage=stage)
    return PLANNED_GROUP if not existing else f"added_{len(existing)}"


def saved_losses(*, output_dir: Path, stage: Stage) -> pl.DataFrame:
    """Return every saved loss of a stage at both settings, stacked."""
    return pl.concat(
        [pl.read_parquet(path) for path in group_files(output_dir=output_dir, stage=stage)],
        how="diagonal",
    )


# --- Rows -----------------------------------------------------------------------------------------


def present(*, column: str) -> pl.Expr:
    """Return whether a column holds a value: neither null nor not-a-number."""
    return pl.col(column).is_not_null() & ~pl.col(column).is_nan().fill_null(value=False)


def ukv_columns(*, domain: DomainType, day: int) -> list[str]:
    """Return UKV-CEDA's weather columns at one lead day."""
    return [f"{PRODUCT}_day{day}_{field}" for field in build.WEATHER_FIELDS[domain]]


def check_init_times(*, frame: pl.DataFrame, domain: DomainType, day: int) -> None:
    """Raise unless every row's stamped run is the 03 UTC run `day` days before its own day.

    Args:
        frame: Rows carrying `time` and `ukv_ceda_day<N>_init_time`.
        domain: `solar` or `wind`.
        day: The lead day.

    Raises:
        ValueError: Naming how many rows read another run than `served_init_time` with `run_hour=3`.
    """
    expected = served_init_time(
        time=pl.col("time"), day=day, domain=domain, run_hour=build.RUN_HOUR
    )
    wrong = frame.filter(pl.col(f"{PRODUCT}_day{day}_init_time") != expected).height
    if wrong:
        msg = (
            f"{domain} day {day}: {wrong} rows read a run other than the 03 UTC run of day D-{day}"
        )
        raise ValueError(msg)


def cut_folds(*, frame: pl.DataFrame) -> tuple[pl.DataFrame, dict[int, int]]:
    """Assign the shared design's folds, or the first covering rotation if the design leaves a gap.

    Args:
        frame: Rows carrying `site`, `time`, and `month`.

    Returns:
        The rows with `era_code`, `era`, and `fold`, and the fold offsets of each era used.

    Raises:
        ValueError: If no rotation covers every calendar month.
    """
    shared = assign_folds_with_eras(frame=frame)
    if uncovered_months(coverage=calendar_month_coverage(frame=shared)).is_empty():
        return shared, dict(NWP_ERA_FOLD_OFFSETS)
    found = search_fold_offsets(frame=frame, first_months=NWP_ERA_START_MONTHS)
    if not found:
        msg = "no fold rotation leaves a training row for every calendar month of every site"
        raise ValueError(msg)
    offsets = dict(found[0])
    return (
        cut_eras(frame=frame, first_months=NWP_ERA_START_MONTHS, fold_offsets=offsets),
        offsets,
    )


def add_copy_columns(*, frame: pl.DataFrame, domain: DomainType, day: int) -> pl.DataFrame:
    """Add exact copies of ENS's mean columns, which pad the reference to the blend's column count.

    Args:
        frame: Rows carrying ENS's mean columns at the lead day.
        domain: `solar` or `wind`.
        day: The lead day.

    Returns:
        `frame` with `ens_mean_day<N>_copy_<field>` equal to `ens_mean_day<N>_<field>`.
    """
    source = f"ens_mean_day{day}"
    return frame.with_columns(
        pl.col(column).alias(f"{source}{fit_aifs.COPY_SUFFIX}_{column.removeprefix(f'{source}_')}")
        for column in build.ens_columns(domain=domain, day=day)
    )


def stage_frame(
    *, stage: Stage, candidates: pl.DataFrame, inputs: pl.DataFrame
) -> tuple[pl.DataFrame, dict[int, int]]:
    """Return the rows every arm of one stage is trained and scored on.

    Args:
        stage: The technology and lead day.
        candidates: `build_ukv_ceda_inputs.published_rows`' result for the technology.
        inputs: The technology's `<domain>_ukv_ceda_inputs.parquet`.

    Returns:
        The rows with folds, ENS's copies, and the shuffled UKV-CEDA copies, and the fold offsets.

    Raises:
        ValueError: If a candidate row has no row in the inputs, a row reads the wrong run, an arm's
            column holds a missing value, or a calendar month is left uncovered.
    """
    domain, day = stage
    keys = ["site", "time"]
    if candidates.select(keys).join(inputs.select(keys), on=keys, how="anti").height:
        msg = f"{domain}: candidate rows are missing from the UKV-CEDA inputs"
        raise ValueError(msg)
    stamp = f"{PRODUCT}_day{day}_init_time"
    joined = candidates.join(
        inputs.select(*keys, stamp, *ukv_columns(domain=domain, day=day)), on=keys, how="left"
    )
    required = [
        TARGET,
        *build.ens_columns(domain=domain, day=day),
        *ukv_columns(domain=domain, day=day),
    ]
    kept = joined.filter(pl.all_horizontal(present(column=column) for column in required)).sort(
        keys
    )
    check_init_times(frame=kept, domain=domain, day=day)
    cut, offsets = cut_folds(frame=kept)
    padded = add_copy_columns(frame=cut, domain=domain, day=day)
    arms = stage_arms(day=day)
    frame = fit_aifs.add_shuffled_columns(
        frame=padded, domain=domain, shuffles=fit_aifs.control_shuffles(arms=arms)
    )
    check_no_missing(frame=frame, columns=fit_aifs.source_columns(arms=arms, domain=domain))
    coverage_table(frame=frame)
    return frame, offsets


def check_arm_columns(*, frame: pl.DataFrame, domain: DomainType, arms: Sequence[str]) -> None:
    """Raise unless every arm's columns are in the frame and hold the count the arm's kind promises.

    Args:
        frame: A stage's rows.
        domain: `solar` or `wind`.
        arms: The arms to be fitted.

    Raises:
        ValueError: Naming the arm and the absent columns or the two counts.
    """
    for arm in arms:
        features = fit_aifs.arm_features(arm=arm, domain=domain)
        absent = [name for name in features if name not in frame.columns]
        if absent:
            msg = f"{domain}: {arm} has no columns {absent} in the rows"
            raise ValueError(msg)
        expected = fit_aifs.expected_column_count(arm=arm, domain=domain)
        if len(features) != expected:
            msg = f"{domain}: {arm} has {len(features)} columns, its kind promises {expected}"
            raise ValueError(msg)


# --- The build gate and the stamp -----------------------------------------------------------------


def read_build_stamp(*, output_dir: Path) -> dict[str, object]:
    """Return `build.json`, after checking the build passed its coverage guard and is unchanged.

    Args:
        output_dir: The folder holding the build's outputs.

    Returns:
        The stamp.

    Raises:
        ValueError: If the build did not pass its coverage guard, or an inputs file's SHA-256 is not
            the one the build recorded.
    """
    stamp = json.loads((output_dir / build.STAMP_NAME).read_text())
    if stamp.get("coverage_guard_passed") is not True:
        msg = "the build did not pass its coverage guard, so no stage may run"
        raise ValueError(msg)
    for domain in fit_aifs.DOMAINS:
        actual = fit_aifs.sha256_of(path=output_dir / f"{domain}_ukv_ceda_inputs.parquet")
        if stamp["inputs_sha256"][domain] != actual:
            msg = f"{domain}: the inputs file is not the one build.json recorded"
            raise ValueError(msg)
    return stamp


def stage_stamp(
    *,
    published_dir: Path,
    day4_dir: Path,
    output_dir: Path,
    stage: Stage,
    arms: Sequence[str],
    offsets: Mapping[int, int],
    snapshot_id: object,
) -> dict[str, str]:
    """Return what a stage's saved losses must match.

    Args:
        published_dir: The folder holding the published inputs.
        day4_dir: The folder holding ENS's day-4 mean.
        output_dir: The folder holding the UKV-CEDA inputs.
        stage: The stage.
        arms: The arms the stamp's `columns` entry lists.
        offsets: The fold offsets of each era.
        snapshot_id: The Icechunk snapshot the inputs were built from.

    Returns:
        `fit_aifs.build_stamp`'s entries, plus the fold offsets and the snapshot.
    """
    return {
        **fit_aifs.build_stamp(
            published_dir=published_dir,
            aifs_dir=output_dir,
            domain=stage.domain,
            arms=arms,
            extra_dirs={"day4_shared": day4_dir},
            inputs_name=PRODUCT,
        ),
        "fold_offsets": json.dumps(dict(sorted(offsets.items()))),
        "icechunk_snapshot": str(snapshot_id),
    }


def check_saved_file(
    *,
    losses: pl.DataFrame,
    frame: pl.DataFrame,
    label: str,
    stamp_file: Path,
    stamp: dict[str, str],
) -> None:
    """Raise unless a saved losses file comes from this build, this device, and exactly these rows.

    Args:
        losses: The saved losses.
        frame: The stage's rows.
        label: The stage, for messages.
        stamp_file: The stamp written beside the losses.
        stamp: The current build's stamp.

    Raises:
        ValueError: If the stamp is missing or differs, or an (arm, setting) of the losses scores
            other (site, time, fold) rows than the stage.
    """
    if not stamp_file.exists() or json.loads(stamp_file.read_text()) != stamp:
        msg = f"{stamp_file} is missing or names another build or device"
        raise ValueError(msg)
    want = frame.select(ROW_KEYS).unique().sort(ROW_KEYS)
    for (arm, setting), group in losses.group_by(["arm", "setting"]):
        if not group.select(ROW_KEYS).unique().sort(ROW_KEYS).equals(want):
            msg = f"{label}: {arm} at {setting} scores other rows or folds than the stage's"
            raise ValueError(msg)


def write_atomically(*, path: Path, frame: pl.DataFrame) -> None:
    """Write a parquet file to a temporary name and rename it, so a crash leaves no partial file."""
    temporary = path.with_name(path.name + ".tmp")
    frame.write_parquet(temporary)
    temporary.replace(path)


def predictions_table(*, losses: pl.DataFrame, frame: pl.DataFrame) -> pl.DataFrame:
    """Return each fit's capped out-of-fold prediction beside the measured power and the fold.

    Args:
        losses: A stage's per-row losses.
        frame: The stage's rows, carrying the target.

    Returns:
        `arm`, `setting`, `site`, `time`, `seed`, `fold`, `actual_mw`, and `prediction_capped_mw`.
    """
    return (
        predictions_from_losses(losses=losses, frame=frame)
        .join(
            losses.select("arm", "setting", "site", "time", "seed", "fold"),
            on=["arm", "setting", "site", "time", "seed"],
        )
        .join(
            frame.select("site", "time", actual_mw=pl.col(TARGET).cast(pl.Float64)),
            on=["site", "time"],
        )
        .select(
            "arm", "setting", "site", "time", "seed", "fold", "actual_mw", "prediction_capped_mw"
        )
    )


# --- Fitting --------------------------------------------------------------------------------------


class PlannedStage(NamedTuple):
    """A stage ready to fit: its rows, fold offsets, and stamp."""

    stage: Stage
    frame: pl.DataFrame
    offsets: dict[int, int]
    stamp: dict[str, str]


def plan_stages(*, published_dir: Path, day4_dir: Path, output_dir: Path) -> list[PlannedStage]:
    """Build every stage's rows and check its arms, after the build gate.

    Args:
        published_dir: The folder holding the published inputs.
        day4_dir: The folder holding ENS's day-4 mean.
        output_dir: The folder holding the UKV-CEDA inputs, and where results are written.

    Returns:
        The stages, solar before wind.
    """
    build_stamp = read_build_stamp(output_dir=output_dir)
    planned: list[PlannedStage] = []
    for domain in fit_aifs.DOMAINS:
        candidates = build.published_rows(
            published_dir=published_dir, day4_dir=day4_dir, domain=domain
        )
        inputs = pl.read_parquet(output_dir / f"{domain}_ukv_ceda_inputs.parquet")
        for day in DAYS:
            stage = Stage(domain=domain, day=day)
            frame, offsets = stage_frame(stage=stage, candidates=candidates, inputs=inputs)
            check_arm_columns(frame=frame, domain=domain, arms=stage_arms(day=day))
            stamp = stage_stamp(
                published_dir=published_dir,
                day4_dir=day4_dir,
                output_dir=output_dir,
                stage=stage,
                arms=stage_arms(day=day),
                offsets=offsets,
                snapshot_id=build_stamp["snapshot_id"],
            )
            planned.append(PlannedStage(stage=stage, frame=frame, offsets=offsets, stamp=stamp))
    return planned


def jobs_to_fit(
    *, output_dir: Path, stage: Stage, only_missing: bool
) -> tuple[str, list[fit_aifs.Job]]:
    """Return the group to write and the (arm, setting) pairs of a stage to fit.

    Args:
        output_dir: The write-once folder.
        stage: The stage.
        only_missing: Whether a stage that holds some but not all of its pairs may fit the rest.

    Returns:
        The group name and the pairs: all of them for a stage with no saved file, the missing ones
        under `--only-missing`, and none for a complete stage.

    Raises:
        ValueError: If a stage holds some pairs but not all and `only_missing` is false.
    """
    saved = saved_pairs(output_dir=output_dir, stage=stage)
    missing = [job for job in planned_jobs(day=stage.day) if job not in saved]
    if saved and missing and not only_missing:
        msg = f"{stem(stage=stage, group='*')}: saved fits lack {missing}; pass --only-missing"
        raise ValueError(msg)
    return next_group(output_dir=output_dir, stage=stage), missing


def fit_stage(
    *, planned: PlannedStage, output_dir: Path, workers: int, only_missing: bool
) -> pl.DataFrame:
    """Fit a stage's missing pairs and write them once, then return all its saved losses.

    Args:
        planned: The stage.
        output_dir: The write-once folder.
        workers: How many (arm, site) fits run at once.
        only_missing: Whether to fit the pairs a partly saved stage lacks.

    Returns:
        Every saved loss of the stage at both settings.
    """
    stage = planned.stage
    group, jobs = jobs_to_fit(output_dir=output_dir, stage=stage, only_missing=only_missing)
    if jobs:
        file = output_dir / f"{stem(stage=stage, group=group)}_losses.parquet"
        predictions = file.with_name(file.name.replace("_losses", "_predictions"))
        refuse_to_overwrite(paths=[file, predictions])
        arms = list(dict.fromkeys(arm for arm, _ in jobs))
        stamp = {
            **planned.stamp,
            "columns": json.dumps(
                {arm: fit_aifs.arm_features(arm=arm, domain=stage.domain) for arm in sorted(arms)}
            ),
        }
        file.with_suffix(".json").write_text(json.dumps(stamp))
        losses = fit_aifs.fit_jobs(
            frame=planned.frame, domain=stage.domain, jobs=jobs, workers=workers
        ).with_columns(device=pl.lit(fit_aifs.DEVICE))
        write_atomically(path=file, frame=losses)
        write_atomically(
            path=predictions, frame=predictions_table(losses=losses, frame=planned.frame)
        )
    return verified_losses(planned=planned, output_dir=output_dir)


def verified_losses(*, planned: PlannedStage, output_dir: Path) -> pl.DataFrame:
    """Return a stage's saved losses after checking each file's stamp and rows.

    Args:
        planned: The stage.
        output_dir: The folder holding the saved files.

    Returns:
        Every saved loss of the stage at both settings.
    """
    stage = planned.stage
    for path in group_files(output_dir=output_dir, stage=stage):
        losses = pl.read_parquet(path)
        arms = sorted(set(losses["arm"].unique().to_list()))
        expected = {
            **planned.stamp,
            "columns": json.dumps(
                {arm: fit_aifs.arm_features(arm=arm, domain=stage.domain) for arm in arms}
            ),
        }
        check_saved_file(
            losses=losses,
            frame=planned.frame,
            label=path.name,
            stamp_file=path.with_suffix(".json"),
            stamp=expected,
        )
    return saved_losses(output_dir=output_dir, stage=stage)


def cpu_noise_floor(*, planned: PlannedStage, output_dir: Path, gpu: pl.DataFrame) -> pl.DataFrame:
    """Refit the wind day-1 blend at one site on the CPU, write it once, and return it.

    Args:
        planned: The wind day-1 stage.
        output_dir: The write-once folder.
        gpu: The stage's saved losses, holding the GPU fit.

    Returns:
        The CPU losses, labelled with the arm, setting, and `cpu` device. A rerun reads the saved
        file after checking its stamp.
    """
    arm = arm_name(day=1, role="")
    file = output_dir / f"{stem(stage=planned.stage, group=CPU_GROUP)}_losses.parquet"
    stamp = {
        **planned.stamp,
        "device": "cpu",
        "columns": json.dumps({arm: fit_aifs.arm_features(arm=arm, domain="wind")}),
    }
    if file.exists():
        if json.loads(file.with_suffix(".json").read_text()) != stamp:
            msg = f"{file.with_suffix('.json')} names another build"
            raise ValueError(msg)
        return pl.read_parquet(file)
    refuse_to_overwrite(paths=[file])
    file.with_suffix(".json").write_text(json.dumps(stamp))
    losses = out_of_fold_losses(
        site_rows=planned.frame.filter(pl.col("site") == CPU_SITE),
        features=list(fit_aifs.arm_features(arm=arm, domain="wind")),
        target=TARGET,
        hyper_parameters=SETTINGS[PRIMARY],
        with_quantiles=False,
        device="cpu",
    ).with_columns(arm=pl.lit(arm), setting=pl.lit(PRIMARY), device=pl.lit("cpu"))
    write_atomically(path=file, frame=losses)
    return losses


def noise_floor_line(*, cpu: pl.DataFrame, gpu: pl.DataFrame) -> str:
    """Describe how far the CPU refit's per-row error is from the GPU fit's, in points of capacity.

    Args:
        cpu: `cpu_noise_floor`'s result.
        gpu: The stage's saved losses.

    Returns:
        A sentence giving the mean absolute and the largest per-row difference, and both fits' mean
        error.
    """
    arm = cpu["arm"][0]
    reference = gpu.filter(
        pl.col("arm") == arm, pl.col("setting") == PRIMARY, pl.col("site") == CPU_SITE
    )
    joined = cpu.join(reference, on=["site", "time", "seed"], suffix="_gpu").with_columns(
        gap=(pl.col(METRIC) - pl.col(f"{METRIC}_gpu"))
    )
    gap = np.abs(joined["gap"].to_numpy()) * PERCENTAGE_POINTS
    cpu_error = joined[METRIC].to_numpy().mean() * PERCENTAGE_POINTS
    gpu_error = joined[f"{METRIC}_gpu"].to_numpy().mean() * PERCENTAGE_POINTS
    return (
        f"The CPU refit of `{arm}` at wind site {CPU_SITE} (primary setting) differs from the GPU "
        f"fit by {gap.mean():.4f} points of capacity per row on average and {gap.max():.4f} at "
        f"most; the two fits' mean errors are {cpu_error:.4f} (CPU) and {gpu_error:.4f} (GPU)."
    )


# --- Reading rule ---------------------------------------------------------------------------------


def setting_verdict(
    *, day: int, p1: BootstrapInterval, p2: Sequence[BootstrapInterval]
) -> dict[str, object]:
    """Return the verdict at one setting.

    The blend lowers the error only if P1 and every P2 contrast (both shuffle seeds) have an upper
    95% bound below zero, so one noisy shuffle cannot decide it.

    Args:
        day: The lead day, for the label.
        p1: The blend minus the padded ENS arm.
        p2: The blend minus each control, one per shuffle seed.

    Returns:
        `verdict`: `lowers the error at day N`, `raises the error at day N` where P1's lower
        bound is above zero, or `no detectable difference`; and `largest_gain_not_excluded`, the
        gain P1's lower bound leaves open, or `None` where the verdict is not `no detectable
        difference`.
    """
    if p1["upper_95"] < 0.0 and all(interval["upper_95"] < 0.0 for interval in p2):
        return {"verdict": f"lowers the error at day {day}", "largest_gain_not_excluded": None}
    if p1["lower_95"] > 0.0:
        return {"verdict": f"raises the error at day {day}", "largest_gain_not_excluded": None}
    return {
        "verdict": NO_DETECTABLE_DIFFERENCE,
        "largest_gain_not_excluded": max(0.0, -p1["lower_95"]),
    }


def reading(
    *,
    day: int,
    p1: Mapping[str, BootstrapInterval],
    p2: Mapping[str, Sequence[BootstrapInterval]],
) -> str:
    """Return the verdict that both settings give, or `no detectable difference` if they differ.

    Args:
        day: The lead day.
        p1: P1 at each setting.
        p2: The P2 contrasts of both shuffle seeds at each setting.

    Returns:
        The verdict, from `studies.bootstrap.combine_setting_verdicts`.
    """
    verdicts = {
        setting: str(setting_verdict(day=day, p1=p1[setting], p2=p2[setting])["verdict"])
        for setting in (PRIMARY, SENSITIVITY)
    }
    return combine_setting_verdicts(
        primary=verdicts[PRIMARY],
        sensitivity=verdicts[SENSITIVITY],
        unresolved=NO_DETECTABLE_DIFFERENCE,
    )


def null_reading(*, interval: BootstrapInterval) -> str:
    """Say what a contrast's interval leaves open, in points of capacity.

    Args:
        interval: A contrast, treatment minus reference, so a negative difference is a gain.

    Returns:
        A sentence naming the largest gain the interval does not exclude, or saying that the
        interval is entirely on one side of zero.
    """
    scale = PERCENTAGE_POINTS
    if interval["upper_95"] < 0.0:
        return "the interval is below zero: the blend's error is lower"
    if interval["lower_95"] > 0.0:
        return "the interval is above zero: the blend's error is higher"
    gain = max(0.0, -interval["lower_95"]) * scale
    return (
        f"an effect as large as {gain:.3f} points of capacity is not excluded, and an effect "
        f"larger than {gain:.3f} points is excluded"
    )


# --- Report ---------------------------------------------------------------------------------------


def record(
    *,
    stage: Stage,
    setting: str,
    contrast: str,
    scope: str,
    interval: BootstrapInterval,
    level: float = 95.0,
) -> IntervalRecord:
    """Return one printed interval as a row of `intervals.parquet`."""
    return {
        "domain": stage.domain,
        "day": stage.day,
        "setting": setting,
        "contrast": contrast,
        "scope": scope,
        "level": level,
        "difference": interval["difference"],
        "lower": interval["lower_95"],
        "upper": interval["upper_95"],
        "n_rows": interval["n_rows"],
        "n_months": interval["n_months"],
    }


def interval_cell(*, interval: BootstrapInterval) -> str:
    """Format an interval as `difference [lower, upper]` in points of capacity."""
    return interval_text(
        point=interval["difference"], lower=interval["lower_95"], upper=interval["upper_95"]
    )


def at_setting(*, losses: pl.DataFrame, setting: str) -> pl.DataFrame:
    """Return the losses of one hyperparameter setting."""
    return losses.filter(pl.col("setting") == setting)


def era_losses(*, losses: pl.DataFrame, frame: pl.DataFrame, eras: Sequence[int]) -> pl.DataFrame:
    """Return the losses on the rows whose era is one of `eras`."""
    rows_in_eras = frame.filter(pl.col("era_code").is_in(list(eras))).select("site", "time")
    return losses.join(rows_in_eras, on=["site", "time"])


def scoped_line(
    *,
    losses: pl.DataFrame,
    treatment: str,
    reference: str,
    label: str,
    stage: Stage,
    records: list[IntervalRecord],
) -> str:
    """Format P1 on a subset of rows as a table row, with an interval only for six months or more.

    Args:
        losses: The subset's losses at the primary setting.
        treatment: The blend.
        reference: The padded ENS arm.
        label: The subset's name.
        stage: The stage.
        records: Where an interval is appended for `intervals.parquet`.

    Returns:
        `| label | difference [interval] | rows | months |`.
    """
    if losses.is_empty():
        return f"| {label} | no rows | 0 | 0 |"
    differences, months = paired_differences(
        losses=losses, treatment=treatment, reference=reference, metric=METRIC
    )
    n_months = len(np.unique(months))
    if n_months >= MIN_MONTHS_FOR_INTERVAL:
        interval = difference(losses=losses, treatment=treatment, reference=reference)
        records.append(
            record(stage=stage, setting=PRIMARY, contrast="p1", scope=label, interval=interval)
        )
        text = interval_cell(interval=interval)
    else:
        text = (
            f"{float(differences.mean()) * PERCENTAGE_POINTS:+.3f} (no interval: {n_months} months)"
        )
    return f"| {label} | {text} | {differences.shape[1]} | {n_months} |"


def generator_error_lines(*, losses: pl.DataFrame, arms: Sequence[str]) -> list[str]:
    """Format each arm's mean absolute error at each generator alone, with no interval.

    Args:
        losses: Per-row losses at one setting, holding every arm of `arms`.
        arms: The arms to show, as columns.

    Returns:
        A Markdown table of percent of capacity, one row per generator in label order.
    """
    means = (
        losses.filter(pl.col("arm").is_in(list(arms)))
        .group_by("site", "arm")
        .agg(error=pl.col(METRIC).mean() * PERCENTAGE_POINTS)
        .pivot(on="arm", index="site", values="error")
        .sort("site")
    )
    lines = [
        "Mean absolute error at each generator alone (primary setting, % of capacity):",
        "",
        "| Generator | " + " | ".join(f"`{arm}`" for arm in arms) + " |",
        "|---|" + "---|" * len(arms),
    ]
    lines += [
        f"| {row['site']} | " + " | ".join(f"{row[arm]:.3f}" for arm in arms) + " |"
        for row in means.iter_rows(named=True)
    ]
    return lines


def stage_lines(
    *, planned: PlannedStage, losses: pl.DataFrame, records: list[IntervalRecord]
) -> tuple[list[str], str]:
    """Format one stage's section of the report.

    Args:
        planned: The stage and its rows.
        losses: The stage's saved losses at both settings.
        records: Where every printed interval is appended for `intervals.parquet`.

    Returns:
        The section's Markdown lines and the stage's reading.
    """
    stage, frame = planned.stage, planned.frame
    pad, blend, control, control_b = stage_arms(day=stage.day)
    per_setting = {setting: at_setting(losses=losses, setting=setting) for setting in SETTINGS}
    p1 = {s: difference(losses=per_setting[s], treatment=blend, reference=pad) for s in SETTINGS}
    p2 = {
        s: [
            difference(losses=per_setting[s], treatment=blend, reference=c)
            for c in (control, control_b)
        ]
        for s in SETTINGS
    }
    wide = {
        s: bootstrap_difference_at_level(
            losses=per_setting[s],
            treatment=blend,
            reference=pad,
            metric=METRIC,
            level=BONFERRONI_LEVEL,
        )
        for s in SETTINGS
    }
    for setting in SETTINGS:
        records.append(
            record(
                stage=stage, setting=setting, contrast="p1", scope="all rows", interval=p1[setting]
            )
        )
        for label, interval in zip(("p2", "p2b"), p2[setting], strict=True):
            records.append(
                record(
                    stage=stage,
                    setting=setting,
                    contrast=label,
                    scope="all rows",
                    interval=interval,
                )
            )
    verdict = reading(day=stage.day, p1=p1, p2=p2)
    lines = [
        f"### {stage.domain.capitalize()}, lead day {stage.day}",
        "",
        (
            f"{frame.height} rows over {frame['month'].n_unique()} months at "
            f"{frame['site'].n_unique()} generators; fold offsets by era "
            f"{json.dumps(dict(sorted(planned.offsets.items())))}."
        ),
        "",
        f"| Contrast (points) | Primary | Sensitivity | Primary, Bonferroni {BONFERRONI_LEVEL}% |",
        "|---|---|---|---|",
        (
            f"| P1: {SPECS['p1']} | {interval_cell(interval=p1[PRIMARY])} "
            f"| {interval_cell(interval=p1[SENSITIVITY])} "
            f"| {wide[PRIMARY][0] * PERCENTAGE_POINTS:+.3f}, "
            f"{wide[PRIMARY][1] * PERCENTAGE_POINTS:+.3f} |"
        ),
        (
            f"| P2: {SPECS['p2']} | {interval_cell(interval=p2[PRIMARY][0])} "
            f"| {interval_cell(interval=p2[SENSITIVITY][0])} | not adjusted |"
        ),
        (
            f"| P2: {SPECS['p2b']} | {interval_cell(interval=p2[PRIMARY][1])} "
            f"| {interval_cell(interval=p2[SENSITIVITY][1])} | not adjusted |"
        ),
        "",
        f"Reading: {verdict}.",
        "",
        (
            f"P1 at the primary setting: {null_reading(interval=p1[PRIMARY])}. "
            f"At the sensitivity setting: {null_reading(interval=p1[SENSITIVITY])}."
        ),
        "",
    ]
    near = [s for s in SETTINGS if fit_aifs.near_line(interval=p1[s])]
    if near:
        lines += [f"P1 is near the 5% line at the {', '.join(near)} setting.", ""]
    signs = {s: np.sign(p1[s]["difference"]) for s in SETTINGS}
    if signs[PRIMARY] != signs[SENSITIVITY]:
        lines += ["P1 changes sign between the two settings.", ""]
    lines += [
        "| Arm | Setting | Error (% of capacity) | 95% interval | Rows | Months |",
        "|---|---|---|---|---|---|",
    ]
    for setting in SETTINGS:
        board = leaderboard(losses=per_setting[setting], arms=list(stage_arms(day=stage.day)))
        for row in board.iter_rows(named=True):
            text = error_text(value=row["value"], lower=row["lower_95"], upper=row["upper_95"])
            value, interval_part = text.split(" [")
            lines.append(
                f"| `{row['arm']}` | {setting} | {value} | [{interval_part} | {row['n_rows']} "
                f"| {row['n_months']} |"
            )
    lines += [
        "",
        *generator_error_lines(losses=per_setting[PRIMARY], arms=stage_arms(day=stage.day)),
    ]
    lines += [
        "",
        (
            "Exploratory: P1 by era (primary setting). Era 0 is before 2025-10, era 1 is 2025-10 "
            "to 2025-12, and era 2 starts in 2026-02, after the UKV upgrade settled."
        ),
        "",
        "| Subset | Difference [interval] (points) | Rows | Months |",
        "|---|---|---|---|",
    ]
    primary = per_setting[PRIMARY]
    eras: dict[str, Sequence[int]] = {
        "era 0": (0,),
        "era 1": (1,),
        "era 2": (2,),
        "before the upgrade (eras 0 and 1)": BEFORE_UPGRADE_ERAS,
        "after the upgrade (era 2)": AFTER_UPGRADE_ERAS,
    }
    lines += [
        scoped_line(
            losses=era_losses(losses=primary, frame=frame, eras=members),
            treatment=blend,
            reference=pad,
            label=f"E4 {name}",
            stage=stage,
            records=records,
        )
        for name, members in eras.items()
    ]
    lines += [
        "",
        "Exploratory: P1 by generator (primary setting).",
        "",
        "| Subset | Difference [interval] (points) | Rows | Months |",
        "|---|---|---|---|",
    ]
    lines += [
        scoped_line(
            losses=primary.filter(pl.col("site") == site),
            treatment=blend,
            reference=pad,
            label=f"E5 generator {site}",
            stage=stage,
            records=records,
        )
        for site in sorted(frame["site"].unique().to_list())
    ]
    lines.append("")
    return lines, verdict


def columns_lines() -> list[str]:
    """Print every arm's feature columns at lead day 1 for each technology, for a reviewer."""
    lines = ["## Columns of every arm at lead day 1", ""]
    for domain in fit_aifs.DOMAINS:
        lines += [f"### {domain}", ""]
        lines += [
            f"- `{arm}` ({len(fit_aifs.arm_features(arm=arm, domain=domain))} columns): "
            + ", ".join(fit_aifs.arm_features(arm=arm, domain=domain))
            for arm in stage_arms(day=1)
        ]
        lines.append("")
    return lines


def report_text(
    *,
    sections: Sequence[tuple[str, list[str]]],
    summary: Sequence[str],
    cpu_line: str,
) -> str:
    """Return `report.md`: the design, the readings, the columns, and every stage.

    Args:
        sections: Each technology's name and its stages' lines.
        summary: The readings table's lines.
        cpu_line: The GPU-CPU noise-floor sentence.

    Returns:
        The report.
    """
    lines = [
        "# UKV-CEDA blends: report",
        "",
        (
            "Every fit is on the GPU. Differences are first arm minus second, in percentage points "
            "of capacity, so a negative difference means the first arm has the lower error. P1 is "
            "the blend minus ENS's mean padded to the blend's column count, and P2 is the blend "
            "minus each control, whose UKV-CEDA columns are shuffled within generator, year-month, "
            "and hour of day under two seeds. The blend lowers the error at a technology and lead "
            "day only if the upper 95% bound of P1 and of both P2 contrasts is below zero at both "
            "settings. The Bonferroni interval corrects P1's 95% level across "
            f"{N_P1_INTERVALS} P1 intervals per setting ({BONFERRONI_LEVEL}%), and P2 is not "
            "adjusted. UKV-CEDA's lead is 3 hours fresher than ENS's at every hour, which favours "
            "the blend, and its wind columns are native 10 m and 925 hPa winds, not 100 m winds. "
            "Rows lost to each cause are in the folder's `README.md`."
        ),
        "",
        "## Readings",
        "",
        *summary,
        "",
        cpu_line,
        "",
        *columns_lines(),
    ]
    for name, section in sections:
        lines += [f"## {name}", "", *section]
    return "\n".join(lines)


def summary_lines(*, readings: Mapping[tuple[str, int], str]) -> list[str]:
    """Format the readings of every stage as one table."""
    lines = ["| Technology | Lead day | Reading |", "|---|---|---|"]
    lines += [f"| {domain} | {day} | {text} |" for (domain, day), text in readings.items()]
    return lines


def report_paths(*, output_dir: Path, name: str) -> tuple[Path, Path]:
    """Return the report and intervals paths: `report.md` and `intervals.parquet` for `report`."""
    if name == "report":
        return output_dir / "report.md", output_dir / "intervals.parquet"
    return output_dir / f"{name}.md", output_dir / f"{name}_intervals.parquet"


def build_report(*, planned: Sequence[PlannedStage], output_dir: Path) -> tuple[str, pl.DataFrame]:
    """Build the report and the interval table from the saved losses, fitting nothing.

    Args:
        planned: Every stage.
        output_dir: The folder holding the saved losses.

    Returns:
        The report text and one row per printed interval.
    """
    records: list[IntervalRecord] = []
    readings: dict[tuple[str, int], str] = {}
    sections: list[tuple[str, list[str]]] = []
    gpu_wind_day1: pl.DataFrame | None = None
    for domain in fit_aifs.DOMAINS:
        lines: list[str] = []
        for item in (p for p in planned if p.stage.domain == domain):
            losses = verified_losses(planned=item, output_dir=output_dir)
            if item.stage == Stage("wind", 1):
                gpu_wind_day1 = losses
            stage_text, verdict = stage_lines(planned=item, losses=losses, records=records)
            lines += stage_text
            readings[(domain, item.stage.day)] = verdict
        sections.append((domain.capitalize(), lines))
    cpu_file = output_dir / f"{stem(stage=Stage('wind', 1), group=CPU_GROUP)}_losses.parquet"
    cpu_line = "No CPU refit is saved."
    if cpu_file.exists() and gpu_wind_day1 is not None:
        cpu_line = noise_floor_line(cpu=pl.read_parquet(cpu_file), gpu=gpu_wind_day1)
    return (
        report_text(sections=sections, summary=summary_lines(readings=readings), cpu_line=cpu_line),
        pl.DataFrame(records),
    )


# --- Modes ----------------------------------------------------------------------------------------


def print_plan(*, planned: Sequence[PlannedStage], output_dir: Path, only_missing: bool) -> None:
    """Print every stage's rows and fits, and the number of (arm, site) fits, without fitting."""
    total = 0
    for item in planned:
        group, jobs = jobs_to_fit(
            output_dir=output_dir, stage=item.stage, only_missing=only_missing
        )
        sites = item.frame["site"].n_unique()
        total += len(jobs) * sites
        sys.stdout.write(
            f"{item.stage.domain} day {item.stage.day}: {item.frame.height} rows, {sites} sites, "
            f"fold offsets {item.offsets}, group {group}: {len(jobs)} (arm, setting) fits\n"
        )
    sys.stdout.write(f"{total} (arm, site) fits in all\n")


def padded_matches_unpadded(*, frame: pl.DataFrame, day: int) -> tuple[bool, float]:
    """Fit ENS's mean alone and its padded copy at the primary setting and compare per-row losses.

    Args:
        frame: A wind stage's rows.
        day: The lead day.

    Returns:
        Whether every row's absolute error is identical, and the largest per-row gap in megawatts.
    """
    alone = f"ens_mean_day{day}"
    padded = arm_name(day=day, role="_pad")
    losses = fit_aifs.fit_jobs(
        frame=frame, domain="wind", jobs=[(alone, PRIMARY), (padded, PRIMARY)], workers=1
    )
    keys = ["site", "time", "seed"]
    columns = [*keys, "absolute_error_mw"]
    first = losses.filter(pl.col("arm") == alone).select(columns).sort(keys)
    second = losses.filter(pl.col("arm") == padded).select(columns).sort(keys)
    gap = float(
        np.abs(first["absolute_error_mw"].to_numpy() - second["absolute_error_mw"].to_numpy()).max()
    )
    return first.equals(second), gap


def run_check(*, planned: Sequence[PlannedStage], output_dir: Path) -> int:
    """Time one arm twice, print the run's estimate, and compare padded with unpadded ENS.

    Args:
        planned: Every stage.
        output_dir: The folder holding the saved fits, which sets the number still to fit.

    Returns:
        0 if the two GPU runs agree, else 1.
    """
    first = next(item for item in planned if item.stage == Stage("wind", 1))
    agree, seconds = fit_aifs.time_two_fits(
        frame=first.frame, arm=arm_name(day=1, role=""), domain="wind"
    )
    n_fits = sum(
        len(jobs_to_fit(output_dir=output_dir, stage=item.stage, only_missing=True)[1])
        * item.frame["site"].n_unique()
        for item in planned
    )
    identical, gap = padded_matches_unpadded(frame=first.frame, day=1)
    sys.stdout.write(
        f"one arm at one site: {seconds:.0f} s; {n_fits} (arm, site) fits are about "
        f"{n_fits * seconds / 3600:.1f} h on one worker\n"
        f"two GPU runs agree: {agree}\n"
        f"ENS's mean alone and padded to the blend's column count have identical per-row losses: "
        f"{identical} (largest gap {gap:.3g} MW)\n"
        f"CHECK {'PASS' if agree else 'FAIL'}\n"
    )
    return 0 if agree else 1


def run_fits(
    *, planned: Sequence[PlannedStage], output_dir: Path, workers: int, only_missing: bool
) -> int:
    """Fit every stage, then the CPU refit, and write the report and intervals once.

    Args:
        planned: Every stage.
        output_dir: The write-once folder.
        workers: How many (arm, site) fits run at once.
        only_missing: Whether to fit the pairs a partly saved stage lacks.

    Returns:
        0.
    """
    report_path, intervals_path = report_paths(output_dir=output_dir, name="report")
    refuse_to_overwrite(paths=[report_path, intervals_path])
    for item in planned:
        losses = fit_stage(
            planned=item, output_dir=output_dir, workers=workers, only_missing=only_missing
        )
        if item.stage == Stage("wind", 1):
            cpu_noise_floor(planned=item, output_dir=output_dir, gpu=losses)
        _LOG.info("%s day %d: fitted", item.stage.domain, item.stage.day)
    text, intervals = build_report(planned=planned, output_dir=output_dir)
    report_path.write_text(text)
    intervals.write_parquet(intervals_path)
    return 0


def check_output_dir(*, output_dir: Path, read_only: Sequence[Path]) -> None:
    """Raise unless `output_dir` is the one folder this script writes to and no input folder.

    Args:
        output_dir: Where the fits write.
        read_only: The folders the script only reads.

    Raises:
        ValueError: If `output_dir` is a read-only folder or is not named like the build's.
    """
    build.check_output_dir(output_dir=output_dir, read_only=read_only)


def workers_argument(text: str) -> int:
    """Parse `--workers`, refusing a count above `MAX_WORKERS`."""
    workers = fit_aifs.workers_argument(text)
    if workers > MAX_WORKERS:
        msg = f"--workers is at most {MAX_WORKERS} under the load rule, not {workers}"
        raise argparse.ArgumentTypeError(msg)
    return workers


def main() -> int:
    """Fit the blends, or list (`--dry-run`), time (`--check`), or report (`--report-only`)."""
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    studies_dir = REPO_DATA_DIR / "studies"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--published-dir", type=Path, default=studies_dir / build.PUBLISHED_DIR_NAME
    )
    parser.add_argument("--day4-dir", type=Path, default=studies_dir / build.DAY4_DIR_NAME)
    parser.add_argument("--output-dir", type=Path, default=studies_dir / build.OUTPUT_DIR_NAME)
    parser.add_argument("--workers", type=workers_argument, default=MAX_WORKERS)
    parser.add_argument("--dry-run", action="store_true", help="List the fits; fit nothing.")
    parser.add_argument("--check", action="store_true", help="Time one fit twice; compare padding.")
    parser.add_argument("--only-missing", action="store_true", help="Fit the pairs no file holds.")
    parser.add_argument("--report-only", action="store_true", help="Write the report; fit nothing.")
    parser.add_argument(
        "--report-name", default="report", help="The report's name, `report` or new."
    )
    parser.add_argument(
        "--verified",
        action="store_true",
        help="Confirm verify_ukv_ceda_inputs.py exited 0 on this build.",
    )
    args = parser.parse_args()
    check_output_dir(output_dir=args.output_dir, read_only=[args.published_dir, args.day4_dir])
    planned = plan_stages(
        published_dir=args.published_dir, day4_dir=args.day4_dir, output_dir=args.output_dir
    )
    if args.dry_run:
        print_plan(planned=planned, output_dir=args.output_dir, only_missing=args.only_missing)
        return 0
    if args.report_only:
        report_path, intervals_path = report_paths(
            output_dir=args.output_dir, name=args.report_name
        )
        refuse_to_overwrite(paths=[report_path, intervals_path])
        text, intervals = build_report(planned=planned, output_dir=args.output_dir)
        report_path.write_text(text)
        intervals.write_parquet(intervals_path)
        return 0
    fit_aifs.check_gpu_visible()
    if args.check:
        return run_check(planned=planned, output_dir=args.output_dir)
    if not args.verified:
        sys.stdout.write("pass --verified once verify_ukv_ceda_inputs.py has exited 0\n")
        return 1
    return run_fits(
        planned=planned,
        output_dir=args.output_dir,
        workers=args.workers,
        only_missing=args.only_missing,
    )


if __name__ == "__main__":
    sys.exit(main())
