"""Tests for `studies/beam_diffuse_split/past_solar_leaderboard.py`, on synthetic losses.

Each test is built to fail on the bug it exists for: a report number the script no longer
reproduces (which must stop the script before it writes), a planned contrast labelled exploratory,
a near-the-line contrast missed, and a second write into the write-once folder.
"""

import importlib.util
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import ModuleType
from typing import Any, Final

import polars as pl
import pytest
from studies.charts import BlockArm, block_contrast_rows, block_leaderboard_rows

REPO_ROOT: Final[Path] = Path(__file__).parent.parent
SCRIPT_DIR: Final[Path] = REPO_ROOT / "studies" / "beam_diffuse_split"
METRIC: Final[str] = "absolute_error_capped_fraction_of_capacity"
SITE_HOURS: Final[int] = 8
ARMS: Final[tuple[BlockArm, ...]] = (
    BlockArm("cams_global", "CAMS", "satellite", reference=True),
    BlockArm("era5_global", "ERA5", "reanalysis", reference=True),
    BlockArm("ens_mean_t3", "ENS", "weather model"),
)


def _load() -> ModuleType:
    sys.path.insert(0, str(SCRIPT_DIR))
    spec = importlib.util.spec_from_file_location(
        "past_solar_leaderboard", SCRIPT_DIR / "past_solar_leaderboard.py"
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _losses() -> pl.DataFrame:
    """Return 2 sites x 4 hours x 2 seeds for three arms at two settings sharing arm names."""
    start = datetime(2025, 1, 1, tzinfo=UTC)
    base = {"cams_global": 0.010, "era5_global": 0.020, "ens_mean_t3": 0.018}
    return pl.DataFrame(
        {
            "arm": arm,
            "setting": setting,
            "site": site,
            "time": start + timedelta(hours=hour),
            "seed": seed,
            "month": "2025-01" if hour < 2 else "2025-02",
            METRIC: value + 0.001 * hour + 0.002 * seed + (0.05 if setting == "sensitivity" else 0),
        }
        for arm, value in base.items()
        for setting in ("pooled", "sensitivity")
        for site in ("a", "b")
        for hour in range(4)
        for seed in (0, 1)
    )


def _printed_report(*, losses: pl.DataFrame, tweak: str | None = None) -> str:
    """Write a report that prints exactly what the script recomputes, optionally with one edit."""
    absolute = block_leaderboard_rows(
        losses=losses, arms=ARMS, setting="pooled", site_hours=SITE_HOURS, metric=METRIC
    )
    contrast = block_contrast_rows(
        losses=losses,
        arms=ARMS[::2],
        reference_arm="era5_global",
        setting="pooled",
        site_hours=SITE_HOURS,
        metric=METRIC,
    )
    lines = [
        f"### Test rows on {SITE_HOURS} common site-hours of solar (2025-01-01 to 2025-02-01)",
        "",
        "| Arm | All sites | 95% interval |",
        "|---|---|---|",
    ]
    by_label = {arm.label: arm.arm for arm in ARMS}
    for row in absolute.iter_rows(named=True):
        value, low, high = (round(row[name], 3) for name in ("value", "lower_95", "upper_95"))
        if tweak == "error" and row["label"] == "ENS":
            value += 0.001
        if tweak == "interval" and row["label"] == "ENS":
            high += 0.001
        lines.append(f"| {by_label[row['label']]} | {value:.3f} | [{low:.3f}, {high:.3f}] |")
    lines += [
        "",
        "#### Planned contrasts",
        "",
    ]
    lines += [
        (
            "| Scope | Contrast | ΔMAE (pp of capacity) | 95% interval | Excludes zero? "
            "| Folds agreeing | Rows |"
        ),
        "|---|---|---|---|---|---|---|",
    ]
    for row in contrast.filter(pl.col("label") == "ENS").iter_rows(named=True):
        difference, low, high = (
            round(row[name], 3) for name in ("difference", "lower_95", "upper_95")
        )
        if tweak == "contrast":
            low -= 0.001
        lines.append(
            f"| all | {by_label[row['label']]} − era5_global | {difference:+.3f} "
            f"| [{low:+.3f}, {high:+.3f}] | no | 2 of 2 | {SITE_HOURS} |"
        )
    return "\n".join(lines) + "\n"


def _row_set(module: ModuleType, tmp_path: Path) -> Any:
    return module.RowSet(
        key="test",
        label="Test rows",
        directory=tmp_path,
        printed_column="All sites",
        arm_suffix="",
        leaderboard_arms=ARMS,
        contrast_arms=(ARMS[0], ARMS[2]),
    )


def _score(*, tmp_path: Path, tweak: str | None = None) -> Any:
    module = _load()
    losses = _losses()
    report_path = tmp_path / "report.md"
    text = _printed_report(losses=losses, tweak=tweak)
    report_path.write_text(text)
    return module.score_row_set(
        row_set=_row_set(module, tmp_path),
        losses=losses,
        report_text=text,
        report_path=report_path,
    )


def test_a_report_the_script_reproduces_scores_and_labels_the_planned_contrast(
    tmp_path: Path,
) -> None:
    result = _score(tmp_path=tmp_path)

    planning = dict(zip(result.contrasts["label"], result.contrasts["planning"], strict=True))
    assert result.site_hours == SITE_HOURS
    assert result.dates == "2025-01-01 to 2025-02-01"
    assert planning == {"CAMS": "exploratory", "ENS": "planned"}


@pytest.mark.parametrize("tweak", ["error", "interval", "contrast"])
def test_one_printed_number_that_differs_stops_the_script(tmp_path: Path, tweak: str) -> None:
    with pytest.raises(ValueError, match="ens_mean_t3"):
        _score(tmp_path=tmp_path, tweak=tweak)


def test_second_setting_is_computed_only_for_planned_or_near_line_rows(tmp_path: Path) -> None:
    result = _score(tmp_path=tmp_path)
    by_label = {row["label"]: row for row in result.contrasts.iter_rows(named=True)}

    assert by_label["ENS"]["second_difference"] is not None
    assert by_label["CAMS"]["near_line"] == (by_label["CAMS"]["second_difference"] is not None)


def test_is_near_line_uses_twenty_percent_of_the_interval_width() -> None:
    module = _load()

    assert module.is_near_line(lower_95=-0.218 - 0.174, upper_95=-0.058)
    assert not module.is_near_line(lower_95=-1.0, upper_95=1.0)
    assert not module.is_near_line(lower_95=-4.294, upper_95=-3.666)


def test_write_outputs_writes_report_and_intervals_and_refuses_a_second_write(
    tmp_path: Path,
) -> None:
    module = _load()
    result = _score(tmp_path=tmp_path)
    output = tmp_path / "solar_leaderboard"

    module.write_outputs(results=[result], output_dir=output)

    assert "exploratory" in (output / "report.md").read_text()
    intervals = pl.read_parquet(output / "intervals.parquet")
    assert sorted(intervals["kind"].unique().to_list()) == ["absolute", "minus_era5"]
    assert intervals.filter(pl.col("kind") == "absolute").height == len(ARMS)
    with pytest.raises(FileExistsError):
        module.write_outputs(results=[result], output_dir=output)
