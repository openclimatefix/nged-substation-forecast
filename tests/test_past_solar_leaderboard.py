"""Tests for `studies/beam_diffuse_split/past_solar_leaderboard.py`, on synthetic losses.

Each test is built to fail on the bug it exists for: a report number the script no longer
reproduces (which must stop the script before it writes), a planned contrast labelled exploratory,
a near-the-line contrast missed, a printed row from another section's fit compared, and a second
write into the write-once folder.
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
from studies.page_numbers import full_precision_values

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
    lines += _contrast_section(
        heading="Planned contrasts", contrast=contrast, label="ENS", by_label=by_label, tweak=tweak
    )
    if tweak != "no_second_setting":
        second = block_contrast_rows(
            losses=losses,
            arms=ARMS[2:],
            reference_arm="era5_global",
            setting="sensitivity",
            site_hours=SITE_HOURS,
            metric=METRIC,
        )
        lines += _contrast_section(
            heading="Planned contrasts at the second hyperparameter setting",
            contrast=second,
            label="ENS",
            by_label=by_label,
            scope="sensitivity",
            tweak="contrast" if tweak == "second_setting" else None,
        )
    lines += _contrast_section(
        heading="Every product against ERA5 (exploratory)",
        contrast=contrast,
        label="CAMS",
        by_label=by_label,
        tweak=None,
    )
    if tweak == "other_fit":
        lines += _contrast_section(
            heading="Leave one site out: the planned contrasts",
            contrast=contrast,
            label="ENS",
            by_label=by_label,
            tweak="contrast",
        )
    if tweak == "exploratory_fit":
        lines += _contrast_section(
            heading="A seasonal refit (exploratory)",
            contrast=contrast,
            label="ENS",
            by_label=by_label,
            tweak="contrast",
        )
    return "\n".join(lines) + "\n"


def _contrast_section(
    *,
    heading: str,
    contrast: pl.DataFrame,
    label: str,
    by_label: dict[str, str],
    tweak: str | None,
    scope: str = "all",
) -> list[str]:
    """Print one report section with a contrast table holding one arm's contrast against ERA5."""
    lines = [
        "",
        f"#### {heading}",
        "",
        (
            "| Scope | Contrast | ΔMAE (pp of capacity) | 95% interval | Excludes zero? "
            "| Folds agreeing | Rows |"
        ),
        "|---|---|---|---|---|---|---|",
    ]
    for row in contrast.filter(pl.col("label") == label).iter_rows(named=True):
        difference, low, high = (
            round(row[name], 3) for name in ("difference", "lower_95", "upper_95")
        )
        if tweak == "contrast":
            low -= 0.001
        lines.append(
            f"| {scope} | {by_label[row['label']]} − era5_global | {difference:+.3f} "
            f"| [{low:+.3f}, {high:+.3f}] | no | 2 of 2 | {SITE_HOURS} |"
        )
    return lines


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


@pytest.mark.parametrize("tweak", ["error", "interval", "contrast", "second_setting"])
def test_one_printed_number_that_differs_stops_the_script(tmp_path: Path, tweak: str) -> None:
    with pytest.raises(ValueError, match="ens_mean_t3"):
        _score(tmp_path=tmp_path, tweak=tweak)


def test_a_printed_row_from_an_exploratory_section_that_differs_stops_the_script(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="A seasonal refit"):
        _score(tmp_path=tmp_path, tweak="exploratory_fit")


def test_a_same_arms_row_in_a_section_that_refits_is_not_compared(tmp_path: Path) -> None:
    """The row prints the planned contrast on the same rows, but from a different fit."""
    result = _score(tmp_path=tmp_path, tweak="other_fit")

    assert result.site_hours == SITE_HOURS


def test_a_planned_contrast_with_a_second_setting_but_no_printed_second_row_stops_the_script(
    tmp_path: Path,
) -> None:
    with pytest.raises(ValueError, match="planned, but the report prints no row"):
        _score(tmp_path=tmp_path, tweak="no_second_setting")


def test_a_leaderboard_arm_with_no_printed_error_stops_the_script(tmp_path: Path) -> None:
    module = _load()
    losses = _losses()
    report_path = tmp_path / "report.md"
    text = _printed_report(losses=losses)
    report_path.write_text(text)
    row_set = _row_set(module, tmp_path)._replace(arm_suffix="_typo")

    with pytest.raises(ValueError, match="prints no error"):
        module.score_row_set(
            row_set=row_set, losses=losses, report_text=text, report_path=report_path
        )


def test_a_contrast_named_only_in_an_exploratory_section_stays_exploratory(
    tmp_path: Path,
) -> None:
    """CAMS is printed in an exploratory section, so only the planned section may label ENS."""
    result = _score(tmp_path=tmp_path)

    planning = dict(zip(result.contrasts["label"], result.contrasts["planning"], strict=True))
    assert planning["CAMS"] == "exploratory"


def test_a_post_hoc_arm_is_labelled_post_hoc(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load()
    monkeypatch.setattr(module, "POST_HOC_ARMS", ["cams_global"])
    losses = _losses()
    report_path = tmp_path / "report.md"
    text = _printed_report(losses=losses)
    report_path.write_text(text)

    result = module.score_row_set(
        row_set=_row_set(module, tmp_path), losses=losses, report_text=text, report_path=report_path
    )

    planning = dict(zip(result.contrasts["label"], result.contrasts["planning"], strict=True))
    assert planning == {"CAMS": "post hoc", "ENS": "planned"}


def _contrasts_with_flags(*, near_line: dict[str, bool]) -> pl.DataFrame:
    """Return exploratory contrast rows for two arms, each flagged near the line or not."""
    return pl.DataFrame(
        {"arm": arm, "planning": "exploratory", "near_line": flag}
        for arm, flag in near_line.items()
    )


def test_second_setting_is_computed_for_an_exploratory_row_near_the_line() -> None:
    module = _load()

    result = module._second_setting(
        contrasts=_contrasts_with_flags(near_line={"cams_global": True, "ens_mean_t3": False}),
        arms=ARMS,
        losses=_losses(),
        site_hours=SITE_HOURS,
    )

    second = dict(zip(result["arm"], result["second_difference"], strict=True))
    assert second["cams_global"] is not None
    assert second["ens_mean_t3"] is None


def test_write_outputs_writes_report_and_intervals_and_refuses_a_second_write(
    tmp_path: Path,
) -> None:
    module = _load()
    result = _score(tmp_path=tmp_path)
    output = tmp_path / "solar_leaderboard"

    module.write_outputs(results=[result], output_dir=output)

    assert "exploratory" in (output / "report.md").read_text()
    intervals = pl.read_parquet(output / "intervals.parquet")
    assert intervals.group_by("section", "setting").len().sort("section", "setting").rows() == [
        ("Mean absolute error", "pooled", len(ARMS)),
        ("Mean absolute error minus ERA5's", "pooled", 2),
        ("Mean absolute error minus ERA5's", "sensitivity", 1),
    ]
    with pytest.raises(FileExistsError):
        module.write_outputs(results=[result], output_dir=output)


def test_the_written_intervals_are_accepted_by_the_page_number_gate(tmp_path: Path) -> None:
    module = _load()
    result = _score(tmp_path=tmp_path)
    module.write_outputs(results=[result], output_dir=tmp_path / "out")
    intervals = pl.read_parquet(tmp_path / "out" / "intervals.parquet")
    lower = float(intervals.filter(pl.col("treatment") == "ens_mean_t3")["lower"][0])

    exact = full_precision_values(
        intervals_path=tmp_path / "out" / "intervals.parquet", print_decimals=frozenset({3})
    )

    assert f"{lower:+.3f}" in exact
