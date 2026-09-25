"""Tests for `studies/beam_diffuse_split/past_solar_leaderboard.py`, on synthetic losses.

Each test is built to fail on the bug it exists for: a report number the script no longer
reproduces (which must stop the script before it writes), a planned contrast labelled exploratory,
a near-the-line contrast missed, a printed row from another section's fit compared, and a second
write into the write-once folder.
"""

import importlib.util
import re
import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import ModuleType
from typing import Any, Final

import polars as pl
import pytest
from studies.charts import (
    BlockArm,
    PlannedContrast,
    block_contrast_rows,
    block_leaderboard_rows,
    planned_contrast_rows,
    report_contrasts,
)
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
ENS_AGAINST_ERA5: Final[PlannedContrast] = PlannedContrast(ARMS[2], ARMS[1])
CAMS_AGAINST_ENS: Final[PlannedContrast] = PlannedContrast(ARMS[0], ARMS[2])


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
    """Return 2 sites x 4 hours x 2 seeds for three arms at two settings sharing arm names.

    Each arm's `sensitivity` losses sit a different distance above its `pooled` losses, so a
    contrast differs between the two settings.
    """
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
            METRIC: value
            + 0.001 * hour
            + 0.002 * seed
            + (0.05 + 0.01 * index if setting == "sensitivity" else 0),
        }
        for index, (arm, value) in enumerate(base.items())
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
        if tweak == "interval_low" and row["label"] == "ENS":
            low -= 0.001
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
        planned_contrasts=(ENS_AGAINST_ERA5,),
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


@pytest.mark.parametrize(
    "tweak", ["error", "interval", "interval_low", "contrast", "second_setting"]
)
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
        ("Planned contrasts, first product minus second", "pooled", 1),
        ("Planned contrasts, first product minus second", "sensitivity", 1),
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


PLANNED_TWO: Final[tuple[PlannedContrast, ...]] = (ENS_AGAINST_ERA5, CAMS_AGAINST_ENS)


def _planned_report(
    *, tmp_path: Path, contrasts: tuple[PlannedContrast, ...], flip: bool = False
) -> pl.DataFrame:
    """Print the planned contrasts as a report would, and read them back; `flip` swaps the sign."""
    rows = planned_contrast_rows(
        losses=_losses(),
        contrasts=contrasts,
        setting="pooled",
        site_hours=SITE_HOURS,
        metric=METRIC,
    )
    lines = [
        "#### Planned contrasts",
        "",
        (
            "| Scope | Contrast | ΔMAE (pp of capacity) | 95% interval | Excludes zero? "
            "| Folds agreeing | Rows |"
        ),
        "|---|---|---|---|---|---|---|",
    ]
    for row in rows.iter_rows(named=True):
        difference, low, high = (
            round(row[name], 3) for name in ("difference", "lower_95", "upper_95")
        )
        if flip:
            difference, low, high = -difference, -high, -low
        lines.append(
            f"| all | {row['arm']} − {row['reference_arm']} | {difference:+.3f} "
            f"| [{low:+.3f}, {high:+.3f}] | no | 2 of 2 | {SITE_HOURS} |"
        )
    path = tmp_path / "planned.md"
    path.write_text("\n".join(lines) + "\n")
    return report_contrasts(report_path=path)


def _planned_frame(*, contrasts: tuple[PlannedContrast, ...]) -> pl.DataFrame:
    return planned_contrast_rows(
        losses=_losses(),
        contrasts=contrasts,
        setting="pooled",
        site_hours=SITE_HOURS,
        metric=METRIC,
    ).with_columns(second_difference=pl.lit(None, dtype=pl.Float64))


def test_a_planned_contrast_against_another_product_is_checked_against_its_printed_row(
    tmp_path: Path,
) -> None:
    module = _load()
    printed = _planned_report(tmp_path=tmp_path, contrasts=PLANNED_TWO)

    module.check_planned_contrasts(
        planned=_planned_frame(contrasts=PLANNED_TWO), printed=printed, site_hours=SITE_HOURS
    )


def test_a_planned_contrast_drawn_with_the_wrong_sign_stops_the_script(tmp_path: Path) -> None:
    module = _load()
    printed = _planned_report(tmp_path=tmp_path, contrasts=PLANNED_TWO, flip=True)

    with pytest.raises(
        ValueError,
        match=r"all difference: bootstrapped -0\.200 but the report says 0\.2",
    ):
        module.check_planned_contrasts(
            planned=_planned_frame(contrasts=PLANNED_TWO), printed=printed, site_hours=SITE_HOURS
        )


def test_a_planned_contrast_with_no_printed_row_stops_the_script(tmp_path: Path) -> None:
    module = _load()
    printed = _planned_report(tmp_path=tmp_path, contrasts=(ENS_AGAINST_ERA5,))

    with pytest.raises(ValueError, match="planned, but the report prints no row"):
        module.check_planned_contrasts(
            planned=_planned_frame(contrasts=PLANNED_TWO), printed=printed, site_hours=SITE_HOURS
        )


def test_the_two_lists_of_planned_contrasts_must_agree_in_both_directions(
    tmp_path: Path,
) -> None:
    module = _load()
    printed = _planned_report(tmp_path=tmp_path, contrasts=PLANNED_TWO)

    dropped = module.unlisted_planned_contrasts(
        contrasts=(ENS_AGAINST_ERA5,), printed=printed, site_hours=SITE_HOURS
    )
    invented = module.unlisted_planned_contrasts(
        contrasts=(*PLANNED_TWO, PlannedContrast(ARMS[1], ARMS[0])),
        printed=printed,
        site_hours=SITE_HOURS,
    )
    agreed = module.unlisted_planned_contrasts(
        contrasts=PLANNED_TWO, printed=printed, site_hours=SITE_HOURS
    )

    assert len(dropped) == 1
    assert "cams_global - ens_mean_t3" in dropped[0]
    assert "the script does not list it" in dropped[0]
    assert len(invented) == 1
    assert "era5_global - cams_global" in invented[0]
    assert "the report does not print it" in invented[0]
    assert agreed == []


def test_a_row_set_whose_report_prints_a_planned_contrast_the_script_omits_stops_the_script(
    tmp_path: Path,
) -> None:
    module = _load()
    losses = _losses()
    report_path = tmp_path / "report.md"
    text = _printed_report(losses=losses)
    report_path.write_text(text)
    row_set = _row_set(module, tmp_path)._replace(planned_contrasts=())

    with pytest.raises(ValueError, match="the script does not list it"):
        module.score_row_set(
            row_set=row_set, losses=losses, report_text=text, report_path=report_path
        )


def test_score_row_set_holds_each_planned_contrast_with_its_second_setting(
    tmp_path: Path,
) -> None:
    result = _score(tmp_path=tmp_path)

    planned = result.planned
    assert planned["label"].to_list() == ["ENS against ERA5"]
    assert planned["difference"].to_list() == pytest.approx([-0.2], abs=0.05)
    assert planned["second_difference"][0] is not None


def test_the_written_report_and_intervals_hold_the_planned_contrasts(tmp_path: Path) -> None:
    module = _load()
    result = _score(tmp_path=tmp_path)
    output = tmp_path / "solar_leaderboard"

    module.write_outputs(results=[result], output_dir=output)

    report = (output / "report.md").read_text()
    assert "#### Planned contrasts, first product minus second" in report
    assert "| ENS against ERA5 |" in report
    intervals = pl.read_parquet(output / "intervals.parquet").filter(
        pl.col("section") == module.PLANNED_CONTRAST_SECTION
    )
    assert intervals.sort("setting").select("setting", "treatment", "reference").rows() == [
        ("pooled", "ens_mean_t3", "era5_global"),
        ("sensitivity", "ens_mean_t3", "era5_global"),
    ]


def test_every_row_set_lists_the_planned_contrasts_the_study_names() -> None:
    module = _load()

    counts = {row_set.key: len(row_set.planned_contrasts) for row_set in module.ROW_SETS}

    assert counts == {"main": 6, "extra": 3, "ens": 2, "station": 3}


def test_the_second_setting_is_bootstrapped_from_the_sensitivity_losses(tmp_path: Path) -> None:
    """Each arm's `sensitivity` losses sit a different distance above its `pooled` ones (1 pp)."""
    module = _load()
    result = _score(tmp_path=tmp_path)

    planned = result.planned.row(0, named=True)
    assert planned["second_difference"] == pytest.approx(planned["difference"] + 1.0)
    ens = result.contrasts.filter(pl.col("label") == "ENS").row(0, named=True)
    assert ens["second_difference"] == pytest.approx(ens["difference"] + 1.0)
    output = tmp_path / "solar_leaderboard"
    module.write_outputs(results=[result], output_dir=output)
    intervals = pl.read_parquet(output / "intervals.parquet").filter(
        pl.col("section") == module.PLANNED_CONTRAST_SECTION, pl.col("setting") == "sensitivity"
    )
    assert intervals["value"].to_list() == pytest.approx([planned["second_difference"]])
    contrast_second = pl.read_parquet(output / "intervals.parquet").filter(
        pl.col("section") == module.CONTRAST_SECTION, pl.col("setting") == "sensitivity"
    )
    assert contrast_second["value"].to_list() == pytest.approx([ens["second_difference"]])


def test_a_contrast_gets_no_second_setting_where_era5_has_no_sensitivity_losses() -> None:
    module = _load()
    losses = _losses().filter(
        ~((pl.col("arm") == "era5_global") & (pl.col("setting") == "sensitivity"))
    )
    contrasts = pl.DataFrame({"arm": ["ens_mean_t3"], "planning": ["planned"], "near_line": [True]})

    result = module._second_setting(
        contrasts=contrasts, arms=ARMS, losses=losses, site_hours=SITE_HOURS
    )

    assert result["second_difference"].to_list() == [None]


def test_the_report_prints_a_positive_difference_with_a_plus_sign(tmp_path: Path) -> None:
    module = _load()
    result = _score(tmp_path=tmp_path)

    lines = module.render_report(results=[result]).splitlines()

    (row,) = [line for line in lines if line.startswith("| ENS against ERA5 |")]
    assert "| -0.2" in row
    assert re.search(r"\+0\.8\d* \[", row)


def test_a_second_setting_row_is_not_compared_with_a_printed_first_setting_row() -> None:
    module = _load()
    contrasts = pl.DataFrame(
        {
            "arm": "ens_mean_t3",
            "second_difference": 0.5,
            "second_lower_95": 0.4,
            "second_upper_95": 0.6,
        }
    )
    printed = pl.DataFrame(
        {
            "scope": "all",
            "treatment": "ens_mean_t3",
            "reference": "era5_global",
            "n_rows": SITE_HOURS,
            "section": "Planned contrasts",
            "difference": 0.1,
            "lower_95": 0.0,
            "upper_95": 0.2,
        }
    )

    problems = module.check_contrasts(
        contrasts=contrasts,
        printed=printed,
        site_hours=SITE_HOURS,
        scope=module.SECOND_SETTING_SCOPE,
        column_prefix="second_",
    )

    assert problems == []


@pytest.mark.parametrize(
    ("lower", "upper", "near"),
    [
        (0.2, 1.2, True),  # the nearer bound is exactly 20% of the width from zero
        (0.21, 1.21, False),
        (-0.10, 0.90, True),  # near through the lower bound, which is negative
        (-0.9, 0.1, True),  # near through the upper bound
        (0.50, 1.50, False),
        (-1.5, -0.5, False),
    ],
)
def test_near_the_line_is_a_bound_within_a_fifth_of_the_width_from_zero(
    lower: float, upper: float, near: bool
) -> None:
    module = _load()
    frame = pl.DataFrame({"lower_95": [lower], "upper_95": [upper]})

    assert frame.select(module.near_line()).item() is near


def test_score_row_set_flags_rows_with_the_near_line_rule(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load()
    monkeypatch.setattr(module, "near_line", lambda: pl.lit(True))

    result = _score_with(module=module, tmp_path=tmp_path)

    assert result.contrasts["near_line"].all()
    assert result.planned["near_line"].all()


def _score_with(*, module: ModuleType, tmp_path: Path) -> Any:
    losses = _losses()
    report_path = tmp_path / "report.md"
    text = _printed_report(losses=losses)
    report_path.write_text(text)
    return module.score_row_set(
        row_set=_row_set(module, tmp_path),
        losses=losses,
        report_text=text,
        report_path=report_path,
    )


def test_the_report_prints_the_near_line_and_reference_flags(tmp_path: Path) -> None:
    module = _load()
    result = _score(tmp_path=tmp_path)
    flagged = result._replace(
        contrasts=result.contrasts.with_columns(near_line=pl.col("label") == "CAMS")
    )

    lines = module.render_report(results=[flagged]).splitlines()

    assert any(line.startswith("| CAMS |") and "| exploratory | yes |" in line for line in lines)
    assert any(line.startswith("| ENS |") and "| planned | no |" in line for line in lines)
    assert any(line.startswith("| CAMS | ") and line.endswith("| yes |") for line in lines)
    assert any(line.startswith("| ENS | ") and line.endswith("| no |") for line in lines)


def test_every_row_set_marks_cams_and_era5_as_reference_rows() -> None:
    module = _load()

    for row_set in module.ROW_SETS:
        flagged = {arm.arm for arm in row_set.leaderboard_arms if arm.reference}
        assert flagged == {"cams_global", "era5_global"}, row_set.key


def test_the_main_contrasts_hold_the_ukv_rebuilds_and_no_row_set_contrasts_era5_with_itself() -> (
    None
):
    module = _load()

    main_arms = {arm.arm for arm in module.ROW_SETS[0].contrast_arms}
    assert {"ukv_trap_global", "ukv_pair_global"} <= main_arms
    for row_set in module.ROW_SETS:
        assert "era5_global" not in {arm.arm for arm in row_set.contrast_arms}


def test_the_intervals_and_the_report_hold_every_row_set(tmp_path: Path) -> None:
    module = _load()
    first = _score(tmp_path=tmp_path)
    second = first._replace(row_set=first.row_set._replace(key="other", label="Other rows"))

    intervals = module.intervals_frame(results=[first, second])
    report = module.render_report(results=[first, second])

    assert set(intervals["row_set"].to_list()) == {"test", "other"}
    assert "### Test rows:" in report
    assert "### Other rows:" in report


def _main_with(
    *, module: ModuleType, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, argv: list[str]
) -> tuple[int, Path]:
    """Run `main` on one synthetic row set, with the output folder under `tmp_path`."""
    data = tmp_path / "data"
    data.mkdir()
    losses = _losses()
    losses.write_parquet(data / "losses.parquet")
    (data / "report.md").write_text(_printed_report(losses=losses))
    output = tmp_path / "solar_leaderboard"
    monkeypatch.setattr(module, "ROW_SETS", (_row_set(module, data),))
    monkeypatch.setattr(module, "SOLAR_LEADERBOARD_DIR", output)
    monkeypatch.setattr(sys, "argv", ["past_solar_leaderboard.py", *argv])
    return module.main(), output


def test_check_only_writes_nothing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    module = _load()

    code, output = _main_with(
        module=module, tmp_path=tmp_path, monkeypatch=monkeypatch, argv=["--check-only"]
    )

    assert code == 0
    assert not output.exists()


def test_a_full_run_writes_the_folder_and_refuses_to_run_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load()

    code, output = _main_with(module=module, tmp_path=tmp_path, monkeypatch=monkeypatch, argv=[])
    assert code == 0
    assert (output / "report.md").exists()
    second_code = module.main()

    assert second_code == 1


def test_a_full_run_refuses_an_existing_folder_before_scoring(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load()
    (tmp_path / "solar_leaderboard").mkdir()

    code, _ = _main_with(module=module, tmp_path=tmp_path, monkeypatch=monkeypatch, argv=[])

    assert code == 1
