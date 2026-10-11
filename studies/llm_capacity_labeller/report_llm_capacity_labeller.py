"""Print every table the page quotes into `report.md`.

Part of the study in <https://github.com/openclimatefix/nged-substation-forecast/issues/1147>. Every
number on the page comes from this report, never from a transcription. The report holds the probe,
the budget estimate, Part 1 (tuning, scores, and the first study's published figures for the
estimators it re-implements), the planned contrasts, the controls, the event scores by arm and
shape, false movement, the exploratory checks, and the run's cost.

Run it with `uv run python studies/llm_capacity_labeller/report_llm_capacity_labeller.py`; `--smoke`
reads and writes under `_smoke/`.
"""

import argparse
import json
import sys
from pathlib import Path

import polars as pl
from labeller_paths import FIRST_STUDY_BUILT, claim_root, log_path, tables_dir
from probe_claude_cli import PROBE_DIR, summarise
from score_labeller import CONTROL_HIT_GATE
from studies.capacity_scaling import SCORED_COLUMN


def _markdown(*, frame: pl.DataFrame) -> str:
    """Render a frame as a Markdown table."""
    with pl.Config(
        tbl_formatting="MARKDOWN", tbl_rows=-1, tbl_cols=-1, tbl_width_chars=400, float_precision=3
    ):
        return str(frame)


def _section(*, title: str, body: str) -> str:
    return f"## {title}\n\n{body}\n"


def _table_section(*, title: str, path: Path) -> str:
    if not path.exists():
        return _section(title=title, body=f"_missing: {path.name} has not been written_")
    return _section(title=title, body=_markdown(frame=pl.read_parquet(path)))


def event_summary(*, tables: Path) -> pl.DataFrame | None:
    """Summarise detection scores by arm, checks, and shape.

    Args:
        tables: The tables directory.

    Returns:
        Hit rate, drop and recovery rates, mean date and size errors, and mean capacity error in
        points, by arm, whether the hard checks were applied, and shape; `None` if none exist.
    """
    path = tables / "event_scores.parquet"
    if not path.exists():
        return None
    return (
        pl.read_parquet(path)
        .group_by("arm", "after_checks", "shape")
        .agg(
            events=pl.len(),
            hit_rate=pl.col("hit").mean(),
            drop_found=pl.col("drop_found").mean(),
            recovery_found=pl.col("recovery_found").mean(),
            mean_date_error_days=pl.col("date_error_days").mean(),
            mean_size_error=pl.col("size_error").mean(),
            mean_capacity_error_points=pl.col("capacity_error").mean() * 100,
        )
        .sort("arm", "after_checks", "shape")
    )


def rejection_share(*, tables: Path) -> pl.DataFrame | None:
    """Return the share of events a hard check turned from a hit into a miss.

    Args:
        tables: The tables directory.

    Returns:
        One row per arm and sample, or `None` if no event scores exist.
    """
    path = tables / "event_scores.parquet"
    if not path.exists():
        return None
    scores = pl.read_parquet(path)
    before = scores.filter(~pl.col("after_checks")).select(
        "arm", "sample", "draw", "plant", "shape", "size", hit_before="hit"
    )
    after = scores.filter(pl.col("after_checks")).select(
        "arm", "sample", "draw", "plant", "shape", "size", hit_after="hit"
    )
    joined = before.join(after, on=["arm", "sample", "draw", "plant", "shape", "size"])
    return joined.group_by("arm", "sample").agg(
        events=pl.len(),
        hits_before_checks=pl.col("hit_before").sum(),
        hits_after_checks=pl.col("hit_after").sum(),
        lost_to_rejection=(pl.col("hit_before") & ~pl.col("hit_after")).sum(),
    )


def round2_sections(*, root: Path) -> list[str]:
    """Return the report's round-2 sections (post hoc), or an empty list before round 2 ran.

    Args:
        root: The output root.

    Returns:
        Sections for the budget, the contrasts, the event scores, the controls, false movement,
        the matching-candidate cases of round 1, and round 2's quoted reasons for no change.
    """
    tables = root / "tables_round2"
    if not tables.exists():
        return []
    parts = ["# Round 2 (post hoc)\n"]
    estimate = tables.parent / "tables" / "round2_budget_estimate.json"
    if estimate.exists():
        data = json.loads(estimate.read_text())
        data.pop("groups", None)
        parts.append(
            _section(
                title="Round 2 budget estimate", body=f"```json\n{json.dumps(data, indent=2)}\n```"
            )
        )
    summary = event_summary(tables=tables)
    parts.append(
        _section(
            title="Round 2 event scores by arm and shape (new draws 03 and 04)",
            body="_missing_" if summary is None else _markdown(frame=summary),
        )
    )
    for title, name in (
        (
            "Round 2 post hoc contrasts (points of capacity; negative favours the left arm)",
            "contrasts.parquet",
        ),
        (
            "Round 2 controls, and round 1's control rescored per run of adjacent months",
            "controls.parquet",
        ),
        ("Round 2 false movement on the 23 quiet plant-months", "false_movement.parquet"),
        (
            "Round 1: planted losses with a matching candidate that the model left unmarked",
            "matching_candidate_cases.parquet",
        ),
        ("Round 2: the model's note on why there was no change", "round2_no_change_notes.parquet"),
    ):
        parts.append(_table_section(title=title, path=tables / name))
    return parts


def round3_sections(*, root: Path) -> list[str]:
    """Return round 3's sections and the re-scores of rounds 1 and 2, or an empty list.

    Args:
        root: The output root.

    Returns:
        Sections for the hit rates with Wilson intervals, the placebo hit rates, the hit rates by
        planted size, the robust contrast intervals, round 3's contrast, the noise floor, the tally
        of round 2's notes, the candidate coverage of round 1, and the real low months.
    """
    tables = root / "tables_round3"
    if not tables.exists():
        return []
    parts = [
        "# Rounds 3 and 4, the easy case, and the re-scores (post hoc, bar the amended check)\n"
    ]
    for title, folder, name in (
        ("Hit rates with Wilson 95% intervals, by round and by check", tables, "hit_rates.parquet"),
        (
            "Placebo hit rates: null-record answers scored against each draw's event dates",
            tables,
            "placebo.parquet",
        ),
        (
            "Hit rates by planted size, real and placebo (round 2 draws)",
            tables,
            "size_bands.parquet",
        ),
        (
            "Contrasts with intervals that resample events, and calendar months, with the powering",
            tables,
            "contrasts_robust.parquet",
        ),
        (
            "Round 3: the amended-check model against the tools-only rule on fresh draws 05 and 06",
            tables,
            "round3_contrast.parquet",
        ),
        (
            "Noise floor: the mean absolute difference between the two samples",
            tables,
            "noise_floor.parquet",
        ),
        (
            "Candidate coverage: planted losses with a candidate or scan episode at the start",
            tables,
            "candidate_coverage_all.parquet",
        ),
        (
            "Round 1 candidate coverage: events with a candidate near the start and the end",
            root / "tables_round2",
            "candidate_coverage.parquet",
        ),
        (
            "Round 2 series on the real low months, share of static power (opinion, not truth)",
            tables,
            "real_low_months_round2.parquet",
        ),
    ):
        parts.append(_table_section(title=title, path=folder / name))
    tally = tables / "notes_tally.json"
    if tally.exists():
        parts.append(
            _section(
                title="Round 2 notes on why there was no change: tally",
                body=f"```json\n{tally.read_text()}\n```",
            )
        )
    return parts


def main() -> int:  # noqa: PLR0915 - one linear script of report sections
    """Write and print the report."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--smoke", action="store_true")
    arguments = parser.parse_args()
    root = claim_root(smoke=arguments.smoke)
    tables = tables_dir(root=root)
    parts = ["# Report: can a language model label a solar plant's capacity changes?\n"]
    parts.append(
        "**Round 1 was planned and ran as designed. Round 2 is post hoc: its tools and prompt were "
        "changed after round 1's draws 01 and 02, and it is judged on new draws 03 and 04. In "
        "round "
        "3 only the amended step check was pre-registered; the easy case, round 4, and the "
        "re-scores are post hoc.**\n"
    )
    if arguments.smoke:
        parts.append(
            "**Smoke run: a stub model answered every prompt. Nothing here is a result.**\n"
        )
    if PROBE_DIR.exists():
        parts.append(_section(title="Probe of `claude -p`", body=_markdown(frame=summarise())))
    estimate = tables / "budget_estimate.json"
    if estimate.exists():
        data = json.loads(estimate.read_text())
        groups = pl.DataFrame(
            [{"group": key, **value} for key, value in data.pop("groups").items()]
        )
        parts.append(
            _section(
                title="Budget estimate",
                body=f"```json\n{json.dumps(data, indent=2)}\n```\n\n{_markdown(frame=groups)}",
            )
        )
    part1 = root / "part1"
    parts.append(
        _section(
            title="Part 1 chosen settings",
            body=f"```json\n{(part1 / 'chosen.json').read_text()}\n```"
            if (part1 / "chosen.json").exists()
            else "_missing_",
        )
    )
    for title, name in (
        ("Part 1 tuning (tuning draw)", "tuning.parquet"),
        ("Part 1 summary (fresh draws and the 23 quiet plant-months)", "summary.parquet"),
        (
            "Part 1 mean error by group (fresh draws, and the first study's draws, exploratory)",
            "summary_by_group.parquet",
        ),
    ):
        parts.append(_table_section(title=title, path=part1 / name))
    first_study = FIRST_STUDY_BUILT.parent / "tables" / "posthoc_stage_a.parquet"
    if first_study.exists():
        published = pl.read_parquet(first_study).select(
            "estimator",
            "mean_selection_error",
            "false_movement_real_windows",
            "mean_step_detection_delay_days",
        )
        parts.append(
            _section(
                title="The first study's published figures (its ten selection draws)",
                body=_markdown(frame=published),
            )
        )
    for title, name in (
        (
            "Planned contrasts C1 and C2 (points of capacity; negative favours the model)",
            "contrasts.parquet",
        ),
        ("Controls", "controls.parquet"),
        ("C3: false movement on the 23 quiet plant-months", "false_movement.parquet"),
    ):
        parts.append(_table_section(title=title, path=tables / name))
    controls = tables / "controls.parquet"
    if controls.exists():
        hit = (
            pl.read_parquet(controls)
            .filter((pl.col("arm") == "tools_llm") & pl.col("hit_rate").is_not_null())["hit_rate"]
            .to_list()
        )
        gate = "no hit rate" if not hit else ("passed" if hit[0] >= CONTROL_HIT_GATE else "failed")
        parts.append(
            _section(
                title="Positive-control gate",
                body=f"Tools plus model must find {CONTROL_HIT_GATE:.0%} of the 30% steps: {gate}. "
                "If it fails, C1 to C3 are untested, not null.",
            )
        )
    summary = event_summary(tables=tables)
    parts.append(
        _section(
            title="Event scores by arm and shape (fresh draws)",
            body="_missing_" if summary is None else _markdown(frame=summary),
        )
    )
    rejected = rejection_share(tables=tables)
    parts.append(
        _section(
            title="Events lost to the hard checks",
            body="_missing_" if rejected is None else _markdown(frame=rejected),
        )
    )
    floor = tables / "noise_floor.json"
    parts.append(
        _section(
            title="Noise floor and missing answers",
            body=f"```json\n{floor.read_text()}\n```" if floor.exists() else "_missing_",
        )
    )
    for title, name in (
        (
            "Agreement with the model's hindsight labels on the real record (opinion, not truth)",
            "agreement.parquet",
        ),
        (
            "Real low-output months: mean capacity as a share of static power",
            "real_low_months.parquet",
        ),
        ("Causal role: stability and failed checks (exploratory)", "causal.parquet"),
        (
            "Plant under active network management, tools only, power with capped hours left in",
            "anm_spot_check.parquet",
        ),
    ):
        parts.append(_table_section(title=title, path=tables / name))
    forecast = root / "forecast" / "outputs" / "losses_forecast_arms.parquet"
    if forecast.exists():
        losses = pl.read_parquet(forecast)
        by_arm = (
            losses.group_by("arm")
            .agg(
                rows=pl.len(),
                mean_error_points=pl.col(SCORED_COLUMN).mean() * 100,
            )
            .sort("arm")
        )
        base = by_arm.filter(pl.col("arm") == "b0_refit")["mean_error_points"].to_list()
        if base:
            by_arm = by_arm.with_columns(minus_b0_points=pl.col("mean_error_points") - base[0])
        parts.append(
            _section(
                title="Forecast-level check (exploratory): error in MW over static power, all rows",
                body=_markdown(frame=by_arm),
            )
        )
    log = log_path(root=root)
    if log.exists():
        entries = [json.loads(line) for line in log.read_text().splitlines()]
        cost = pl.DataFrame(entries).select("model", "input_tokens", "output_tokens", "cost_usd")
        total = cost.group_by("model").agg(
            calls=pl.len(),
            input_tokens=pl.col("input_tokens").sum(),
            output_tokens=pl.col("output_tokens").sum(),
            cost_usd=pl.col("cost_usd").sum(),
        )
        parts.append(
            _section(
                title="Calls made and their cost (from the log, which stays under data/)",
                body=_markdown(frame=total),
            )
        )
    parts += round2_sections(root=root)
    parts += round3_sections(root=root)
    text = "\n".join(parts)
    (root / "report.md").write_text(text)
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
