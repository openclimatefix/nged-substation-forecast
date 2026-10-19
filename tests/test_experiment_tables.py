"""Tests for `scripts/mkdocs_hooks/experiment_tables.py`.

Each test builds a throwaway `docs/ml-experiments/` tree under `tmp_path`, so none of them depends
on the real experiments staying put. The strict MkDocs build in CI runs the same validation over
the real tree.
"""

from pathlib import Path
from typing import Any

import experiment_tables
import pytest
import yaml


def _write_experiment(experiments_dir: Path, experiment_id: str, **overrides: Any) -> Path:
    front_matter: dict[str, Any] = {
        "id": experiment_id,
        "slug": f"slug-{experiment_id}",
        "status": "done",
        "theme": "alpha",
        "also_relevant_to": [],
        "hypothesis": f"Hypothesis {experiment_id}.",
        "verdict": "supported",
        "production_status": "not-for-production",
        "take_home_message": f"Take-home {experiment_id}.",
    }
    front_matter.update(overrides)
    front_matter = {key: value for key, value in front_matter.items() if value is not None}
    folder = experiments_dir / f"{experiment_id}-{front_matter['slug']}"
    folder.mkdir(parents=True)
    result = folder / "result.md"
    result.write_text(f"---\n{yaml.safe_dump(front_matter)}---\n\nBody.\n")
    return result


@pytest.fixture
def experiments_dir(tmp_path: Path) -> Path:
    directory = tmp_path / "docs" / "ml-experiments"
    for theme in ("alpha", "beta"):
        (directory / "themes" / theme).mkdir(parents=True)
    return directory


def _table(experiments_dir: Path, theme: str) -> str:
    experiments = experiment_tables.load_experiments(experiments_dir=experiments_dir)
    return experiment_tables.experiments_table(experiments=experiments, theme=theme)


def test_table_rows_and_order(experiments_dir: Path) -> None:
    _write_experiment(experiments_dir, "00010", verdict="refuted")
    _write_experiment(experiments_dir, "00012", theme="beta", also_relevant_to=["alpha"])
    _write_experiment(experiments_dir, "00011", theme="beta")

    rows = _table(experiments_dir, "alpha").splitlines()[2:]

    assert len(rows) == 2
    assert rows[0].startswith("| [00012 slug-00012](../../00012-slug-00012/result.md) |")
    assert "also relevant" in rows[0]
    assert rows[1].startswith("| [00010 slug-00010](../../00010-slug-00010/result.md) |")
    assert "| refuted |" in rows[1]
    assert "Take-home 00010." in rows[1]
    assert "primary" in rows[1]
    assert "00011" not in "\n".join(rows)


def test_older_experiment_without_take_home_message_uses_headline(experiments_dir: Path) -> None:
    _write_experiment(
        experiments_dir,
        "00050",
        take_home_message=None,
        headline="First sentence. Second sentence.",
    )

    row = _table(experiments_dir, "alpha").splitlines()[2]

    assert "First sentence." in row
    assert "Second sentence" not in row


def test_check_fails_on_bad_front_matter(experiments_dir: Path) -> None:
    cases = {
        "00001": {"theme": None},
        "00002": {"theme": "no-such-theme"},
        "00003": {"verdict": "great"},
        "00103": {"take_home_message": None},
        "00004": {"also_relevant_to": ["no-such-theme"]},
    }
    for experiment_id, overrides in cases.items():
        _write_experiment(experiments_dir, experiment_id, **overrides)

    errors = experiment_tables.validate_experiments(
        experiments=experiment_tables.load_experiments(experiments_dir=experiments_dir),
        experiments_dir=experiments_dir,
    )

    assert len(errors) == len(cases)
    for experiment_id in cases:
        assert any(f"{experiment_id}-slug-{experiment_id}" in error for error in errors)


def test_check_mode_exits_non_zero_and_names_the_file(
    experiments_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_experiment(experiments_dir, "00003", verdict="great")

    exit_code = experiment_tables.main(["--check"], experiments_dir=experiments_dir)

    assert exit_code == 1
    assert "00003-slug-00003/result.md" in capsys.readouterr().out


def test_check_mode_passes_on_good_front_matter(experiments_dir: Path) -> None:
    _write_experiment(experiments_dir, "00003")

    assert experiment_tables.main(["--check"], experiments_dir=experiments_dir) == 0


def test_theme_print_mode_prints_the_table(
    experiments_dir: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    _write_experiment(experiments_dir, "00003")

    exit_code = experiment_tables.main(["--theme", "alpha"], experiments_dir=experiments_dir)

    assert exit_code == 0
    assert capsys.readouterr().out.strip() == _table(experiments_dir, "alpha").strip()


def test_candidates_table_lists_only_candidates(experiments_dir: Path) -> None:
    _write_experiment(experiments_dir, "00003", production_status="candidate")
    _write_experiment(experiments_dir, "00004")

    table = experiment_tables.candidates_table(
        experiments=experiment_tables.load_experiments(experiments_dir=experiments_dir)
    )

    assert "00003" in table
    assert "00004" not in table


def test_sort_is_numeric_on_string_ids(experiments_dir: Path) -> None:
    _write_experiment(experiments_dir, "00099")
    _write_experiment(experiments_dir, "00100")

    rows = _table(experiments_dir, "alpha").splitlines()[2:]

    assert "00100" in rows[0]
    assert "00099" in rows[1]


def test_finished_status_is_valid(experiments_dir: Path) -> None:
    _write_experiment(experiments_dir, "00003", status="finished")
    _write_experiment(experiments_dir, "00004", production_status="adopted")

    assert experiment_tables.main(["--check"], experiments_dir=experiments_dir) == 0


def test_planned_result_needs_no_verdict(experiments_dir: Path) -> None:
    _write_experiment(
        experiments_dir,
        "00200",
        status="planned",
        verdict=None,
        production_status=None,
        take_home_message=None,
    )
    experiments = experiment_tables.load_experiments(experiments_dir=experiments_dir)

    assert experiment_tables.main(["--check"], experiments_dir=experiments_dir) == 0
    assert "00200" in experiment_tables.planned_table(experiments=experiments, theme="alpha")
    assert "00200" not in experiment_tables.experiments_table(
        experiments=experiments, theme="alpha"
    )


def test_primary_theme_is_not_repeated_as_also_relevant(experiments_dir: Path) -> None:
    _write_experiment(experiments_dir, "00003", also_relevant_to=["alpha", "beta"])

    rows = _table(experiments_dir, "alpha").splitlines()[2:]

    assert len(rows) == 1
    assert "also relevant" not in rows[0]


def test_missing_placeholder_is_an_error(experiments_dir: Path) -> None:
    experiments = experiment_tables.load_experiments(experiments_dir=experiments_dir)

    with pytest.raises(ValueError, match="experiments-table"):
        experiment_tables.fill_placeholders(
            markdown="# Experiments\n\nNo placeholder here.\n",
            src_path="ml-experiments/themes/alpha/experiments.md",
            experiments=experiments,
        )


def test_placeholder_is_replaced_on_theme_pages(experiments_dir: Path) -> None:
    _write_experiment(experiments_dir, "00003", production_status="candidate")
    experiments = experiment_tables.load_experiments(experiments_dir=experiments_dir)

    experiments_page = experiment_tables.fill_placeholders(
        markdown="# Experiments\n\n<!-- experiments-table -->\n",
        src_path="ml-experiments/themes/alpha/experiments.md",
        experiments=experiments,
    )
    overview = experiment_tables.fill_placeholders(
        markdown="<!-- candidates-table -->\n",
        src_path="ml-experiments/README.md",
        experiments=experiments,
    )
    other_page = experiment_tables.fill_placeholders(
        markdown="Nothing.\n", src_path="index.md", experiments=experiments
    )

    assert "(../../00003-slug-00003/result.md)" in experiments_page
    assert "(00003-slug-00003/result.md)" in overview
    assert other_page == "Nothing.\n"
