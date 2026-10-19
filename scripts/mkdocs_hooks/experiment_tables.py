"""MkDocs hook that builds the experiment tables from the front matter of every `result.md`.

A theme page under `docs/ml-experiments/themes/<theme>/` carries one placeholder line, and this
hook replaces it with a table at build time, so concurrent experiment pull requests never conflict
on a committed table:

- `<!-- experiments-table -->` in `experiments.md`: every finished experiment whose `theme` is the
  page's theme or whose `also_relevant_to` lists it.
- `<!-- planned-table -->` in `plan.md`: every experiment with `status: planned` in the same way.
- `<!-- candidates-table -->` in `docs/ml-experiments/README.md`: every finished experiment with
  `production_status: candidate`.

The hook also validates every front matter block in `on_config`, so a bad block fails
`mkdocs build --strict` whether or not any page renders it. The same checks run from the command
line: `--check` validates, and `--theme <theme>` prints a theme's experiments table as markdown.
This is docs tooling with no production caller, so it raises rather than degrades.
"""

import argparse
import re
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import yaml

REPO_ROOT: Final[Path] = Path(__file__).resolve().parents[2]
EXPERIMENTS_DIR: Final[Path] = REPO_ROOT / "docs" / "ml-experiments"
"""Folder holding the experiment folders (`<id>-<slug>/result.md`) and `themes/`."""

VERDICTS: Final[frozenset[str]] = frozenset(
    {"inconclusive", "refuted", "supported", "partly supported", "small gain", "not supported"}
)
PRODUCTION_STATUSES: Final[frozenset[str]] = frozenset(
    {"candidate", "not-for-production", "adopted"}
)
STATUSES: Final[frozenset[str]] = frozenset({"planned", "done", "finished"})
FIRST_ID_NEEDING_TAKE_HOME_MESSAGE: Final[int] = 103
"""Older experiments show the first sentence of `headline` where `take_home_message` is absent."""

EXPERIMENTS_PLACEHOLDER: Final[str] = "<!-- experiments-table -->"
PLANNED_PLACEHOLDER: Final[str] = "<!-- planned-table -->"
CANDIDATES_PLACEHOLDER: Final[str] = "<!-- candidates-table -->"

_FRONT_MATTER: Final[re.Pattern[str]] = re.compile(r"\A---\n(.*?)\n---\n", re.DOTALL)
_FIRST_SENTENCE: Final[re.Pattern[str]] = re.compile(r"^.*?[.!?](?=\s|$)", re.DOTALL)


@dataclass(frozen=True)
class Experiment:
    """One `result.md` file: its folder name and raw front matter."""

    folder: str
    front_matter: dict[str, Any]

    @property
    def number(self) -> int:
        """The integer id, read from the folder name so a bad `id` field cannot break sorting."""
        return int(self.folder.split("-", maxsplit=1)[0])

    @property
    def is_planned(self) -> bool:
        """Whether the experiment is only planned and has no result yet."""
        return self.front_matter.get("status") == "planned"

    def themes(self) -> list[str]:
        """The primary theme followed by the `also_relevant_to` themes, without repeats."""
        primary = self.front_matter.get("theme")
        also = self.front_matter.get("also_relevant_to") or []
        return [theme for theme in [primary, *also] if theme]

    def take_home(self) -> str:
        """The take-home message, or the first sentence of `headline` for older experiments."""
        message = self.front_matter.get("take_home_message")
        if message:
            return " ".join(str(message).split())
        headline = " ".join(str(self.front_matter.get("headline") or "").split())
        match = _FIRST_SENTENCE.match(headline)
        return match.group(0) if match else headline


def _error(experiment: Experiment, message: str) -> str:
    return f"{experiment.folder}/result.md: {message}"


def load_experiments(*, experiments_dir: Path) -> list[Experiment]:
    """Read the front matter of every `<id>-<slug>/result.md`, sorted by id descending.

    Args:
        experiments_dir: The folder holding the experiment folders.

    Returns:
        The experiments, highest id first.

    Raises:
        ValueError: If a `result.md` has no front matter block.
        TypeError: If the front matter block is not a mapping.
    """
    experiments = []
    for path in experiments_dir.glob("[0-9]*/result.md"):
        match = _FRONT_MATTER.match(path.read_text())
        if match is None:
            raise ValueError(f"{path}: no front matter block")
        front_matter = yaml.safe_load(match.group(1))
        if not isinstance(front_matter, dict):
            raise TypeError(f"{path}: front matter is not a mapping")
        experiments.append(Experiment(folder=path.parent.name, front_matter=front_matter))
    return sorted(experiments, key=lambda experiment: experiment.number, reverse=True)


def validate_experiments(*, experiments: Sequence[Experiment], experiments_dir: Path) -> list[str]:
    """Check every front matter block against the rules, naming the file and the field.

    Args:
        experiments: The experiments to check.
        experiments_dir: The folder whose `themes/` subfolder must hold every theme named.

    Returns:
        One message per problem, empty when every block is valid.
    """
    errors = []
    for experiment in experiments:
        fields = experiment.front_matter
        status = fields.get("status")
        if status not in STATUSES:
            errors.append(
                _error(experiment, f"`status` {status!r} is not one of {sorted(STATUSES)}")
            )
        if str(fields.get("id")) != experiment.folder.split("-", maxsplit=1)[0]:
            errors.append(
                _error(experiment, f"`id` {fields.get('id')!r} does not match the folder")
            )
        if not fields.get("theme"):
            errors.append(_error(experiment, "missing `theme`"))
        errors.extend(
            _error(experiment, f"theme {theme!r} has no folder under themes/")
            for theme in experiment.themes()
            if not (experiments_dir / "themes" / theme).is_dir()
        )
        if status in ("done", "finished"):
            if fields.get("verdict") not in VERDICTS:
                errors.append(
                    _error(
                        experiment,
                        f"`verdict` {fields.get('verdict')!r} is not one of {sorted(VERDICTS)}",
                    )
                )
            if fields.get("production_status") not in PRODUCTION_STATUSES:
                errors.append(
                    _error(
                        experiment,
                        f"`production_status` {fields.get('production_status')!r} is not one of "
                        f"{sorted(PRODUCTION_STATUSES)}",
                    )
                )
        needs_take_home = experiment.number >= FIRST_ID_NEEDING_TAKE_HOME_MESSAGE
        if needs_take_home and status != "planned" and not fields.get("take_home_message"):
            errors.append(_error(experiment, "missing `take_home_message`"))
    return errors


def _cell(text: str) -> str:
    return " ".join(text.split()).replace("|", "\\|")


def _table(*, headers: Sequence[str], rows: Sequence[Sequence[str]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(_cell(cell) for cell in row) + " |" for row in rows),
    ]
    return "\n".join(lines) + "\n"


def _link(experiment: Experiment, *, link_prefix: str) -> str:
    folder = experiment.folder
    return f"[{folder.replace('-', ' ', 1)}]({link_prefix}{folder}/result.md)"


def experiments_table(
    *, experiments: Sequence[Experiment], theme: str, link_prefix: str = "../../"
) -> str:
    """Markdown table of the finished experiments relevant to `theme`, highest id first."""
    rows = []
    for experiment in experiments:
        if experiment.is_planned or theme not in experiment.themes():
            continue
        fields = experiment.front_matter
        rows.append(
            [
                _link(experiment, link_prefix=link_prefix),
                str(fields.get("hypothesis", "")),
                str(fields.get("verdict", "")),
                str(fields.get("production_status", "")),
                experiment.take_home(),
                "primary" if fields.get("theme") == theme else "also relevant",
            ]
        )
    headers = [
        "Experiment",
        "Hypothesis",
        "Verdict",
        "Production status",
        "Take-home message",
        "Relevance",
    ]
    return _table(headers=headers, rows=rows)


def planned_table(
    *, experiments: Sequence[Experiment], theme: str, link_prefix: str = "../../"
) -> str:
    """Markdown table of the planned experiments relevant to `theme`, highest id first."""
    rows = [
        [
            _link(experiment, link_prefix=link_prefix),
            str(experiment.front_matter.get("hypothesis", "")),
            "primary" if experiment.front_matter.get("theme") == theme else "also relevant",
        ]
        for experiment in experiments
        if experiment.is_planned and theme in experiment.themes()
    ]
    return _table(headers=["Experiment", "Hypothesis", "Relevance"], rows=rows)


def candidates_table(*, experiments: Sequence[Experiment], link_prefix: str = "") -> str:
    """Markdown table of every finished experiment with `production_status: candidate`."""
    rows = [
        [
            _link(experiment, link_prefix=link_prefix),
            str(experiment.front_matter.get("theme", "")),
            str(experiment.front_matter.get("verdict", "")),
            experiment.take_home(),
        ]
        for experiment in experiments
        if not experiment.is_planned
        and experiment.front_matter.get("production_status") == "candidate"
    ]
    return _table(headers=["Experiment", "Theme", "Verdict", "Take-home message"], rows=rows)


def _replace_placeholder(*, markdown: str, placeholder: str, table: str, src_path: str) -> str:
    if placeholder not in markdown:
        raise ValueError(f"{src_path}: the line {placeholder} is missing, so no table can be built")
    return markdown.replace(placeholder, table.rstrip("\n"))


def fill_placeholders(*, markdown: str, src_path: str, experiments: Sequence[Experiment]) -> str:
    """Replace the page's placeholder line with its table; leave any other page unchanged.

    Args:
        markdown: The page's markdown source.
        src_path: The page's path relative to `docs/`, with forward slashes.
        experiments: Every experiment, as returned by `load_experiments`.

    Returns:
        The markdown with the table in place of the placeholder line.

    Raises:
        ValueError: If a theme or Overview page lacks its placeholder line.
    """
    path = Path(src_path)
    if src_path == "ml-experiments/README.md":
        return _replace_placeholder(
            markdown=markdown,
            placeholder=CANDIDATES_PLACEHOLDER,
            table=candidates_table(experiments=experiments),
            src_path=src_path,
        )
    if path.parent.parent == Path("ml-experiments/themes"):
        theme = path.parent.name
        if path.name == "experiments.md":
            return _replace_placeholder(
                markdown=markdown,
                placeholder=EXPERIMENTS_PLACEHOLDER,
                table=experiments_table(experiments=experiments, theme=theme),
                src_path=src_path,
            )
        if path.name == "plan.md":
            return _replace_placeholder(
                markdown=markdown,
                placeholder=PLANNED_PLACEHOLDER,
                table=planned_table(experiments=experiments, theme=theme),
                src_path=src_path,
            )
    return markdown


def on_config(config: Any) -> Any:
    """MkDocs hook: fail the build if any `result.md` front matter is invalid."""
    experiments = load_experiments(experiments_dir=EXPERIMENTS_DIR)
    errors = validate_experiments(experiments=experiments, experiments_dir=EXPERIMENTS_DIR)
    if errors:
        raise ValueError("Invalid experiment front matter:\n" + "\n".join(errors))
    return config


def on_page_markdown(markdown: str, page: Any, config: Any, files: Any) -> str:
    """MkDocs hook: fill the page's table placeholder line."""
    return fill_placeholders(
        markdown=markdown,
        src_path=page.file.src_path.replace("\\", "/"),
        experiments=load_experiments(experiments_dir=EXPERIMENTS_DIR),
    )


def on_serve(server: Any, config: Any, builder: Any) -> Any:
    """MkDocs hook: rebuild the tables when any experiment folder changes."""
    server.watch(str(EXPERIMENTS_DIR))
    return server


def main(argv: Sequence[str] | None = None, *, experiments_dir: Path = EXPERIMENTS_DIR) -> int:
    """Command line entry point: `--check` validates, `--theme <theme>` prints a table.

    Args:
        argv: The arguments, defaulting to `sys.argv[1:]`.
        experiments_dir: The folder holding the experiment folders.

    Returns:
        The process exit code: 1 when `--check` finds a problem, otherwise 0.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--check", action="store_true", help="validate every front matter block")
    group.add_argument("--theme", help="print this theme's experiments table as markdown")
    args = parser.parse_args(argv)
    experiments = load_experiments(experiments_dir=experiments_dir)
    if args.check:
        errors = validate_experiments(experiments=experiments, experiments_dir=experiments_dir)
        print("\n".join(errors))
        return 1 if errors else 0
    print(experiments_table(experiments=experiments, theme=args.theme))
    return 0


if __name__ == "__main__":
    sys.exit(main())
