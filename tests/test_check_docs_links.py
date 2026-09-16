"""Tests for `scripts/check_docs_links.py`.

The script resolves an anchor by converting the page with the real `markdown.Markdown`
converter rather than a hand-rolled slugify, because Python-Markdown's `toc` extension preserves
underscores — a guessed `_` -> `-` rule produced 15 false failures the first time this was tried.
`test_underscore_anchor_resolves` is the regression for that guessed slug rule.

Each test builds a throwaway `mkdocs.yml` + `docs/` tree under `tmp_path` rather than depending on
the real docs staying put. Most call `main()` with explicit file arguments;
`test_whole_repo_scan_finds_a_bad_link` covers the no-argument path CI uses, which needs a real
git repository to list.
"""

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType
from typing import Final

import pytest

REPO_ROOT: Final[Path] = Path(__file__).parent.parent
"""The repo root, one level above this `tests/` directory."""

SCRIPT_PATH: Final[Path] = REPO_ROOT / "scripts" / "check_docs_links.py"
"""The script under test, imported by path because `scripts/` is not an importable package."""


def _load_script() -> ModuleType:
    """Import `check_docs_links.py` from its path in `scripts/`."""
    spec = importlib.util.spec_from_file_location("check_docs_links", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


check_docs_links = _load_script()

SITE_PREFIX: Final[str] = check_docs_links.DOCS_SITE_PREFIX


def _make_docs_site(tmp_path: Path) -> None:
    """Write a minimal `mkdocs.yml` + `docs/` tree, exercising both the file and folder URL forms.

    `docs/architecture/performance.md` is the `<path>.md` form and carries a plain heading plus one
    whose id contains an underscore. `docs/guide/index.md` is the `<path>/index.md` form, which most
    of the real `docs/` tree uses, and its heading slugifies to an id *ending* in an underscore.
    `docs/api/contracts.md` mimics a real mkdocstrings page.
    """
    (tmp_path / "mkdocs.yml").write_text("markdown_extensions: []\n", encoding="utf-8")
    docs = tmp_path / "docs"
    (docs / "architecture").mkdir(parents=True)
    (docs / "api").mkdir(parents=True)
    (docs / "index.md").write_text("# Home\n", encoding="utf-8")
    (docs / "architecture" / "performance.md").write_text(
        "# Performance and Scale\n\n"
        "## Tuning\n\n"
        "some text\n\n"
        "## register_experiment_job\n\n"
        "some text\n",
        encoding="utf-8",
    )
    (docs / "api" / "contracts.md").write_text(
        "# Contracts API\n\n::: contracts.common\n", encoding="utf-8"
    )
    (docs / "guide").mkdir(parents=True)
    (docs / "guide" / "index.md").write_text(
        "# Guide\n\n## Storage roots (`DATA_STORE_*`)\n\nsome text\n", encoding="utf-8"
    )


def _write_consumer(tmp_path: Path, text: str) -> Path:
    """Write `text` to a throwaway file the checker will scan, and return its path."""
    consumer = tmp_path / "consumer.md"
    consumer.write_text(text, encoding="utf-8")
    return consumer


@pytest.fixture
def docs_site(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """A throwaway `mkdocs.yml` + `docs/` tree, with the cwd pointed at it."""
    _make_docs_site(tmp_path)
    monkeypatch.chdir(tmp_path)
    return tmp_path


def test_trailing_full_stop_and_markdown_link_both_resolve(docs_site: Path) -> None:
    consumer = _write_consumer(
        docs_site,
        f"See {SITE_PREFIX}architecture/performance/.\n\n"
        f"Also [tuning]({SITE_PREFIX}architecture/performance/#tuning) for detail.\n",
    )
    exit_code = check_docs_links.main([str(consumer)])
    assert exit_code == 0


def test_url_before_closing_triple_quote_resolves(docs_site: Path) -> None:
    """Regression: a Python docstring closing right after a URL leaves three trailing quotes.

    Stripping one quote at a time and re-checking parity after each removal flips odd to even
    after the first strip, so a naive implementation stops with two spurious quotes still
    attached and reports a bad anchor for a link that is actually fine.
    """
    consumer = _write_consumer(
        docs_site,
        f'rule: <{SITE_PREFIX}architecture/performance/#tuning>."""\n',
    )
    exit_code = check_docs_links.main([str(consumer)])
    assert exit_code == 0


def test_underscore_anchor_resolves(docs_site: Path) -> None:
    consumer = _write_consumer(
        docs_site,
        f"{SITE_PREFIX}architecture/performance/#register_experiment_job\n",
    )
    exit_code = check_docs_links.main([str(consumer)])
    assert exit_code == 0


def test_bad_page_is_reported_and_fails(
    docs_site: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    consumer = _write_consumer(docs_site, f"{SITE_PREFIX}no-such-page/\n")
    exit_code = check_docs_links.main([str(consumer)])
    assert exit_code == 1
    out = capsys.readouterr().out
    assert "BAD PAGE" in out
    assert f"{SITE_PREFIX}no-such-page/" in out


def test_bad_anchor_is_reported_and_fails(
    docs_site: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    consumer = _write_consumer(
        docs_site, f"{SITE_PREFIX}architecture/performance/#no-such-anchor\n"
    )
    exit_code = check_docs_links.main([str(consumer)])
    assert exit_code == 1
    out = capsys.readouterr().out
    assert "BAD ANCHOR" in out


def test_mkdocstrings_page_skips_anchor_check(
    docs_site: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    consumer = _write_consumer(docs_site, f"{SITE_PREFIX}api/contracts/#anything\n")
    exit_code = check_docs_links.main([str(consumer)])
    assert exit_code == 0
    out = capsys.readouterr().out
    assert "SKIPPED" in out


@pytest.mark.parametrize(
    "template",
    [
        "Have you read {url}?",
        "Go and read {url}!",
        "**{url}**",
        "*{url}*",
        "_{url}_",
        "| {url} |",
        "|{url}|",
        "**[tuning]({url})**",
    ],
)
def test_link_resolves_through_surrounding_punctuation(docs_site: Path, template: str) -> None:
    """Punctuation a regex sweep picks up must not be mistaken for part of the anchor.

    Every case here is a false positive rather than a false negative: the link is good and a
    sloppy stripper reports it broken. That is the failure that gets the hook deleted, because
    the first person it fires on cannot commit and can see the link is fine.
    """
    url = f"{SITE_PREFIX}architecture/performance/#tuning"
    consumer = _write_consumer(docs_site, template.format(url=url) + "\n")
    assert check_docs_links.main([str(consumer)]) == 0


def test_whole_repo_scan_finds_a_bad_link(
    docs_site: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """With no arguments the checker scans every git-tracked file, which is how CI runs it."""
    _write_consumer(docs_site, f"{SITE_PREFIX}no-such-page/\n")
    for command in (["git", "init", "-q"], ["git", "add", "-A"]):
        subprocess.run(command, cwd=docs_site, check=True)

    assert check_docs_links.main([]) == 1
    assert "BAD PAGE" in capsys.readouterr().out


def test_folder_form_page_resolves(docs_site: Path) -> None:
    """`docs/guide/index.md` is reached as `guide/`, the form most of the real docs tree uses."""
    consumer = _write_consumer(docs_site, f"{SITE_PREFIX}guide/\n")
    assert check_docs_links.main([str(consumer)]) == 0


def test_site_root_resolves(docs_site: Path) -> None:
    """The bare site root is `docs/index.md`."""
    consumer = _write_consumer(docs_site, f"{SITE_PREFIX}\n")
    assert check_docs_links.main([str(consumer)]) == 0


def test_anchor_ending_in_underscore_resolves(docs_site: Path) -> None:
    """Regression: peeling trailing punctuation up front breaks a real anchor.

    Python-Markdown's slugify drops the `*` from a heading like ``Storage roots (`DATA_STORE_*`)``
    and leaves the underscore before it, so the anchor genuinely ends in `_`. The live docs carry
    one of these on the AWS setup page. Trying the URL exactly as written before peeling anything
    is what keeps it working.
    """
    consumer = _write_consumer(docs_site, f"{SITE_PREFIX}guide/#storage-roots-data_store_\n")
    assert check_docs_links.main([str(consumer)]) == 0


def test_wrapped_url_is_reported(docs_site: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A URL reflowed across two lines loses its anchor, so the link is broken for the reader too.

    The page half still resolves, so without this check the scan passes a link whose anchor was
    never looked at.
    """
    consumer = _write_consumer(
        docs_site,
        f"See {SITE_PREFIX}architecture/performance/\n#no-such-anchor for detail.\n",
    )
    assert check_docs_links.main([str(consumer)]) == 1
    assert "WRAPPED URL" in capsys.readouterr().out


def test_bare_hash_comment_line_is_not_a_wrapped_url(docs_site: Path) -> None:
    """A commented file header puts a bare `#` under a link; that is a separator, not an anchor."""
    consumer = _write_consumer(
        docs_site, f"# See {SITE_PREFIX}architecture/performance/\n#\n# More prose.\n"
    )
    assert check_docs_links.main([str(consumer)]) == 0


def test_http_scheme_is_checked_not_skipped(
    docs_site: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """A link typo'd as `http://` still points at a page that can be renamed away."""
    consumer = _write_consumer(docs_site, f"{SITE_PREFIX.replace('https', 'http')}no-such-page/\n")
    assert check_docs_links.main([str(consumer)]) == 1
    assert "BAD PAGE" in capsys.readouterr().out


def test_bad_anchor_names_the_closest_real_anchor(
    docs_site: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """The suggestion is the whole value of the failure message, so assert on its content."""
    consumer = _write_consumer(docs_site, f"{SITE_PREFIX}architecture/performance/#tunning\n")
    assert check_docs_links.main([str(consumer)]) == 1
    assert "tuning" in capsys.readouterr().out


def test_mkdocs_yml_extension_entries_are_parsed(tmp_path: Path) -> None:
    """`markdown_extensions` mixes bare names with single-key `{name: config}` mappings.

    Reading both forms is what keeps this script's anchors identical to the built site's, so it is
    worth a test that does not go through the `markdown_extensions: []` fixture.
    """
    names, configs = check_docs_links._markdown_extensions_from_mkdocs_yml(
        {"markdown_extensions": ["pymdownx.highlight", {"pymdownx.arithmatex": {"generic": True}}]}
    )
    assert names == [
        *check_docs_links._BUILTIN_MARKDOWN_EXTENSIONS,
        "pymdownx.highlight",
        "pymdownx.arithmatex",
    ]
    assert configs == {"pymdownx.arithmatex": {"generic": True}}
