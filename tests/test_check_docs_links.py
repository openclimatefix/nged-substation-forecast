"""Tests for `scripts/check_docs_links.py`.

The script resolves an anchor by converting the page with the real `markdown.Markdown`
converter rather than a hand-rolled slugify, because Python-Markdown's `toc` extension preserves
underscores — a guessed `_` -> `-` rule produced 15 false failures the first time this was tried.
`test_underscore_anchor_resolves` is the regression for that.

Each test builds a throwaway `mkdocs.yml` + `docs/` tree under `tmp_path` rather than depending on
the real docs staying put, and calls `main()` directly with explicit file arguments so no test
needs a real git repository.
"""

import importlib.util
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

    `docs/architecture/performance.md` carries three headings: a plain one, one whose id contains
    an underscore, and one that only a folder-form API page's `:::` sibling should be confused
    with. `docs/api/contracts.md` mimics a real mkdocstrings page.
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
