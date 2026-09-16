"""`apply_findings.py` splices a sweep's replacement in without moving the page's markup.

The script lives in `.claude/skills/prose-review/scripts/`, which pytest does not collect — the
directory is hidden — so its tests live here, where CI runs them.

Every test below is a regression. A docs sweep found the script writing a serial comma *inside* the
code span it followed, `` `n_h3_cells,` `` for `` `n_h3_cells`, ``, in four places across four
files; the same boundary wrote a comma inside a bold span, refused edits that landed on a
`](url)`, and stripped the trailing newline off any file whose last paragraph was edited. All four
came from one cause: an offset map that recorded where each stripped character sat and not where
the markup around it ended.
"""

import importlib.util
import sys
import textwrap
from pathlib import Path
from types import ModuleType
from typing import Final

REPO_ROOT: Final[Path] = Path(__file__).parent.parent
"""The repo root, one level above this `tests/` directory."""

SCRIPT_PATH: Final[Path] = (
    REPO_ROOT / ".claude" / "skills" / "prose-review" / "scripts" / "apply_findings.py"
)
"""The script under test, imported by path because `.claude/` is not an importable package."""


def _load_script() -> ModuleType:
    """Import `apply_findings.py` from its path in the skill directory."""
    spec = importlib.util.spec_from_file_location("apply_findings", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


apply_findings = _load_script()

WRAP_WIDTH: Final[int] = 95
"""The width the fixtures below are wrapped at, inside the script's own 88-to-104 search range."""

MARKUP_COUNTS: Final[tuple[str, ...]] = ("**", "`", "[", "](")
"""The markers whose count a splice must never change, as `check_structure.py` counts them."""

EVERY_MARKUP: Final[str] = (
    "**Name the fault in the telemetry.** The grid carries `n_h3_cells`, `n_series` and `weight` "
    "columns, which the [operations page](https://example.com/ops) and the "
    "[aws page](https://example.com/aws) both describe, so the **on-call engineer** reads the "
    "alert rather than the logs when the overnight ingest degrades."
)
"""One paragraph carrying a bolded lead, three code spans, two links and a mid-sentence bold."""


def _write(path: Path, *paragraphs: str) -> str:
    """Write the paragraphs to `path`, hard-wrapped the way this repo's pages are, and return it."""
    wrapped = "\n\n".join("\n".join(textwrap.wrap(para, width=WRAP_WIDTH)) for para in paragraphs)
    raw = f"{wrapped}\n"
    path.write_text(raw, encoding="utf-8")
    return raw


def _apply(path: Path, raw: str, quote: str, replacement: str) -> tuple[str, str]:
    """Run one finding against `raw`, returning the new text and the status the script reported."""
    finding = {"file": str(path), "quote": quote, "replacement": replacement}
    return apply_findings.apply_one(raw=raw, finding=finding, merge_base=None)


def _counts(text: str) -> dict[str, int]:
    """How many of each markup marker `text` carries."""
    return {marker: text.count(marker) for marker in MARKUP_COUNTS}


def test_a_serial_comma_after_a_code_span_lands_outside_the_backticks(tmp_path: Path):
    path = tmp_path / "page.md"
    raw = _write(
        path,
        "The h3 grid carries `n_h3_cells`, `n_series` and `weight` columns for every cell, which "
        "the spatial join then uses to aggregate the gridded weather onto each substation feeder.",
    )
    updated, status = _apply(
        path,
        raw,
        "The h3 grid carries n_h3_cells, n_series and weight columns",
        "The h3 grid carries n_h3_cells, n_series, and weight columns",
    )
    assert status == "applied"
    assert "`n_series`, and `weight`" in updated
    assert "`n_series,`" not in updated
    assert _counts(updated) == _counts(raw)


def test_a_serial_comma_after_a_bold_span_lands_outside_the_markers(tmp_path: Path):
    path = tmp_path / "page.md"
    raw = _write(
        path,
        "Each feeder reading carries **power**, **voltage** and **current** at half-hourly "
        "resolution, and the ingest rejects a row that is missing any one of the three columns.",
    )
    updated, status = _apply(
        path,
        raw,
        "Each feeder reading carries power, voltage and current at half-hourly",
        "Each feeder reading carries power, voltage, and current at half-hourly",
    )
    assert status == "applied"
    assert "**voltage**, and **current**" in updated
    assert "**voltage,**" not in updated
    assert _counts(updated) == _counts(raw)


def test_a_serial_comma_after_a_link_lands_outside_the_target(tmp_path: Path):
    path = tmp_path / "page.md"
    raw = _write(
        path,
        "The runbook is the [operations page](https://example.com/ops), the "
        "[aws page](https://example.com/aws) and the alert rules, which together say what the "
        "on-call engineer does when the overnight ingest has degraded the forecast.",
    )
    updated, status = _apply(
        path,
        raw,
        "The runbook is the operations page, the aws page and the alert rules",
        "The runbook is the operations page, the aws page, and the alert rules",
    )
    assert status == "applied"
    # The fixture wraps the second label across a line break, so match from its target on.
    assert "](https://example.com/aws), and the alert rules" in updated
    assert _counts(updated) == _counts(raw)


def test_a_bolded_leads_full_stop_is_pulled_inside_its_markers(tmp_path: Path):
    path = tmp_path / "page.md"
    raw = _write(
        path,
        "**Name the fault**, and the operator reads the alert rather than the logs when the "
        "overnight ingest degrades and the forecast falls back to yesterday's weather run.",
    )
    updated, status = _apply(
        path,
        raw,
        "Name the fault, and the operator reads the alert",
        "Name the fault. The operator reads the alert",
    )
    assert status == "applied"
    assert updated.startswith("**Name the fault.** The operator reads")
    assert _counts(updated) == _counts(raw)


def test_a_full_stop_after_a_mid_sentence_bold_stays_outside_its_markers(tmp_path: Path):
    path = tmp_path / "page.md"
    raw = _write(
        path,
        "The runbook is set out on the **operations page** and the alert rules name the series "
        "at fault, so the engineer on call reads the alert rather than the pipeline's own logs.",
    )
    updated, status = _apply(
        path,
        raw,
        "set out on the operations page and the alert rules name",
        "set out on the operations page. The alert rules name",
    )
    assert status == "applied"
    assert "**operations page**. The alert rules" in updated
    assert _counts(updated) == _counts(raw)


def test_a_word_inserted_before_a_code_span_stays_outside_it(tmp_path: Path):
    path = tmp_path / "page.md"
    raw = _write(
        path,
        "The h3 grid carries `n_h3_cells` and `weight` for every cell of the boundary, which the "
        "spatial join then uses to aggregate the gridded weather onto each substation feeder.",
    )
    updated, status = _apply(
        path,
        raw,
        "The h3 grid carries n_h3_cells and weight for every cell",
        "The h3 grid carries the n_h3_cells and weight for every cell",
    )
    assert status == "applied"
    assert "carries the `n_h3_cells`" in updated
    assert _counts(updated) == _counts(raw)


def test_renaming_the_first_word_of_a_link_label_keeps_the_link(tmp_path: Path):
    path = tmp_path / "page.md"
    raw = _write(
        path,
        "See the [operations page](https://example.com/ops) for the runbook that the on-call "
        "engineer follows each morning when the overnight ingest has degraded the forecast.",
    )
    updated, status = _apply(
        path,
        raw,
        "See the operations page for the runbook",
        "See the outage page for the runbook",
    )
    assert status == "applied"
    assert "[outage page](https://example.com/ops)" in updated
    assert _counts(updated) == _counts(raw)


def test_the_trailing_newline_survives_an_edit_in_the_last_paragraph(tmp_path: Path):
    path = tmp_path / "page.md"
    raw = _write(
        path,
        "# Reading the h3 grid",
        "The h3 grid carries `n_h3_cells`, `n_series` and `weight` columns for every cell, which "
        "the spatial join then uses to aggregate the gridded weather onto each substation feeder.",
    )
    updated, status = _apply(
        path,
        raw,
        "The h3 grid carries n_h3_cells, n_series and weight columns",
        "The h3 grid carries n_h3_cells, n_series, and weight columns",
    )
    assert status == "applied"
    assert updated.endswith("feeder.\n")


def test_an_edit_in_the_last_paragraph_keeps_the_pages_own_wrap_width(tmp_path: Path):
    path = tmp_path / "page.md"
    raw = _write(
        path,
        "# Reading the h3 grid",
        "The h3 grid carries `n_h3_cells`, `n_series` and `weight` columns for every cell, which "
        "the spatial join then uses to aggregate the gridded weather onto each substation feeder.",
    )
    updated, status = _apply(
        path,
        raw,
        "The h3 grid carries n_h3_cells, n_series and weight columns",
        "The h3 grid carries n_h3_cells, n_series, and weight columns",
    )
    assert status == "applied"
    assert max(len(line) for line in updated.splitlines()) <= WRAP_WIDTH


def test_renaming_a_code_spans_whole_content_keeps_it_spanned(tmp_path: Path):
    path = tmp_path / "page.md"
    raw = _write(
        path,
        "The h3 grid carries `n_h3_cells` and `weight` for every cell of the boundary, which the "
        "spatial join then uses to aggregate the gridded weather onto each substation feeder.",
    )
    updated, status = _apply(
        path,
        raw,
        "The h3 grid carries n_h3_cells and weight for every cell",
        "The h3 grid carries h3_cell_count and weight for every cell",
    )
    assert status == "applied"
    assert "`h3_cell_count`" in updated
    assert _counts(updated) == _counts(raw)


def test_cutting_the_last_word_of_a_bolded_lead_leaves_the_lead_bolded(tmp_path: Path):
    path = tmp_path / "page.md"
    raw = _write(
        path,
        "**Name the fault in the telemetry properly.** The operator reads the alert rather than "
        "the logs, so whatever reaches Sentry carries the tag an alert rule can route on.",
    )
    updated, status = _apply(
        path,
        raw,
        "Name the fault in the telemetry properly. The operator reads",
        "Name the fault in the telemetry. The operator reads",
    )
    assert status == "applied"
    assert updated.startswith("**Name the fault in the telemetry.** The operator reads")
    assert _counts(updated) == _counts(raw)


def test_a_splice_that_would_leave_an_empty_bold_span_is_refused(tmp_path: Path):
    path = tmp_path / "page.md"
    raw = _write(
        path,
        "The engineer reads **the alert and the run identifier** before opening the logs at all, "
        "because the alert already names the series, the run and the asset that is at fault.",
    )
    updated, status = _apply(
        path,
        raw,
        "The engineer reads the alert and the run identifier before opening",
        "The engineer reads before opening",
    )
    assert status == "markup refused"
    assert updated == raw


def test_punctuation_inserted_at_any_point_lands_between_the_markup_and_nothing_else_moves(
    tmp_path: Path,
):
    """Sweep every insertion point in a paragraph carrying all four kinds of markup.

    The reported defect was one insertion point out of hundreds writing a serial comma on the wrong
    side of a closing backtick, so the guard that matters is the exhaustive one. A semicolon stands
    in for the comma because the paragraph already carries commas, and the assertion needs to know
    which character the splice wrote.
    """
    path = tmp_path / "page.md"
    raw = _write(path, EVERY_MARKUP)
    assert ";" not in raw
    projected, spans = apply_findings.project(raw)
    # The projection's trailing space comes from the file's final newline, and no quote ever
    # carries it: `locate` matches a stripped needle.
    limit = len(projected.rstrip())
    for point in range(1, limit):
        spliced = apply_findings.splice(
            raw=raw,
            start=0,
            end=limit,
            replacement=f"{projected[:point]};{projected[point:limit]}",
        )
        written_at = next(index for index, char in enumerate(spliced) if char != raw[index])
        assert spliced[written_at] == ";", f"the splice rewrote more than one character at {point}"
        assert spliced[written_at + 1 :] == raw[written_at:], f"text moved for a splice at {point}"
        assert spans[point - 1].right <= written_at <= spans[point].left, (
            f"the semicolon landed inside the markup at {point}"
        )


def test_every_character_is_bracketed_by_its_own_markup_and_no_two_overlap(tmp_path: Path):
    path = tmp_path / "page.md"
    raw = _write(path, EVERY_MARKUP)
    projected, spans = apply_findings.project(raw)
    assert len(projected) == len(spans)
    for index, span in enumerate(spans):
        assert span.left <= span.text < span.right
        if not projected[index].isspace():
            assert raw[span.text] == projected[index]
        if index:
            assert spans[index - 1].right <= span.left


FENCED_PAGE: Final[str] = """\
Install the workspace and its dependencies before running anything else, because the Dagster
definitions import every package in `packages/` at load time.

```bash
uv sync     # create the virtualenv and install all workspace packages
```

The definitions load from `src/` and the assets appear in the Dagster UI, so the ingest and the
forecast can then be materialised by hand.
"""
"""A page whose fenced block carries prose a reviewer's quote can match: "and install"."""


def test_a_finding_that_reaches_into_a_fenced_block_is_refused(tmp_path: Path):
    path = tmp_path / "page.md"
    path.write_text(FENCED_PAGE, encoding="utf-8")
    updated, status = _apply(
        path,
        FENCED_PAGE,
        "create the virtualenv and install all workspace packages",
        "create the virtualenv, and install all workspace packages",
    )
    assert status == "code block"
    assert updated == FENCED_PAGE


def test_a_finding_in_the_prose_of_a_page_that_has_a_fenced_block_still_applies(tmp_path: Path):
    path = tmp_path / "page.md"
    path.write_text(FENCED_PAGE, encoding="utf-8")
    updated, status = _apply(
        path,
        FENCED_PAGE,
        "The definitions load from src/ and the assets appear",
        "The definitions load from src/, and the assets appear",
    )
    assert status == "applied"
    assert "`src/`, and the assets appear" in updated


def test_a_fence_indented_under_a_list_item_still_bounds_a_code_block(tmp_path: Path):
    """Every fence on the code-style page is indented under a list item, and two of six here."""
    page = (
        "Install the workspace before running anything else:\n\n"
        "1. Sync the environment, which creates the virtualenv:\n\n"
        "    ```bash\n"
        "    uv sync     # create the virtualenv and install all workspace packages\n"
        "    ```\n\n"
        "The definitions then load and the assets appear in the Dagster UI.\n"
    )
    path = tmp_path / "page.md"
    path.write_text(page, encoding="utf-8")
    updated, status = _apply(
        path,
        page,
        "create the virtualenv and install all workspace packages",
        "create the virtualenv, and install all workspace packages",
    )
    assert status == "code block"
    assert updated == page


def test_a_bolded_lead_in_a_blockquote_keeps_its_stop_inside_the_markers(tmp_path: Path):
    path = tmp_path / "page.md"
    raw = (
        "> **Name the fault**, and the operator reads the alert rather than the logs when the\n"
        "> overnight ingest degrades and the forecast falls back to yesterday's weather run.\n"
    )
    path.write_text(raw, encoding="utf-8")
    updated, status = _apply(
        path,
        raw,
        "Name the fault, and the operator reads the alert",
        "Name the fault. The operator reads the alert",
    )
    assert status == "applied"
    assert updated.startswith("> **Name the fault.** The operator reads")
    assert _counts(updated) == _counts(raw)


def test_a_bolded_lead_on_a_list_item_keeps_its_stop_inside_the_markers(tmp_path: Path):
    path = tmp_path / "page.md"
    raw = (
        "- **Name the fault**, and the operator reads the alert rather than the logs when the\n"
        "  overnight ingest degrades and the forecast falls back to yesterday's weather run.\n"
    )
    path.write_text(raw, encoding="utf-8")
    updated, status = _apply(
        path,
        raw,
        "Name the fault, and the operator reads the alert",
        "Name the fault. The operator reads the alert",
    )
    assert status == "applied"
    assert updated.startswith("- **Name the fault.** The operator reads")


def test_an_arrow_before_a_bold_span_is_not_read_as_a_list_marker(tmp_path: Path):
    """An arrow `->` is not a bullet, so the span after it opens nothing and keeps its stop
    outside its markers.
    """
    path = tmp_path / "page.md"
    raw = _write(
        path,
        "-> **Retry the run**, and the queue backs off before the next attempt, so a run that has "
        "failed on a transient network error is retried without an operator touching anything.",
    )
    updated, status = _apply(
        path,
        raw,
        "Retry the run, and the queue backs off",
        "Retry the run. The queue backs off",
    )
    assert status == "applied"
    assert updated.startswith("-> **Retry the run**. The queue backs off")


def test_an_inline_triple_backtick_span_does_not_open_a_fenced_block(tmp_path: Path):
    """A hard wrap can push such a span to the start of a line, where it looks like a fence.

    Read as one, it opens a region no later line closes, and every finding in the rest of the file
    is refused. CommonMark forbids a backtick in a fence's info string, which tells the two apart.
    """
    page = (
        "The retry header is spelled\n"
        "```retry_after``` in the response, which the client reads before scheduling the next\n"
        "attempt at the download.\n"
        "\n"
        "The engineer reads the alert and the run identifier before opening the logs at all,\n"
        "because the alert already names the series and the asset that is at fault.\n"
    )
    path = tmp_path / "page.md"
    path.write_text(page, encoding="utf-8")
    assert apply_findings.fenced_regions(page) == ()
    _, status = _apply(
        path,
        page,
        "The engineer reads the alert and the run identifier before opening",
        "The engineer reads the alert, and the run identifier, before opening",
    )
    assert status == "applied"


def test_a_stop_after_single_asterisk_emphasis_is_left_where_the_replacement_put_it(
    tmp_path: Path,
):
    """Only `**` was counted, so only `**` takes a stop inside it."""
    path = tmp_path / "page.md"
    raw = _write(
        path,
        "*Freshest run wins*, and the join falls back to the same run for a future target time, "
        "which is what keeps the lag features free of any lookahead into the forecast horizon.",
    )
    updated, status = _apply(
        path,
        raw,
        "Freshest run wins, and the join falls back",
        "Freshest run wins. The join falls back",
    )
    assert status == "applied"
    assert updated.startswith("*Freshest run wins*. The join falls back")


def test_a_tilde_fence_and_an_unclosed_fence_both_bound_a_code_block():
    tilde = "Prose here.\n\n~~~bash\nuv sync\n~~~\n\nMore prose.\n"
    ((low, high),) = apply_findings.fenced_regions(tilde)
    assert tilde[low:high] == "~~~bash\nuv sync\n~~~\n"
    unclosed = "Prose here.\n\n```bash\nuv sync\n"
    assert apply_findings.fenced_regions(unclosed) == ((13, len(unclosed)),)


def test_a_bullet_marker_is_not_read_as_emphasis(tmp_path: Path):
    path = tmp_path / "page.md"
    raw = "* the first feeder\n* the second feeder\n"
    path.write_text(raw, encoding="utf-8")
    projected, _ = apply_findings.project(raw)
    assert projected == "the first feeder the second feeder "


def test_a_code_span_keeps_the_characters_the_projection_would_otherwise_strip(tmp_path: Path):
    path = tmp_path / "page.md"
    raw = "Name the files individually, because `packages/dashboard/*.py` also silences the rest.\n"
    path.write_text(raw, encoding="utf-8")
    projected, _ = apply_findings.project(raw)
    assert "packages/dashboard/*.py" in projected
