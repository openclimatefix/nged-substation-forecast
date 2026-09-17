"""`apply_findings_py.py` splices a sweep's replacement into a docstring or a comment block.

A docstring is markdown, so the ways a splice can corrupt one are the ways it corrupts a `.md`
page: a comma written inside the code span it follows, a replacement that deletes the backtick
closing a span, a match found inside a code sample rather than in prose. Those three are what
`prose_splice.py` now guards for both appliers, and the first three tests below are the Python
copies of the markdown regressions in `test_prose_review_apply_findings.py`.

The rest are faults only a Python file has. A re-wrap that grows a paragraph onto the line below
it can swallow a `# noqa` pragma, a hand-aligned diagram or the next bullet of a list, and a
re-wrap reaching into a fenced block or a reST `::` block rewrites a command a reader is meant to
copy and run.
"""

import textwrap
from pathlib import Path

from _prose_review_scripts import load

apply_findings_py = load("apply_findings_py")

LONG_PROSE = (
    "A first line of ordinary prose that runs past one hundred characters so the re-wrap decides "
    "it has to reflow this paragraph."
)
"""One line long enough for `wrap_overlong` to pick it up, and prose end to end."""


def _apply(raw: str, quote: str, replacement: str) -> tuple[str, str]:
    """Run one finding against `raw`, returning the new text and the status reported."""
    finding = {"file": "mod.py", "quote": quote, "replacement": replacement}
    return apply_findings_py.apply_one(raw=raw, finding=finding, merge_base=None)


def _wrapped(tmp_path: Path, source: str) -> str:
    """Write `source`, re-wrap its over-long prose lines, and return the file's new text."""
    path = tmp_path / "mod.py"
    path.write_text(source, encoding="utf-8")
    apply_findings_py.wrap_overlong(path)
    return path.read_text(encoding="utf-8")


def test_a_serial_comma_after_a_code_span_lands_outside_the_backticks():
    raw = (
        '"""The join carries `n_h3_cells`, `weight` and `time_series_id` across the boundary."""\n'
    )
    updated, status = _apply(
        raw,
        "The join carries n_h3_cells, weight and time_series_id across the boundary.",
        "The join carries n_h3_cells, weight, and time_series_id across the boundary.",
    )
    assert status == "applied"
    assert "`weight`, and `time_series_id`" in updated
    assert "`weight,`" not in updated
    assert updated.count("`") == raw.count("`")


def test_a_splice_that_would_delete_a_closing_backtick_is_refused():
    raw = '"""The asset calls `write_nwp` helper before it returns the frame to the caller."""\n'
    updated, status = _apply(
        raw,
        "The asset calls write_nwp helper before it returns the frame to the caller.",
        "The asset calls the writer before it returns the frame to the caller.",
    )
    assert status == "markup refused"
    assert updated == raw


def test_a_multi_word_link_label_keeps_its_spaces_so_the_sentence_matches():
    raw = '"""See the [inherent stability](https://example.org/x) page for the ladder."""\n'
    updated, status = _apply(
        raw,
        "See the inherent stability page for the ladder.",
        "See the inherent stability page for the full ladder.",
    )
    assert status == "applied"
    assert "[inherent stability](https://example.org/x) page for the full ladder." in updated


def test_a_link_label_projects_with_its_spaces_intact():
    raw = '"""See the [inherent stability](https://example.org/x) page."""\n'
    (unit,) = apply_findings_py.prose_units(raw)
    projected, _spans = apply_findings_py.project_unit(raw=raw, unit=unit)
    assert "inherent stability" in projected


def test_a_finding_appending_at_the_end_of_a_unit_does_not_raise():
    raw = '"""Foo bar."""\n'
    updated, status = _apply(raw, "Foo bar.", "Foo bar. Baz.")
    assert status == "applied"
    assert updated == '"""Foo bar. Baz."""\n'


def test_a_quote_matching_inside_a_fenced_block_is_refused():
    raw = textwrap.dedent('''\
        """Summary line.

        ```bash
        # create the virtualenv and install all workspace packages
        uv sync
        ```
        """
    ''')
    updated, status = _apply(
        raw,
        "create the virtualenv and install all workspace packages",
        "create the virtual environment and install all workspace packages",
    )
    assert status == "code block"
    assert updated == raw


def test_a_quote_matching_inside_a_rest_literal_block_is_refused():
    raw = textwrap.dedent('''\
        """Summary line.

        Usage::

            python3 count_prose_words.py --rev HEAD one path and another path
        """
    ''')
    updated, status = _apply(
        raw,
        "python3 count_prose_words.py --rev HEAD one path and another path",
        "python3 count_prose_words.py --rev HEAD one path, and another path",
    )
    assert status == "code block"
    assert updated == raw


def test_a_re_wrap_does_not_weld_a_noqa_pragma_into_the_paragraph(tmp_path: Path):
    source = f"# {LONG_PROSE}\n# sample-key/2024-01-01.json  # noqa: E501\nx = 1\n"
    updated = _wrapped(tmp_path, source)
    assert "# sample-key/2024-01-01.json  # noqa: E501\n" in updated


def test_a_re_wrap_does_not_collapse_a_hand_aligned_diagram(tmp_path: Path):
    diagram = "# key/2024-01-01.json  ->  date\n#                 ^^^^     ext\n"
    updated = _wrapped(tmp_path, f"# {LONG_PROSE}\n{diagram}x = 1\n")
    assert diagram in updated


def test_a_re_wrap_keeps_two_bullets_separate(tmp_path: Path):
    source = f'"""Summary line.\n\n- {LONG_PROSE}\n- The second bullet is short.\n"""\n'
    updated = _wrapped(tmp_path, source)
    assert "\n- The second bullet is short.\n" in updated
    assert "\n  to reflow this paragraph.\n" in updated


def test_a_re_wrap_leaves_a_fenced_code_block_alone(tmp_path: Path):
    call = (
        "result = some_function(argument_one=1, argument_two=2, argument_three=3, "
        "argument_four=4, argument_five=5)"
    )
    source = f'"""Summary line.\n\n```python\n{call}\n```\n"""\n'
    updated = _wrapped(tmp_path, source)
    assert f"\n{call}\n" in updated


def test_a_re_wrap_leaves_a_rest_literal_block_alone(tmp_path: Path):
    command = (
        "python3 apply_findings_py.py findings.json --apply --merge-base 7657db1c "
        "--and-one-more-argument --and-yet-another-argument"
    )
    source = f'"""Summary line.\n\nUsage::\n\n    {command}\n"""\n'
    updated = _wrapped(tmp_path, source)
    assert f"\n    {command}\n" in updated


def test_a_re_wrap_still_fixes_an_over_long_args_line(tmp_path: Path):
    entry = (
        "        source: The full text of a Python file, which this helper reads end to end "
        "before it decides anything at all."
    )
    source = f'def f(source: str) -> None:\n    """Summary.\n\n    Args:\n{entry}\n    """\n'
    updated = _wrapped(tmp_path, source)
    assert all(len(line) <= 100 for line in updated.split("\n"))


def test_a_file_needing_no_wrapping_is_left_byte_identical(tmp_path: Path):
    source = '"""Summary line."""\n\nVERTICAL_TAB = "a\\x0bb"\n'
    path = tmp_path / "mod.py"
    path.write_text(source, encoding="utf-8")
    before = path.read_bytes()
    apply_findings_py.wrap_overlong(path)
    assert path.read_bytes() == before


def test_a_quote_matching_twice_is_refused():
    raw = '"""Foo bar."""\n\n\ndef f() -> None:\n    """Foo bar."""\n'
    updated, status = _apply(raw, "Foo bar.", "Foo baz.")
    assert status == "ambiguous"
    assert updated == raw


def test_a_splice_crossing_a_blank_line_is_refused():
    raw = '"""Summary line.\n\nSecond paragraph here.\n"""\n'
    updated, status = _apply(
        raw, "Summary line. Second paragraph here.", "A summary line. A second paragraph."
    )
    assert status == "crosses blank line"
    assert updated == raw


def test_a_quote_wrapped_across_a_comment_block_is_matched():
    raw = (
        "# The ingest degrades rather than raising, because an\n"
        "# absent input is not our bug.\nx = 1\n"
    )
    updated, status = _apply(
        raw,
        "The ingest degrades rather than raising, because an absent input is not our bug.",
        "The ingest degrades rather than raising, because an absent input is not our defect.",
    )
    assert status == "applied"
    assert "our defect." in updated


def test_a_finding_missing_its_quote_is_reported_rather_than_raising_a_key_error():
    problems = apply_findings_py.invalid_findings([{"file": "mod.py", "replacement": "x"}])
    assert problems == ["finding 1 is missing a string quote"]


def test_a_findings_file_that_is_not_a_list_is_reported():
    assert apply_findings_py.invalid_findings({"file": "mod.py"})


def test_a_re_wrap_of_a_comment_with_no_space_after_the_hash_keeps_every_word(tmp_path: Path):
    """`#text` is legal Python, and the prefix has to be read off the line rather than assumed."""
    source = f"#{LONG_PROSE}\nx = 1\n"
    updated = _wrapped(tmp_path, source)
    assert LONG_PROSE.split() == "".join(updated.split("x = 1")[0]).replace("#", " ").split()
