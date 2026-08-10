---
name: mkdocs-authoring
description: >-
  Ways Python-Markdown (MkDocs' renderer) renders a page visibly wrong while `pymarkdown scan`
  and `mkdocs build --strict` both pass: nested sub-bullets need a full 4-space indent, a sibling
  list item needs a blank line after an indented continuation, and a wrapped link whose
  continuation line starts with `#` becomes a heading. Load before editing any page under `docs/`
  — especially nested lists, list items containing code blocks, or wrapped links — and when a
  rendered page looks wrong but the linters are clean.
---

# Authoring markdown that MkDocs renders correctly

Python-Markdown is stricter and weirder than CommonMark in ways this repo's linters do not catch.
The standing rule that follows: **any docs change that touches links or non-trivial lists should
run `uv run mkdocs build --strict` and then actually read the generated HTML under `site/`**. A
clean `pymarkdown scan` plus a successful build can both pass on rendering that is visibly wrong;
only reading the HTML catches it.

## Nested sub-bullets need a full 4-space indent

Python-Markdown (without the `sane_lists` extension) only treats a sub-list as nested when it is
indented a full 4 spaces. A 2-space indent — a bare list's natural default, and what most other
renderers accept — silently renders as a flat, unnested list. This one *is* enforced, by the
`pml101` rule configured in `pyproject.toml`, but write it correctly in the first place rather
than waiting for the linter. `pml101` anchors every level at (depth − 1) × 4 spaces regardless
of whether the parent marker is a bullet or a number.

## A list item needs a blank line before it if it follows an indented continuation

Python-Markdown doesn't let a list item interrupt a paragraph the way GitHub-flavored Markdown
does. If a bullet's continuation content ends with an indented paragraph (e.g. a clarifying
sentence after a fenced code block inside the item) and the next sibling bullet immediately
follows with no blank line in between, Python-Markdown treats the new list-marker line as more
paragraph text rather than a new list item — the marker renders as a literal hyphen, merged into
the previous sentence's prose. `pymarkdown scan` does **not** catch this: a markdown source with
the missing blank line lints clean.

**How to apply:** always put a blank line between a list item's continuation content (paragraphs,
fenced code blocks) and the next sibling item. For any non-trivial list item — one that embeds a
code block or multiple paragraphs — spot-check with `uv run mkdocs build --strict` and inspect
the rendered HTML rather than trusting the linter alone.

## A wrapped link whose continuation line starts with `#` renders as a heading

CommonMark requires a space after `#` for a line to start an ATX heading (`#5` is just text,
`# 5` is a heading). Python-Markdown does not enforce that space, so any line that happens to
start with `#` — for any reason — is parsed as a heading. A markdown link wrapped across the
80-ish-character line length this repo's prose otherwise isn't held to can put the `#123](url)`
half of `[issue #123](url)` at the start of a line, and Python-Markdown reads it as a heading
rather than as the second half of a link. The rendered page gets a stray `<h1>`/`<h2>` containing
the raw URL, the link text before the wrap point left dangling as plain text, and the paragraph
split in two. Neither `pymarkdown scan` nor `mkdocs build --strict` catches this — both pass on a
source file that renders visibly broken.

**How to apply:** when a link's markdown source wraps across a line break, make sure the
continuation line does not begin with `#`; keep `[text](url)` together on one line, or wrap
before `[` rather than inside it.
