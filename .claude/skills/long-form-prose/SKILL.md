---
name: long-form-prose
description: >-
  How to draft new prose long enough to need its own structure — a docs/ page, a roadmap section,
  a literature-review chapter, a PR description explaining a design. Claude drafts prose in one
  pass with uniform access to everything it knows, so it cannot enact the seriality a reader
  experiences and often gets the order wrong even when every sentence is correct. This skill is
  the fix: write a section-by-section outline before any prose exists, review it by reading its
  conclusions as a flat list and checking each section's prerequisites against what came before,
  then draft from the approved outline and check the result against a fresh reader instead of a
  reviewer who already knows the ending. Load before drafting any prose longer than a few
  paragraphs of connected argument, or when a draft keeps needing structural rewrites after review
  rather than sentence-level fixes.
---

# Planning before drafting long-form prose

The prose rules in [`CLAUDE.md`](../../../CLAUDE.md) under "Prose style" apply throughout and are
not repeated here. Those rules fix words and sentences. This skill fixes the order of a whole
document — that order needs a plan made before any prose exists, not a review of prose after the
fact.

## Why an outline comes first

**A badly ordered document reads fine sentence by sentence, so drafting-then-reviewing rarely
catches it.** A reviewer holding the whole document in context is a poor judge of ordering, because
that reviewer already knows what section seven says while reading section two — exactly the
knowledge a first-time reader doesn't have. Expository writing is a planning problem over the
reader's knowledge state: pick an order over the ideas that keeps what the reader must hold in mind
small, and for each fact, decide whether the reader can use it yet. Fixing the order is cheap in an
outline and expensive in 4,000 words of drafted prose, which is why the plan has to come first.

## Write the outline before drafting a sentence

**Write down a conclusion and a prerequisite list for each section, and nothing else:**

- **The one-sentence conclusion** — the claim that section's bolded lead will eventually state.
- **The prerequisites** — the terms, numbers, and earlier claims the reader must already hold to
  follow that conclusion.

Order the sections so each section's prerequisites are all satisfied by sections that came before
it. Leave supporting detail out of the outline entirely; an outline that already contains the
detail has stopped being a planning tool.

## Review the outline before it becomes prose

- **Extract the section conclusions into a flat list and read only that list.** If the argument
  doesn't hold together read this way, reorder or cut sections — don't plan to fix it with better
  sentences later.
- **Check each section's prerequisite list against what earlier sections actually establish.** A
  section that needs a term or number nobody has introduced yet is in the wrong place, or the fact
  is.

## Draft from the approved outline

- **Introduce a fact only in the section whose outline entry needs it.** If a section's draft wants
  a fact that belongs to a later section, that's a sign the outline order is wrong, not a fact to
  smuggle in early.
- **Apply the CLAUDE.md prose-style rules one at a time, as separate passes over the drafted text**,
  rather than trying to satisfy all of them while drafting. A single named-rule pass — for example,
  a pronoun-only pass whose output is a line number and a proposed replacement — does better work
  than a general "make this good" pass, and does much better work than trying to draft
  rule-conforming prose from a blank page.

## Check the draft against a reader, not a reviewer

**Spawn a fresh sub-agent that can see only the draft, not this repository or this conversation.**
Paste the draft into the prompt, or point the sub-agent at a single scratchpad file, and instruct it
to read nothing else — left unconstrained, a sub-agent in this repo auto-loads `CLAUDE.md` and can
read any file in it, which defeats the isolation. Give it a stated knowledge boundary too: a persona
matched to the document's actual audience, such as "a distribution-network planner who has never
trained a machine-learning model." Ask it to read from the top and stop at the first sentence it
cannot follow, and to name the earlier sentence that would have had to exist for that sentence to
work. Forbid it from rewriting anything — its job is to report where a reader stalls, not to fix the
prose. Run it with two or three personas that cover the document's real readers; a stop point one
persona reports and another doesn't is a gap specific to that persona's background, not a fault in
the document as a whole.

**Fix every stop point at the outline level, then re-extract the conclusion list and check it
again.** Move the missing prerequisite earlier, or move the stalling section later, rather than
patching the one sentence the reader tripped over — a patch fixes that sentence and leaves the
outline's order wrong for the next reader who trips somewhere else.

## See also

`restructure-prose` runs the same flat-list and first-stumble-reader checks over prose that already
exists, for when there's a draft to fix rather than one to write.
