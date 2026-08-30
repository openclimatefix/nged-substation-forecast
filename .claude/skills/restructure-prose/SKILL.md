---
name: restructure-prose
description: >-
  How to restructure and simplify prose that already exists — most of it in `docs/`, most of it
  drafted originally by Claude — for readability, as a pass kept separate from fact-checking.
  Extract the document's bolded-lead sentences into a flat list and read only that list to test
  whether the argument survives; run a first-stumble-reader sub-agent to find where an unfamiliar
  reader gets lost; apply one CLAUDE.md prose-style rule at a time rather than a blanket rewrite.
  Load before reordering or simplifying prose someone already wrote, especially a docs/ page nobody
  has audited for structure since it was first drafted.
---

# Restructuring prose that already exists

The prose rules in [`CLAUDE.md`](../../../CLAUDE.md) under "Prose style" apply throughout and are
not repeated here. This skill is for prose that already exists and reads correctly sentence by
sentence, but is hard to follow as a whole — badly ordered, front-loaded with detail a reader
doesn't need yet, or written in one uniform pass with no sense of what the reader has and hasn't
been told so far.

## This is a structure pass, not a fact-check

Run it separately from any accuracy review. Accuracy and readability find different defects, and a
reviewer asked to check both at once does neither well. Do the structure pass first: reordering
paragraphs moves sentences around, and a fact-check done before the reorder gets redone once
paragraphs move.

## Test 1: the bolded-lead extraction

Pull every bolded lead sentence (the ones CLAUDE.md's "lead each paragraph with a bolded sentence"
rule asks for) out of the document into a flat list, in order, and read only that list. If the
argument doesn't hold together read this way, the paragraphs are in the wrong order, or a paragraph
doesn't belong where it sits — fix that before touching a single sentence inside any paragraph.
Extracting the list is mechanical, not a judgement call, and it catches what a reviewer holding the
whole document in context cannot see, because such a reviewer already knows what a later section
says while reading an earlier one.

## Test 2: the first-stumble reader

Spawn a fresh sub-agent with the document alone — no repository, no source papers, no conversation
history — plus a stated knowledge boundary matched to the document's actual audience, such as "an
NGED engineer who has never trained a machine-learning model" or "a funder reading this for the
first time." Ask it to read from the top and stop at the first sentence it cannot follow, and to
name the earlier sentence that would have had to exist for that sentence to work. Forbid it from
rewriting anything — its job is to report where a reader stalls, not to fix the prose. Run it a few
times with personas that cover the page's real readers.

The output is a list of stop points: an ordering-bug report, not a style critique.

## Fix at the level the test found the problem

- A stop point traced to a missing prerequisite → move the missing fact earlier, or move the
  stalling sentence later.
- A bolded lead that doesn't fit the argument's order → move or cut the paragraph, not the sentence.
- Only once ordering is settled, apply CLAUDE.md's sentence- and word-level rules, one rule per
  pass — a pronoun-only pass whose output is a line number and a proposed replacement beats a pass
  that tries to fix pronouns, numerals, and metaphor together.

## Re-run after a restructure

Moving paragraphs can break a bolded lead that referred to "the previous section," or introduce a
fact before its new position's prerequisites are met. Re-run both tests after a restructure, not
just once at the start.

## See also

`long-form-prose` runs the same planning discipline before any prose exists, for drafting rather
than reordering.
