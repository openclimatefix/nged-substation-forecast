---
name: restructure-prose
description: >-
  How to restructure and simplify prose that already exists — most of it in `docs/`, most of it
  drafted originally by Claude — for readability, as a pass kept separate from fact-checking.
  Extract the document's bolded-lead sentences into a flat list and read only that list to test
  whether the argument survives, summarising each paragraph's conclusion first on the many pages
  that predate the bolded-lead rule; run a first-stumble-reader sub-agent to find where an
  unfamiliar reader gets lost; apply one CLAUDE.md prose-style rule at a time rather than a blanket
  rewrite. Load before reordering or simplifying prose someone already wrote, especially a docs/
  page nobody has audited for structure since it was first drafted.
---

# Restructuring prose that already exists

The prose rules in [`CLAUDE.md`](../../../CLAUDE.md) under "Prose style" apply throughout and are
not repeated here. **This skill is for prose that already exists and reads correctly sentence by
sentence, but is hard to follow as a whole** — badly ordered, front-loaded with detail a reader
doesn't need yet, or written in one uniform pass with no sense of what the reader has and hasn't
been told so far.

## This is a structure pass, not a fact-check

**Run the structure pass separately from any accuracy review.** Accuracy and readability find
different defects, and a reviewer asked to check both at once does neither well. Do the structure
pass first: reordering paragraphs moves sentences around, and a fact-check done before the reorder
gets redone once paragraphs move.

## Test 1: the bolded-lead extraction

Pull every bolded lead sentence (the ones CLAUDE.md's "lead each paragraph with a bolded sentence"
rule asks for) out of the document into a flat list, in order, and read only that list. If the
argument doesn't hold together read this way, the paragraphs are in the wrong order, or a paragraph
doesn't belong where it sits — fix that before touching a single sentence inside any paragraph.
Where bolded leads exist, extracting the list is mechanical rather than a judgement call, and it
catches what a reviewer holding the whole document in context cannot see, because such a reviewer
already knows what a later section says while reading an earlier one.

**Many pages have no bolded leads to extract — summarise each paragraph instead.** The bolded-lead
rule is recent, so a page written before it exists can lack bolded leads entirely, and even a page
that has some can have paragraphs without one. Where a paragraph has no bolded lead, write a
one-sentence summary of its conclusion — not its topic — standing in for the lead it should have
had, then build the same flat list from those summaries and read it the same way. Unlike the
mechanical extraction above, this summarising step is a judgement call: do it paragraph by
paragraph, blind to the document's overall argument as far as that's practical, so a summary
reports what the paragraph actually concludes rather than what the summariser already expects the
page to say. Once a restructure lands, adding the bolded lead to each paragraph outright is worth
doing — it makes this test free to re-run next time.

## Test 2: the first-stumble reader

**Spawn a fresh sub-agent that can see only the document, not this repository or this
conversation.** Paste the document into the prompt, or point the sub-agent at a single scratchpad
file, and instruct it to read nothing else — left unconstrained, a sub-agent in this repo
auto-loads `CLAUDE.md` and can read any file in it, which defeats the isolation. Give it a stated
knowledge boundary too: a persona matched to the document's actual audience, such as "an NGED
engineer who has never trained a machine-learning model" or "a funder reading this for the first
time." Ask it to read from the top and stop at the first sentence it cannot follow, and to name the
earlier sentence that would have had to exist for that sentence to work. Forbid it from rewriting
anything — its job is to report where a reader stalls, not to fix the prose. Run it with two or
three personas that cover the page's real readers; a stop point one persona reports and another
doesn't is a gap specific to that persona's background, not a fault in the document as a whole.

The output is a list of stop points: an ordering-bug report, not a style critique.

## Fix at the level the test found the problem

- A stop point traced to a missing prerequisite → move the missing fact earlier if it exists
  elsewhere in the document, move the stalling sentence later if it doesn't, or write the missing
  prerequisite in where the reader first needs it if it exists nowhere at all.
- A lead (bolded, or summarised for this test) that doesn't fit the argument's order → move or cut
  the paragraph, not the sentence.
- Only once ordering is settled, apply CLAUDE.md's sentence- and word-level rules, one rule per
  pass — a pronoun-only pass whose output is a line number and a proposed replacement beats a pass
  that tries to fix pronouns, numerals, and metaphor together.

## Re-run after a restructure

Moving paragraphs can break a bolded lead that referred to "the previous section," or introduce a
fact before its new position's prerequisites are met. Re-run both tests after a restructure, not
just once at the start.

## See also

`long-form-prose` runs the same planning discipline before any prose exists, for drafting rather
than reordering. For a rewrite that adds whole new sections to an existing page, run this skill on
the existing text first, then switch to `long-form-prose` to outline the new sections against the
result — outlining new material against a page whose own structure hasn't been checked risks
building the new sections on prerequisites the existing page never actually establishes.
