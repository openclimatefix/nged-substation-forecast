---
name: long-form-prose
description: >-
  How to draft new prose long enough to need its own structure — a docs/ page, a roadmap section,
  a literature-review chapter, a PR description explaining a design. Claude drafts prose in one
  pass with uniform access to everything it knows. Claude therefore cannot enact the seriality a
  reader experiences, and often gets the order wrong even when every sentence is correct. This
  skill is the fix: write a section-by-section outline before any prose exists, review it by
  reading its conclusions as a flat list and checking each section's prerequisites against what
  came before, then draft from the approved outline and check the result against a fresh reader
  instead of a reviewer who already knows the ending. Load before drafting any prose longer than
  a few paragraphs of connected argument, or when a draft keeps needing structural rewrites after
  review rather than sentence-level fixes.
---

# Planning before drafting long-form prose

The prose rules in [`CLAUDE.md`](../../../CLAUDE.md) under "Prose style" apply throughout and are
not repeated here. Those rules fix words and sentences. This skill fixes the order of a whole
document. That order needs a plan made before any prose exists, not a review of prose after the
fact.

## Why an outline comes first

**A badly ordered document reads fine sentence by sentence, so drafting-then-reviewing rarely
catches it.** A reviewer holding the whole document in context is a poor judge of ordering, because
that reviewer already knows what section seven says while reading section two — exactly the
knowledge a first-time reader doesn't have. Expository writing is a planning problem over the
reader's knowledge state: pick an order over the ideas that keeps what the reader must hold in mind
small. For each fact, decide whether the reader can use it yet. Fixing the order is cheap in an
outline and expensive in 4,000 words of drafted prose. That is why the plan has to come first.

## Write the outline before drafting a sentence

**Write down a conclusion and a prerequisite list for each section, and nothing else:**

- **The one-sentence conclusion** — the claim that section's bolded lead will eventually state.
- **The prerequisites** — the terms, numbers, and earlier claims the reader must already hold to
  follow that conclusion.

Order the sections so each section's prerequisites are all satisfied by sections that came before
it. Leave supporting detail out of the outline entirely. An outline that already contains the
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
- **Decide prose or bullets per passage, on how complex the concept is rather than how long the
  passage runs.** A parallel set of simple, independent facts — the options a setting takes, what a
  table holds, a run of short design notes — is a list, and drafting it as prose only makes a
  reader work harder for the same content. A passage introducing a complex new concept is prose,
  because the connective tissue between the sentences is what makes a new concept followable. The
  outline is where this is easiest to see: an entry whose content is a set of peers is a list, and
  an entry whose content is an argument reaching a conclusion is not.
- **Apply the CLAUDE.md prose-style rules one at a time, using `prose-review`'s one-rule-per-pass sweep, as separate passes over the drafted text** rather than trying to satisfy every rule while drafting. A single named-rule pass does better work than a general "make this good" pass, and much better work than trying to draft rule-conforming prose from a blank page.

## Check the draft against a reader, not a reviewer

**Run `prose-review`'s first-stumble reader against the draft.** A fresh sub-agent, isolated from this repository and this conversation, reads from the top with a persona matched to the draft's real audience — such as "a distribution-network planner who has never trained a machine-learning model" — and stops at the first sentence it cannot follow. The isolation requirement, the multi-persona rule and the instruction not to rewrite anything are the same whether the text is a fresh draft or an existing page. `prose-review` documents all three.

**Fix every stop point at the outline level, then re-extract the conclusion list and check it
again.** Move the missing prerequisite earlier, or move the stalling section later, rather than
patching the one sentence the reader tripped over. A patch fixes that sentence and leaves the
outline's order wrong for the next reader who trips somewhere else.

## See also

`prose-review` runs the same flat-list and first-stumble-reader checks over prose that already
exists, for when there's a draft to fix rather than one to write, and owns the one-rule-per-pass
sweep this skill hands off to once a draft exists. For a rewrite that adds whole new sections to
an existing page, run `prose-review` on the existing text first, then outline the new sections
here against the result. Outlining new material against a page whose own structure hasn't been
checked risks building the new sections on prerequisites the existing page never actually
establishes.
