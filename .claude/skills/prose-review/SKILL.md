---
name: prose-review
description: >-
  How to audit prose that already exists against the "Prose style" rules in CLAUDE.md, at the level
  of words and sentences rather than document order. Claude reads those rules as a general standard
  while drafting and does not enact them, so the violations are found only by a sweep that takes one
  rule at a time: a combined sweep of one section reported zero findings and a one-rule-at-a-time
  sweep of the same text found thirty. This skill owns the sweep procedure — the order to take the
  rules in, what is deliberately not a finding, how to chunk a long file across sub-agents, which
  model to give the work to, and how to triage findings before any of them are applied. Load before
  auditing a docs/ page, a README, a SKILL.md, or a literature review against the prose rules, and
  whenever a reviewer has been asked to check prose and reported little or nothing.
---

# Auditing prose against the style rules

**The rules live in [`CLAUDE.md`](../../../CLAUDE.md) under "Prose style" and are not repeated
here.** Two copies of a style guide drift apart and then nobody knows which copy is binding. This
skill owns the *procedure* for auditing prose against those rules; the rules themselves stay in one
place.

This is the word-and-sentence pass. `restructure-prose` owns document order, and a badly ordered
page reads correctly sentence by sentence, so the two passes find different faults and neither
substitutes for the other. Run the ordering pass first when both are needed: reordering moves
sentences, and a word-level sweep done before the move is partly wasted.

## One pass per rule, or the sweep finds nothing

**A reviewer asked to check every rule at once finds the loudest fault in each paragraph and moves
on, so the quieter faults survive.** The evidence is direct rather than theoretical. A combined
sweep of one section of the literature review reported **zero** findings. A one-rule-at-a-time
sweep of the same text, by the same model, found **thirty** — twenty-six pronouns, two unscoped
superlatives, two umbrella nouns, and a money metaphor. Extended across the whole review, six
sub-agents working one rule per pass returned 469 findings on a file that had already passed
several reviews.

**Say "one pass per rule" explicitly in the brief.** A brief that lists nine rules together gets a
sweep that honours none of them, because listing the rules is not the same instruction as
sequencing them.

**Require reasoning on the passes that return nothing.** A silent pass and a skipped pass look
identical in a report. Asking for the reasoning makes the difference visible, and it catches a
real failure mode: one model reported a clean acronym pass on a range containing two acronyms that
are expanded nowhere outside a table and the reference list.

## The order to sweep in

Highest yield first, so that the expensive passes run while attention is freshest:

1. Pronouns and demonstratives, including "one", "ones", and "such a"
2. Unenumerated singletons and superlatives
3. Umbrella nouns — "thing", "something", "anything", "metadata"
4. Money metaphors for performance
5. Ambiguous "network"
6. Numerals
7. Serial commas
8. Acronyms expanded on first use
9. Sentences readable two ways, and noun-piles

Pronouns dominate every sweep run so far, by roughly an order of magnitude over any other rule.

## What is deliberately not a finding

**Most of the triage cost is one category, so put it in the brief rather than discovering it
afterwards.** In the review sweep, roughly 250 of 469 findings were a single pattern that should
never have been reported.

- **An author possessive whose owner is already the subject of its own clause.** "Kaas et al.
  report the median across their 200 feeders" is correct as written. The rule's test is whether a
  pronoun makes the reader look backwards to work out the referent, and a possessive with no
  competing candidate does not. Repeating the name there — "Kaas et al.'s 200 feeders" — costs
  readability and buys no precision. Left unstated in the brief, this pattern produces proposals as
  bad as "Enedis has forecast all 2,300 of Enedis's substations".
- **"One" as a determiner in front of the noun it counts.** "the one review we found" both scopes a
  claim and names its noun.
- **A demonstrative that already names its noun**, such as "those principles". The fault is a *bare*
  demonstrative the reader must resolve backwards.
- **Direct quotations from a cited source.** A pronoun outside the quotation marks is in scope; the
  source's own wording is not.
- **Reference-list entries and author strings**, which follow the citation convention rather than
  the serial-comma rule.
- **Headings**, unless renaming is genuinely needed — a heading rename changes the anchor slug and
  breaks inbound links from elsewhere in the docs.

Findings worth keeping are the mirror image: "one"/"ones"/"theirs"/"ours" standing in place of a
noun, a pronoun or demonstrative *opening* a sentence, a referent with two or more plausible
candidates, and a bare "that" or "this" standing for a whole clause.

## Briefing the sub-agents

**Give each agent one contiguous line range and the whole rule list, and require a line number, the
verbatim quote, and concrete replacement wording for every finding.** A finding without replacement
wording cannot be triaged and is worth nothing.

- **Chunk by section boundary, not by equal line count**, so no agent owns half an argument.
  Roughly 5,000 to 7,000 words per agent worked well; the reference list needs no sweep.
- **Warn about hard wrapping.** In a wrapped file a sentence spans several lines, so a plain grep
  for a phrase misses most matches. Tell the agent to normalise first:
  `tr '\n' ' ' < FILE | tr -s ' ' | grep -o 'phrase'`.
- **Report only. Edit no file, run no `git` command, spawn no sub-agents, use no browser tools.**
  Concurrent agents editing one file collide, and a finding that lands before triage cannot be
  rejected.
- **Ask for absolute line numbers in the file**, not offsets into an extracted chunk.

## Which model to give the sweep to

**Use Sonnet 5. Do not use Haiku 4.5.** Both were given a byte-identical brief over an identical
433-line range. Sonnet returned 116 findings; Haiku returned 8, and Haiku's 6 pronoun findings were
a strict subset of Sonnet's 108 — Haiku found nothing Sonnet missed. Haiku also underperformed on
the mechanical rules it was being trialled for, returning 1 numeral finding against 3 and no
acronym findings against 2, and reported a clean acronym pass that was demonstrably wrong. Neither
model invented a quote, and both honoured the one-pass-per-rule structure, so the failure is recall
and self-verification rather than discipline.

## Triage before applying anything

**`restructure-prose` covers the general rule that roughly half of any reviewer's findings are
wrong; two failure modes are specific to this sweep.**

- **Verify every quote exists and is unique before editing.** Run the whole batch as a dry run that
  asserts a match count per finding. A quote can be stale because a paragraph was rewritten, and it
  can match more than once.
- **A quote that matches more than once is usually a lifted sentence, and every copy needs the same
  fix.** A review with a summary lifts sentences verbatim from its own body — one sentence had three
  copies. Fixing the body alone makes the summary silently disagree with the section it came from.
  Different agents own those copies, so the same finding also arrives twice and needs deduplicating.
- **Check a proposed replacement against the rules too.** Two replacements in one sweep swapped a
  banned "one" for a banned "it", and another defined a term using the term itself.

## Applying and checking the edits

Edit through the wrap-tolerant `rsub` substitution in the `literature-review` skill, which asserts
its match count, and follow `restructure-prose` for the balanced-bold check and for re-wrapping only
the paragraphs that actually changed. Then:

```bash
uv run pymarkdown scan -r docs README.md CLAUDE.md packages/*/README.md
uv run mkdocs build --strict
```

**Neither command checks an anchor inside an absolute URL.** `mkdocs build --strict` validates
relative links only, so a link to
`https://openclimatefix.github.io/nged-substation-forecast/...#wrong-anchor` passes every check in
the repo. Compare a changed anchor against the `id=` attributes in the built page under `site/`, or
against how another docs page spells the same anchor. A heading whose text contains an em dash
generates a single separator, not two: `## H2 — a hundred experiments` becomes
`#h2-a-hundred-experiments`.

One further check catches a fault no linter reports, because Python-Markdown treats any line
starting with `#` as a heading even without the space CommonMark requires:

```bash
python3 -c "import re;[print(i+1,l[:60]) for i,l in enumerate(open('FILE')) if l.startswith('#') and not re.match(r'#{1,6} ',l)]"
```

## See also

`restructure-prose` fixes document order and owns the hard-wrapped-edit machinery this skill points
at. `long-form-prose` runs the planning discipline before any prose exists. `literature-review`
owns the accuracy round, which is a separate pass from this one: accuracy and readability find
different defects, and a reviewer asked for both does neither well.
