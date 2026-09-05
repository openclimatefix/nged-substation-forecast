---
name: prose-review
description: >-
  How to review prose that already exists against the "Prose style" rules in CLAUDE.md — structure
  and word choice alike. Claude reads those rules as a general standard while drafting and does not
  enact them, so the violations are found only by a review that takes one rule at a time. A
  combined sweep of one section reported zero findings; a one-rule-at-a-time sweep of the same text
  found 30. This skill sizes the review to the text, then runs the structural passes (cross-page
  redundancy, bolded-lead extraction, paragraph splitting, count closure, first-stumble reader)
  before the word-and-sentence sweep. It also owns what is deliberately not a finding, how to chunk
  a long file across sub-agents, which model to give the work to, how to triage findings before
  applying any, and how to check that a restructure lost no information, and the scripts that apply
  a batch of findings and prove the page kept its structure. Load whenever prose is to be reviewed,
  reordered or simplified — a docs/ page, a README, a SKILL.md, a docstring, a literature review —
  and whenever a reviewer asked to check prose has reported little or nothing. Sweeping the
  docstrings and comments in Python files has its own section: the word-count rule reverses, the
  README collides with the module docstrings on the API page, and there are three guards to run
  that a docs/ page never needs.
---

# Reviewing prose that already exists

**The rules live in [`CLAUDE.md`](../../../CLAUDE.md) under "Prose style" and are not repeated
here.** Two copies of a style guide drift apart and then nobody knows which copy is binding. This
skill owns the *procedure*; the rules themselves stay in one place.

## Size the review to the text

**A structural pass needs more than one paragraph of connected argument, because a single paragraph
cannot be mis-ordered relative to itself.** Size the review before starting rather than running
every pass over everything:

| What is being reviewed | Passes to run |
|---|---|
| A docstring, a comment, a pull-request body, one or two paragraphs | The sentence sweep only |
| Several paragraphs under one heading | Pass B, then the sentence sweep |
| A whole page, or anything with headings | Passes A, B, C, and D, then the sentence sweep |
| A page nobody has audited, or beyond roughly 5,000 words | Every pass, chunked across sub-agents |
| A whole section of the docs, or the docs as a whole | Pass F first, then the rest per page |

**Structure is settled before the sentence sweep runs, never after.** Reordering and splitting move
sentences, so a word-level sweep done first is partly wasted and its line numbers go stale.

**Keep the review separate from any accuracy round.** Accuracy and readability find different
defects, and a reviewer asked to check both at once does neither well. `literature-review` owns the
accuracy round.

## Pass A: the bolded-lead extraction

Pull every bolded lead sentence (the ones CLAUDE.md's "lead each paragraph with a bolded sentence"
rule asks for) out of the document into a flat list, in order, and read only that list. If the
argument doesn't hold together read that way, the paragraphs are in the wrong order, or a paragraph
doesn't belong where it sits — fix that before touching a single sentence inside any paragraph.
Where bolded leads exist, extracting the list is mechanical rather than a judgement call. The
extraction also catches what a reviewer holding the whole document in context cannot see, because
that reviewer already knows what a later section says while reading an earlier one.

**Many pages have no bolded leads to extract — summarise each paragraph instead.** The bolded-lead
rule is recent, so a page written before the rule can lack bolded leads entirely. Even a page that
has some can have paragraphs without one. Where a paragraph has no bolded lead, write a
one-sentence summary of its conclusion — not its topic — standing in for the lead it should have
had, then build the same flat list from those summaries and read it the same way. Unlike the
mechanical extraction above, summarising is a judgement call: do it paragraph by paragraph, blind
to the document's overall argument as far as that's practical, so a summary reports what the
paragraph actually concludes rather than what the summariser already expects the page to say.

**Check whether a missing bolded lead is a convention before proposing to add one.** A paragraph
with no bolded lead usually wants one, but not always: a page can deliberately give a whole class
of paragraph no lead — the short paragraph that states a problem ahead of the sections answering
it, the sentence that exists only to introduce the table under it. Three separate reviewers in one
run each proposed adding a lead to a paragraph of one such class, because each saw a single
instance and none checked the rest of the page. Before recording the finding, count how many
paragraphs of the same kind the document holds and how many of those carry a lead. Where none
does, the absence is a convention and the finding is wrong.

## Pass B: paragraphs carrying more than one claim

**A paragraph that makes two distinct claims can carry a bolded lead for only one of them, which
makes an oversized paragraph a structural defect rather than a stylistic one.** Nothing else in
this repo looks for it. Until this pass existed, a 479-word paragraph, a 525-word paragraph and a
352-word paragraph all survived every review round the literature review had been given.

Roughly 150 words or 10 wrapped lines is the prompt to look, not the test. The test is whether the
paragraph reaches more than one conclusion. A 180-word paragraph running one continuous argument to
one conclusion stays whole; splitting it would cut a sentence away from the evidence it needs.

For every split, give the exact sentence to split at and write the bolded lead for the new
paragraph. A split that leaves the second half without a lead has moved the defect rather than
fixed it.

**The mirror-image finding is a run of bullets that should be prose.** CLAUDE.md prefers
sub-headings and short paragraphs over bullet lists, because a list flattens an argument into items
of equal weight. A bulleted item carrying several sentences and a citation is a paragraph wearing a
hyphen; a genuinely parallel set of short design notes is a list. Both directions are findings.

**Prose that should be bullets is the harder direction to spot, and the rule that decides it is
the complexity of the concept, not the length of the passage.** CLAUDE.md's bullets rule allows a
list wherever the items really are of equal weight and each one is simple — the options a setting
takes, what a table holds, a run of short design notes — and forbids one wherever the passage
introduces a complex new concept, because the connective tissue between the sentences is what
makes a new concept followable. So the finding is not "this paragraph is long". The finding is
that a reader meeting this material for the first time would lose nothing if the sentences stopped
joining up. Where a paragraph is building an argument towards a conclusion, leave it as prose
however long it runs.

## Pass C: does every count close?

**An enumeration that promises N items and delivers a different N is invisible to every other pass
and mechanically detectable by this one.** It reads perfectly well sentence by sentence, so the
word-level sweep passes over it. A reviewer holding the whole document already knows what the
author meant, too.

For every "the four in use", "three further", "six spokes", "the five are not", "nine challenges":

- Count the enumerated items and check the number matches.
- Check the later discussion names the **same** members, not merely the same number. One paragraph
  enumerated six substitutes for ground truth and then discussed a technique it had never
  introduced, having taken an item from a different document's list. Two documents each owned a
  six-item protocol, and the counts hid the swap.
- Check any heading that states a count against the section under it. `literature-review`'s own
  heading promised five traps and listed six.

**When prose restates a list another document owns, diff it item by item against the source list**
rather than transcribing it from memory.

## Pass D: do the headings work read cold?

**Read the table of contents alone, as a reader who has not read the page.** Pass A reads the bolded
leads to test the argument's order; this pass reads the headings to test whether a reader can
navigate. The failure is invisible to the author, who cannot un-know what each section says.

Two tests per heading, per CLAUDE.md's heading rule:

- **Can a first-time reader parse every word?** A term the section itself defines has no meaning yet
  in the heading above it.
- **Can that reader tell why the claim would matter to them?** A heading whose significance is only
  visible from inside the section sends the reader on a detour that returns nothing.

A heading failing either test is rewritten as a plain descriptor of its subject, and the conclusion
moves into the section's bolded lead. A heading passing both tests is left alone, however long it
is. Often only one phrase is failing, and replacing that phrase keeps the conclusion: "the field"
became "MLOps research" because a bare "the field" is the referent fault Rule 1 already forbids.
The rewritten phrase lands in the one sentence a skim-reader is guaranteed to read.

**The two tests apply to a navigation entry with more force than to a heading.** A published site's
navigation is where a reader arrives, so a section or page name that only makes sense once the page
is open sends every reader down the wrong branch. Read the navigation on its own, the way Pass A
reads the bolded leads: `mkdocs.yml`'s `nav` block is the whole list, and any name a first-time
reader cannot place is a finding. Keep the capitalisation consistent across the list too — a
navigation list mixing title case with sentence case reads as two lists stapled together — and
where two entries in different sections name the same subject, that duplication is a Pass F
finding, not a naming one.

**Renaming a heading changes its anchor slug, so grep for inbound links to the old slug first** —
across `docs/`, the skills, and any absolute link to the published site — and update every one in the
same commit. That cost is real: of three headings renamed in this repo, two carried three and two
inbound links respectively. The link text is usually a fragment of the sentence around it rather than
the heading itself, so only the anchor needs changing.

## Pass E: the first-stumble reader

**Spawn a fresh sub-agent that can see only the document, not this repository or this
conversation.** Paste the document into the prompt, or point the sub-agent at a single scratchpad
file, and instruct it to read nothing else. Left unconstrained, a sub-agent in this repo auto-loads
`CLAUDE.md` and can read any file in it, which defeats the isolation. Give it a stated knowledge
boundary too: a persona matched to the document's actual audience, such as "an NGED engineer who
has never trained a machine-learning model" or "a funder reading this for the first time." Ask it
to read from the top and stop at the first sentence it cannot follow, and to name the earlier
sentence that would have had to exist for that sentence to work. Forbid it from rewriting anything
— its job is to report where a reader stalls, not to fix the prose. Run it with two or three
personas that cover the page's real readers. A stop point one persona reports and another doesn't
is a gap specific to that persona's background, not a fault in the document as a whole.

The output is a list of stop points: an ordering-bug report, not a style critique.

## Pass F: what does another page already say?

**When the unit under review is a set of pages rather than one page, the biggest cut is the idea
explained on four pages instead of one.** Every other pass looks inside a page and cannot see this.
Run Pass F before the per-page passes, because restructuring a page that is about to lose half its
content wastes the restructuring.

**Each idea belongs on the one page a reader would look for it on, and every other page links
there.** Durable explanation belongs on a permanent page; a plan belongs with the plan; a
step-by-step procedure belongs with the procedure. An idea that currently exists only inside a page
scheduled for deletion has to be promoted to a permanent page before that page goes.

**A cross-reference must carry a few words saying what is on the other end.** A bare `H2`, a bare
"see the design principles", or a link whose visible text is only a label leaves the reader unable
to tell whether they need to follow it, so they either break off and read the other page or skip a
claim they should have checked. Write "the model must beat the incumbent forecast at day-ahead
(H2)" with the link on the words, not "see H2". A label is an address, not a description, and a
link whose text is only the address is as opaque as no link at all.

**Find the candidates with `scripts/find_duplication.py`, then find the real duplication by
reading.** The script counts the 8-word runs each pair of pages shares, which is a cheap way to
choose what to read side by side:

```bash
uv run python .claude/skills/prose-review/scripts/find_duplication.py docs
```

**Verbatim overlap is the small half of the problem, so never treat that output as the work list.**
Across this repo's docs, verbatim runs account for 3% to 15% of a page, while the redundancy that
costs a reader is the same idea written out twice in different words. Give a sub-agent a pair of
related pages and ask what each says that the other also says, in whatever words — not which
sentences match.

**Two pages named for the same subject are a redundancy and a navigation fault at once.** Where one
subject has a short page in one section and a long page in another, decide which page a reader
arrives at first and what that page owes them, rather than merging by default: an overview that
genuinely routes a reader earns its length, and one that restates the page it links to does not.

**Before deleting any passage, grep for inbound links to it, anchors included.** A heading that
disappears takes its anchor slug with it, and `mkdocs build --strict` checks relative links only,
so a link from outside the repo breaks silently. **And never cut a passage that limits a claim in
the project's own favour** — material that makes the project look more modest is almost always
there on purpose.

## Fix at the level the pass found the problem

- A stop point traced to a missing prerequisite → move the missing fact earlier if it exists
  elsewhere in the document, move the stalling sentence later if it doesn't, or write the missing
  prerequisite in where the reader first needs it if it exists nowhere at all.
- A lead (bolded, or summarised for Pass A) that doesn't fit the argument's order → move or cut the
  paragraph, not the sentence.
- A paragraph reaching two conclusions → split it and write the second lead.
- An idea explained on a second page → delete the weaker explanation and link to the stronger one,
  with a few words saying what the link leads to.
- Only once ordering is settled, run the sentence sweep below.

## The sentence sweep: one pass per rule, or it finds nothing

**A reviewer asked to check every rule at once finds the loudest fault in each paragraph and moves
on, so the quieter faults survive.** The evidence is direct rather than theoretical. A combined
sweep of one section of the literature review reported **zero** findings. A one-rule-at-a-time
sweep of the same text, by the same model, found **30** — 26 pronouns, 2 unscoped superlatives, 2
umbrella nouns, and a money metaphor. Extended across the whole review, 6 sub-agents working one
rule per pass returned 469 findings on a file that had already passed several reviews.

**Say "one pass per rule" explicitly in the brief.** A brief that lists all eleven rules together gets a
sweep that honours none of them, because listing the rules is not the same instruction as
sequencing them.

**Require reasoning on the passes that return nothing.** A silent pass and a skipped pass look
identical in a report. Asking for the reasoning makes the difference visible, and catches a real
failure mode: one model reported a clean acronym pass on a range containing two acronyms that are
expanded nowhere outside a table and the reference list.

### The order to sweep in

The first two positions are fixed by dependencies; the rest run highest yield first, so that the
expensive passes get the freshest attention:

1. Clauses that can be deleted without the sentence losing anything
2. Long sentences carrying two claims, which a full stop would split
3. Pronouns and demonstratives, including "one", "ones", and "such a"
4. Unenumerated singletons and superlatives
5. Umbrella nouns — "thing", "something", "anything", "metadata"
6. Counting nouns that never say what was counted — "records", "sources", "items",
   "entries", "studies", "results"
7. Money metaphors for performance
8. Ambiguous "network"
9. Numerals
10. Serial commas
11. Acronyms expanded on first use
12. Sentences readable two ways, and noun-piles

Pronouns dominate every sweep run so far, by roughly an order of magnitude over any other rule.

**The deletion pass goes first because a clause that is about to be deleted is not worth
splitting, naming a noun in, or expanding an acronym in.** Every later pass costs less on text the
deletion pass has already thinned, and a sentence that loses a dead clause often stops being a
two-claim sentence at all, so the split pass has less to do. The deletion finding is a whole
clause the sentence can lose: the contrast that restates what the definition already settled, the
half-sentence saying what a passage is *not* about, the aside that repeats the subject. Delete it,
re-read the sentence, and keep the deletion unless the reader lost something. Where the reviewer
cannot tell whether the clause carries anything, it stays — this pass is for the clauses whose
deletion is obviously safe, not for close calls.

**The split pass goes second because splitting a sentence manufactures work for the passes behind
it.** A split leaves the second claim needing a subject, and the obvious subject is a pronoun
standing for whatever the first half named: "it plays on substation load the role the combined
reference plays on irradiance, and it is the bar that decides whether the project is worth its
money" splits into a second sentence opening "It is also the bar", whose referent is now three
nouns back. Seven of 94 splits in one pass did this. Running the pronoun pass next turns every one
of those stranded pronouns into an ordinary pronoun finding, and the numeral pass later catches a
split that leaves a sentence opening with a numeral. Sweep in the old order and each of those
faults has to be hunted afterwards by hand, on text that no pass is looking at any more.

**Where naming the stranded noun reads as pure repetition, the split was the wrong call.** Two
verbs sharing one subject carry one claim between them, so the sentence was never a finding and
the pronoun pass is telling you so. Rejoin it rather than inventing a subject for the second half.

**The reorder costs the split pass its cheapest way in.** Naming a noun, naming what a count
counted, and expanding an acronym all make sentences longer, so a sweep that ran those passes
first would hand the split pass a longer and more conspicuous corpus. Going first means the length
grep below sees every sentence at its shortest. Set the threshold lower than the 160 characters
that worked when splitting ran last, and treat the grep as a way in rather than as the pass: the
finding is two claims, not a character count.

**Rule 1 has a cheap way in: find the long sentences.** `grep -oE '[^.]{130,}\.'` over a
whitespace-normalised copy returns the sentences worth reading, and the finding is real where the
sentence carries two claims that read better apart. The joins to look for are "and" and "but", a
semicolon, an em dash, a "so", a "which", and a trailing participle — the last three are the ones a
sweep briefed on conjunctions alone will miss. A conjunction joining two verbs that share one
subject is not a finding, and neither is a split that would leave a fragment.

**Rule 5 has a cheap way in too: find the sentences carrying two or more numerals.** A count chain
is where the fault lives, and a methods sentence reporting a screening funnel is where a count
chain lives. `grep -oE '[^.]*[0-9]+[^.]*[0-9]+[^.]*\.'` over a whitespace-normalised copy finds
them, and most will be fine. The ones that are not hand the reader a different unit at each number
and define none of them.

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
- **Code blocks and the code inside them**, including comments.
- **Headings**, which Pass D owns. The sentence sweep leaves them alone. A heading rename changes
  the anchor slug and breaks inbound links from elsewhere in the docs, including absolute links
  from other pages, so it is worth doing deliberately rather than as a by-product.

Findings worth keeping are the mirror image: "one"/"ones"/"theirs"/"ours" standing in place of a
noun, a pronoun or demonstrative *opening* a sentence, a referent with two or more plausible
candidates, and a bare "that" or "this" standing for a whole clause.

## Briefing the sub-agents

**Give each agent one contiguous line range and the whole rule list, and require a line number, the
verbatim quote, and concrete replacement wording for every finding.** A finding without replacement
wording cannot be triaged and is worth nothing.

- **Chunk by section boundary, not by equal line count**, so no agent owns half an argument.
  Roughly 5,000 to 7,000 words per agent worked well; the reference list needs no sweep.
- **A per-section pass finds the defects inside a section; only a whole-document pass finds a
  paragraph sitting in the wrong section.** What a split cannot see is the join. An agent holding
  one section cannot notice that its section repeats a paragraph from an earlier one, or that a
  fact it asks for arrives two sections later. Run Pass A over the complete document as well, and
  treat that whole-document pass as the only one entitled to move a paragraph between sections.
- **Warn about hard wrapping.** In a wrapped file a sentence spans several lines, so a plain grep
  for a phrase misses most matches. An agent that greps will report a present passage as absent.
  Tell the agent to normalise first: `tr '\n' ' ' < FILE | tr -s ' ' | grep -o 'phrase'`.
- **Warn that a diff overstates what changed, when the review is scoped to a diff.** Hard wrapping
  means an edit to one word reflows the whole paragraph, so `git diff` presents untouched sentences
  as added lines. Tell the agent to check any finding against the merge-base version before
  reporting it, or the sweep returns pre-existing prose as new findings.
- **Report only. Edit no file, run no `git` command that writes, spawn no sub-agents, use no
  browser tools.** Concurrent agents editing one file collide, and a finding that lands before
  triage cannot be rejected.
- **Ask for absolute line numbers in the file**, not offsets into an extracted chunk.
- **Ask for the findings as JSON** — a list of objects carrying `file`, `quote` and `replacement` —
  so `scripts/apply_findings.py` can apply the whole batch without anything being retyped. Tell the
  agent to quote the sentence as the page reads; the script matches whether or not the agent kept
  the markdown, and rejects a quote that occurs twice rather than guessing which one was meant.
- **Do not edit a file while a sweep of it is in flight.** Every line number in every report still
  running is measured against the file as the agents found it.

**Where the sweep covers pages a branch only partly wrote, gate each finding on the merge-base.**
Tell the agent to skip prose that predates the branch, then check it mechanically too, because the
instruction leaks. A finding that is already on `main` belongs to whatever issue owns the unswept
rest of the docs, not to the branch in hand. `scripts/apply_findings.py --merge-base <ref>` runs
the check: it normalises the whitespace, projects the markup away, and tests the quoted sentence
against `git show <ref>:<path>` before applying anything.

**The leak is small, and an earlier estimate of it here was wrong.** Of 171 findings gated this
way, the gate caught one. The figure this skill carried before — about one in nine — came from
sampling sentence fragments picked by hand rather than the quotes the agents actually filed, and
overstated the leak by more than an order of magnitude. Measure a yield on the findings that were
really filed, or quote no yield at all. The gate stays worth running at one in 171, because it
costs one flag and the finding it catches is prose the branch has no business touching.

## Which model to give the review to

**Use Opus 5 for anything an outside reader will see, Sonnet 5 for routine internal sweeps, and
never Haiku 4.5.** All three were given a byte-identical brief over an identical 433-line range.

| | Haiku 4.5 | Sonnet 5 | Opus 5 |
|---|---|---|---|
| Total findings | 8 | 116 | 100 |
| Pronouns | 6 | 108 | 73 |
| Unenumerated singletons and superlatives | 1, misfiled | 0 | 8 |
| Acronyms | 0, with a false all-clear | 2 | 3 |
| Sentences readable two ways | 0 | 1 | 8 |
| Sub-agent tokens | 93,000 | 174,000 | 100,000 |

**Opus returned fewer findings than Sonnet, and better findings.** Opus applied the
author-possessive carve-out above without being told, and listed every candidate it had rejected
with a line number. Listing the rejections makes the restraint auditable rather than assumed.
Sonnet reported that whole category as findings.

**The gap that matters most is the singleton-and-superlative rule.** Sonnet reported no findings
there, with confident reasoning. Opus found eight, one of which contradicted another passage of the
same file 370 lines earlier. On a document a funder publishes, an unscoped absence claim is the
costliest fault in the rule list. A confident empty pass on that rule is therefore the wrong kind of
wrong.

**Opus also found a defect outside the rule list**, in a passage that had survived several earlier
review rounds: the enumeration mismatch Pass C now looks for. A sweep is worth reading for what it
notices as well as for what it was asked to find.

**Haiku's failure was recall and self-verification, not discipline.** Haiku honoured the
one-pass-per-rule structure and invented no quotes, and its six pronoun findings were a strict
subset of Sonnet's. Haiku reported a clean acronym pass over a range containing two acronyms that
are expanded nowhere outside a table and a reference list. A false all-clear is worse than a low
count, because it is indistinguishable from a real all-clear.

## Triage before applying anything

**Assume roughly half of any reviewer's findings are wrong, and read the line a finding quotes
before acting on it.** Reviewers in these runs variously reported a cross-reference pointing the
wrong way when the document already pointed the right way, proposed renaming a metric to a word
other than the one the cited paper uses for it, three times proposed a bolded lead the page
deliberately omits, and proposed a replacement sentence repeating a factual error a different agent
had disproved in the same run. An agent reading a 3,000-line document reports what it remembers,
and what it remembers is sometimes not what the line says.

- **Triage the whole set together, not agent by agent.** Agents contradict each other, and the
  contradiction is the signal.
- **Verify every quote exists and is unique before editing.** Run the whole batch as a dry run that
  asserts a match count per finding. A quote can be stale because a paragraph was rewritten, and it
  can match more than once.
- **A quote that matches more than once is usually a lifted sentence, and every copy needs the same
  fix.** A review with a summary lifts sentences verbatim from its own body — one sentence had three
  copies. Fixing the body alone makes the summary silently disagree with the section it came from.
  Different agents own those copies, so the same finding also arrives twice and needs deduplicating.
- **Check a proposed replacement against the rules too.** Two replacements in one sweep swapped a
  banned "one" for a banned "it", and another defined a term using the term itself.
- **When two agents independently propose the same cut, that is evidence the passage reads badly,
  not evidence the cut is right.** The third option is usually a rewrite.
- **A cut that makes the document more self-serving is almost always wrong.** Material that limits
  the author's own claim is usually there on purpose.

## Applying and checking the edits

**Apply a batch with `scripts/apply_findings.py`, which refuses the edits that would corrupt the
page.** Write the findings to a JSON file — one object per finding, carrying `file`, `quote` and
`replacement` — then dry-run the script, read what it refused, and re-run it with `--apply`:

```bash
uv run python .claude/skills/prose-review/scripts/apply_findings.py findings.json --merge-base REF
uv run python .claude/skills/prose-review/scripts/apply_findings.py findings.json --apply --merge-base REF
```

The paragraphs below say what the script is defending against. Read them before hand-editing
anything the script refused, because what the refusal was for decides how the edit has to be made
instead. Every defect named below was written by an earlier apply script and then passed
`pymarkdown scan`, `mkdocs build --strict` and `check_information_loss.py` unnoticed.

**A sub-agent quotes the sentence with the markdown stripped, so a wrap-tolerant substitution still
misses it.** `[Gijon et al. (2025)](https://doi.org/…) write` comes back as `Gijon et al. (2025)
write`, and 57 of 98 findings failed to match on that alone. The script matches against a
markup-stripped projection of the file that keeps an offset map back to the raw text, then splices
the replacement in run by run, copying raw characters wherever the wording is unchanged so the
links and bold markers the agent dropped survive. The replacement is written without markup too,
so splicing it in whole would delete every link in the sentence.

**The offset map records where each character's markup ends, not only where the character sits.** A
map of bare character positions resumes the raw text before a closing backtick, so a serial comma
inserted after `n_h3_cells` is written as `` `n_h3_cells,` `` — the comma inside the code span. The
backtick count is unchanged, so every markup check below is satisfied and only `check_structure.py`
catches it. Each projected character therefore carries the opening backtick, `[` or `**` before it
and the closing backtick, `](url)` or `**` after it, and a splice writes between those bounds
rather than across them.

**A quote that stops short of a trailing clause matches nothing at all.** A sub-agent routinely
ends its quote before a parenthetical the file actually carries, while every change it proposes
sits in the head of the sentence. Trimming the words the quote and the replacement share at the end
makes the truncated quote match, which the script does before searching.

**A bolded lead is the one span whose full stop belongs inside its markers, and the splice puts it
there.** Splitting is the pass that reaches this boundary, because the join it breaks is often the
comma right after a bolded phrase or a code span. `**the same information**, and they are harder to
spot` becomes `**the same information.** They are harder to spot`, with the stop pulled inside the
`**` that opens the block — through a blockquote's `>` and a list item's bullet alike. Every other
span takes its punctuation outside: a serial comma after a code span, a link or a mid-sentence bold
is written after the closing marker. Counts over the 78 markdown files under `docs/`, in the
repository root and in `.claude/skills/` are what set that rule. A lead opening a paragraph carries
the stop inside its `**` 451 times against 6 that do not, a lead on a list item 404 times against
1, and a lead in a blockquote 38 times against 1. A bold span in the middle of a sentence goes the
other way, with 144 commas and 70 full stops after its closing `**` against no comma and 2 full
stops inside one. Single-asterisk emphasis was never counted, so a stop after one is left where the
replacement put it. The script still counts `**`, backticks and links in the paragraph either side of the splice,
and refuses any edit that changes a count.

**An insertion anchored on a sentence can land inside a bolded lead**, between the lead's opening
`**` and its closing `**`, leaving both markers unbalanced. The rendered page then turns bold on
and leaves it on for the rest of the section. The same marker count catches this case, and the
per-paragraph check below catches one that reaches the file by another route.

**A quote can match inside a fenced code block, where the words are a command rather than prose.**
The getting-started page carries the comment `# create the virtualenv and install all workspace
packages` inside a fenced block, and a serial-comma finding quoting those words rewrites the
command. Nothing downstream notices: the page still lints, still builds, and `check_structure.py`
sees no marker move. The script reports such a finding as `code block` and writes nothing, the way
it already refuses one landing in a skill file's YAML frontmatter. Reword the finding to quote the
prose it meant, or leave the block alone. A fence indented under a list item counts, because that
is where most of this repo's fenced blocks sit — every one on the code-style page, and two of the
six on the getting-started page.

**A replacement spanning a different number of lines from the text it replaced invalidates every
line index taken before the splice.** A three-line span rewritten as one line moves every following
line up by two, so a re-wrap driven by the old indices reflows the wrong lines: one such re-wrap
merged two numbered list items into `…instead. 2. **No translation gap.**`, and another swallowed
the closing `---` of a skill file's YAML frontmatter into the description above it. Recompute the
unit boundaries on the spliced text, never on the text as it was before, and leave the YAML
frontmatter out of the re-wrap altogether — its indented lines read as list markers.

**Reflow only the unit the change landed in, and solve that unit's width rather than assuming
one.** Re-wrapping a whole file buries the edit: one 8-edit batch produced a 412-line diff before
the reflow was narrowed. A unit is finer than a paragraph, because a bullet list written without
blank lines between its items is a single paragraph, and re-wrapping the whole of one reflows every
sibling bullet. The width is not uniform across this repo either — the pages this skill has swept
are wrapped anywhere between 94 and 100 characters — so the script solves each unit's width by
finding the width that reproduces the unit exactly, and falls back to the width most of the file's
other units solve to.

**Each guard above has a regression test in `tests/test_prose_review_apply_findings.py`**, run by
the repo's own `uv run pytest`. The tests live in the root `tests/` directory because pytest skips
hidden directories, so a suite inside `.claude/skills/` would never run. Change the splice, the
projection or the re-wrap and run them.

**Then check that the batch changed the words and nothing else**, before running the repo's own
gates:

```bash
uv run python .claude/skills/prose-review/scripts/check_structure.py HEAD
uv run pymarkdown scan -r docs README.md CLAUDE.md packages/*/README.md
uv run mkdocs build --strict
```

`check_structure.py` compares each changed file against a git ref and fails when a link, a bold or
code span, a list item, a heading, a table row or a blank line has been *lost*, when a paragraph
gains an unpaired `**` or backtick, when the trailing newline goes, or when a YAML frontmatter
block stops closing. A count that *rises* is reported and not gated, because naming the noun a
pronoun stood for legitimately adds a code span — writing `file` or `prose-review` where the
sentence said "it" — while no sweep can legitimately lose one.

**Renaming a heading breaks every absolute link to its old anchor, and `scripts/check_docs_links.py`
is what tells you.** `mkdocs build --strict` validates relative links only, so a link from a
docstring or a comment to the published site survives a heading rewrite with nothing to catch it.
Run `uv run python scripts/check_docs_links.py` after any heading change; it reports the file, the
line and the closest real anchors on the target page. A heading whose text contains an em dash
generates a single separator, not two: `## H2 — a hundred experiments` becomes
`#h2-a-hundred-experiments`.

**Neither command catches content dropped at render time either.** Python-Markdown treats any line
starting with `#` as a heading even without the space CommonMark requires, and it silently discards
a table row's extra cells when the row carries more cells than the header. A whole paragraph of
sourced prose sat invisible on the published site for exactly that reason. Both are greppable in
the built output:

```bash
python3 -c "import re;[print(i+1,l[:60]) for i,l in enumerate(open('FILE')) if l.startswith('#') and not re.match(r'#{1,6} ',l)]"
uv run python .claude/skills/prose-review/scripts/check_render_loss.py
```

## Did the restructure lose anything?

**Run this after any large structural change, and skip it after a sentence-level sweep.** Splitting,
merging, moving and cutting paragraphs moves text in bulk. The loss is silent: nothing fails, the
page still reads well, and the missing caveat is noticed only by the reader who needed it.

Three checks, cheapest first. The first two are mechanical and take seconds:

```bash
uv run python .claude/skills/prose-review/scripts/check_information_loss.py <old-ref> <path>
```

1. **Diff the inventory of things that cannot survive being dropped** — every number, every link
   and citation, every direct quotation, every bolded term. Losing one is always a defect, never a
   rewording. The script above extracts and diffs all four between a git ref and the working tree.
2. **Shingle the old text against the new.** Every 9-word run of the original appearing nowhere in
   the rewrite is either a deliberate rewording or a deletion. Only a human can tell which, but the
   list is short enough to read, and the script prints it.
3. **Ask a fresh sub-agent what went missing.** Give it the before and after as two scratchpad
   files and one question: what does the old text state that the new text does not? Tell it to
   report **hedges, caveats, scope limits and attributions first** — those are the losses that
   matter and the ones a rewrite drops most easily. A restructure that quietly makes the document
   more confident than its evidence supports is the failure mode this check exists for, and it is
   the same fault as the triage rule above about self-serving cuts.

## Re-run after a restructure

Moving paragraphs can break a bolded lead that referred to "the previous section", or introduce a
fact before its new position's prerequisites are met. Re-run Passes A to E after a restructure, not
just once at the start, and re-run the balanced-bold count with them.

## Sweeping the prose inside Python files

**A sweep of docstrings and comments follows every pass above, plus the mechanics below, and one
rule that reverses the instinct the rest of this skill trains.** The prose lives in files that
`ruff`, `ty`, and `pytest` all have opinions about, two-thirds of it renders into the API docs
beside the READMEs, and a docstring can describe behaviour the code stopped having. None of that
applies to a `docs/` page.

**Aim for a net-neutral or higher word count, and cut only excessive duplication.** A net-neutral
target is the opposite of the default instinct on a prose task, and it is not negotiable: both
attempts to shorten this repo's code prose drew the same objection. The first cut `src/` by 29% and
was closed unmerged; the second targeted duplication rather than word count, cut 6%, and still drew
review comments asking for `main`'s fuller wording back. Losing information from a docstring is
worse than a little duplication with `docs/`. **Say this at the top
of every sub-agent brief**, because a reviewer asked to improve prose will otherwise recommend
tightening, and every one of those findings has to be thrown away. The rules themselves — the
duplication bar, worked examples, load-bearing links, Dagster docstrings, the README collision —
are on [the code-style page](https://openclimatefix.github.io/nged-substation-forecast/architecture/code-style/)
and are not repeated here.

**Never delete prose in the same edit that adds a link.** Add the link beside the prose. Where
prose genuinely has to go, that is a separate deletion, justified on its own and visible as such in
the diff. The audit that produced this rule found 10 passages whose reasoning had been *replaced*
by a link mid-pull-request, six of which still had that reasoning on `main`.

**One search shape finds one shape of fault.** A grep for `why X: <link>` found 10 load-bearing
links; a later pass reading the passages found four more that stated a claim and left its
justification to the link, which no template would have matched. Budget a second sweep that reads
rather than greps.

**Check which packages render their README beside their module docstrings.** `docs/api/<pkg>/`
pages `{% include %}` the README and then emit `::: module` directives, so a reader meets both at
once and a README that restates a module docstring is the one deletion the duplication rule
sanctions. Only the packages with a page under `docs/api/` are affected; the others are not, and
their READMEs should usually grow rather than shrink.

**Verify which worktree you are in before reading a single file, and give sub-agents the absolute
path.** This repo keeps a worktree per branch under `.claude/worktrees/`, and a session's primary
directory is often a bridge worktree on `main` rather than the branch under review. The failure is
silent and *inverts* your conclusion: `main` legitimately contains the text the branch removed,
so a correct finding reads as false. Triaging one audit against the wrong worktree nearly rejected
10 valid findings. Run `git rev-parse --abbrev-ref HEAD` first, and tell every sub-agent not to
`cd` to the repository root.

**Fix obviously-wrong prose you meet outside the nominal scope.** A prose sweep is the one time
anybody reads these files closely, so filing a defect for later wastes the pass that found it. When
a claim is checkably wrong — six passages called `init_time` "the NWP partition key" when
`delta_store` partitions on two columns — correct it, and say in the pull-request body why the
change reaches outside its stated scope.

### Guards to run on a code prose sweep

Four checks, none of which a `docs/` sweep needs:

- **The abstract-syntax-tree guard**, which proves a prose-only change really was prose-only: parse
  each file before and after, blank every string constant, and compare `ast.dump()`. Anything that
  survives is a behavioural change, and belongs in the pull-request body as a list a reviewer can
  reject as a unit — or in its own pull request.
- **`pydoclint`**, for a docstring whose `Args:` or `Returns:` section disagrees with the
  signature. Ruff's `D417` sees only an `Args:` section that is present and incomplete, so it is
  silent on the two failures a rename actually produces. `pydoclint` runs as a pre-commit hook and
  as a CI step, so a sweep only has to read its output.
- **A grep for reStructuredText cross-reference roles**, which reach the API pages as literal
  markup because nothing interprets them. A `pygrep` pre-commit hook rejects them now, so a sweep
  inherits the guard rather than running the grep itself. The lesson generalises past the one hook:
  **a sweep that changes how a docstring renders has to read the rendered page.** The same
  blindness hides an empty section heading, a nested list that flattens, and prose in a private
  function that mkdocstrings never renders at all.
- **Link resolution against the *built* site**, not a guessed slug: `uv run mkdocs build` and then
  check each URL's page and `#anchor` against the generated HTML. `scripts/check_docs_links.py`
  does this repo-wide and is also a hook.

Then the ordinary green-before-push set — `ruff check`, `ruff format`, `ty check`, `pytest`,
`pymarkdown scan`, `mkdocs build --strict`.

### What to run, and when

**Sweep a package when that package changes materially, not on a calendar.** A monthly pass over
unchanged code re-reads what it read last month and finds almost nothing, while the drift that
matters is caused by events: a refactor that changes public signatures, a docs restructure that
moves anchors, a measurement that supersedes a number quoted in three places. Trigger on those.

**Use sub-agents to author the findings and a different, fresh sub-agent to review the diff.** An
authoring agent cannot adversarially review its own work. Two reviews caught a wrong claim the
authoring pass had introduced *and* an overclaim written during triage; one of them ran a mutation
to check a comment's assertion about which test catches a bug, and found the comment named the
wrong test. Verify every finding against the code before applying it — a wrong "fix" to a docstring
costs more than a missed fix, because the next reader trusts the docstring.

## See also

`long-form-prose` runs the planning discipline before any prose exists, for drafting rather than
reviewing. For a rewrite that adds whole new sections to an existing page, run this skill on the
existing text first, then switch to `long-form-prose` to outline the new sections against the
result. Outlining new material against a page whose own structure hasn't been checked risks
building the new sections on prerequisites the existing page never actually establishes.

`code-style` holds the rules a code prose sweep is measured against, and is where a new rule
belongs; this skill holds only the procedure for applying them.

`literature-review` owns the accuracy round, which is a separate pass from this one. Reach for its
`rsub` helper only for a single hand-edit; a batch of findings goes through
`scripts/apply_findings.py` here, which `rsub` predates and which handles the markup, the wrap
width and the merge-base gate that a bare substitution does not.
