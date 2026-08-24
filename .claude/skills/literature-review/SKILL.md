---
name: literature-review
description: >-
  How to research, write and adversarially review a literature review that a third party will
  publish under their own name: assembling a local library of full texts through legitimate
  open-access routes, the `pdftotext` traps that silently drop or scramble the text you then quote,
  what a useful entry on a paper contains, the four classes of claim that go wrong (absence,
  superlative, number, attribution), the reviewer personas and how to brief sub-agents so their
  findings are checkable, and the mechanical checks — citation closure, wrap-tolerant edits,
  render verification. Load before starting a literature review or a state-of-the-art section,
  before adding a paper to one, or before running a review round over one.
---

# Writing a literature review that someone else will publish

This skill is for a review that leaves the building: a state-of-the-art section in a report a
funder publishes, a survey for an external audience, anything where the cited authors will read
what you wrote about their work. **The worst possible outcome is a wrong claim about someone's
published work**, and the whole routine below is built around making that outcome unlikely.

The prose rules in [`CLAUDE.md`](../../../CLAUDE.md) under "Prose style" apply throughout and are
not repeated here. Two of them do most of the work in a review: say what the source found rather
than what is always true, and name the thing instead of writing "it".

## The shape of the work

1. **Fix the questions first.** Write down the problems the project actually has to solve, then
   review the literature against those problems. A review organised around the questions is
   useful to the funder; a review organised around the papers is a reading list.
2. **Assemble the library** — full texts on local disk, not abstracts.
3. **Draft**, one problem at a time, with the sources open rather than a summary of them.
4. **Fact-check new prose before you commit it**, not after. A section that lands and is then
   corrected twenty times leaves a history that is honest but avoidable, and every uncorrected
   hour is an hour the wrong claim could have been read.
5. **Review in rounds**, each round asking a different question. Accuracy and readability are
   separate rounds, because they find different defects and a reviewer asked for both does
   neither well.
6. **Triage every finding against the source yourself** before changing a word.

Rounds 5 and 6 repeat. Expect the review to keep finding real defects for longer than feels
reasonable.

## Assembling the library

**Get the full text. An abstract is not a source you can quote from.** An abstract states what was
done and that it worked; it almost never carries the number, the baseline, the horizon or the
caveat that makes a finding worth citing. Several of the worst errors caught in review came from
drafting off an abstract and inferring the rest.

**Draft from the source, not from a summary of the source.** A research agent's report — or your
own notes, or a previous draft — is a map of where to look, never a substitute for looking. Writing
a section against notes rather than against the papers is the single most productive way to
introduce errors that read perfectly well: a number drifts (a source's "approximately five years"
became "more than a decade"), a hedge evaporates ("their internal benchmark was easier" acquired
"and saying by how much", which the source never says), and the summariser's paraphrase hardens into
your assertion. One section written this way needed twenty corrections in two thousand words, an
error rate several times worse than anything drafted with the PDFs open. If you must write against
notes to get the shape down, treat that draft as unpublishable until every sourced sentence has been
opened against its source.

**A source you are citing for one claim may bear on the claim next to it.** The same section cited a
paper for one fact while contradicting that paper's account of a second, three sentences away —
the absence claim it made was refuted on a page it had already quoted from. When you open a source,
read what it says about the neighbouring claims too, not only the sentence you came for.

**Only legitimate routes.** Never use a pirate mirror. In rough order of yield:

- **Unpaywall** — `https://api.unpaywall.org/v2/<doi>?email=<address>` names every open location
  a paper has, including repository copies the publisher page does not link to.
- **OpenAlex** — `https://api.openalex.org/works/doi:<doi>` gives `best_oa_location` and the
  full location list.
- **Semantic Scholar** — `https://api.semanticscholar.org/graph/v1/paper/DOI:<doi>?fields=title,openAccessPdf,externalIds`
- **arXiv** — the preprint is often the same content; check the version and say which you read
  if the published version differs.
- **CORE** and **Crossref** for repository copies and metadata.
- **Institutional repositories** directly, when an author's university is known.
- **The author's own conference slides**, which are often posted openly and sometimes carry the
  number the paper puts behind a paywall.
- **Network-operator innovation projects** are usually published openly: in GB, the Energy
  Networks Association Smarter Networks Portal and Ofgem's site, free and without registration.

**When a paper cannot be obtained, say so in the review at the point of citation**, and record it
in the library's README with what was tried. A documented negative is a result; a quiet reliance
on an abstract is a defect waiting for a reviewer to find. Do not let an unobtainable paper carry
a load-bearing claim.

**Never bypass a bot challenge** to reach a document, and never let an agent do it. If a route is
gated, it is closed.

## Reading the PDFs: three traps that silently corrupt what you quote

Each of these produces plausible-looking text, so nothing warns you that the sentence you are
about to quote is not the sentence in the paper.

**Form-feed page breaks swallow records.** `pdftotext` separates pages with `\014`, and a line-based
tool reads the last line of one page and the first of the next as a single line. When extracting a
table or a list that spans pages, convert the form feeds first:

```bash
pdftotext paper.pdf - | tr '\014' '\n' | grep -n 'RMSE'
```

**Two-column papers interleave.** Default `pdftotext` reads across the page, so a sentence in the
left column continues into an unrelated sentence in the right. Always use `-layout`, and extract
one page at a time when quoting precisely:

```bash
pdftotext -layout -f 7 -l 7 paper.pdf -
```

**Soft hyphens break every match.** A word hyphenated across a line break carries `\xc2\xad`, so
grepping for "forecasting" misses "fore-casting". Strip them before searching:

```bash
pdftotext -layout paper.pdf - | sed 's/\xc2\xad//g' | tr '\014' '\n' > paper.txt
```

Cache the cleaned text next to the PDF. Reviewers will need to check the same passages, and
re-extracting per query wastes their time and yours.

## What a useful entry on a paper contains

**For a closely relevant paper, extract the conclusions, not a nod.** A sentence saying that a
paper "investigated substation load disaggregation" tells the reader nothing they could act on.
Give the method that won, the margin it won by, the baseline it beat, and the lesson that
transfers. If a paper is worth citing at all, it is worth two or three sentences of what it found.

**Give every number the four things that make it comparable**: what was forecast, at what
aggregation level, over what horizon, and against what baseline. A percentage error with no
baseline is not evidence, and a review that reprints one lends it credibility it has not earned.
Where a paper's headline number fails this test, either say so at the point of citation or leave
the number out.

**Prefer errors normalised by something physical** — a rating, a capacity — over errors normalised
by the load that happened to occur, and say which normalisation each quoted number used. Numbers
normalised differently cannot be put in the same table without a note saying so.

**Describe scope, not failure.** A paper that did not do X because X was not its question has not
"failed to" do X, "only" done Y, or "stopped short". Loaded verbs and adverbs — only, merely, just,
fails to, barely, quietly, does not even — turn a description of scope into a criticism, and the
author will read it that way. Describe what the paper did and what question remains open, and let
the gap speak for itself.

## The four classes of claim that go wrong

**Absence claims** — "nobody has", "no published work", "the first". These are only ever as good
as the search behind them. State what was searched so the reader can judge, and prefer a narrower
true claim to a wider one you cannot defend. Several such claims in this session's review survived
three accuracy rounds and were refuted by a fourth with a wider net.

**Superlatives** — "the closest paper to our problem", "the largest study". Two of these in one
document contradict each other, and a reader who notices stops trusting the rest. Scope every
superlative to a set you have actually enumerated: "the closest paper among the ones this exclusion
covers".

**Numbers.** Check every number against the source text, not against your notes. Check whether the
same fact appears twice in different units — a ratio in one place and the two figures behind it in
another is one fact stated twice, and the reader who divides them wonders which is wrong.

**Attribution.** Which organisation did which part? A tool built by one group and deployed by
another, a method named after a project that did not invent it, a benchmark run by a competition
rather than by its host — these are easy to blur and embarrassing to get wrong in a document the
parties will read. **When the claim is about software, check the tagged source, not the current
release.** A review once said an open-source stack "carries no trace" of a method its own
maintainers had published; the method shipped in every release across a whole major version, and
the current release still ships its simplified descendant pre-trained. Fetching three tags from the
project's repository settled in a minute what the sentence had got backwards.

## Consistency with your own project's documents

**A review that states a commitment must match the document that owns that commitment.** The most
valuable single finding in this session was that the review told the funder how a metric would be
computed, and the project's own metrics page specified something different. Before publishing, list
every commitment the review makes about what the project will do or measure, find the internal
document that owns each one, and reconcile them. When the two disagree, decide which is right — the
review is sometimes the document that is correct, and then the internal one changes.

**Do not narrow the project's own strand to fit a tidy sentence.** A claim that some approach is
"the least well supported in the literature" may be true of one half of that approach and false of
the other. Check which half the project is actually betting on.

## Reviewing: the personas

Run these as separate sub-agents, in parallel, each reading the whole document. They find
different things and merging them into one brief loses most of the value.

- **The junior colleague.** Knows the domain, not the method. Every sentence they read twice is a
  defect; every unexplained piece of jargon is a defect. Ask them to state the document's argument
  in three sentences, which tells you whether the spine landed.
- **The senior manager.** Will not read end to end. Reads the headings, then three paragraphs at
  random. Ask specifically for every paragraph opening with an unresolved pronoun or demonstrative,
  and whether the headings alone tell the story.
- **The cited author.** Looks for their own work described as a failure rather than a scope,
  for loaded verbs, for a caveat they were careful to state and the review dropped, and for credit
  taken for their insight.
- **The regulator or funder.** Looks for unsupported novelty claims, for scope not under control,
  for duplication of work already funded, for whether the benefit to the end customer is visible,
  and for commitments too vague to be held to.
- **The paying customer** — the engineer or operator the work is for. Asks what they get, when, and
  what in the review changes what they would do on Monday morning.
- **The house-style auditor.** A mechanical, exhaustive sweep, not a taste review: pronouns,
  citation format, spelling variant, units, dashes, acronym expansion, heading parallelism. Ask for
  every instance, not a representative sample, and ask for the replacement wording each time.

Two further rounds ask different questions and belong on their own:

- **Accuracy.** Split the document into chunks, one agent per chunk, each checking every claim
  against the source PDFs on disk. Tell each agent explicitly that the full texts are local and
  where they are, or it will work from what it can find online.
- **Relevance and bloat.** Is the review as a whole, and each paper in it, relevant to the
  questions the project has to answer? This finds things an accuracy round cannot: a correct
  passage about a paper that does not matter.

## Briefing a sub-agent

Every reviewer brief needs all of these, or the findings come back unusable.

- **The audience and the stakes.** Who publishes it, who reads it, what a wrong claim costs.
- **The hard constraints** — passages that must not change, structural conventions that are
  deliberate, and the fact that length is or is not a limit. Without these, a third of the findings
  will propose undoing a decision already made, and you will spend the triage rejecting them.
- **Line numbers and quoted text for every finding, plus a concrete replacement sentence.** A
  finding that says "tighten this" cannot be triaged and is worth nothing. Insist on the actual
  wording.
- **Report only; do not edit.** Concurrent agents editing one file collide, and a finding you
  cannot triage before it lands is a finding you cannot reject.
- **Do not spawn sub-agents.** A grandchild agent cannot report back once its parent has finished,
  so the work is simply lost.
- **No browser tools.** A browser tool can open a file-download dialog on an unattended screen.
  Command-line only: `curl`, `grep`, `sed`, `awk`, `python3`, `pdftotext`.
- **Warn about the wrapping.** In a hard-wrapped file a sentence spans several lines, so a plain
  grep misses most matches. Tell the agent to normalise whitespace first.

## Triage: assume roughly half the findings are wrong

**Check every finding against the document and the source before acting on it.** In this session's
rounds, agents variously: audited a slide deck and reported a conclusion about a journal paper of
the same authorship; called a set of figures unverifiable when the full text was cached on local
disk; proposed deleting a table row that another section cross-referenced; and twice proposed
cutting a passage that an earlier round had added deliberately to *limit* a claim in the project's
own favour.

**A search that returns nothing is not a finding until you have proved the search ran.** A grep
against a mistyped path, a wrong directory or a PDF that extracted to zero lines prints exactly what
a genuine absence prints. Before recording any "we found no…", confirm the corpus was actually
read — count the lines the extraction produced, or grep for a string you know is present. This is
the same failure as the hard-wrapping trap: both return a confident empty result from a search that
never happened.

Three triage rules earn their keep:

- **A cut that makes the document more self-serving is almost always wrong.** Material that limits
  your own credit is usually there on purpose.
- **Before deleting anything, grep for inbound cross-references to it.**
- **When two agents independently propose the same cut, that is evidence the passage reads badly,
  not evidence the cut is right.** Look for the third option: usually a rewrite rather than a
  deletion.

**Your own drafting is in scope for the same scepticism.** The single clearest error in this
session was mine, not an agent's: writing a plausible mechanism into a sentence about a paper's
method when the abstract did not state it. The failure mode is inference dressed as citation, and
it does not announce itself.

## Mechanical checks

**Citation closure, both directions.** Every in-text citation must appear in the reference list,
and every reference must be cited in the body. Do this with a script rather than by eye — match on
canonicalised URL, since the link text varies. `scripts/check_citations.py` in this skill's
directory does it — `python3 .claude/skills/literature-review/scripts/check_citations.py <file>` —
and the checks it runs are:

- every in-text citation is hyperlinked (find citation-shaped text outside a link)
- every body link resolves to a reference entry, and every reference entry's URL appears in the
  body
- no duplicate URLs in the reference list
- the reference list is in alphabetical order by first author
- the in-text label's surname and year agree with the reference entry they link to
- citation form matches house style: `Author et al. (year)` for three or more authors,
  `Author and Other (year)` for two

**Edit a hard-wrapped file through a wrap-tolerant substitution**, never with a literal string
match, or every edit whose target spans a line break fails:

```python
def rsub(pat, rep, n=1):
    """Replace `pat` (whitespace-insensitive) with `rep`, asserting it matched exactly n times."""
    global t
    r = re.compile(r"\s+".join(re.escape(w) for w in pat.split()))
    t2, c = r.subn(lambda m: rep, t, count=n)
    assert c == n, f"count={c}: {pat[:70]!r}"
    t = t2
```

The assertion is the point: a silent zero-match edit is the commonest way a batch of fixes half
lands.

**A span anchored by its opening and closing words deletes the wrong thing when the closing words
recur.** Deleting by line number goes stale after the first cut, so the natural fix is to anchor a
span by the words it starts and ends with. That fix has its own failure: `opening + .*? + closing`
finds the *next* occurrence of the closing words, and if those words appear again later, the
deletion swallows everything in between and reports success. Twice in one session that removed
whole sections — once about 250 words, once about 550 — with nothing in the output to show it. Two
guards between them make the tool safe, and neither costs anything:

- **Cap the match length** and refuse anything longer, because a legitimate sentence-level cut is
  tens of words and a runaway one is hundreds.
- **Print what was removed** — the word count, the first few words and the last few — so a cut that
  ran past its intended end is visible in the log rather than in the finished document.

The related failure is quieter still: when the closing words are a *suffix of the opening words*,
the pattern can never match and the cut is simply skipped. That one at least reports zero matches,
which is why an assertion on the match count is not optional.

## Writing a short version of a long review

**Build the short version by lifting whole sentences out of the long one, not by writing a summary
from memory.** Every sentence in the long review has already survived fact-checking; a freshly
written paraphrase has not, and drafting from notes rather than from sources is what produced
twenty errors in two thousand words the last time it was tried. Lifting also keeps the two documents
saying the same thing, which matters when a reader has both.

**Write the extraction as a script, not as retyping.** A script that pulls each span out of the
source file and refuses anything that does not match exactly once cannot mistype a number or drop a
hedge. The connective prose written as literals in that script is then the only text in the short
version that has never been checked, and it is small enough to check by hand.

**Audit the result with shingles.** Split the short version into overlapping nine-word runs and
check each against the long review. Every run that is not found is either connective prose you
wrote or a join between two lifted passages — and the joins are where the faults are: a stranded
"the ones" whose referent was in the sentence you cut, a bare "0.07" whose result sentence went
with it, a space left before a full stop. Read every miss.

**Expect the cuts to strand referents.** A pronoun or a demonstrative that was clear in the long
review points at nothing once the sentence naming its subject is gone. After each round of cutting,
list every sentence-initial "it", "this", "that", "these", "those", "they" and "the ones" and check
that the noun is still in the previous sentence.

**Verify the render, not just the lint.** A clean lint and a successful build both pass on markdown
that renders visibly wrong — see the `mkdocs-authoring` skill. After each batch, rebuild and count
the elements in the generated HTML: headings at each level, table count, reference list items, and
a grep for stray `**` or `](http` that a broken edit leaves behind.

```bash
uv run pymarkdown scan -r docs
uv run mkdocs build --strict
```

**Reflow after editing**, so the wrapping stays consistent and the diff of the next edit stays
readable.
