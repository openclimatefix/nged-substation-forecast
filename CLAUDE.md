# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this
repository.

## Commands

```bash
# Install dependencies
uv sync

# Linting & formatting
uv run ruff check .            # check
uv run ruff check . --fix      # fix (never over a marimo notebook - see marimo-notebooks skill)
uv run ruff format .           # format
uv run ty check                # type checking
uv run pymarkdown scan -r docs README.md CLAUDE.md packages/*/README.md  # markdown lint

# Testing
uv run pytest                                # all tests
uv run pytest path/to/test_foo.py::test_bar  # single test

# Run Dagster UI
uv run dg dev                  # open http://localhost:3000

# Marimo notebooks
uv run marimo edit packages/notebooks/some_notebook.py
```

Markdown (README.md files, docs/*.md, and Python docstrings) is linted automatically by the
pre-commit hook, but when developing code or docs it's a good idea to run the markdown lint
command above yourself before committing, for faster feedback than waiting on the commit-time
hook.

**Testing conventions** — where test dependencies and fixtures live, how discovery works, mocking
with `monkeypatch`, network-gated tests, the moto S3 reset-per-test rule, and the Patito assertion
house style — are documented on the **[Testing](docs/architecture/testing.md)** page.

**Never create a `uv venv` or run `uv sync` with a target under `/tmp`.** `/tmp` on this machine is
tmpfs, a different filesystem from `~/.cache/uv`, so uv can't hardlink packages from its cache
across that boundary and silently falls back to a full byte-for-byte copy per package — costing
real RAM instead of the near-zero marginal cost a same-filesystem venv gets. This has been seen to
exhaust the tmpfs quota mid-install and abandon the venv half-built. The session scratchpad
directory is under `/tmp` too, so it has the same problem — put throwaway venvs (a
mutation-testing worktree, a version-bisection scratch build, a one-off repro) on the home
partition instead, e.g. a worktree under `.claude/worktrees/`.

## Skills

Detail that only matters while you are touching one specific thing lives in
`.claude/skills/<name>/SKILL.md` and is loaded on demand. **Load the relevant skill *before* you
start** — most of what they hold are traps that fail silently, so by the time you notice you
needed one, the mistake is already written.

| Skill | Load it before… |
|---|---|
| `code-style` | writing or editing **any** Python in this repo |
| `polars-patito-gotchas` | writing Polars/Patito code that joins, casts, filters a `pt.LazyFrame`, declares a Patito field, or reads/writes Delta |
| `mkdocs-authoring` | editing markdown MkDocs renders — `docs/`, READMEs, `SKILL.md`, docstrings — especially nested lists, list items with code blocks, or wrapped links |
| `marimo-notebooks` | creating or editing a Marimo notebook (`packages/dashboard/*.py`, `packages/notebooks/*.py`) |
| `ty-workarounds` | acting on a `ty` error in Altair chart code or numpy `.view()` code, or adding any `# ty: ignore` |
| `plan-wave` | choosing the next batch of issues under an epic to run in parallel (`/plan-wave <EPIC>`) |
| `plan-issue` | deciding what to build for a GitHub issue (`/plan-issue <N>`) — sizes the issue, then writes a reviewed plan unless it is trivial, no code |
| `simplicity-clean-room` | testing whether an existing module is more complicated than its problem requires |
| `implement-issue` | writing code for an approved plan: worktree, verify set, PR, up to two adversarial reviews, stop |
| `github-issue-pr-workflow` | `gh issue create`, `gh pr create`, `gh pr merge`, or ship-time triage |
| `github-graphql` | any `gh api graphql` call — sub-issue attach/reorder, issue Type, project fields |
| `long-form-prose` | drafting new prose longer than a few paragraphs of connected argument — a `docs/` page, a roadmap section, a PR description explaining a design |
| `prose-review` | reviewing, reordering or simplifying prose that already exists — structure first, then one pass per rule |
| `literature-review` | researching, writing or reviewing a literature review or state-of-the-art section that an outside party will publish |

## Docs

`docs/` contains a lot of useful information beyond API reference: forward-looking plans and
their ordering (`docs/roadmap/`), the portable design principles, engineering hypotheses and
inherent-stability argument (`docs/design-philosophy/`), durable explainers of solution methods
(`docs/techniques/`), background/requirements context (`docs/background/`), design rationale for
what's already built (`docs/architecture/`), and step-by-step operational how-to for what's already built
(`docs/ml_experimentation/`, `docs/live_service/` — design and how-to are deliberately separate
pages, cross-linked via "See also"). When planning new features, check `docs/` for relevant prior
discussion before proposing an approach.

The docs are published at <https://openclimatefix.github.io/nged-substation-forecast>. When
linking to a docs page from anywhere outside `docs/` itself (GitHub issues, PR bodies, code
docstrings), link to that rendered site (e.g.
`https://openclimatefix.github.io/nged-substation-forecast/roadmap/live-service/#anchor`),
never to a `github.com/.../blob/main/docs/...` path.

**Chart images — optimise an SVG before committing it.** A Vega/Altair chart exported straight to
SVG carries one path point per reading, at more decimal places than the viewport can express, so
the file is far larger than it needs to be. Run the export through

```bash
npx svgo@4 --multipass --precision=1
```

which took `docs/example_power_forecast.svg` from 571 KB to 296 KB with no visible change (verified
by rendering both to PNG at 2× and comparing pixel by pixel). Unoptimised exports tend to trip
`check-added-large-files`' 500 KB limit, which is the signal that this step was skipped.

### Prose style

These rules apply to everything we write in prose: `docs/` pages, READMEs, `SKILL.md` files,
docstrings, code comments, GitHub issue and PR bodies, and anything we write for an outside reader.

**The prose rules in this section govern words and sentences; getting a whole document's order
right needs a planning step of its own.** A badly ordered document reads fine sentence by sentence,
so the rules below won't catch it and neither will a reviewer holding the whole document in
context. That reviewer already knows what a later section says while reading an earlier one —
exactly the knowledge a first-time reader doesn't have. Load `long-form-prose` before drafting new
prose longer than a few paragraphs.

**When reviewing prose against these rules, load `prose-review` before starting.** A reviewer asked
to check everything at once finds the loudest fault in each paragraph and moves on, so the quieter
faults survive. A combined sweep of one section of the literature review reported nothing, and a
one-rule-at-a-time sweep of the same text found thirty. The `prose-review` skill owns the procedure
— the order to sweep in, what is deliberately not a finding, how to chunk a long file across
sub-agents, which model to use, and how to triage findings before applying any of them.

**This is technical writing, not poetry: precision first, concision second, elegance last.** Every
page here is a reference document, read by someone who is about to act on it. A sentence that can
be read two ways will eventually be read the wrong way and built on. Precision wins every contest
it enters: repeat the noun, restate the qualifier, name the units, and accept a sentence flatter
than a writer would like. Concision comes next, and comes from cutting whole sentences rather than
from clipping words out of a sentence that needs them. Most of the rules below are that order of
priorities applied to one recurring case.

**Lead each paragraph with a bolded sentence that states its conclusion.** The reader should get
the argument from the bolded leads alone, then read on only where they want the reasoning. That
skim-reading pattern is why we prefer sub-headings and short paragraphs over bullet lists. A list
flattens the argument into items of equal weight; a bolded lead says which claim matters, and the
sentences under it say why.

**Bullets are right where the items really are of equal weight and each one is simple.** The
preference for prose above is about an argument, not about every list: where a passage is a
parallel set of simple, independent facts — the options a setting takes, what a table holds, a run
of short design notes — flattening costs nothing and the bullets read faster than the prose would.
The test is whether a reader meeting the material for the first time needs the connective tissue
between the sentences. **Never condense a passage that introduces a complex new concept**, because
the connective tissue is what makes a new concept followable, and a bulleted argument reads as a
set of assertions nobody joined up. Judge it per passage rather than per page, in both directions:
a bulleted item carrying several sentences and a citation is a paragraph wearing a hyphen, and a
paragraph listing five simple settings is a list wearing prose.

**Prefer a heading that states the section's conclusion, but fall back to a plain descriptor the
moment that conclusion needs explaining.** A heading is read cold, by someone who has not read the
section and is deciding whether to, so it has no room to define its own terms. Two tests: can a
first-time reader parse every word, and can that reader tell why the claim would matter to them?
"More detailed weather data has not always improved the forecast" passes both tests. Every word is
ordinary, and a reader knows at once whether that finding is their problem. "Energy forecasting has
platform descriptions and no retraining cadence" fails both tests, because "platform" and
"retraining cadence" only acquire their meaning inside the section. Where the conclusion cannot be
stated without a term the section itself has to introduce, name the subject instead — "MLOps in
energy forecasting". Leave the conclusion to the bolded lead, which has a whole sentence in which
to land it. Helping a reader find the section they want is the heading's job; summarising the
section is what a heading earns when every word survives being read cold. Renaming a heading
changes its anchor slug, so grep for inbound links to the old slug first and update every one in
the same commit.

**Be concrete and plain; write for a skim-reader.** Assume the reader is skimming and wants the
meaning to jump off the page, not to spend effort decoding a clever, abstract or metaphorical
phrase. Name the actual thing — the asset, the column, the number, the failure — rather than
gesturing at it. Prefer "if the ECMWF download fails, the forecast reuses yesterday's NWP run and
widens the uncertainty bands" over "the pipeline weathers upstream turbulence"; prefer "one Delta
table per data source" over "a constellation of storage primitives". Short everyday words beat long
Latinate ones, and a plain sentence beats an elegant one. A useful shorthand for the target: *The
Economist*'s house style — short words, active voice, concrete nouns, British spelling, every
acronym expanded on first use — but without the two journalistic habits that would hurt a reference
doc, so no punning or whimsical headings, and no scene-setting opening: state the conclusion first,
then explain it.

**Name the thing; don't write "it".** Wherever a pronoun or a demonstrative makes the reader look
backwards to work out what it refers to, repeat the noun instead. "It", "this", "that", "these",
"those", "they", "such", "one", "ones", "the former" and "the latter" are the usual offenders. A
paragraph that *opens* with a pronoun is the worst case, because a skim-reader landing there has
nothing to look back at. **"One" and "such a" are the two that slip past a careless sweep**,
because both read smoothly: write "an energy-forecasting lifecycle rather than a generic
lifecycle", never "rather than a generic one"; write "the only paper of the three", never "the only
one of the three"; write "a measure of effectiveness", never "such a metric" where a metric was
named a clause earlier. Repeating the noun is always available and always correct. "One" is fine as
a determiner in front of the noun it counts — "the one review we found" both scopes a claim and
names its noun — and wrong only when it stands in place of the noun. Prefer "the NWP download" over
"it", "the threshold-weighted score" over "this". A little repetition beats an ambiguous sentence
every time: never make the reader refer backwards to decode a sentence, and never buy elegance with
a referent the reader has to hunt for. The same rule covers version numbers: write "Flexpectation
v1" and "Flexpectation v2", never a bare "v1" or "v2", which could be a version of anything.

**"Thing" is never the right noun.** Every use of "thing" or "things" has a specific noun waiting
behind it, and the specific noun carries information the placeholder throws away: "the two
contaminants that must be filtered out", not "the two things"; "Two caveats temper both figures",
not "Two things temper"; "the closest work already published", not "the closest thing already
published". A bolded lead opening "Two things follow" wastes the one sentence a skim-reader is
guaranteed to read. **"Something" and "anything" throw away the same information.** They are harder
to spot, because they read like ordinary English: "a decision to agree with NGED", not "something
to agree with NGED"; "any occasion a human had to intervene in the running service", not "had to do
something to it"; "when the ingest fails", not "when something breaks". Where the sentence
genuinely means an unknown of unknown kind — a placeholder in a rule about future cases — say what
kind of unknown: "an input we have not anticipated", not "something unexpected".

**"Metadata" hides the fields that carry the information; list them.** The fault is the same as
"thing": a reader told that a model was given "the site's metadata" learns nothing, because the
fields are the information. Write "the panel tilt, the panel azimuth, and the ratio of
direct-current to alternating-current rating", not "the site's metadata"; write "six columns
describing each low-voltage feeder — among them how many housing units it serves", not "metadata
covariates". Where the fields are not worth listing in full, name the fields that matter and say
how many there are. The same goes for every other umbrella noun that stands in for a list the
reader wants: "parameters", "attributes", "characteristics", "data quality issues".

**A noun that carries a count has to say what was counted.** A sentence chaining counts — "screened
256 records to 31 sources and mapped 13 general-purpose platforms" — hands the reader three units
and defines none of them. The reader cannot tell whether a record, a source and a platform are one
kind of object counted at three stages or three different kinds. Name each unit where it first
appears, which usually takes two or three words: "screened 256 candidate documents — vendor
documentation, open-source repositories, and academic papers — down to the 31 they kept, and mapped
the 13 machine-learning-operations platforms those documents describe". The offenders are the nouns
a methods section reaches for: records, sources, items, entries, results, studies, works, cases,
instances, observations, points, and units. This is the counting cousin of the "metadata" rule
above: there an umbrella noun hides a list of fields, here it hides what is being tallied.

**Say which kind of network you mean, every time.** This project forecasts an electricity network
using neural networks, so a bare "network" makes the reader stop and work out which one is meant.
Qualify it on both sides: "electricity network", "distribution network", "network operator" for the
wires, and "neural network", "graph neural network", "long short-term memory neural network" for
the model. Where a sentence would otherwise pile up the qualifier, name the specific noun instead —
"a model trained on the feeders' own history" beats "a model given a network's whole history". Any
other word this project uses for two different meanings gets the same treatment.

**Describe performance in performance terms, not in money metaphors.** A forecast does not "pay",
an input does not "buy" accuracy, and a modelling choice does not "cost" anything unless real money
changes hands. Write what actually moved: "the inputs that improve skill at short range", "adding
the physics model made the forecast interpretable without making it less accurate", "rejected the
gradient-boosted tree on the effort of tuning it". Keep "cost" and "price" for money: what NGED
spends procuring flexibility is a cost. Calling a lost percentage point of skill a cost as well
makes the page ambiguous exactly where it has to be exact.

**Put the words in the order that cannot be misread.** "73 wind farms in GB" says what it means;
"73 GB wind farms" makes the reader parse a noun-pile and can be read as a unit of measure. Where a
qualifier can attach to more than one noun, move it or add the word that pins it down.

**Use numerals when the number carries a unit, is 10 or more, or sits beside another numeral in the
same phrase; use words otherwise.** So "5 years of half-hourly data", "50,000 substations" and "6
solar farms, 3 wind farms", but "the eight challenges" and "nine ideas in ten". Reserving numerals
for measured quantities makes the figures the reader cares about jump off the page on a skim, and
stops one horizon appearing as "day four" in one paragraph and "day 4" in the next. Never open a
sentence with a numeral: recast so the number falls inside the sentence ("Of the 32 series, 12 are
metered generators"), or spell the number out. Counts of the document's own structure ("three model
families") and idiomatic ratios ("nine ideas in ten") stay in words, because neither is a
measurement.

**Use the serial comma — the comma before the final "and" or "or" in a list of three or more
items.** So "solar, wind, and dispatchable generators", never "solar, wind and dispatchable
generators". The serial comma is the one deliberate departure from *The Economist*'s house style
above. The serial comma also removes a real ambiguity: without the comma, the last two items can
read as a pair belonging to the item before them. In a list of assets or of data sources, that
reading changes the meaning. A list of two items takes no comma, and an author string in a
reference list follows the citation convention rather than this rule.

**Be concise by cutting whole sentences, not words.** Prose should be as short as it can be
without losing readability, but the compressible material is rarely inside a sentence. It is whole
sentences and paragraphs that carry no information: restating the heading, summarising what the
reader has just read, hedging ("it is worth noting that"), listing what we are *not* doing, or a
closing paragraph that repeats the opening. Delete those outright, and leave the surviving
sentences intact — buying brevity by clipping words out of a sentence that needs them is the
mistake the next rule forbids.

**Prefer short sentences. Where a sentence carries two claims, split it into two sentences.** A
joined sentence makes the reader hold the first claim in mind while parsing the second, and the
second claim usually carries the conclusion. The joins worth checking are "and" and "but", a
semicolon, an em dash, a "so", a "which", and a trailing participle. None of those joins is wrong
by itself. The test is whether splitting makes the passage easier to read, never whether a joining
word is present. Two limits keep the rule from doing harm. A split must leave full sentences rather
than a fragment, which is the rule below. And a conjunction joining two verbs that share one
subject joins no second claim: "the forecast reuses yesterday's NWP run and widens the uncertainty
bands" is one sentence and stays one.

**Write full sentences; don't drop the subject.** Don't clip words for terseness
if it leaves a sentence without a clear subject/verb. Prefer "We split storage across two
buckets so that..." over "Two buckets, not one — split so that...". The full form is more
readable and no less concise in practice.

**Write about the present, not the past.** The docs describe how the code works *now*. Don't write
about how it used to work, what a change replaced, or which issue changed it — that history lives
in git, in the PR and in the issue tracker, and repeating it here turns every page into a running
changelog and makes the docs unreadable. When a change invalidates a passage, rewrite the passage
to describe the new behaviour rather than appending a note about what changed. This is the
"comments and docs must reflect current state only" rule in
[`docs/architecture/code-style.md`](docs/architecture/code-style.md), applied to prose.

**Every citation is a hyperlink to the work cited.** Writing "Sculley et al. (2015)" as plain text
makes the reader go and find the paper. The link costs only a few characters, and a digital object
identifier is a stable address. Wrap the author-and-year label itself — `[Sculley et al.
(2015)](https://doi.org/...)` — and prefer a DOI to a publisher's landing page. This holds
everywhere we write, not only in the literature review: a GitHub issue or a pull-request body
quoting a paper links it too, because the reader there has even less context than a docs reader.
The exception is repetition. Once a work is linked, later mentions of the same work in the same
passage drop both the year and the link and name the authors alone — "Sculley et al. report" — so a
paragraph does not carry the same link four times. `check_citations.py` in the `literature-review`
skill enforces the linking half of this rule on the review, and nothing enforces it anywhere else.

**Say what the source found, not what is always true.** When prose rests on a paper, a measurement
or a trial, state the finding with its scope attached: "in the studies we read, a gradient-boosted
model beat a same-time-yesterday rule by 10 to 20%", not "sophisticated models beat naive ones".
A law-like sentence claims far more than the evidence supports, and the first reader who knows a
counter-example stops trusting the rest of the page. The same applies to claims that something does
*not* exist: an absence claim is only ever as good as the search behind it. Say what was searched
and let the reader judge, rather than asserting that nobody has tried the approach.

**Don't claim a set has exactly one member unless you have enumerated the set.** "The only study",
"the one paper", "the first network to publish", "nobody has done this", "the closest precedent" —
each asserts that a search was exhaustive. A search almost never is. Absence of evidence is not
evidence of absence: we may only be aware of one instance, which is a different claim from there
being only one. Say what was looked at and what turned up — "the one paper we found that measured
properly", "the closest of the four", "labels that none of the GB projects we checked published".
Superlatives need the same treatment, because "the most useful method" ranks a whole field on a
survey nobody ran: scope the superlative to a set you have listed, or state the basis of the
judgement instead, as in "the published method that fits NGED's telemetry most closely".

**Don't commit the project to work it has not agreed to.** A page explaining what the literature
found, what a technique does, or how a subsystem works is not a project plan. A sentence like
"Flexpectation will therefore label the telemetry by hand" turns a description into a promise a
funder can hold us to. Describe what is known and what the options are, and leave what we will do
to the roadmap, the issue tracker, and the documents that own those commitments.

**Don't introduce a name, a number, or an acronym before the reader has a use for it.** A fact that
exists only to justify a claim belongs after the claim, not before it — a reader who meets the
justification first has nothing yet to hang it on, and has to re-read once the claim finally
arrives. Prefer "ECMWF ENS coarsens from a 3-hour to a 6-hour step width at day 6, so a forecast
built on days 6 to 10 sits on the coarse half of the grid" over opening the paragraph with the step
widths and only later saying why they matter. This is the document-level form of leading with the
conclusion: state the claim, then supply the detail that backs it, never the reverse.

**Don't name individuals.** Write the rule, not who asked for it: "get a contract change agreed
before making it", never "ask so-and-so before changing a contract". One person maintains this repo
today, but the docs and skills outlive that, and a name that reads as "the person responsible" to
us reads as an unknown third party to whoever picks the work up next. Where the sentence needs an
actor, name the role — the reviewer, the maintainer, whoever runs the pipeline. Real GitHub
handles used as data are fine (the `JackKelly` assignee, a commit's `Co-Authored-By`); it is prose
about a named person that this forbids.

## How planning works

Full description and a "which place do I use?" table: `docs/documentation-guide.md`. In brief:

- **GitHub** (issues + the OCF Project board) is the *complete, ordered* task list — task-level
  priority lives only there. When current priorities matter, query it with `gh` (epics map 1:1
  to roadmap milestones; dependencies are `blocked by` issue links).
- **`docs/roadmap/`** holds design, dependencies, and the milestone arc. Step-by-step mechanics
  sit inside each page under an "Implementation details (deleted when this ships)" section.
- **`docs/design-philosophy/engineering-hypotheses.md`** holds the falsifiable claims the
  engineering is meant to deliver. Cite them by label (`H1`, `T1.2`); labels are append-only —
  never renumber.
- **`plans/`** holds at most one file: the in-flight branch's implementation plan, written by the
  `plan-issue` skill before any code is touched and deleted on merge. One worktree per branch is
  what keeps it to one file, so parallel sessions never collide. Usually empty on `main`, and empty
  on a branch whose issue was simple enough to need no plan.

**Creating an issue or a PR has a checklist** — labels, org issue Type, OCF project membership
and its fields, sub-issue ordering, the `JackKelly` assignee — and none of it can be set by `gh
issue create` / `gh pr create`. It lives in the `github-issue-pr-workflow` skill, along with the
never-squash-merge rule and ship-time triage. Load it before you run either command.

## How work gets done

Three skills, in order. The last two are deliberately separate so that a design is approved
before any code moves:

1. **`plan-wave`** (`/plan-wave <EPIC>`) chooses the next one-to-five issues under an epic that
   can run concurrently without editing the same files, and dispatches each as a chip to be
   launched as its own Claude Code session. It plans one wave and stops, because the epic gains
   issues while a wave is in flight. Skip it when the issue to work on has already been named.
2. **`plan-issue`** (`/plan-issue <N>`) reads the issue, decides whether it is worth implementing
   at all, and sizes how much process it needs. It writes `plans/<branch-name>.md`, links to the
   plan as soon as it is committed and pushed, has up to two fresh sub-agents adversarially review
   that plan in turn — the first hunting for a simpler approach, the second checking correctness
   and testability — and stops for human review. It writes no code.
3. **`implement-issue`** picks up an approved plan: worktree, implement, the green-before-push
   verification set, PR with labels and assignee, then up to two *further independent* adversarial
   reviews of the diff — the first for correctness and for cutting the code, tests and prose
   down to what the change needs, the second mutation-testing the change — committing, triaging
   and pushing after each, stop for human review. **Never merge.**

**How much process an issue gets is sized to the issue**, in step 3 of `plan-issue`:

- **Simple** — a mechanical change with one obvious way to do it, touching no contract, no
  production degradation path and nothing stored, where the verification set is the whole of the
  risk. It gets **no plan and no agentic review**: implement it, open the PR saying that no
  sub-agent reviewed it, and stop for human review.
- **Complex** — anything that changes what gets stored, touches the production serving path or a
  degradation rule, or admits more than one defensible design. It gets the plan and **all four**
  reviews.
- **Medium** — everything else. It gets a plan, and Claude chooses between zero and two of the
  plan reviews and between zero and two of the diff reviews, running the earlier of each pair
  first and erring towards running one more when the call is close.

Stay inside the issue's scope; report unrelated design mistakes rather than fixing them.

**Ask before changing a Patito data contract.** The schemas in `packages/contracts/` are the
authoritative account of what the data means, so code that violates one is usually the thing at
fault. Widening a field to `| None` or relaxing a range to make a failing `validate()` pass hides
the defect in the one place the rest of the system trusts. Reasoning and the rest of the rule:
[`packages/contracts/README.md`](packages/contracts/README.md).

**Why:** diffs are reviewed in GitHub's UI, and a PR should already have survived an
adversarial pass by the time a human opens it, so that human review is the last line of defence
rather than the first. The fresh-reviewer requirement exists so the reviewer cannot be anchored by the
implementer's rationale; the triage step exists because reviewer findings are often wrong and
must not be applied uncritically. Simplicity gets its own reviewer, and gets it first, because a
plan that is more complicated than the issue requires is the failure mode that survives a
correctness review intact. Mutation testing gets the last reviewer because a green suite proves
nothing on its own: whether a test would catch the bug it exists for is only settled by writing
that bug and watching. The sizing exists because that machinery costs wall-clock time and a round
of triage: on a change whose correctness is visible in the diff it finds nothing the diff did not
already show, and the process then delays the change instead of protecting it.

## Architecture

This is a `uv` workspace monorepo. The root `src/nged_substation_forecast/` is the Dagster
application; all reusable logic lives in `packages/`.

**A short list of design principles** governs architectural decisions:
[`docs/design-philosophy/design-principles.md`](docs/design-philosophy/design-principles.md).
Read them before proposing a structural change. If a change violates one, that is not a veto, but
say which principle is being traded away and what is bought in return.

### Inherent stability (production code)

**In production, never raise because an input is absent or stale — degrade, widen the uncertainty
bands, and record the degradation on the row.** Raising is reserved for states that are our own bug
(an empty promoted model, a contract violation), not for the outside world misbehaving. Corollaries
that come up constantly when editing `defs/`:

- **Liberal about missing inputs, strict about malformed ones.** Absent data routes into the
  always-output path; malformed data is rejected at the Patito boundary. Detectably-*wrong* input
  (a stuck meter) is treated as missing, not as data.
- **Asset checks warn, they do not block** — `AssetCheckSeverity.WARN` with `blocking=False`. There
  is deliberately no `ERROR`-severity check anywhere in the repo. A warning function must never be
  able to raise, or fail-open silently becomes fail-closed.
- **Measure degradation in missed NWP runs, never in hours of age.** We ingest one ECMWF run per
  day, so healthy NWP is 12–30 hours old depending on the 6-hourly slot.
- **R&D is the opposite**: the CV, training and metrics assets fail fast, because a quietly-degraded
  training run poisons every comparison built on it.
- **When a capability could live in the training loop or in the production service, put it in the
  training loop.** Keep the serving path close to "load a model, call `predict`".
- **Make the telemetry name the fault.** Whatever reaches Sentry — a swallowed exception, a
  degradation warning — carries the tag an alert rule routes on, and names the series, the run or
  the asset at fault rather than only the type of error. The operator reads the alert, not the
  logs.

Full rationale, the degradation ladder and the numbered rules:
[`docs/design-philosophy/inherent-stability.md`](docs/design-philosophy/inherent-stability.md). The
falsifiable claims it is meant to deliver — cite them as `H1`/`T1.2` and never renumber them — are
in [`docs/design-philosophy/engineering-hypotheses.md`](docs/design-philosophy/engineering-hypotheses.md).

### Packages

| Package | Purpose |
|---|---|
| `contracts` | Patito data schemas (the single source of truth for all data shapes) |
| `delta_store` | Physical storage policy for Delta tables: parquet writer properties, sort orders, significand rounding, write helpers |
| `ml_core` | Feature engineering and `BaseForecaster` abstract class |
| `nged_data` | Reading NGED JSON files from S3 and writing to Delta Lake |
| `dynamical_data` | Downloading ECMWF ensemble NWP from Dynamical.org |
| `geo` | H3 spatial indexing utilities |
| `weather_utils` | Shared NWP query helpers used by both the dashboard and the feature pipeline (the analysis-proxy selection, `NWP_PUBLICATION_DELAY_HOURS`) |
| `xgboost_forecaster` | Concrete `BaseForecaster` implementation using XGBoost |
| `plotting` | The OCF-brand Altair theme and shared plotting helpers |
| `dashboard` | Marimo web apps for visualisation (`view_forecasts.py`, `map_and_timeseries.py`) plus their shared helpers in `src/dashboard/` |
| `notebooks` | Marimo exploration notebooks |

### Dagster Assets (`src/nged_substation_forecast/defs/assets.py`)

Three main assets:

- `power_time_series_and_metadata` — pulls NGED telemetry from S3, appends to Delta Lake, upserts metadata parquet
- `h3_grid_weights` — computes fractional H3 cell overlap with the GB boundary for spatial NWP aggregation
- `ecmwf_ens` — daily-partitioned asset that downloads ECMWF ENS NWP and writes it to Delta Lake via `delta_store.nwp.write_nwp`, which replaces that `(nwp_model_id, init_time)` partition

### Data Contracts (`packages/contracts/`)

All tabular data flowing through the system is validated with **Patito** models. Key schemas:

- `PowerTimeSeries` — half-hourly power observations (MW/MVA) per `time_series_id`
- `TimeSeriesMetadata` — substation metadata including lat/lon, H3 index, substation type
- `Nwp` — NWP weather data in physical-unit `Float32`, on disk and in memory alike (rounded to a 13-bit significand at write time by `delta_store.nwp`)
- `AllFeatures` — the final joined dataset handed to ML models; primary key is `(time_series_id, power_fcst_init_time, valid_time[, ensemble_member])`
- `PowerForecast` — model output schema

### Feature Engineering (`packages/ml_core/src/ml_core/features/`)

`_engineer_features()` (in `tabular_feature_engineer.py`) is the central tabular pipeline function: given a `set[str]` of requested feature names, it joins power observations with NWP and metadata, then applies features. Feature names are parsed by `ParsedFeatures.from_strings()` (in `_parsed_features.py`) into typed `LagFeature`, `RollingFeature`, `StaticFeature`, `TimeFeature`, or `WeatherFeature` objects. Callers reach this via `FeatureEngineer.engineer()` — see the ML Model Interface section below.

**Critical design invariant — no lookahead bias:** `power_fcst_init_time` (when we make the
forecast) is distinct from `nwp_init_time` (when the NWP model ran). Power lag features are
nullified via `_nullify_leaky_lags()` when the lag is shorter than or equal to the forecast lead
time. Weather lags use a dual-strategy join: same NWP run for future target times, freshest NWP run
for past target times.

Two operating modes:

- **Bulk training and multi-run backtesting** (recommended for most callers): `power_fcst_init_time` is `None`; it is derived per-row as `nwp_init_time + nwp_publication_delay_hours`.
- **Single-run inference or backfilling**: `power_fcst_init_time` is provided; NWP is joined on `(time_series_id, valid_time, nwp_init_time)` for the one matching NWP run.

### ML Model Interface (`packages/ml_core/src/ml_core/base_forecaster.py`)

All forecasting models subclass `BaseForecaster`, which defines `train(AllFeatures)`,
`predict(AllFeatures) -> PowerForecast`, `save(Path)`, and `load(Path) -> Self`. Each subclass owns
its own persistence format; `XGBoostForecaster` writes one `.ubj` file per `time_series_id` plus a
`meta.json` with the full `XGBoostConfig`.

Identity is split across two levels. **Model-family identity** — `MODEL_NAME` and `MODEL_VERSION` —
are class-level constants on each `BaseForecaster` subclass (properties of the implementation;
bumping `MODEL_VERSION` is a deliberate code change). **Experiment identity** — `experiment_name`
and `ml_flow_experiment_id` — lives in `BaseForecasterConfig` so it travels with the saved model.
Both levels are stamped onto every `PowerForecast` row at predict time:
`power_fcst_model_name`/`power_fcst_model_version` from the class, and the dedicated
`experiment_name`/`ml_flow_experiment_id` columns from the config. Do not collapse experiment
identity into `power_fcst_model_name`.

Each `BaseForecaster` also carries a `feature_engineer: ClassVar[FeatureEngineer]` — a strategy
object (composition, not inheritance) that owns the full data-preparation pipeline from raw inputs
to an `AllFeatures` frame, including the NWP spatial join. The default `TabularFeatureEngineer`
maps each gridded NWP H3 cell to the nearest time series then runs the tabular `_engineer_features`
pipeline. A future model needing a different view of the data (e.g. a CNN wanting a spatial NWP
crop) overrides `feature_engineer` with a different `FeatureEngineer` subclass — it does not change
`_engineer_features` or `BaseForecaster`. Both classes live in
`packages/ml_core/src/ml_core/features/`.

## Code Style

**Load the `code-style` skill before writing or editing any Python.** It takes you to
[`docs/architecture/code-style.md`](docs/architecture/code-style.md), the single source of truth —
Python version, ruff configuration and its traps, naming, how expressive a signature has to be,
comments and doc links, Polars and Patito conventions, data handling, error handling. None of it is
repeated here, so skipping it means guessing at rules that are written down.

## This is a young project

The project is a new, green-field project. No one else is using this code yet. Which means:

- It's 100% fine to make breaking changes, if doing so improves the code. (And as long as we update
  all the downstream code.)
- Our aim is to make the code well-organised and easy to use.
- None of this code is "written in stone" or battle-tested.
- We haven't trained any "serious" ML models yet, so a change that invalidates an existing trained
  model or its saved config costs us a retrain, not a migration path. Don't design for backwards
  compatibility with models we've already trained.
- If you see a design mistake _anywhere_ in the code, then please flag that design mistake to me.
  I'd much rather end up with a project that's well engineered. (That said, if we're working on
  feature X, and you spot a mistake in some code that isn't obviously in scope for X, then please
  discuss the change with me first. Definitely don't make out-of-scope changes with asking me!)
