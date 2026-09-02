# Cut the prose in `src/` down to the code (#691)

**The problem.** Prose is 28.9% of the Python lines in this repository, and much of that prose is
background and design rationale that `docs/` already owns or should own. `CLAUDE.md` and [the
code-style page](https://openclimatefix.github.io/nged-substation-forecast/architecture/code-style/)
already state the governing rule — "One home per argument" — and the code does not follow it. Two
copies of an argument drift silently, because no linter, type checker or test can notice when a
docstring goes on asserting reasoning a docs page has since superseded.

**The planned solution.** This pull request cuts the docstrings and comments in
`src/nged_substation_forecast/` — the Dagster application, 12 files and 13,337 prose words — down to
what a reader needs to understand the code in front of them, and promotes every surviving argument
to the `docs/` page that owns its subject with a rendered-site link left behind. Issue #691 covers
the whole repository, so the remaining packages follow in later pull requests under the same issue,
each one applying the cutting rule this pull request settles.

## Verdict, size and departures

**Verdict: worth implementing, with the measure of success restated.** The duplication is real and
the rule against it is already written down. What the issue gets wrong is the metric.

**Size: complex.** The change spans every file in a directory, the correct amount to cut is a
judgement call with more than one defensible answer, and getting it wrong is lossy in a way no test
catches. Complex buys the plan, both plan reviews, and both diff reviews.

Departures from the issue body, each with its reason:

- **The percentage of Python lines that are prose is a report, not a target.** A repository with
  Google-convention docstrings carrying `Args:` and `Returns:` sections on every public function is
  naturally 20% to 30% docstring lines, and driving that number down for its own sake deletes the
  argument documentation the `D` ruff rules exist to require. The acceptance criterion "prose as a
  share of Python lines is reported before and after, per package" is kept as a report. The target
  is the first acceptance criterion instead: no docstring or comment restates a subject a `docs/`
  page owns.
- **The issue's headline measurement is wrong, and the plan reports corrected numbers.** The issue
  claims 139 files, 32,916 Python lines, 6,546 docstring lines and 2,029 comment lines. Measuring
  `src/`, `packages/*/src/`, `packages/*/*.py`, `scripts/` and `tests/` with `ast` and `tokenize`
  gives 93 files, 22,955 lines, 5,260 docstring lines and 1,375 comment lines — 28.9% prose, 57,632
  prose words. The overcount does not change the conclusion, and the corrected figures are what the
  pull request reports.
- **Two of the issue's three named examples overstate what `docs/` already covers.** The issue says
  the feature-engineering documentation owns the subjects in
  `packages/ml_core/src/ml_core/features/tabular_feature_engineer.py`. No feature-engineering page
  exists: `docs/architecture/performance.md` owns lazy evaluation and nothing owns the lookahead-bias
  rule. Rationale in that position is promoted, not deleted. The Sentry example holds up —
  `docs/architecture/production-deployment.md` already carries the design argument that
  `src/nged_substation_forecast/_sentry.py`'s module docstring repeats.
- **The second diff review is a lossiness audit, not a mutation pass.** Mutation testing asks
  whether a test would catch the bug it exists for, and a prose-only change introduces no behaviour
  a mutant could break. The failure mode here is deleting an explanation that exists nowhere else,
  so the second reviewer reads the deleted prose against `docs/` and reports every removal it cannot
  account for.
- **The issue's "out of scope: any change to behaviour, signatures or contracts" is tightened to
  "no executable line changes at all"**, which the section below states in full.
- **This pull request covers `src/` only.** The issue asks for one pull request per package. `src/`
  is the largest single group of prose words, holds two of the issue's three named worst offenders,
  and is not rendered into the API pages by mkdocstrings, so a malformed docstring cannot break the
  docs build while the cutting rule is still being agreed. Doing it first makes the first pull
  request a proposal about how aggressive to be, cheap to correct before the same rule is applied to
  41,000 further prose words.

**One rule in `docs/architecture/code-style.md` is deliberately overridden for this work.** That page
says "Do not remove existing comments unless they are misleading or out of date". Issue #691 is an
explicit, one-off instruction to remove comments on a third ground: the comment duplicates a
subject `docs/` owns. The plan adds that ground to the code-style page rather than leaving the two
instructions contradicting each other, so the next reader of the page is not told the opposite of
what this pull request did.

## Nothing but docstrings and comments changes

**No line of executable Python changes in this pull request.** Not a signature, not a default, not
an import, not a rename, not a reordering, not a `# noqa`. Every hunk in the diff touches a
docstring, a comment, or a `docs/` markdown file, and a diff filtered to non-prose lines is empty.
That constraint is what makes the change reviewable: the reviewer reads the diff asking only
whether an explanation was lost, and never whether behaviour moved.

Two consequences worth stating, because both are tempting while reading a file closely:

- **A design mistake found while cutting is reported, not fixed.** `CLAUDE.md` asks for
  out-of-scope mistakes to be raised rather than corrected, and this pull request has an unusually
  wide surface for finding them. The findings go in the pull-request body and, where they deserve
  their own work, into new issues.
- **A docstring that is simply wrong about the code is corrected in place.** Correcting prose to
  match current behaviour is a prose change, not a code change, and leaving a known-false docstring
  behind would be worse than the duplication this pull request is removing. Every such correction
  is called out individually in the pull-request body, because it is the one class of prose edit a
  reviewer cannot check by reading the deleted text alone.

## The cutting rule

Every docstring and comment block in scope is classified into exactly one of three outcomes. The
rule is written here once so that every file, and every later pull request under this issue, applies
the same one.

**Keep** — what the reader needs to call or edit the thing correctly:

- What the function does, and what each argument, return value and raised exception means, including
  units and sign conventions.
- A precondition or an invariant a caller must not break, and what happens if a caller breaks it.
- A trap that would bite someone reading only this file — a Polars or Patito gotcha, an ordering
  that looks arbitrary but is not, a dtype that must not be widened.
- The one-clause "why this guard exists" that the code-style page already mandates for validation
  defending a package's public API rather than a reachable production state.
- A comment that pins a rule from
  [inherent stability](https://openclimatefix.github.io/nged-substation-forecast/design-philosophy/inherent-stability/)
  onto a specific line — that a warning path must never raise, that an absent input degrades rather
  than fails. Deleting an explanation of that kind invites a later editor to break the rule.

**Cut** — prose whose removal costs the reader nothing:

- A narration of the function body that a fluent Python reader gets from the code.
- History: what a design replaced, which alternatives were considered and rejected, which issue
  changed it.
- Background on the domain or on the data, where it is not needed to read the code.
- Measured benchmark numbers and their justification.
- A second copy of an argument a `docs/` page already makes.

**Promote** — rationale worth keeping that no `docs/` page owns yet. Move it to the page that owns
the subject, or to a new page where none does, and leave a rendered-site link behind. This is the
step that stops the exercise being lossy, and it is where the reviewer's attention goes.

Prose that survives is still governed by `CLAUDE.md`'s prose-style rules. Prose newly written into
`docs/` gets the `prose-review` skill's sentence sweep; a surviving one-line docstring does not.

## What changes, file by file

The classification below is the plan's prediction, made from reading each module's docstrings and
the `docs/` pages covering its subject. The implementer verifies each one against the code and the
page rather than applying it blind, and reports where the prediction was wrong.

- **`_sentry.py`** (442 lines, 48.9% prose, 1,982 words). The 411-word module docstring restates
  `docs/architecture/production-deployment.md`'s "Send telemetry to Sentry, and alarm on absence"
  section, which already carries the error-versus-degradation split, the `fault_category` tag and
  the "read the tag, never the level" rule. Cut it to a summary naming the three mechanisms and
  linking to that section. Keep the reason the Dagster failure hook is used instead of Sentry's
  `LoggingIntegration` only if the design page does not already carry it; check, and promote it if
  it does not. `init_sentry` (284 words) and `sentry_capture_failure` (199 words) keep their
  contracts and lose their rationale.
- **`defs/checks.py`** (1,021 lines, 31.6% prose, 3,199 words). The 603-word module docstring is the
  largest in the repository. The asset-check design — warn, never block; no `ERROR` severity; a
  warning function must never raise — is owned by
  `docs/design-philosophy/inherent-stability.md`, and the operator-facing account is owned by
  `docs/live_service/operations.md`. Cut to a summary and links. Each check function keeps what it
  measures, what threshold it warns at, and the sentence saying it cannot raise.
- **`defs/cv_assets.py`** (1,075 lines, 34.0% prose, 3,064 words). Cross-validation design is owned
  by `docs/ml_experimentation/cross-validation-folds.md` and
  `docs/ml_experimentation/dagster-workflow.md`; metric definitions are owned by
  `docs/techniques/evaluation-metrics.md`. `_score_forecast_group` (366 words) and `metrics` (203
  words) are the two to check against those pages first. Keep the fail-fast rule: R&D assets raise
  where production degrades, and that inversion surprises a reader who has just been in `defs/`.
- **`defs/assets.py`** (813 lines, 22.3% prose, 1,701 words). Ingestion. Subjects covered by
  `docs/roadmap/data-sources.md`, `docs/architecture/nwp-variable-conventions.md` and
  `docs/architecture/ecmwf-ens-known-issues.md`. Roadmap pages are deleted when their work ships, so
  code must not link to them: rationale whose only home is `docs/roadmap/` is promoted to a durable
  page instead.
- **`defs/production_assets.py`** (375 lines, 37.9% prose, 1,337 words). `live_forecasts` carries a
  445-word docstring. The degradation ladder it describes is owned by
  `docs/design-philosophy/inherent-stability.md` and the operational view by
  `docs/live_service/operations.md`. Keep every clause that names which degradation is recorded on
  which row, because that is a contract on the output rather than a rationale.
- **`defs/jobs.py`** (403 lines, 37.2% prose, 1,227 words). `_resolve_forecaster_config` (224 words)
  and `_reject_changed_identity` (206 words) explain experiment identity, which
  `docs/ml_experimentation/model-configuration.md` and the `CLAUDE.md` model-identity rule own. Keep
  the statement of what is rejected and why the rejection is not a degradation path.
- **`defs/_engineering_inputs.py`** (126 lines, 53.2% prose, 583 words). `load_engineering_inputs`
  carries a 486-word docstring. Lazy evaluation and input pruning are owned by
  `docs/architecture/performance.md#bounding-feature-engineering-memory-prune-the-inputs-not-the-output`.
  Keep the argument contract and the pruning invariant; cut the argument for it.
- **`defs/schedules.py`** (83 lines, 149 words), **`defs/_tags.py`** (34 lines, 37 words),
  **`definitions.py`** (32 lines, 47 words), **`defs/__init__.py`**, **`__init__.py`**. Small
  enough that the sweep is a read-and-confirm rather than a rewrite. Change nothing unless a
  duplication is found.

## How the work is executed

**Sonnet-5 sub-agents do the per-file cutting; this session orchestrates and checks.** Twelve files
read closely against a dozen `docs/` pages is exactly the work that parallelises, and a fresh agent
per file cannot carry another file's calibration into the next one. The orchestrating session owns
the judgement the sub-agents cannot: whether a promotion was needed, whether a deletion was lossy,
and whether the calibration held across files.

The dispatch, in three rounds:

1. **Calibration round, one sub-agent.** One agent rewrites `_sentry.py`, the smallest of the two
   worst offenders, against the cutting rule and the docs pages named for it. This session reads
   the result in full and corrects the calibration before anything else is dispatched. Whatever
   comes back becomes the worked example quoted in every later brief.
2. **Bulk round, one sub-agent per file, run concurrently.** `defs/checks.py`, `defs/cv_assets.py`,
   `defs/assets.py`, `defs/production_assets.py`, `defs/jobs.py` and `defs/_engineering_inputs.py`
   get one agent each. Each brief carries the cutting rule verbatim, the calibrated example, the
   docs pages that own the file's subjects, and the no-code-changes constraint. One file per agent
   means no two agents edit the same file, so the concurrent runs cannot collide.
3. **Sweep round, one sub-agent.** The five small files — `defs/schedules.py`, `defs/_tags.py`,
   `definitions.py`, and the two `__init__.py` files — go to a single agent, because between them
   they hold 244 prose words.

Each sub-agent returns a report rather than only a diff: every block it cut with the reason from the
three-way classification, every promotion it proposes with the target page and anchor, and every
sentence it was unsure about. **A sub-agent proposes promotions and does not write them.** Prose
written into `docs/` has to read in the receiving page's voice and sit in the right section, which
needs the whole page in view; six agents each appending a paragraph to
`docs/design-philosophy/inherent-stability.md` would also collide on one file. This session
consolidates the proposals and writes the docs edits itself, once.

**Checking the sub-agents is the orchestrating session's real work, and it is not a formality.**
Each returned file is read against its own diff with three questions: does any deleted sentence say
something no `docs/` page and no surviving line says; did any non-prose line change; and does what
survives still let a reader call the thing correctly. A sub-agent that over-cuts and a sub-agent
that changes nothing are both common, and both are only visible on a read.

## Docs to update

- **`docs/architecture/code-style.md`** — add the third ground for removing a comment, as argued
  above, next to the existing "misleading or out of date". This is the only rule change in the pull
  request, and it is what keeps the page describing the present.
- **Every page receiving promoted rationale** gets the new material written into the section that
  owns the subject, in that page's voice, not pasted as a block quoted from a docstring. The
  candidate pages are `docs/architecture/production-deployment.md`,
  `docs/design-philosophy/inherent-stability.md`, `docs/live_service/operations.md`,
  `docs/ml_experimentation/dagster-workflow.md` and `docs/architecture/performance.md`. Which pages
  actually receive material is settled during implementation and listed in the pull-request body.
- **No new docs page is expected for `src/`.** If the implementer finds rationale that no existing
  page can host, that is a finding to report rather than a licence to add a page: an orphan page is
  worse than the docstring it replaced.
- **No link may point at `docs/roadmap/` or at an "Implementation details" section**, per the
  code-style page's durable-docs rule. Every link is written as its rendered `https://` URL.

## Design-philosophy check

**No code path changes, so the production-versus-R&D distinction is untouched.** No asset check is
added or edited, no severity changes, and no warning function gains a statement that could raise.
The plan's one interaction with `docs/design-philosophy/inherent-stability.md` runs the other way:
comments that pin an inherent-stability rule onto a line are explicitly in the keep list, because
deleting them makes the rule easier to break later. No principle in
`docs/design-philosophy/design-principles.md` is traded away, and no engineering hypothesis is
delivered or affected.

## Tests

**No test is added or changed, and that is the correct outcome for this pull request.** The change
edits docstrings and comments, which no test asserts on. Adding a test that counts prose words would
pin the metric this plan has just argued is not the target.

The existing suite still has to pass unchanged, which is the assertion that matters: a docstring
edit that breaks an import, a doctest-shaped example or a marimo cell shows up there. Two specific
risks the suite does cover — `scripts/check_marimo_notebooks.py` for any name a notebook cell binds,
and `scripts/lint_docstring_markdown.py` for markdown that renders wrong.

## Verification

The green-before-push set, plus what this change specifically needs:

```bash
uv run ruff check .
uv run ruff format --check .
uv run --all-packages ty check --project .
uv run pytest
uv run pymarkdown scan -r docs README.md CLAUDE.md packages/*/README.md
uv run python scripts/lint_docstring_markdown.py <changed files>
uv run mkdocs build --strict
```

`mkdocs build --strict` is required even though `src/` is absent from the API pages, because the
pull request edits `docs/` pages and adds links to them. After it passes, read the rendered HTML for
each edited page rather than trusting the exit code — a link into a heading anchor that no longer
exists is exactly the failure a strict build does not catch when the target page is present and only
the anchor is wrong.

Rerun the measurement script over `src/` before and after, and put the per-file before-and-after
table in the pull-request body.

## Risks and open questions

- **Is `src/` the right first slice, and should the rest of issue #691 follow as separate pull
  requests under the same issue?** Recommendation: yes to both. `src/` is the largest coherent group
  and cannot break the docs build, which makes it the cheapest place to settle how aggressive the
  cutting rule is. The alternative — one pull request for all 41,000 non-test prose words — produces
  a diff nobody can review against the docs pages it claims to defer to.
- **How aggressive should the cut be?** The plan's own answer is the cutting rule above, and the
  human reviewer's judgement on the first two files rewritten (`_sentry.py` and `defs/checks.py`)
  should settle the calibration for the rest. Recommendation: the implementer rewrites those two
  first and says so in the pull-request body, so the reviewer can read them as the worked example.
- **Should the promoted rationale go in with this pull request or into a docs-only pull request
  first?** Recommendation: in with this one. Splitting the promotion from the deletion means `main`
  briefly holds neither copy, and reviewing the deletion needs the promotion visible in the same
  diff.
- **Should `scripts/` join this slice?** Recommendation: no. `scripts/export_baseline_forecasts.py`
  and `scripts/run_baseline_experiment.py` document experiment reproduction, whose docs home is
  `docs/ml_experimentation/`, and folding them in mixes two audiences in one diff for the sake of
  1,508 words.
