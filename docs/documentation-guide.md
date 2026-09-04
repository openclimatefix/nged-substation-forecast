# Documentation Guide

How this repository's documentation and planning content is organised — where to look for
existing content, and where to put new content.

> **Status legend** (used throughout the design docs below):
> ✅ **Implemented** — exists in code today ·
> 🚧 **Planned** — designed, not yet built ·
> 🔬 **Research** — exploratory / v2.

## How planning works

Planning content lives in five places with deliberately non-overlapping jobs:

| Place | Job |
|---|---|
| **[`docs/design-philosophy/engineering-hypotheses.md`](design-philosophy/engineering-hypotheses.md)** | **Our own falsifiable claims** — what we assert the engineering will achieve, the threshold that decides it, and what would falsify it. Not design (that is `roadmap/` or `architecture/`) and not NGED-derived requirement (that is `background/`): a claim we are on the hook for, and a natural Network Innovation Allowance (NIA) deliverable. One page; append tests, never renumber them. |
| **GitHub** ([issues](https://github.com/openclimatefix/nged-substation-forecast/issues) + the OCF Project board) | The **complete, ordered task list** — including quick tweaks and non-code tasks — plus all discussion. **Fine-grained prioritisation lives only in GitHub.** Epics map 1:1 to the [roadmap milestones](roadmap/index.md#milestones); dependencies are recorded as `blocked by` issue relationships. |
| **[`docs/roadmap/`](roadmap/index.md)** | Design depth: What we plan to build and *why*. The milestone arc and inter-plan dependencies are recorded here; fine-grained task-level ordering is not. |
| **`docs/`[techniques](techniques/index.md), [background](background/index.md), [architecture](architecture/overview.md), [ml_experimentation](ml_experimentation/index.md), [live_service](live_service/index.md)** | What is already built — design (`architecture/`) and operational how-to (`ml_experimentation/`, `live_service/`) alike. These pages are where content moves to from `docs/roadmap/` after implementation. |
| **`plans/`** (repo root, not published) | At most **one** file per branch: the implementation plan for the work in flight on that branch, written before any code is touched and deleted when it merges. One worktree per branch is what keeps it to one file. Usually empty on `main`. |

**Relationship between `docs/roadmap/` and GitHub**: Every substantial 🚧 plan in the
`docs/roadmap/` folder has a GitHub issue, and every dependency stated in `docs/roadmap/` exists as
a `blocked by` link on GitHub — but GitHub freely contains small issues with no counterpart here in
the docs. (🔬 research ideas are exempt from GitHub until they are promoted to a milestone.) The
litmus test for needing a design doc in `docs/roadmap/`: *does it take more than a few sentences to explain?*

When a piece of work ships, its design content **moves out** of `roadmap/` to its permanent home
— and the roadmap page shrinks; when a page's last 🚧 item ships, the page is deleted. That
permanent home splits along a **why vs. how** line: `architecture/` holds system design — the
decisions and rationale, written once and rarely re-read step-by-step — while
[`ml_experimentation/`](ml_experimentation/index.md) and
[`live_service/`](live_service/index.md) hold operational how-to — step-by-step recipes for
running what's already built, one per area (ML backtesting vs. the live production service).
Each `architecture/` design page names its how-to counterpart (and vice versa) in a "See also"
section — e.g. [ML Orchestration Design](architecture/ml-orchestration.md) ↔
[ML Experimentation](ml_experimentation/index.md), and
[Production Deployment — Design](architecture/production-deployment.md) ↔
[Setting up the live service on AWS](live_service/aws.md). A page mixing the two — design
rationale followed by a runbook with literal commands — is a sign it should split along this line.
The `docs/roadmap/` folder therefore contains **only design for work that is not yet implemented**,
and is never a mirror of the code. Because roadmap pages are deletable, **code must never link into
`roadmap/`** — instead, code docstrings link to the durable sections (`design-philosophy/`,
`techniques/`, `architecture/`, `background/`, `ml_experimentation/`, `live_service/`) instead. The *methods*
behind these plans — differentiable physics, learned encoders, the disaggregation-evaluation
protocol — live in [Techniques](techniques/index.md) for exactly this reason: they survive the
roadmap items that apply them.

## Docstrings, READMEs and `docs/` hold three different jobs

The section above covers the pages under `docs/`. Docstrings and package READMEs are documentation
too — mkdocstrings renders every module listed in `docs/api/<package>/index.md` onto the published
site — so the same "one home per argument" question applies to them, and it is settled by asking
what the reader already has in their hand when they arrive.

**A docstring holds everything that dies when the symbol dies**: units, preconditions, invariants,
failure modes, and the argument for this particular implementation. The test is whether the prose
would go with the function if the function were deleted. `round_to_significand_bits` in
`delta_store.precision` is the model — its Veltkamp-splitting proof and the preconditions that make
the identity hold are meaningless away from that function, and they are exhaustive on it. A reader
arrives here from an editor, a traceback, or the API page, already holding the symbol.

**A package README holds what a reader needs in order to decide whether to open the package at
all**: what the package owns, what it deliberately does not own and which neighbouring package does,
and one line per module pointing down into the docstrings. The test is whether the sentence would
still be true if every function inside were reimplemented. `delta_store`'s README is the model, both
for drawing the boundary — `contracts` owns each table's logical shape, `delta_store` owns its
physical layout — and for deferring rather than repeating.

**`docs/` holds the arguments that outlive any one symbol, or that span more than one package**: the
design principles, the degradation ladder, end-to-end measurements, and the operator how-to. The
test is whether the argument references two or more packages, or would survive a rewrite of the
module it is nearest to.

**Links run in one direction, so the three homes cannot loop.** Docstrings and READMEs link *up*
into `docs/` for rationale that spans the package. Pages under `docs/` link *down* to the API page
for authoritative signatures and semantics, and do not restate them, because mkdocstrings has
already published them.

**A measurement lives where the decision it justifies is made, and is cited everywhere else.** The
choice to round `power_fcst` to a 13-bit significand is made in `delta_store.power_forecasts`, so
the measured effect of that choice belongs in that module's docstring;
[Performance](architecture/performance.md) cites it as a system-level consequence rather than
repeating the figures, and the README says the package makes the trade without giving the numbers a
third time. A measurement no single symbol owns — peak memory across a cross-validation fold, say —
belongs on the `docs/` page alone.

### Which place do I use?

| I want to… | Go to |
|---|---|
| Decide what to work on this morning | The GitHub Project board (complete, ordered) |
| Discuss / challenge a plan | GitHub issue comments (fold conclusions back into the roadmap page) |
| Think through a substantial design | A `docs/roadmap/` page, reviewed via PR |
| Communicate direction to NGED / leadership | The [milestones](roadmap/index.md#milestones) (published site) |
| Give an AI coding tool context on the broader plan | `docs/roadmap/` (plus `gh` for live task priorities) |
| Understand a method (differentiable physics, encoders, …) | [`docs/techniques/`](techniques/index.md) |
| Understand the principles the whole design answers to | [`docs/design-philosophy/`](design-philosophy/index.md) — the portable argument, readable without knowing the codebase |
| Understand *why* an existing system works the way it does | [`docs/architecture/`](architecture/overview.md) — the local rationale, recorded next to each component |
| State — or check — a measurable claim about the engineering | [`docs/design-philosophy/engineering-hypotheses.md`](design-philosophy/engineering-hypotheses.md). Add a test with a threshold and a resolution point; never renumber an existing one |
| Record an assessment of work we decided **not** to do | [`docs/architecture/`](architecture/overview.md), with a `Status:` banner saying so — e.g. [Why Dagster, not Airflow?](architecture/why-dagster-not-airflow.md). Not `docs/roadmap/`, which implies intent to build and is deleted on ship. |
| Learn *how* to run or operate an existing system, step by step | [`docs/ml_experimentation/`](ml_experimentation/index.md), [`docs/live_service/`](live_service/index.md) |
| File a quick tweak or a non-code task | A GitHub issue only — no markdown needed |
| Plan how to implement an issue, before writing code | `plans/<branch-name>.md` on that issue's branch (one file per branch, deleted on merge) |
| Explain what a function guarantees, what a caller must not assume, or why *this* implementation | The symbol's **docstring** — everything that would die with the symbol. Rendered onto the API page by mkdocstrings |
| Explain what a package is for, where its boundary against neighbouring packages falls, or what its modules are | The **package README** — the contents page for the docstrings rendered beneath it. Never restate a docstring here; both land on one page |
| Record a measured number | Wherever the decision it justifies is made. Cite it from the other two homes rather than repeating the figures |
