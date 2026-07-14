# Documentation Guide

How this repository's documentation and planning content is organized — where to look for
something, and where to put something new.

> **Status legend** (used throughout the design docs below):
> ✅ **Implemented** — exists in code today ·
> 🚧 **Planned** — designed, not yet built ·
> 🔬 **Research** — exploratory / v2.

## How planning works

Planning content lives in four places with deliberately non-overlapping jobs:

| Place | Job |
|---|---|
| **GitHub** ([issues](https://github.com/openclimatefix/nged-substation-forecast/issues) + the OCF Project board) | The **complete, ordered task list** — including quick tweaks and non-code tasks — plus all discussion. **Fine-grained prioritisation lives only in GitHub.** Epics map 1:1 to the [roadmap milestones](roadmap/index.md#milestones); dependencies are recorded as `blocked by` issue relationships. |
| **[`docs/roadmap/`](roadmap/index.md)** | Design depth: What we plan to build and *why*. The milestone arc and inter-plan dependencies are recorded here; fine-grained task-level ordering is not. |
| **`docs/`[techniques](techniques/index.md), [background](background/network.md), [architecture](architecture/overview.md), [ml_experimentation](ml_experimentation/index.md), [live_service](live_service/index.md)** | What is already built — design (`architecture/`) and operational how-to (`ml_experimentation/`, `live_service/`) alike. This is where content moves to from `docs/roadmap/` after implementation. |
| **`plans/`** (repo root, not published) | At most **one** file: the mechanical checklist for the PR currently in flight, deleted when it merges. Usually empty. |

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
`roadmap/`** — instead, code docstrings link to the durable sections (`techniques/`,
`architecture/`, `background/`, `ml_experimentation/`, `live_service/`) instead. The *methods*
behind these plans — differentiable physics, learned encoders, the disaggregation-evaluation
protocol — live in [Techniques](techniques/index.md) for exactly this reason: they survive the
roadmap items that apply them.

### Which place do I use?

| I want to… | Go to |
|---|---|
| Decide what to work on this morning | The GitHub Project board (complete, ordered) |
| Discuss / challenge a plan | GitHub issue comments (fold conclusions back into the roadmap page) |
| Think through a substantial design | A `docs/roadmap/` page, reviewed via PR |
| Communicate direction to NGED / leadership | The [milestones](roadmap/index.md#milestones) (published site) |
| Give an AI coding tool context on the broader plan | `docs/roadmap/` (plus `gh` for live task priorities) |
| Understand a method (DP, encoders, …) | [`docs/techniques/`](techniques/index.md) |
| Understand *why* something already built works the way it does | [`docs/architecture/`](architecture/overview.md) |
| Learn *how* to run/operate something already built, step by step | [`docs/ml_experimentation/`](ml_experimentation/index.md), [`docs/live_service/`](live_service/index.md) |
| File a quick tweak or a non-code task | A GitHub issue only — no markdown needed |
| Write the mechanical checklist for the PR in flight | `plans/` (single file, deleted on merge) |
