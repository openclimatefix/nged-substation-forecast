# Restructure the planning docs

Consolidate three overlapping planning surfaces — `plans/*.md`, `docs/roadmap/`, and GitHub —
into a system with non-overlapping roles. Designed 2026-07-02; this file is the phased
implementation plan (one phase per PR, per `plans/README.md`). When the final phase lands,
this file is deleted and `plans/` will be empty — which is itself part of the design.

## Diagnosis

`plans/` was meant to hold one PR-scoped plan at a time, deleted on merge. It overflowed:
it now holds a 13-item ordered roadmap (durable, multi-month content duplicating
`docs/roadmap/index.md`'s job) *and* per-PR mechanics (the content it was designed for).
Meanwhile `docs/roadmap/index.md` is stale (still "as of Milestone 1, 28 May 2026"; doesn't
know about the July AWS-first reprioritisation), and GitHub sub-issues have drifted from the
markdown (7+ plans have no issue; #207 references a deleted plan file).

## Target roles (the core design)

Three places, non-overlapping jobs:

| Place | Job | Volatility |
|---|---|---|
| **GitHub** | The *complete*, *ordered* task list (the morning triage view), plus discussion. Epics = milestones; sub-issues = tasks; Project board order = priority; `blocked by` links = dependencies. | High — reordered freely, no doc edits needed |
| **`docs/`** | Design depth: `background/` = the problem, `techniques/` = solution methods (durable explainers), `roadmap/` = what we'll build and why, `architecture/`/`ml_experimentation/` = what's built | Low — changes via reviewed PRs |
| **`plans/`** | At most **one** file: the mechanical checklist for the PR currently in flight. Deleted on merge. Usually empty. | Ephemeral |

### Ordering lives at three tiers

1. **Milestones** (v0.1 → v0.2 → … → v1.0 → v2.0) — in `roadmap/index.md`. Strategic arc;
   changes ~quarterly; each milestone section links its GitHub epic (1:1, so drift is
   near-impossible).
2. **Dependencies among substantial plans** — inherent facts ("live asset before container
   before deployment"), stated in the roadmap pages/map table **and mirrored in GitHub as
   native `blocked by` issue relationships**.
3. **Task-level priority** — GitHub Project board order *only*. Markdown never claims task
   order; the map table in `index.md` carries an explicit disclaimer.

### Invariants (all one-directional: GitHub ⊇ markdown)

- Every 🚧 *scheduled* markdown plan has a GitHub issue. (🔬 research items are exempt until
  promoted to a milestone — no speculative tracker noise.)
- Every dependency stated in markdown exists as a `blocked by` link in GitHub.
- GitHub may freely contain small issues with no markdown counterpart ("quick little tweaks",
  non-code tasks). The litmus for needing markdown: *does it take more than a few sentences
  to explain?*
- **Code links only to durable docs** (`background/`, `techniques/`, `architecture/`,
  `ml_experimentation/`) — never to `roadmap/` pages, `plans/` files, or
  "Implementation details" sections. Everything in `roadmap/` eventually moves out or is
  deleted, so a code link into it is guaranteed to rot. Links *between docs* into `roadmap/`
  are fine: retargeting them is part of ship-time triage.

### Use-cases: which place do I use?

(This table lands in `roadmap/index.md`'s "How planning works" section in Phase 2, with a
pointer from CLAUDE.md in Phase 4.)

| I want to… | Go to |
|---|---|
| Decide what to work on this morning | GitHub Project board (complete, ordered) |
| Discuss / challenge a plan | GitHub issue comments (fold conclusions back into the roadmap page) |
| Think through a substantial design | A `docs/roadmap/` page, reviewed via PR |
| Communicate direction to NGED / leadership | `roadmap/index.md` milestones (published site) |
| Give Claude context on the broader plan | `docs/roadmap/` (+ `gh` for live priorities) |
| Understand a method (DP, encoders, …) | `docs/techniques/` |
| Understand what's already built | `docs/architecture/`, `docs/ml_experimentation/` |
| File a quick tweak or non-code task | GitHub issue only — no markdown needed |
| Write the mechanical checklist for the PR in flight | `plans/` (single file, deleted on merge) |

### Lifecycle: what happens when work ships

Ship-time triage, performed in the PR that lands the work:

1. **Promote** surviving design decisions to their permanent home (`architecture/`,
   `ml_experimentation/`, …) — the existing roadmap move-out rule.
2. **Delete** the plan's `## Implementation details (deleted when this ships)` section
   (and any `plans/` file); **paste it (or a summary) into the PR body** for archaeology.
   Git history preserves the full text regardless.
3. **Close** the GitHub issue; update the map table row and any status emoji.

Roadmap pages shrink as their items ship (the promoted content is replaced by a one-line
pointer to its permanent home). When a page's *last* 🚧 item ships, **delete the page** —
no mostly-empty stubs in the nav. Deleting means: remove the `mkdocs.yml` nav entry and the
map-table row, and retarget inbound doc links to the permanent homes. Code is never affected
because code never links into `roadmap/` (see invariants). If published-URL stability ever
matters, add the `mkdocs-redirects` plugin at that point — not before.

Status systems stay deliberately two-resolution: roadmap emoji (✅🚧🔬) are feature-level
documentation, updated when pages are edited; GitHub open/closed is task-level and always
current. No automatic syncing. Discussion happens on issues; when a thread reaches a
conclusion that changes a design, the *conclusion* is folded into the roadmap page in the PR
that acts on it.

## Target `docs/` tree

```text
docs/
├── background/                        # the PROBLEM (unchanged)
├── techniques/                        # NEW — solution METHODS; durable, never deleted
│   ├── index.md                       # one paragraph: what lives here
│   ├── differentiable-physics.md      # the method (split from roadmap/differentiable-physics.md)
│   ├── encoders.md                    # moved wholesale from roadmap/
│   └── disaggregation-evaluation.md   # moved wholesale from roadmap/
├── architecture/                      # unchanged
├── ml_experimentation/                # unchanged
└── roadmap/
    ├── index.md                       # REWRITTEN — milestones + map table (see below)
    ├── live-service.md                # NEW ← plans 02 + 03 + 04 + 05
    ├── xgboost-improvements.md        # NEW ← plan 09 (+ v0.5 detail from old index)
    ├── capacity-estimation.md         # NEW ← the plan half of differentiable-physics.md
    ├── engineering-health.md          # NEW ← plans 01, 08, 10, 12, 13
    ├── metrics-and-leaderboard.md     # absorbs plans 06, 07, 11
    ├── switching-events.md            # unchanged for now (split into techniques/ later if useful)
    ├── delivery-tables.md             # unchanged
    ├── forecast-building-blocks.md    # unchanged
    └── data-sources.md                # unchanged
```

Nav order: Background → **Techniques** → Architecture → ML Experimentation → Roadmap
(problem → methods → what exists → how to experiment → what's next).

**Page naming: topics, never numbers or versions** — ordering lives only in the `index.md`
milestone narrative and in GitHub, so reprioritising never renames files or breaks links.

### `roadmap/index.md` skeleton

```markdown
# Roadmap
(intro + status legend, as now)

## How planning works
(the three-place system above: GitHub is the complete ordered tracker —
 check the Project board / `gh` for current priorities; this folder holds
 design and dependencies; plans/ holds at most the current PR's checklist)

## Map of substantial work                ← grouped BY MILESTONE, not rank-ordered
| Work | Milestone | Design | GitHub | Depends on |
(one row per juicy plan; explicit disclaimer: "the complete, current
 ordering is the GitHub Project; this table maps work to its design docs")

## Milestones (v0.1 … v2.0)
(existing sections updated for the July 2026 AWS-first reprioritisation;
 each milestone header links its epic; v0.5 inline detail replaced by a
 pointer to xgboost-improvements.md)
```

## Content allocation (nothing is thrown away)

Every `plans/` file's full detail is preserved inside a
`## Implementation details (deleted when this ships)` section of its destination page.
Design-level content (context, decisions, invariants) sits above that section as the durable
part of the page.

| Source | Destination |
|---|---|
| `plans/00_review_findings.md` | Ordering table → `index.md` map table. Verdict + verified-findings list → `live-service.md` context or PR body. "Findings with no plan": NMAE-aggregation note → `metrics-and-leaderboard.md`; Patito friction budget → `architecture/code-style.md`; Delta-atomicity + partition-derivation findings → `live-service.md`; global-model normalisation → already in plan 09 item 16 |
| `plans/02_live_forecasts.md` | `live-service.md` — live/replay semantics and train==predict invariant are durable design; file-placement/test mechanics go in the Implementation details section |
| `plans/03_production_model_artifacts.md` | `live-service.md` (container + champion-model section) |
| `plans/04_aws_deployment.md` | `live-service.md` (the five costed architecture options + four workstreams — reviewable design worth publishing) |
| `plans/05_production_monitoring.md` | `live-service.md` (monitoring scope, sensor, retirement job) |
| `plans/06_baseline_forecasters.md` | `metrics-and-leaderboard.md`, new "Baselines" section (why scores need them + which baselines; mechanics in Implementation details) |
| `plans/07_probabilistic_evaluation.md` | `metrics-and-leaderboard.md` — the page already lists PICP/CRPS/spread-skill; plan 07 adds status + phased delivery |
| `plans/11_leaderboard_fold_hygiene.md` | `metrics-and-leaderboard.md`, "Cross-fold validation" section (a caveat on the existing protocol) |
| `plans/09_xgboost_quick_wins.md` | `xgboost-improvements.md` (the four tiers, in full) |
| `plans/01_ci.md`, `08_reproducibility_stamping.md`, `10_nwp_clip_logging.md`, `12_drop_hydra.md`, `13_rigor_tests_and_cleanup.md` | `engineering-health.md` — one H2 per item, full content. Common theme: tooling/reproducibility/rigour improvements that don't change the forecast |
| `roadmap/differentiable-physics.md` | Split: §2 core idea, §4–§7 building blocks/fleet machinery, MVA handling → `techniques/differentiable-physics.md`. §3 phases + capacity-estimation specifics (metered vs unmetered, curtailment, cheap baseline to beat) → `roadmap/capacity-estimation.md`. §1 problem statement → compressed to a framing paragraph linking `background/switching-events.md` |
| `roadmap/encoders.md`, `roadmap/disaggregation-evaluation.md` | Move wholesale to `techniques/` (already pure explainers) |

Cross-references to update: `index.md` and `switching-events.md` currently deep-link
`differentiable-physics.md` sections (e.g. "§8"); retarget after the split.

## Phases

### Phase 1 (PR 1): create `docs/techniques/`

- Move `encoders.md` and `disaggregation-evaluation.md`; add `techniques/index.md`.
- Split `differentiable-physics.md` per the allocation table (creates
  `roadmap/capacity-estimation.md`).
- Update `mkdocs.yml` nav (insert Techniques section; reorder as above) and all
  cross-links (`grep -r` for old paths).
- Pure docs move; no `plans/` involvement.

### Phase 2 (PR 2): rewrite `roadmap/index.md`

- New skeleton above: "How planning works", milestone-grouped map table, updated milestones
  (July reprioritisation; epic links on milestone headers).
- Absorb and delete `plans/00_review_findings.md` (scatter the no-plan findings per the
  allocation table — the `code-style.md` and `metrics-and-leaderboard.md` edits happen here).

### Phase 3 (PR 3): absorb the remaining plans

- Create `live-service.md`, `xgboost-improvements.md`, `engineering-health.md`; extend
  `metrics-and-leaderboard.md` (plans 06/07/11).
- Every absorbed plan keeps its full detail under
  `## Implementation details (deleted when this ships)`.
- Delete `plans/01–13`. `plans/` is now empty except `README.md` and this file.
- Update `mkdocs.yml` nav for the new roadmap pages.

### Phase 4 (PR 4): conventions

- Rewrite `plans/README.md`: at most one plan at a time — the in-flight PR's mechanical
  checklist; deleted on merge; usually empty; everything durable belongs in `docs/roadmap/`.
- CLAUDE.md updates: mention `docs/techniques/` in the Docs section; add the
  planning-system pointer ("GitHub Project = complete ordered task list; check with `gh`
  when current priorities matter; `docs/roadmap/` = design + dependencies; use-case table
  in `roadmap/index.md`"); add the ship-time triage checklist; tighten the linking rule —
  code links only to durable docs sections, never to `roadmap/`, `plans/`, or
  "Implementation details" sections (the current "linking between code and the permanent
  docs under `docs/` is encouraged" wording must exclude `roadmap/`).
- **Retarget the 7 existing code→roadmap references** (found 2026-07-02: `cv_assets.py`
  ×2, `power_schemas.py` ×4, `metrics.py` ×1). Each either points to the content's new
  durable home (e.g. `docs/roadmap/differentiable-physics.md` →
  `docs/techniques/differentiable-physics.md` after Phase 1) or, where the referenced
  rationale is roadmap-only (the metrics/leaderboard notes), inlines a short version of the
  rationale in the docstring instead of linking.
- Delete this plan file (its own ship-time triage).

### Phase 5 (GitHub pass — not a PR; alongside/after Phase 3)

Fill the gaps found in the 2026-07-02 audit of epic #132's sub-issue tree:

- **Create missing sub-issues** (batch-drafted for Jack's review before creation; each body
  ~2 lines + link to its roadmap section): live inference asset (under #137; #208 is only
  its verification step), container + champion model (under #137), production monitoring,
  probabilistic evaluation (under #6), reproducibility stamping, leaderboard fold hygiene,
  drop Hydra, rigor tests (under #138), and the plan-09 Tier-1 quick wins (under #145).
- **Mirror dependencies** as native `blocked by` links (live asset → container → deployment;
  monitoring blocked by live forecasts accumulating; xgboost item 15 blocked by 14; …).
  Confirm `gh`/GraphQL support for the dependencies API at execution time; fall back to a
  "Blocked by #n" body-text convention if the API isn't available yet.
- **Fix drift**: close or rewrite #207 (references the deleted `dagster_plan.md`);
  re-parent #144 (data cleaning) from #136 to the v0.4 epic #150.
- **Slim epic bodies** to ~2 lines + roadmap-page link; roadmap pages link back.
- **Triage calls for Jack**: are #96 (NGED-agreed schema) and #5 (backups) v0.1-gating
  (→ map-table rows)? Does #161's placement under the v0.1 epic reflect real intent, or
  should it move to match the review's later ordering? Do the #136 strays
  (#179, #197, #153) stay GitHub-only?

### Later (deliberately deferred)

- Splitting `switching-events.md` into technique vs roadmap halves — revisit once
  `techniques/` has proven its shape.
- A freshness audit (script or occasional Claude pass) for the two mechanical drift forms:
  🚧 markdown plan with no open issue; closed issue whose markdown section still exists.
  Wait a few weeks of living with the system first.
- Whether `delivery-tables.md` / `forecast-building-blocks.md` migrate to `architecture/`
  as they ship — the existing move-out rule handles this naturally.

## Verification

- `uv run pymarkdown scan -r docs README.md CLAUDE.md metadata/README.md packages/*/README.md`
  after each phase.
- `uv run mkdocs build --strict` (catches broken nav entries and dead cross-links).
- `grep -rn "plans/" docs/ src/ packages/` returns nothing (no doc/code references to
  `plans/`); after Phase 4, `grep -rn "roadmap/" src/ packages/ --include="*.py"` also
  returns nothing; `grep -rn "differentiable-physics.md#" docs/` finds only valid anchors.
- After Phase 5: every 🚧 map-table row has an open issue; #207 is closed or rewritten;
  and #144 sits under #150.
