# Plan: remove the maintainer's name from the skills and docs (#583)

**The problem.** `CLAUDE.md` tells us to write the rule, not who asked for it, and to name the role
where a sentence needs an actor. Seven files still name one person 40 times, mostly inside
behavioural instructions that agents follow: `plan-wave` (15), `plan-issue` (10),
`simplicity-clean-room` (6), `implement-issue` (4), `docs/roadmap/live-service.md` (2),
`github-issue-pr-workflow` (2) and `code-style` (1). Four more sentences carry a `he`/`him`/`his`
that becomes an orphan once the name goes.

**The solution.** A read-through rewrite, file by file, replacing each occurrence with the phrase
its own sentence needs. Six sites mean "the routine halts and a person looks at the work", and
those become **"stop for human review"** — the issue's headline case, worded to distinguish a
person from the reviewing sub-agents the same routines spawn. The other thirty-eight sites mean
something else — a blocking question, an audience, a decision-maker, someone clicking a chip — and
each gets the role name that fits. No routine is told to do anything different from what it does
today.

## Verdict

**Worth implementing, roughly as described.** This is the repo's own written prose rule applied to
the files that escaped the pass which produced it. The name appears in the frontmatter
`description:` of four skills, which is loaded into every session's context, so the cost of leaving
it is paid on every turn, not just when a skill is opened.

## Departures from the issue body

1. **The per-file counts in the body are stale.** Measured on `main` at `66de9c6c`, excluding the
   `JackKelly` GitHub handle: 40 occurrences, not the body's 46. `implement-issue` has 4 (body says
   7) and `github-issue-pr-workflow` has 2 (body says 5) — PR #581 and #582 took the rest.

2. **"stop for human review" is the right replacement for six of the forty sites, not all forty.**
   The comment gives the headline case; `CLAUDE.md`'s "Don't name individuals" rule gives the
   general one. Substituting the headline phrase everywhere would turn "ask the human which number
   to start from" into "ask stop-for-human-review", and "a chip is inert until it is clicked" into
   nonsense. The mapping below is the actual deliverable.

3. **`docs/roadmap/live-service.md:569` is left alone.** The name there sits inside
   `[#5 Backup procedure for data & models on Jack's workstation](…)` — a GitHub issue title quoted
   verbatim as a link label. That is data in the same sense as the `JackKelly` assignee, and editing
   it makes the docs table disagree with the issue it links to. See the open questions.

4. **The `authors` entries in five `pyproject.toml` files are out of scope.** `{ name = "Jack
   Kelly", email = "jack@openclimatefix.org" }` is package metadata, not prose.

## What changes, file by file

Forty name occurrences plus four pronoun sentences. Grouped by what the sentence means, because the
replacement follows the meaning:

### Class A — the routine halts and a person looks (6 sites) → "stop for human review"

| File:line | Now | After |
|---|---|---|
| `implement-issue` frontmatter:9 | `stop for Jack, never merge` | `stop for human review, never merge` |
| `implement-issue`:19 | `and stops for Jack` | `and stops for human review` |
| `implement-issue`:113 | `**Stop and wait for Jack's review. Never merge.**` | `**Stop for human review. Never merge.**` |
| `implement-issue`:118–119 | `**Why:** Jack reviews diffs in GitHub's UI and wants a PR to already have survived an adversarial pass by the time he looks at it, so his review is the last line of defence rather than the first.` | Adopt `CLAUDE.md`:182–184 verbatim: `**Why:** diffs are reviewed in GitHub's UI, and a PR should already have survived an adversarial pass by the time a human opens it, so that human review is the last line of defence rather than the first.` |
| `plan-issue` frontmatter:8 | `triage the findings, stop for Jack` | `triage the findings, stop for human review` |
| `github-issue-pr-workflow`:80 | `you stop and wait for Jack's review rather than merging at all` | `you stop for human review rather than merging at all` |

Row 4 deliberately reuses wording already committed in `CLAUDE.md`, so the canonical statement of
the routine and the skill that implements it read the same.

### Class B — decision authority and approval (10 sites) → "the human reviewer" / "a human" / the rule itself

| File:line | Now | After |
|---|---|---|
| `plan-issue`:16–17 | `Jack approves a plan before any code moves, so his review of the eventual diff checks execution` | `A human approves a plan before any code moves, so the review of the eventual diff checks execution` |
| `plan-issue`:147 | `the things Jack should decide` | `the things the human reviewer should decide` |
| `plan-issue`:218 | `**A proposed rearchitecture is Jack's call, not yours**` | `**A proposed rearchitecture is the human reviewer's call, not yours**` |
| `plan-issue`:222 | `unless Jack says otherwise` | `unless the human reviewer says otherwise` |
| `plan-issue`:260 | `Once Jack approves the plan` | `Once a human approves the plan` |
| `simplicity-clean-room`:23 | `and Jack decides which findings are worth it` | `and a human decides which findings are worth it` |
| `simplicity-clean-room`:226–227 | `Then ask Jack which findings he wants acted on` | `Then ask which findings should be acted on` |
| `github-issue-pr-workflow`:78 | "Jack wants the full commit history preserved in `main`, so use a merge commit" | "We keep the full commit history in `main`, so use a merge commit" |
| `code-style` frontmatter:4 | `This repo's code conventions, which Jack cares about and expects to be followed:` | `This repo's code conventions, which every change is expected to follow:` |
| `live-service`:541 | `Terraform vs CDK is Jack's call to make when Stage 2 work starts` | `Terraform vs CDK is a call for whoever starts Stage 2 work` |

"the human reviewer" rather than bare "the reviewer" in `plan-issue`, because that file's steps 4
and 6 call its sub-agents "the reviewer" four times; a bare noun there would read as an instruction
to let a sub-agent decide.

### Class C — a blocking question to a person (5 sites) → "ask the human"

`plan-wave`:42 (`ask Jack which number to start from`), :55 (`ask Jack before dispatching`), :121,
:152 and :156 (three instances of `stop and ask Jack rather than edit it` / `ask Jack rather than
plan on top of code that is about to move`) all become `ask the human`. This is the load-bearing
case after Class A: a sub-agent asking its parent agent is not the same act as halting for a person,
and `ask for a decision` would blur them.

### Class D — audience for output, or reader (13 sites) → "the user", or the actor dropped

| File:line | Now | After |
|---|---|---|
| `plan-wave` frontmatter:6 | `chips Jack launches as separate Claude Code sessions` | `chips the user launches as separate Claude Code sessions` |
| `plan-wave` frontmatter:9 | `Load whenever Jack asks what to work on next` | `Load whenever the user asks what to work on next` |
| `plan-wave`:17 | `plus one chip per issue for Jack to launch` | `plus one chip per issue to launch` |
| `plan-wave`:30 | `Jack uses it to tell this wave's sessions apart from the last one's` | `The wave number is what tells this wave's sessions apart from the last one's` |
| `plan-wave`:125 | `the wave costs Jack five plan reviews` | `the wave costs five plan reviews` |
| `plan-wave`:129 | `Show Jack the wave as a table` | `Present the wave as a table` |
| `plan-wave`:163 | `it is what Jack reads in the tooltip` | `it is what the user reads in the tooltip` |
| `plan-issue`:66 | `Jack picks sessions out of the desktop app's session list` | `The user picks sessions out of the desktop app's session list` |
| `plan-issue`:72 | `a title Jack can copy in one gesture` | `a title the user can copy in one gesture` |
| `plan-issue`:215 | `so Jack can see what was considered and dismissed` | `so the human reviewer can see what was considered and dismissed` |
| `plan-issue`:256 | `Report to Jack: the verdict from step 2` | `Report: the verdict from step 2` |
| `simplicity-clean-room` frontmatter:7,9 | `Load when Jack suspects a module is over-built` … `before promising him anything` | `Load when the user suspects a module is over-built` … `before promising anything` |
| `simplicity-clean-room`:27, :194, :219 | `Say this to Jack up front` / `the half of the answer Jack's question actually needs` / `before repeating any of them to Jack` | `Say this to the user up front` / `the half of the answer the question actually needs` / `before repeating any of them to the user` |

"the user" is the role in the skill-triggering `description:` fields and wherever the sentence means
"the person in this conversation". It matches how the frontmatter of every other skill in the repo
already reads, so skill matching is not disturbed.

### Class E — an actor performing an action (4 sites)

| File:line | Now | After |
|---|---|---|
| `plan-wave`:25 | `If Jack names the epic by milestone ("v0.2") instead, find it` | `If the epic is named by milestone ("v0.2") instead, find it` |
| `plan-wave`:131 | `A chip is inert until Jack clicks it` | `A chip is inert until it is clicked` |
| `plan-wave`:142 | `write no code until Jack approves the plan` | `write no code until a human approves the plan` |
| `plan-issue` frontmatter:8 | `Load whenever Jack asks for a plan for an issue` | `Load whenever the user asks for a plan for an issue` |

## Design-philosophy check

Nothing here runs in production or in R&D — the change touches only markdown prose in
`.claude/skills/` and one `docs/` page. No asset, asset check, Patito schema, config field or Python
symbol is touched, so the degrade-don't-raise rules in `inherent-stability.md` and the `WARN` /
`blocking=False` rule do not apply, and no hypothesis label in `engineering-hypotheses.md` is
delivered or affected. The change *implements* one existing rule — `CLAUDE.md`, "Prose style" →
"Don't name individuals" — and trades away no principle from `design-principles.md`.

## Tests

**No test is added, and none can be.** There is no test that would fail on `main` and pass after,
because nothing executable changes: the repo has no linter rule, pre-commit hook or test that reads
prose for named individuals, and adding one to guard 40 sites in seven files would be more machinery
than the rule needs. Saying so is the honest answer; inventing a test here would be the
"generalising for a caller that does not exist" failure.

What stands in for a test is a mechanical completeness check, run before pushing:

```bash
grep -rniE "\bjack\b|\b(he|him|his)\b" --include='*.md' . | grep -v '^\./\.git/' | grep -v JackKelly
```

It must return exactly two lines — `docs/roadmap/live-service.md:569` (the quoted issue title) and
nothing else — plus whatever `plans/blissful-bohr-a635db.md` itself contains, which is deleted at
ship time. The five `pyproject.toml` author entries are outside the `*.md` filter.

## Docs to update

`docs/roadmap/live-service.md` is the only `docs/` page in the surface, and it is in the file list
above. `CLAUDE.md` and `docs/documentation-guide.md` were checked and contain zero occurrences of
the name — but see open question 1 about `CLAUDE.md`:167.

This issue does not complete a roadmap item, so there is no ship-time triage: no "Implementation
details" section to delete and no status banner to move.

## Verification commands

The green-before-push set, plus what a markdown-only change needs:

```bash
uv run ruff check . && uv run ruff format . && uv run --all-packages ty check && uv run pytest
uv run pymarkdown scan -r docs README.md CLAUDE.md packages/*/README.md
uv run pymarkdown scan -r .claude/skills
uv run mkdocs build --strict
```

`mkdocs build --strict` is needed because `live-service.md` renders on the published site. If any
list or wrapped link is reflowed, load the `mkdocs-authoring` skill first and read the rendered HTML
under `site/`, not just the linter — the nested-list and wrapped-link gotchas pass both linters and
render wrong. The intended edits are all in-line word substitutions, so no list structure should
move; if one does, that is the signal to check.

Beyond the tooling, the real verification is a read-through of the four skill files end to end,
asking of each rewritten sentence whether it instructs the same act as before.

## Risks and open questions

1. **`CLAUDE.md`:167 says "stops for review", which is the flattening this issue exists to
   prevent.** The name is already gone from `CLAUDE.md`, so it is not in the issue's file list — but
   line 167's `and stops for review` and line 171's `after each, stop.` describe the same two halts
   that this plan is wording as "stop for human review" in the skills, and `CLAUDE.md` is the
   canonical statement of the routine. Leaving it means the two files describe the halt differently
   in the one place where the human-versus-agent distinction is the point.
   **Recommendation: make it consistent** — `and stops for human review` on line 167, and
   `after each, stop for human review.` on line 171. Two words in a file already in the verification
   set. **Ask before doing it**, since it is outside the issue's stated surface.

2. **`live-service.md`:569 quotes issue #5's title, which contains the name.** The plan leaves it,
   on the grounds that a verbatim issue title in a link label is data. The alternative is
   `gh issue edit 5 --title "Backup procedure for data & models on the maintainer's workstation"`
   and updating the label to match. **Recommendation: leave both alone for now.** Renaming a GitHub
   issue is a change to something outside the repo, and #5 is already marked deferred and largely
   superseded. Worth a one-line follow-up issue if the name should go from GitHub titles too.

3. **Three other wave-5 sessions are following these very skills while this rewrites them.** They
   read their own worktree copies, so the edits cannot reach them mid-run, and this change alters no
   instruction's meaning. The residual risk is that a rewritten sentence *does* change an
   instruction — which is exactly what the correctness review in step 6 and the read-through in the
   verification section are for.

4. **"the user" versus "the human" in skill frontmatter.** The plan uses "the user" for triggering
   descriptions and audience, "the human" for blocking questions, and "human review" for the halts.
   A single word everywhere would be simpler to state but would lose the distinction the issue asks
   for. **Recommendation: keep the three.** Flag it here because it is the one wording decision in
   the change that is a judgement call rather than a mechanical substitution.
