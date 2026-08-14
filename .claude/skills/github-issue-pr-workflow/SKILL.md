---
name: github-issue-pr-workflow
description: >-
  The checklist for creating a GitHub issue or PR in openclimatefix/nged-substation-forecast —
  the fields `gh issue create` / `gh pr create` cannot set (labels, org issue Type, OCF project 33
  and its Status/Project/Area fields, sub-issue attachment and ordering, the JackKelly assignee) —
  plus the never-hard-wrap rule for anything posted to GitHub, the never-squash-merge rule and
  ship-time triage. Load before running `gh issue create`, `gh pr create`, `gh pr comment`,
  `gh issue comment` or `gh pr merge`, before writing any issue/PR body or comment, or when a PR
  completes a roadmap item.
---

# Creating issues and PRs, and shipping them

## Creating GitHub issues

Whenever you create an issue, also set:

- **Labels** and **Type** (org issue type: Task / Bug / Feature / Spike / Epic / …) — pick
  whatever fits the issue.
- Add it to the **OCF project** (org project 33, `gh project item-add 33 --owner
  openclimatefix --url <issue-url>`) and set the project fields **Status = Todo**,
  **Project = NGED**, **Area = ML**.
- If it is a sub-issue, attach it to its parent epic **and position it appropriately in the
  parent's sub-issue order** (execution order, respecting `blocked by` chains) — the
  `reprioritizeSubIssue` GraphQL mutation with `afterId`/`beforeId`.
- **Body** — if (and only if) the docs already contain a plan for the issue (e.g. a
  `docs/roadmap/` section), the body may be *just* a link to that rendered docs section and
  nothing more; don't duplicate the plan. Otherwise, write a self-contained body.
- When the body links to a docs page, link to the **rendered site**
  (`https://openclimatefix.github.io/nged-substation-forecast/...`), never a `github.com`
  blob path.

`gh issue create` can't set any of these: use `gh issue edit --add-label` for labels, the
`updateIssueIssueType` GraphQL mutation for Type, and `gh project item-edit` (or the
`updateProjectV2ItemFieldValue` mutation) for the project fields.

## Creating pull requests

Whenever you create a PR, also set:

- **Labels** — pick whatever fits (e.g. `documentation`, `enhancement`, `bug`), same label set as
  issues.
- **Assignees = JackKelly**.

`gh pr create` can't set either: use `gh pr edit --add-label` and `gh pr edit --add-assignee
JackKelly` right after creating the PR.

## Never hard-wrap a GitHub body or comment

**Write one line per paragraph. No hard wraps, at any width.** This applies to every issue body,
PR body, issue comment, PR comment and review comment — everything posted to GitHub rather than
committed to the repo. Blank lines still separate paragraphs, and list items still get their own
line; it is only wrapping *within* a paragraph that is forbidden.

GitHub renders comment-shaped content with hard line breaks turned on, so a single newline inside
a paragraph becomes a literal `<br>`. Repo `.md` files are rendered without that setting, which is
why the same wrapping is correct there and wrong here. GitHub's own API shows the two renderers
disagreeing on identical input:

```bash
printf '{"text":"line one\\nline two","mode":"gfm"}' | gh api /markdown --input -
```

`mode=gfm` (the issue/PR/comment renderer) returns `<p>line one<br>\nline two</p>`;
`mode=markdown` (the repo-file renderer) returns `<p>line one\nline two</p>` with no `<br>`. A body
wrapped at this repo's 100-character prose width therefore reaches a reviewer as a ragged block of
forced breaks, one per source line.

**The trap is the draft file, not the typing.** A long body wants to be written to a file and
passed with `gh pr create --body-file`, and a draft written *inside the repo* gets reflowed to the
house line length by the markdown linter or by habit — at which point the wrapping is baked in
before the body is ever posted. Write the draft to the session scratchpad directory instead, where
no linter touches it, and pass that path to `--body-file`. Never commit a body draft.

## Merging pull requests

Never squash-merge. We keep the full commit history in `main`, so use a merge commit
(`gh pr merge --merge`) or rebase (`gh pr merge --rebase`), not `gh pr merge --squash`. Under the
`implement-issue` routine you stop for human review rather than merging at all.

**Check what the merge will close, before you merge.** Issues are closed by two independent routes,
the **PR body** and the **commit messages**, and neither is reliably visible in
`closingIssuesReferences` beforehand. Read the text yourself:

```bash
KW='\b(clos|fix|resolv)[a-z]*\b.{0,60}#[0-9]+|#[0-9]+.{0,60}\b(clos|fix|resolv)[a-z]*\b'
gh pr view <N> --json body --jq .body | grep -inE "$KW"
git log origin/main..HEAD --format='%B' | grep -inE "$KW"
```

The word boundaries are load-bearing: without them `openclimatefix` matches on its own trailing
"fix", so every body carrying a GitHub URL near a `#number` looks like a hit, and a guard that
fires on every PR stops being read. They cost nothing in coverage — every keyword GitHub honours
is a standalone word.

**`closingIssuesReferences` proves nothing before the merge.** It stays `[]` right up to the point
the merge lands and only then fills in. A PR body reading "tracked in issue \#593, which this does
not close" reported `[]` when checked minutes before merging, registered a closing link on merge,
and closed 593. Commit-message keywords are invisible to the field for the same reason. Treat an
empty result as no evidence at all.

**Word order does not save you, and neither does a negation.** GitHub matches the keyword whether
it falls before or after the reference, and it ignores the "not": both `closes #512` and "#512,
which this does not close" register the link. Prose *about* a closure counts too — "Merging #514
closed #512" in a commit message closes 512. When writing about an issue the PR leaves open, keep
close, fix and resolve out of the sentence entirely. "Tracked separately in #512" and "issue #512
stays open" are safe; anything pairing the word with the number is not.

**The link is sticky once registered.** GitHub records it when the text containing the keyword is
first saved and does not drop it when the text is edited away, so a body that now reads "filed
rather than fixed, see \#512" can still hold a closing link from an early draft. A link you did not
intend cannot be edited out — close the PR and open it again from the same branch with a clean
body.

**Check every issue the PR mentioned once the merge lands.** The pre-merge checks tell you about
the text, not about what GitHub decided to do with it.

**Don't pass `--delete-branch`.** The repo has `delete_branch_on_merge` turned on, so GitHub deletes
the head branch on merge by itself. The flag only adds a *local* branch deletion, and that step
first checks out another branch — usually `main`, which the primary worktree already holds. Git
refuses, and `gh` reports the refusal as its own exit status:

```text
failed to run git: fatal: 'main' is already used by worktree at '/home/jack/dev/python/...'
```

The merge has already gone through by then. **A non-zero exit from `gh pr merge` does not mean the
merge failed**, so confirm the outcome rather than retrying the command or reporting a failure:
`gh pr view <N> --json state,mergeCommit` and the issue's own state say what actually landed.
Delete the local branch afterwards, from a worktree that is not sitting on it.

When something is closed that should not have been: `gh issue reopen <N>`, then put its project
Status back explicitly. The board automation moves a closed issue to Done, and reopening lands it
on In Progress rather than Todo — see the `github-graphql` skill for `gh project item-edit`.

## GraphQL calls

Attaching and reordering sub-issues, setting an issue's Type, and setting a project field all need
`gh api graphql`. The `github-graphql` skill has the exact invocations and how to obtain the node
IDs they need.

## Ship-time triage

When a PR lands a roadmap item, that PR (or an immediate follow-up) must also:

1. Promote surviving design decisions to their permanent home (`docs/architecture/`,
   `docs/ml_experimentation/`, …).
2. Delete the item's "Implementation details" section (and any `plans/` file), pasting it (or
   a summary) into the PR body. When a roadmap page's last 🚧 item ships, delete the page
   (nav entry, inbound doc links).
3. Close the GitHub issue; update the status banner on the roadmap page (and the milestone
   section in `docs/roadmap/index.md` if the arc changed).
