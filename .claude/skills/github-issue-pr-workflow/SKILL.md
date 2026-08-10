---
name: github-issue-pr-workflow
description: >-
  The checklist for creating a GitHub issue or PR in openclimatefix/nged-substation-forecast —
  the fields `gh issue create` / `gh pr create` cannot set (labels, org issue Type, OCF project 33
  and its Status/Project/Area fields, sub-issue attachment and ordering, the JackKelly assignee) —
  plus the never-squash-merge rule and ship-time triage. Load before running `gh issue create`,
  `gh pr create` or `gh pr merge`, or when a PR completes a roadmap item.
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

## Merging pull requests

Never squash-merge. Jack wants the full commit history preserved in `main`, so use a merge commit
(`gh pr merge --merge`) or rebase (`gh pr merge --rebase`), not `gh pr merge --squash`. Under the
`implement-issue` routine you stop and wait for Jack's review rather than merging at all.

**Check what the merge will close, before you merge:**

```bash
gh pr view <N> --json closingIssuesReferences --jq '.closingIssuesReferences[].number'
```

Every number listed is closed the moment the PR merges. The list is *sticky*: a closing keyword
in an early draft of the PR body, or in any commit message on the branch, registers the link
permanently, and later editing that text away does not remove it. So a PR whose body now says
"filed rather than fixed, see #512" can still be holding a closing link to #512 from a draft —
reading the current body is not enough, and neither is grepping the commits.

If the list contains an issue you did not mean to close, either sort it out before merging or
watch for it afterwards: `gh issue reopen <N>`, then put its project Status back (the board
automation moves a closed issue to Done, and reopening it lands on In Progress, not Todo — see
the `github-graphql` skill for `gh project item-edit`).

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
