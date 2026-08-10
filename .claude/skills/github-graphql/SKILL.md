---
name: github-graphql
description: >-
  Concrete `gh api graphql` invocations for the four GitHub operations plain `gh issue`/`gh pr`
  can't do: attaching a sub-issue to its parent epic, reordering sub-issues, setting an issue's
  org-level Type, and setting a Projects (v2) field (Status/Project/Area). Load before running any
  of them for openclimatefix/nged-substation-forecast, or whenever you'd reach for `gh api graphql`
  and aren't sure of the mutation name, its input fields, or how to get the node IDs it needs.
---

# GitHub GraphQL cheatsheet

Every mutation below operates on GraphQL **node IDs**, not the issue/PR numbers or URLs `gh`
normally works with. Get a node ID with:

```bash
gh issue view <number> --json id --jq .id   # works for PRs too via `gh pr view`
```

All commands assume `openclimatefix/nged-substation-forecast`; swap the owner/name for another
repo. Every mutation's input fields below were confirmed against GitHub's live schema by
introspection. If a call starts failing, re-check the same way — GitHub occasionally adds or
renames fields: `{ __type(name: "<MutationName>Input") { inputFields { name } } }`.

## Attach a sub-issue to its parent

`addSubIssue` — `issueId` is the **parent's** node ID, `subIssueId` is the **child's**:

```bash
gh api graphql -f query='
  mutation($issueId: ID!, $subIssueId: ID!) {
    addSubIssue(input: {issueId: $issueId, subIssueId: $subIssueId}) {
      subIssue { number }
    }
  }' -f issueId="<parent node id>" -f subIssueId="<child node id>"
```

Do this *before* trying to reorder the sub-issue — `reprioritizeSubIssue` below assumes the
attachment already exists.

## Reorder a sub-issue within its parent's list

`reprioritizeSubIssue`. `afterId`/`beforeId` take a **sibling sub-issue's** node ID (not the
parent's) and are mutually exclusive — pass whichever one expresses where the issue should land:

```bash
gh api graphql -f query='
  mutation($issueId: ID!, $subIssueId: ID!, $afterId: ID) {
    reprioritizeSubIssue(input: {issueId: $issueId, subIssueId: $subIssueId, afterId: $afterId}) {
      issue { number }
    }
  }' -f issueId="<parent node id>" -f subIssueId="<child node id>" -f afterId="<sibling node id>"
```

## Set an issue's Type

`updateIssueIssueType`. Issue-type IDs are stable per-org, so look them up once and reuse:

```bash
gh api graphql -f query='
  { repository(owner: "openclimatefix", name: "nged-substation-forecast") {
      issueTypes(first: 10) { nodes { id name } }
  } }'
```

then:

```bash
gh api graphql -f query='
  mutation($issueId: ID!, $issueTypeId: ID!) {
    updateIssueIssueType(input: {issueId: $issueId, issueTypeId: $issueTypeId}) {
      issue { number }
    }
  }' -f issueId="<issue node id>" -f issueTypeId="<issue type node id>"
```

## Set a GitHub Projects (v2) field

Prefer `gh project item-edit` over raw GraphQL — it wraps `updateProjectV2ItemFieldValue` for you
and only drop to the raw mutation if it doesn't cover the field type you need (e.g. iteration
fields it can't set). Gather the IDs it needs:

```bash
gh project view 33 --owner openclimatefix --format json --jq .id          # project node ID
gh project field-list 33 --owner openclimatefix --format json             # field + option IDs
gh project item-add 33 --owner openclimatefix --url <issue-url> \
  --format json --jq .id                                                  # item node ID
```

Then, for a single-select field (Status, Project, Area are all single-select in this repo's
board):

```bash
gh project item-edit --id <item node id> --project-id <project node id> \
  --field-id <field node id> --single-select-option-id <option id>
```

If raw GraphQL is genuinely needed, the mutation is:

```bash
gh api graphql -f query='
  mutation($projectId: ID!, $itemId: ID!, $fieldId: ID!, $optionId: String!) {
    updateProjectV2ItemFieldValue(input: {
      projectId: $projectId, itemId: $itemId, fieldId: $fieldId,
      value: {singleSelectOptionId: $optionId}
    }) {
      projectV2Item { id }
    }
  }' -f projectId="<project node id>" -f itemId="<item node id>" \
     -f fieldId="<field node id>" -f optionId="<option id>"
```
