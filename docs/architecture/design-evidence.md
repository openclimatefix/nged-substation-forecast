# The experiments behind the production design

**This page lists each production setting that an ml-experiment decided, and links the experiment
that decided it.** A reader who asks "why does the wind forecast use these features?" finds the
answer here as one line and follows the link to the evidence. [Design
principles](../design-philosophy/design-principles.md) and the other `architecture/` pages say how
the system works and why in general. Each experiment's `result.md` holds the hypothesis, the
decision rule, the verdict, and the numbers.

## What belongs here

**An entry exists only when an ml-experiment's verdict is `supported` or `refuted` and the
production code or configuration it informs is named.** A study, a literature finding, or an
experiment whose winner has not been ported into production does not get an entry. The
[`ml-experiment` skill](https://github.com/openclimatefix/nged-substation-forecast/blob/main/.claude/skills/ml-experiment/SKILL.md)
tells whoever ports a winner into production to add the entry in the same change, so the page cannot
fall behind the code. A later experiment that overturns an entry changes the entry in place, and an
entry whose setting production drops is deleted. The experiment's `result.md` names the production
setting in its `informs` front matter field, so the experiment and this page point at each other.

## The entries

| Production setting | What production does | Experiment | Verdict |
|---|---|---|---|
| None yet | No ml-experiment's winner has been ported into production | | |
