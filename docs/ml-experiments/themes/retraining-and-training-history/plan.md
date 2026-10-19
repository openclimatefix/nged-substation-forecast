# Retraining and training history: research plan

**This page lists the experiments in this theme that are planned or proposed.** The
[Experiments](experiments.md) page holds the finished ones. A lower number in the "Priority" column
runs sooner.

## Priority

| Id | Question | Priority | Waits for | State |
|---|---|---|---|---|
| [P16](#p16-a-longer-ens-history) | Does the ENS backfill (2021-03 to 2024-03) give an extra validation fold, or improve the single-stage or stage 2 models? | 5 | The maintainer, who runs the fetch on a cloud machine | Deferred; tracked in [issue #959 (Extend ECMWF ENS training history)](https://github.com/openclimatefix/nged-substation-forecast/issues/959), and the 51-member fetch cannot run autonomously because it needs a cloud machine in `us-west-2` |

## Planned experiments

<!-- planned-table -->

## Proposals in detail

### P16: a longer ENS history

**The ENS backfill from Source Cooperative (2021-03 to 2024-03) could give an earlier validation
fold, and the full fetch cannot run autonomously.** [Issue #959 (Extend ECMWF ENS training
history)](https://github.com/openclimatefix/nged-substation-forecast/issues/959) describes the
fetch. The archive is in `us-west-2`. Fetching the control member alone is about 530 GB and 4 to 5
hours, and fetching all 51 members is about 30 TB, which needs a cloud machine in that region and
the maintainer's approval of the cost. The issue also ties autonomous use of the fetch to the
scorer-protection work in issue 958. Every experiment since 00003 scores the same year on the same
16 series, so choices tuned there overfit it, and an earlier fold is the remedy. For demand, a
longer training history has hurt so far, so the backfill is valuable first as validation data.
