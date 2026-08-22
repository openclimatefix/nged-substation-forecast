# Roadmap changes the literature review surfaced

Working note, written 2026-08-22, to be deleted with the rest of `plans/` before this branch
merges. It exists because the relevance review of
[`docs/background/energy-forecasting-review.md`](../docs/background/energy-forecasting-review.md)
turned up work on the roadmap side that we are deliberately not doing yet: the next step is a
manual read of the review itself.

Nothing here is a decision. Each item says what was found, where, and what was verified.

## Not yet done

**`docs/roadmap/cost-savings-metrics.md` cites nothing at all.** Verified: zero DOIs or links to
published work in the whole file. The review says the literature has priced forecast decisions in
energy volumes or spare capacity but never in money on a real distribution network, and this page
is our attempt at exactly that — so it is the page where the gap should be named and the two
relevant papers cited: Bernecker et al. (2025) and Richardson (2000)'s relative-economic-value
curve. This is the clearest single instance of the wider step-2 job below.

**Problem 3 has no owner for the fully-metered validation site.** The review lists six substitutes
for ground truth when estimating something nobody measures, one of which is a substation where
every feeder and embedded generator is metered for a period, used only as validation.
`docs/techniques/disaggregation-evaluation.md` carries it for problems 7 and 8. Verified:
`docs/roadmap/capacity-estimation.md` does not carry it for problem 3. Worth deciding whether such
a site exists anywhere in NGED's data, or whether it is something to ask NGED for.

**The two halves of the differentiable-physics strand are in opposite evidential positions, and the
roadmap does not distinguish them.** The review now says the demand-side half — aggregating the
thermal response of a few thousand buildings up to a substation, inside a probabilistic forecast —
has no published precedent we could find, while the generation-side half has several. Both
`capacity-estimation.md` (Candidate B) and `disaggregation.md` (the v2 engine) lean on the strand
without saying which half they are relying on. The generation side is the well-supported one, so
this is mostly a matter of saying so.

**Fold overlap, if we ever run more than one fold.** `cross-validation-folds.md` reasons about
non-overlapping walk-forward *between* folds; the section added this session covers the overlap
between reissued forecasts *within* a fold. Nobody has checked whether the two interact.

**Step 2 proper.** Every `docs/roadmap/` page making a claim the literature bears on should cite it
and point at the review. Note the direction: **inbound only.** The review links to nothing of ours
and must stay that way until the Milestone 2 report has gone to NGED, so the work happens entirely
on the roadmap side.

## Already done, so it does not need doing again

- `docs/roadmap/data-sources.md` gained an AIFS-ENS research row. The review said whether a
  machine-learned ensemble forecasts substation load better is something we can measure, and no
  roadmap page recorded that question.
- `docs/techniques/evaluation-metrics.md` named only ambient temperature and season as what moves a
  thermal rating; it now names wind and thermal mass too.
- `docs/ml_experimentation/cross-validation-folds.md` and
  `docs/techniques/disaggregation-evaluation.md` already link to the review.

## Open questions for the manual read of the review

Review-side rather than roadmap-side, collected here so they are in one place.

- **OCF's £30m imbalance-cost and 300,000-tonne CO₂ figures** are currently left out, on the
  grounds that they are unaudited marketing claims. Include them, and if so with what caveat?
- **Eight reference years follow the in-text citation rather than the journal issue date.** Body and
  reference list agree with each other, so nothing is internally inconsistent. Flip both, or leave?
- **Was Sheffield Solar involved in NIA_UKPN0104?** The claim was cut for lack of evidence and can
  be restored on your word.
- **A one-sentence version of "NGED gets three things regardless of how the research goes"** was
  removed. Restore?
- **The Austrian row in problem 1's table** (Ruhhütl et al.) is the one figure that fails the
  review's own transfer test — a mean absolute percentage error with no baseline named. It is kept,
  with note 5 saying so, because it is the only substation-level study from a comparable European
  network and the exclusions section cross-references it. Delete the row instead?
- **Hypothesis labels (`H2`, `T2.1`) are not cited** in the review, against the repo convention,
  because they mean nothing to an ENA reader. Confirmed 2026-08-22; recorded here in case the
  reasoning needs revisiting after submission.
