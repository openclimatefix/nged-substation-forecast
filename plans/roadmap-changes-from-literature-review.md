# Roadmap changes the literature review surfaced

Working note, started 2026-08-22 and extended 2026-08-23, to be deleted with the rest of
`plans/` before this branch merges. It exists because the relevance review of
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

## Decisions the review round of 2026-08-23 left for you

Six Opus reviewers read the whole review — as a junior colleague, a skimming senior manager, a
cited author, an Ofgem analyst, an NGED planning engineer and a mechanical house-style auditor —
and six more fact-checked it chunk by chunk against the sources on disk. Most of what they found
was applied. These are the findings that were **not** applied, because each one commits the project
to something, and that is not a reviewer's call or Claude's.

**The consumer benefit is invisible, and the review does not commit to the metric it says the field
lacks.** Across 28,000 words the only money linking forecasting to bill-payers belongs to other
people: Northern Powergrid's £60m value case, Bernecker et al.'s euros, Angus et al.'s 10 to 12%
extra transformer capacity. The review spends four paragraphs arguing that what the literature is
missing is the price on a real network, and then makes no commitment to supply one. The Ofgem
reviewer called this the single thing most likely to read as research for its own sake. The
suggested fix was a tenth publication commitment: a relative-economic-value curve in the shape of
Richardson (2000), per substation, across the range of ratios between the cost of acting and the
loss avoided. That is a real commitment on NGED's behalf, so it needs your decision rather than
Claude's.

**Four of the eight problem sections never say what Flexpectation will do.** Problems 1, 2, 3 and 6
survey the literature and then state a plan. Problems 4, 5, 7 and 8 survey and stop. The NGED
planning engineer's verdict was that the plan is the half they are paying for. Adding those four
paragraphs means writing down project commitments that do not exist anywhere yet.

**The review never says what NGED gets, or when.** The pieces are scattered across five passages up
to 1,700 lines apart, and three facts appear nowhere in the file at all: the size of the
network-wide scale-up (about 2,500 series), the project end date (March 2028), and the form the
forecast arrives in — no dashboard, alert, API or control-room hand-off is described anywhere. The
suggested fix was a short "What Flexpectation delivers, and when" block after the voltage ladder.
Every sentence of it would be a delivery commitment.

**A funding-boundary sentence in the Open Climate Fix interest declaration.** The review declares
that Open Climate Fix is a partner in both NIA_UKPN0104 and Flexpectation. It does not say that
Flexpectation is not paying again for the capacity-inference work funded there. Duplicate funding
across two Network Innovation Allowance projects with a shared delivery partner is exactly what an
Ofgem analyst looks for, and the sentence would be cheap — but only you can confirm it is true.

**The third counter-finding ends by conceding the project might not be worth doing.** "A model
trained on none of NGED's data may match a model trained on all of it" closes on that possibility
without saying what NGED still gets if it holds. The other two counter-findings each end with a
test and a reason the result may not carry over.

**Two structural moves, both reversible and both left alone.** Moving "What GB networks have
already built" to sit before the eight problems, so the material the customer trusts most arrives
first rather than fourth from last; and moving the CIRED search out of "What this review excluded"
into "What we read", since searching 3,600 papers in full is the review's strongest completeness
claim and is currently filed under housekeeping. Both are pure reordering, no words lost. Left for
your manual read because a section move is the kind of change you would want to see before it
happens.

## From the leaderboard section, added 2026-08-23

**One question only you can answer: did the Presumed Open Data data science challenge exist, and
what was it?** A research agent recalled a Western Power Distribution or NGED challenge on
forecasting and battery scheduling at a real GB primary substation. If that is right, it is a
*substation-level* leaderboard run by this network's own predecessor, and it would change the new
section's claim that we found no published leaderboard at distribution-substation level — a claim
that is otherwise well evidenced and that the section now makes. I searched OpenAlex several ways,
tried Crossref-style bibliographic queries and guessed at Energy Networks Association portal
identifiers, and found no indexed account; the nearest hit was a Swansea paper that may be a
participant write-up rather than the challenge itself. Rather than publish an absence claim over
something you may simply know, this is flagged for your Monday read.

**`docs/roadmap/metrics-and-leaderboard.md` now has literature behind it, and cites none of it.**
The new review section draws on TS-Arena's pre-registration protocol, Energy-Arena's
deadline-defines-the-information-set rule, Doubleday et al.'s two-benchmark bracket, Blum and
Hardt's Ladder, and Messner et al.'s demonstration that a several-month window can rank the wrong
model first. The roadmap page independently arrived at several of the same positions — most
strikingly it already names "classic leaderboard overfitting" and tracks a final-test window under
issue #226 — so this is a matter of citing the support rather than changing the design. Inbound
only, as with the rest of step 2.

**Two mechanisms in the literature that the roadmap does not currently have.** Blum and Hardt's
Ladder — publish a new best only when it beats the standing best by more than a margin, and report
it rounded to that margin — is a cheap, implementable guard for a leaderboard one team queries
repeatedly, and it is stronger than the submission-rate caps that ImageNet and the M5 competition
used. And CAMEO freezes its baseline pipelines while their underlying databases keep updating, so
that data growth cannot be mistaken for method improvement; the analogue here is rerunning frozen
persistence and climatology baselines on every evaluation window.

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
