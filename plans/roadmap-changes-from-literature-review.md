# Decisions the literature review left for you

Working note, started 2026-08-22, to be deleted with the rest of `plans/` before this branch
merges. The stage-3 job it was written to track — linking the rest of `docs/` to
[`docs/background/energy-forecasting-review.md`](../docs/background/energy-forecasting-review.md) —
is done, and those items have been removed. What is left is the set of decisions that commit the
project to something, which is not Claude's call.

Two design additions surfaced in that work and are not made, for the same reason. Blum and Hardt's
Ladder — publish a new best only when it beats the standing best by more than a margin, and report
it rounded to that margin — is a cheap guard for a leaderboard one team queries repeatedly, and it
is stronger than the submission-rate caps that ImageNet and the M5 competition used. It is not in
the review, so it would arrive on the roadmap page as a new mechanism rather than as a citation.
CAMEO freezes its baseline pipelines while their underlying databases keep updating, so that data
growth cannot be mistaken for method improvement; the analogue here is rerunning frozen persistence
and climatology baselines on every evaluation window.

One tension is recorded but not written into the roadmap. `metrics-and-leaderboard.md` plans a
final-test window, a fixed hold-out reused once per champion candidate. TS-Arena avoids reusing any
fixed evaluation window at all, scoring every submission against outturn that did not exist when
the model was frozen. The page's own mitigations are real — the window is read only immediately
before promotion, never to choose between candidates — but saying *why* we accept the narrower
trade would be writing a rationale the project has not agreed.

Two gaps are reported rather than filled. `capacity-estimation.md` has no counterpart to the
fully-metered validation site that `disaggregation-evaluation.md` carries for challenges 7 and 8;
adding one would commit the project to finding or building such a site. And
`docs/ml_experimentation/mlops-approach.md` is titled "MLops" where the review writes "MLOps"
throughout.

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

## Open questions for the manual read of the review

Review-side rather than roadmap-side, collected here so they are in one place.

- **Fitted plant parameters are effective, not true, and the forecast score is the test.**
  [Saint-Drenan et al. (2015)](https://doi.org/10.1016/j.solener.2015.07.024), read in full this
  session, fit a photovoltaic plant's tilt, azimuth, and angular-loss coefficient to its own power
  history, and report that an azimuth 5° from the operator's records — a value they checked against
  an aerial photograph — simulated *better* than the true one, because the fit balances the
  systematic error of the physical model. `docs/techniques/differentiable-physics.md` now says so,
  and says that comparing a fitted tilt against a surveyed one is a diagnostic rather than a test,
  and that the priors exist to keep posterior spreads honest and regularise sites with little data
  rather than to pin a posterior to a survey. What that leaves open is identifiability: a loose
  prior on orientation can trade off against capacity, soiling, or the fleet mixture weights, so
  prior widths are a hyperparameter to tune on the forecast score, watching that the fit stays
  identifiable.
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
