# Energy forecasting benchmarks and competitions: notes for the Flexpectation SOTA review

> **DEPRECATED — 22 August 2026. Do not cite this file, and do not trust it against the report.**
>
> These are working notes from the literature search. They have been superseded by
> [`milestone-2-report-SOTA-section.md`](milestone-2-report-SOTA-section.md), which is the
> deliverable and is the authoritative text. Where the two disagree, **the report is right and this
> file is wrong** — the report has since been through five adversarial fact-checks and this file has
> not. At least two errors are known to have propagated from here into an earlier draft of the
> report (the Huyghues-Beaufond entry, which had the paper's method backwards, and the Pinheiro
> entry, which had the wrong grounds for rejecting a gradient booster); both are corrected in place
> below, but others may remain.
>
> **Four sections are still live**, and they are what to harvest when `docs/roadmap/` is updated
> with the ideas this review surfaced:
>
> - "How Flexpectation could contribute to this literature"
> - "Conclusions for Flexpectation", and inside it "The recommendations, ranked and costed"
> - "Sources found by the 2026-08-22 search agents", and inside it "Leads not yet closed"
> - "Tool notes for the next search" — how to reach the databases and which hosts block us
>
> Everything else here is history: per-paper reading notes whose conclusions have already been
> folded into the report, and earlier coverage reviews that the report has since overtaken.
>
> This file is deliberately not being reconciled with the report. It will be deleted when the
> roadmap additions have been folded in.

**Written by Claude Opus 5 (Anthropic), prompted and directed by Jack, and revised through several
rounds of adversarial review in which the model was asked to attack its own accuracy, sourcing and
reasoning. Several substantive errors were caught and corrected that way; others may remain, so
treat every factual claim as checkable rather than checked.**

Research notes, August 2026. These are working notes to inform the NGED "state of the art"
document — not draft text for it.

## How to read these notes

Each paper gets four sections:

1. **Summary** — what the paper is and what it found.
2. **Setup in detail** — data, NWP, splits, models, tuning, compute. The things that decide
   whether a headline number means anything.
3. **Read against Flexpectation** — where the comparison is fair, where it isn't, and what it
   tells us about our own design choices.
4. **Does it measure what NGED care about?** — the decisive question. NGED's objective is
   minimising the cost of flexibility procurement and curtailment at an approximately fixed risk
   level, which makes the upper tail of the predictive distribution (p95, p99) far more important
   than MAE. Very few of these papers score anything close to that.

**Provenance and confidence.**

- **Read in full, August 2026 — ten papers.** Browell et al. (arXiv:2507.01579v2 and the Glasgow
  eprint); Kaas et al. (arXiv:2607.01966v1); Hertel et al. (arXiv:2607.15705v1); Shukla & Hong (IET
  Smart Grid, CC BY-NC-ND, supplied by Jack); Haben et al. (arXiv:2106.00006); Browell & Fasiolo
  (Glasgow accepted manuscript, eprints.gla.ac.uk/250372/1/250372.pdf — note this is the accepted
  version, not the published one, and may differ); Pinheiro, Madeira & Francisco
  (escholarship.org/content/qt0s14445q/qt0s14445q.pdf, CC BY); and the two supporting papers,
  Gilbert, Browell & Stephen (arXiv:2206.11745) and the FeederBW dataset paper, Treutlein et al.
  (arXiv:2602.03521). Also supplied by Jack and read in full, all Northern Powergrid / Faculty: the
  **Artificial Forecasting SIF Beta Annual Progress Report, March 2026** (52 pages), and the two
  **Alpha WP2-D2 technical reports of early 2024** — Scope Item 1.2, customer export and net demand at
  EHV-HV (82 slides), and Scope Item 2, active power at HV-LV (48 slides). Between them these
  corrected eight claims this document had been making about that project, including two of the five
  differentiators it had been treating as ours, and they are the reason §10a is now the longest
  section in the review. **Both are published openly** on the ENA Smarter Networks Portal under
  project NPG_SIF_006, downloadable without registration, so they are ordinary citable sources
  despite the stale "CONFIDENTIAL" stamp Faculty left on every slide (§10a).
- **Abstract, one section excerpt, and the live platform.** Energy-Arena (arXiv:2604.24705). The
  arXiv PDF returned no machine-readable text to my fetcher; §4 is built from the abstract, a
  search-index excerpt of the challenge-design section, and energy-arena.org itself, which answers
  most of what the paper would have. Anyone with the PDF should check §4 against it.
- **Still second-hand: GEFCom2017 alone.** It remains paywalled, and §5 is assembled from the
  abstract, the organisers' own instruction pages (which turn out to be a better source than the
  secondary literature, and are cited where used), competitors' method papers, and citing works.
  **Every factual claim in §5 should be treated as unverified whether or not it carries an inline
  tag** — the tags mark where I was conscious of guessing, which is not the same as marking
  everywhere I might be wrong.

**Three sources contradict themselves, so quote ranges rather than figures.**

*Kaas et al.* — four table-versus-prose disagreements, all in their §5: XGBoost+'s combined pinball
loss is 0.6219 kW in Table 3 and 0.6268 kW in prose; Chronos-Bolt's pinball is 0.8746 versus 0.8805;
its MAE is 4.11 versus 4.16; and its interval width and coverage are 8.652 kW / 62.11% in the table
against 8.839 kW / 62.38% in prose. This document quotes the table values throughout. None of the
four changes a ranking.

*Pinheiro et al.* — two of their figures for the share of substations beating a 24-hour-naive
forecast disagree outright, and the third arbitrates. Their §3.6 prose says 82.8% of DSO-owned and
66.0% of client-owned; the Figure 14 caption says 86.5% and 70.0%; the conclusion gives a single
pooled figure of 82.1%. Weighting by their own asset counts (61,689 DSO-owned, 22,974 client-owned),
the caption's pair pools to 82.0% and the prose's pair to 78.2% — so **the conclusion reconciles with
the caption, not the prose**, and the caption's percentages also match its own raw counts (53,391 and
16,083). The caption is therefore probably right. This document quotes the range 83–87% and 66–70%
across both versions, which is conservative; if you need one figure, use the caption's. Note this is
the one discrepancy here that moves a headline number, by three to four percentage points.

*Shukla & Hong* — the conclusion describes "9 of the 13 teams" building separate models per track and
then refers to "seven of these ten teams". Almost certainly a typo for nine; it changes nothing.

Between them these mean the four-significant-figure precision in this document is spurious and should
not be reproduced in the NGED text.

**One cross-paper disagreement, which is a different thing.** Kaas et al. describe the FeederBW
weather as short-term forecasts of one to three hours lead time; the FeederBW dataset paper describes
the same field as hourly ICON-D2 output that "does not include forecasts". Neither is contradicting
itself — they characterise the same data differently — and §2 sets out the likely reconciliation.

**Status of this document.** These are working research notes, around twenty times the length of
the two-to-three page deliverable they feed. They are not a draft, and the balance of attention here does not reflect
the balance the NGED document should have. The recommendations in the conclusions are ranked and
costed at the end of that section; treat everything above it as evidence rather than instruction.

**Known gap.** This is a selective set — nine numbered paper sections, two supporting papers, and a
tenth section covering three adjacent items admitted under a deliberately high bar — not a systematic
review, and the selection was mine. §10 ends with a log of what was searched and rejected, so at least
the exclusions are auditable; papers may still be missing for no better reason than that I did not
find them.
Anything the NGED document claims about what the literature does *not* contain should be phrased
against the Haben review's systematic count rather than against this selection.

**Two terms that recur below.** They are Latin: *ex ante* means "from before" and *ex post* means
"from after" (the fuller form *ex post facto*, "from a thing done afterwards", makes the sense
clearer). The vantage point is what differs — judging from before the event, or from after it. An
**ex-ante** forecast uses only information that genuinely existed at the moment the forecast was
issued — so if it needs tomorrow's temperature, it must use a *weather
forecast* of tomorrow's temperature, with whatever error that carries. An **ex-post** forecast is
evaluated using information that only became available later, most often the *observed* weather for
the period being forecast, or a reanalysis of it. Ex-post evaluation isolates the weather-to-power
model by removing weather-forecast error from the picture, which is useful for answering "how well
does this model convert weather into power", but it systematically flatters the numbers relative to
what the same model would achieve in operation. The gap widens with horizon, since weather forecasts
decay while observations do not. Several papers here are explicit that their results are ex-post and
therefore optimistic; the distinction is why the table below has a column for it, and it is the single
biggest reason published error figures are not a bar we should expect to clear.

The same words also describe *competitions*: an ex-ante competition requires entrants to submit
before the outcome is known, which removes any possibility of tuning against the answer, whether
accidental or deliberate. Our ERA5-only leaderboard category is deliberately ex-post in the weather
sense, for exactly the isolating reason above, while the live shadow deployment is ex-ante in both.

---

## Papers at a glance

| Paper | Task | Horizon | Probabilistic? | Tail / decision metric? | Live or ex-post? |
|---|---|---|---|---|---|
| Browell et al. 2025 (HEFTCom24) | Hybrid wind+solar generation, GB | Day-ahead | Yes, deciles 10–90% | Trading revenue (value, not tail) | Live, daily submissions |
| Kaas et al. 2026 | Net load, 200 LV feeders, DE | 4 days | Yes, p05/p95 | **Yes** — fuse-derived F1 on p95/p05 | Ex-post weather |
| Hertel et al. 2026 | Load at 3 grid levels, DE/PT | 4 days | No | No | Ex-post weather |
| Kleinebrahm et al. 2026 (Energy-Arena) | Platform: prices, load, wind, solar for AT and DE-LU | Day-ahead | Yes — point/RMSE, quantile/WIS, ensemble/CRPS | No — no threshold or cost metric | Live, ex-ante by design |
| Hong, Xie & Black 2019 (GEFCom2017) | Hierarchical load, ISO-NE | ~2–6 weeks (monthly rounds) | Yes, deciles 10–90% | No | Live, ex-ante |
| Shukla & Hong 2024 (BigDEAL22) | Peak load, 3 US LDCs | Rolling calendar months (final); 1 year ex-post (qualifying) | No | **Yes** — peak magnitude, timing (WMAE), shape (PSE) | Ex-post then ex-ante |
| Haben et al. 2021 | Review, LV | — | — | — | — |
| Browell & Fasiolo 2021 | Regional net load, GB | Day-ahead | Yes, QR body + conditional GPD tails | **Yes** — reserve volume at fixed risk | Ex-ante (HRES forecasts) |
| Pinheiro, Madeira & Francisco 2023 | Load, 96,989 PT secondary substations | Day-ahead | No | **Yes** — peak-aware adjusted *p*-norm error | Ex-ante (NWP, 7–8 h delay) |
| Gilbert, Browell & Stephen 2023 (in §7) | Load, 4-level hypothetical LV hierarchy, GB | Day-ahead | Yes, GAMLSS densities | **Yes** — CRPS restricted to peak periods | No weather at all |
| Treutlein et al. 2026 — FeederBW (in §2) | Dataset: 200 LV feeders, DE, 1-min | — | — | — | NWP analysis, not forecasts |

The last two are supporting papers rather than numbered sections, but both were read in full and
both carry findings the document leans on — Gilbert et al. supply the clearest peak-versus-average
evidence in the review, and the FeederBW paper documents the selection filters applied to the
population both KIT papers evaluate on.

Two entries in this table are easy to get wrong from the secondary literature. GEFCom2017's
qualifying match asked for **nine quantiles, not the 99 of GEFCom2014**, and its rounds were
**month-ahead**, making it the longest-horizon competition in this set rather than a short-term one.
And Browell & Fasiolo do evaluate a decision, in reserve volume at a fixed risk level, which is
easy to miss from the abstract.

## What "good" looks like, and why these numbers do not compare

The best reported result from each source, kept separate from the table above because each entry
needs its baseline and its normaliser attached to mean anything. Artificial Forecasting is included
even though it is not a paper (§10a), because it is the only entry that is a GB DNO operating at our
voltage levels.

| Source | Best method | Headline result, and what it beat | Weather | Read across to us? |
|---|---|---|---|---|
| Browell et al. 2025 (HEFTCom24) | Gradient boosted trees (team SVK) | **22.18 MWh** mean pinball against the supplied benchmark's **53.58**. Revenue £88.9m against a £105.2m perfect-decision ceiling | Ex-ante, live | No — wind and solar generation, not substation load |
| Kaas et al. 2026 | Chronos-2, zero-shot | Median **3.839 kW** MAE per feeder against **4.184 kW** for the best trained baseline (XGBoost with lags); p05/p95 pinball **0.5545** against 0.6219 kW; coverage 89.75% against a nominal 90%. But on their own decision metric Chronos-**Bolt** wins (F1 0.6745) with 62% coverage | Ex-post | Closest on asset size, 4-day horizon. Absolute kW is meaningless at our scale |
| Hertel et al. 2026 | Encoder-decoder Transformer | Improvement over a naive baseline of **59.6%** on the TSO control area, **42.3%** on LV feeders, **23.3%** on individual clients. Only **6.6%** over the best non-Transformer on FeederBW | Ex-post | Yes, for the shape rather than the level: headroom shrinks as you disaggregate, and our series sit at the feeder-to-primary end |
| Kleinebrahm et al. 2026 (Energy-Arena) | No single winner — a rolling leaderboard | RMSE, Winkler interval score and CRPS, updated continuously at energy-arena.org | Ex-ante by design | A method for running comparison, not a number |
| Hong, Xie & Black 2019 (GEFCom2017) | Kanda & Quintana (final match) | Quantile score relative to a benchmark. **No score appears in anything we have read** — §5 is second-hand throughout | Ex-ante | Do not cite a number from here |
| Shukla & Hong 2024 (BigDEAL22) | Differs by track; nine of thirteen finalists built separate models per track | Winning scores are not given in the paper as read. The transferable result is that magnitude ranks correlate least with timing and shape ranks | Ex-post, then ex-ante | Metric design, not a number |
| Haben et al. 2021 | Review | The structural number: of **221** LV forecasting papers, **3** used weather *forecasts* and **none** used ensembles | — | The gap statement the rest of this review rests on |
| Browell & Fasiolo 2021 | GAM quantile regression with conditional GPD tails | Conditional GPD needs **24.6%** less upward reserve than a static GPD and **16.5–18.0%** less than a naive historical-error method; downward reserve only **0.8–10.8%** better than static GPD. Adding wind and irradiance features cut pinball **40%** overall — 10% in Greater London, 60% in North Scotland | Ex-ante (HRES) | The decision metric is the model for ours; the series are far larger (GSP regions) |
| Pinheiro et al. 2023 | GAM, inside a regime-weighted ensemble | System level: **42–47%** below a Tao Hong GLM benchmark on MAE, MAPE, RMSE and NRMSE; the regime ensemble cut RMSE from **203 MW to 154 MW**. Substation level, far more sober: beats a 24-hour-naive forecast on only **83–87%** of DSO-owned and **66–70%** of client-owned sites. Adding 24-hour and 1-week lags cut MAPE **4.09% → 2.53%** | Ex-ante (NWP, 7–8 h delay) | **The closest analogue in the review** — day-ahead, 96,989 secondary substations, a production system |
| Gilbert et al. 2023 | Forecast fusion across a four-level hierarchy | CRPS gain over the advanced model: **0.0–0.4% averaged over all periods**, but **5.7%** primary, **9.0%** secondary, **8.2%** feeder and **6.0%** household *restricted to peaks*. At household level during peaks both their models are **worse** than a time-of-day KDE by 1–5% | None at all | The clearest peak-versus-average evidence here; no weather, so read it as a floor |
| Artificial Forecasting (§10a) | EHV-HV: stepwise Bayesian linear regression. HV-LV: TCN | **Beta, EHV-HV:** ~**8%** lower MAE of utilisation rate than growth, persistence and a rolling four-week baseline; **83%** of top-10% demand values inside p5–p95; better at **8/8** near-capacity substations. **Alpha, HV-LV:** TCN at **11.38%** Peak MAPE and **11.47%** daily-peak MAPE (**2.77%** and **3.63%** normalised by transformer rating) against a four-week-average baseline's 12.46% and 11.39% | EHV-HV ex-ante, three point locations. **HV-LV: none at all** | **The benchmark that matters** — a GB DNO, our voltage levels, in live operation |

**Three things make a number look good here without any modelling skill behind it**, and all three
are live in this table. **Aggregation**: Hertel et al. measure the same models beating the same naive
baseline by 59.6% at TSO level and 23.3% at individual clients, so a headline percentage says more
about where it was measured than about the model. **Ex-post weather**: Kaas et al. and Hertel et al.
both feed models the weather that actually happened, which removes the NWP error our 3–10 day band
is dominated by. **Averaging over all periods**: Gilbert et al.'s fusion looks worthless at 0.0–0.4%
CRPS until you restrict to peaks and it becomes 5.7–9.0%.

**So only two kinds of number transfer.** Ratios against a stated baseline on a stated population —
which is why Pinheiro et al.'s *fraction of substations beating persistence* (66–70% for
client-owned) is the single most useful figure in the table, and why it is recommendation 3. And
errors normalised by something physical rather than by outturn — which is why Artificial
Forecasting's 2.77% Peak MAPE against transformer rating is the number our own capacity-normalised
results should be reported next to. Neither of those is an absolute error in kW or MW, and none of
the absolute figures above should be quoted as a target for us.

---

## 1. Browell, van der Meer, Kälvegren, Haglund, Simioni, Bessa & Wang (2025) — HEFTCom 2024

*The hybrid renewable energy forecasting and trading competition 2024.* International Journal of
Forecasting 42(3), 709–723. Preprint: arXiv:2507.01579. Data: Zenodo 10.5281/zenodo.13950764.
Analysis code: github.com/jbrowell/HEFTcom24-Analysis.

### Summary

The competition asked participants to forecast and trade the output of a roughly 3.6 GW hybrid
portfolio in Great Britain over three months in 2024, submitting genuine day-ahead forecasts and
market bids every day. The portfolio combined the Hornsea 1 wind farm with the aggregate solar
capacity of East England, forecasts were the nine deciles for each half-hour due by 09:20 UTC, and
scoring was by pinball loss. Of more than 170 registered teams, 66 submitted at least once and 24
completed. Team SVK won all three tracks. The headline conclusion is that gradient boosted trees
remain competitive for day-ahead wind and solar forecasting, that other methods also produced
strong results, and that performance in all cases depended heavily on implementation.

The most discussed finding is not the winner but the spread. The methods table shows that gradient
boosting, model combination, feature selection and hyper-parameter tuning were used by most teams,
top and bottom alike. What the authors identify as separating the leaders is that they selected
features using training and validation experiments or feature importance, while lower-ranked teams
selected them from exploratory data analysis. The spread is wide: SVK won on 22.18 MWh average
pinball, the supplied benchmark scored 53.58, and completing teams ranged across that interval.

**This is weaker evidence for the experiment-throughput thesis than it first appears, and the
document should not lean on it hard.** It is an observational correlation across roughly 24
self-selected teams, with confounders the paper itself names: several top teams were experienced
operational forecasters, team sizes varied from one to four, and SVK's largest single reported
gain — 8% of pinball — came from buying access to an extra NWP source rather than from running more
experiments. More awkwardly, the winning method cuts against the reading. SVK tuned exactly one
hyper-parameter, the number of boosting iterations, and left everything else at defaults. What won
was a well-chosen *structure* — per-source models, quantile stacking, availability clipping,
missing-input fallback — not an extensive search. A fair summary is that HEFTCom shows measured
iteration beating intuition on one specific decision (feature selection), while also showing that
structural choices dominated tuning. Both halves belong in the NGED document if either does.

The competition also produced the first detailed exploration of the relationship between forecast
skill and forecast value. A regression of revenue on pinball score gives a gradient of about
−£0.18m per MWh of pinball. Teams that bid strategically rather than bidding their median added
over £500,000, which by that regression is equivalent to a forecast improvement of more than 10%.
A perfect deterministic forecast would have earned about £3.2m more than the winner, while
"perfect decisions" would have earned £105.2m against the winner's £88.9m. The tempting summary —
that better decisions were worth roughly five times better forecasts — overstates it: their
perfect-decision counterfactual assumes exact foreknowledge of the day-ahead-to-imbalance price
spread, which is itself a forecasting problem rather than decision skill. The defensible claim is
narrower, and is the one the authors make: effectiveness of decision-making under forecast
uncertainty was of similar importance to forecast skill.

### Setup in detail

The target was the sum of Hornsea 1 (1.2 GW offshore wind, metered via Elexon) and the estimated
aggregate solar output of East England (~2.4 GW, from Sheffield Solar PV Live, because individual
solar units fall below the Elexon reporting threshold). Weather inputs were three years of historic
and operational forecasts from **DWD's ICON-EU and NCEP's GFS**, both hourly with four runs per day,
supplied by rebase.energy as gridded fields around the wind farm and around East England, plus point
forecasts at GB population centres for price modelling. Notably, the use of gridded rather than
point NWP is called out as standard practice in industry but rare in competitions.

There was no cross-validation in the usual sense, because there was nothing to cross-validate
against: the evaluation period was the live future. Teams had a two-month testing phase from
November 2023, the competition ran 20 February to 19 May 2024, and scoring used data that did not
exist when the models were built. Teams could use any external data they liked, which several did.
SVK reported an 8% pinball improvement from adding MET Norway's MEPS ensemble to the supplied GFS
and ICON-EU forecasts, validated on 2023. Two of the top five used no extra weather data at all.

The winning method is worth recording in full because it is a good reference architecture. CatBoost
models were fitted per NWP source and separately for wind and solar, using a MultiQuantile loss
targeting the nine required quantiles, with features being the NWP grid points raw, lagged and
differenced, plus calendar features. Only the number of boosting iterations was tuned; everything
else was left at defaults and unimportant features were dropped after initial testing. Quantiles
were clipped to available capacity using REMIT outage messages. The 27 predicted quantiles from the
three CatBoost models were then combined by a linear quantile regression meta-model per target
quantile, and wind and solar quantiles were added with a correlation adjustment that testing showed
was worth very little. When an NWP source was missing, its model's quantiles were filled from the
available models.

Nine of the top ten forecast wind and solar separately before combining. The most distinctive
non-tree entry was team Rnt, who used embeddings from in-house AI weather models — extended from
Andrychowicz et al. (2023) to include irradiance and day-ahead lead times, with station
observations, radar and satellite imagery as inputs — feeding downstream neural networks. They
finished 3rd in forecasting and 2nd in trading.

### Read against Flexpectation

The NWP-fallback behaviour in the winning solution is direct support for our approach to input
failure, and it is a cleaner design than ours. SVK's structure gives graceful degradation for free,
because each NWP source has its own model and a missing source simply drops out of the stack. Our
current plan relies on training a single model with outage-shaped examples so that it survives a
total NWP failure. Both are defensible, but SVK's is worth registering as an arm: a per-source
model plus a quantile-combining meta-model is a structural answer to missingness rather than a
learned one, and it also gives per-source attribution for free. The counter-argument in our
setting is that we have one NWP source (ECMWF ENS) rather than three, so the ensemble members are
the axis along which this would generalise — which pushes towards the "ensemble statistics as
features versus member-by-member rows" fork already on the roadmap.

The Hornsea cable fault is the closest analogue in the literature to our switching events, and it
behaved exactly as our two-stage design predicts. A step change in effective capacity appeared
partway through the record; teams that forecast the combined total directly struggled to adapt,
while teams that could post-process a single component adapted quickly; and the supplied benchmark,
which ignored it, collapsed. That is an argument for the disaggregation work and for the
capacity-estimation asset, and it is a citable one rather than a hypothetical. It also shows the
value of an exogenous availability signal: SVK clipped to REMIT. We have no REMIT equivalent for
substation switching, which is precisely why the switching detector exists — worth saying plainly
in the document, because it turns an apparent weakness into a statement about why our problem is
harder than the competition's.

The mismatches are substantial and should be stated rather than glossed. HEFTCom is generation
rather than net demand, so there is no behavioural or calendar component and no unmetered
generation to disentangle. It is a single 3.6 GW portfolio rather than 32 series scaling to 2,500,
so nothing in it speaks to global-versus-per-series modelling or to transfer across data-poor
sites. And it is day-ahead. Browell's own closing recommendation is that intraday and medium-term
horizons of days to weeks ahead have had relatively little attention in past competitions, which is
exactly the 3–10 day band our users act on.

### Does it measure what NGED care about?

Only partially, and the gap is instructive. The forecasting track scores mean pinball over deciles
10% to 90% — the quantile grid stops at 90%, so **the competition says nothing at all about p95 or
p99 behaviour**. Worse, the paper contains a direct warning about reading tail quality off an
aggregate pinball score: team UI BUD achieved a competitive average pinball with substantial bias,
and the authors note that calibration and sharpness can be traded off if the objective is purely
pinball minimisation. Their reliability diagrams, split into daytime and overnight, are the thing
doing the real diagnostic work, and they are worth copying in spirit — our horizon-sliced metrics
should be paired with per-slice reliability, not just per-slice NMAE.

The trading track is a value metric, but it is not our value metric. Its loss is roughly symmetric
around the optimal bid, whereas flexibility procurement is asymmetric by construction: buying too
much costs the availability price on the excess, while buying too little risks a breach. HEFTCom's
"same task, priced in pounds" framing is the right idea for us, but the arithmetic does not
transfer. What does transfer is the discipline of reporting both the statistical score and the
decision score, and the finding that rank correlation between them is high but imperfect.

---

## 2. Kaas, Treutlein, Gerber, Neumann, Phatthanakhuha, Resch, Mikut & Hagenmeyer (2026)

*Probabilistic Low-Voltage Peak Load Forecasting with Time Series Foundation Models Evaluated on
Application-Oriented Metrics.* arXiv:2607.01966, 2 July 2026. Authors from Netze BW (a German DSO)
and KIT.

### Summary

This is the closest paper in the literature to what Flexpectation is doing, and it should probably
be the centrepiece of the LV section. The authors produce four-day-ahead probabilistic net-load
forecasts for 200 real low-voltage feeders, comparing three time-series foundation models
(Chronos-Bolt, Chronos-2, TabPFN-TS) against six models trained from scratch, and they invent an
application-oriented metric derived from the time-current curve of the protecting fuse. Chronos-2
came out best on the conventional metrics, and the weather-covariate ablation showed that
foundation models degrade materially when weather is withheld.

On the conventional metrics, Chronos-2 achieved a median MAE across feeders of 3.839 kW against
4.184 kW for the best trained baseline (XGBoost with lags), a combined p05/p95 pinball of 0.5545 kW
against 0.6219 kW, and an empirical coverage of 89.75% against a nominal 90% — the closest to
calibrated of any model tested. All three foundation models beat XGBoost+ on the point metrics.
Removing weather covariates cost roughly 1 kW of MAE across all three, and widened intervals by
1.5–2 kW, with the damage concentrated at midday where PV feed-in lives.

The result in their Table 4 is the most interesting thing in the paper, and it needs reading
carefully because the obvious interpretation is wrong. On the **quantile variant** of the
application-oriented F1 — consumer side, scored at each model's p95 — the best model is
Chronos-Bolt at 0.6745, and second is WeekNaive at 0.6283, ahead of Chronos-2 at 0.5515. On the
producer side, scored at p05, Chronos-Bolt again leads at 0.7445 with XGBoost+ second at 0.6478 and
Chronos-2 at 0.5962. Chronos-Bolt has a worse pinball loss than every trained baseline except the
naive one, and WeekNaive has the worst pinball of anything tested.

**The tempting reading — that the decision metric ranks models differently from the distributional
one, so the field is measuring the wrong thing — does not survive the calibration column.**
Chronos-Bolt's empirical coverage is 62.1% against a nominal 90%, and WeekNaive's is 58.1%. They
are the two most under-dispersed models in the study by a wide margin. Their nominal p95 therefore
sits far closer to their median than a calibrated p95 would, so it crosses the overload threshold
less often, generates fewer false positives, and wins a metric that weights false positives and
false negatives equally. The bad pinball and the good F1 are the same phenomenon seen twice, not
two independent facts. Note also that the inversion appears only in the quantile variant: on the
point-estimate F1, Chronos-2 still wins both consumer and producer sides, so this is a property of
one of their two scoring variants rather than of decision metrics in general.

**Read correctly, the finding is more useful to us, not less.** It shows that a threshold metric
evaluated at a fixed *nominal* quantile, with symmetric error costs and no risk calibration, can be
won by being badly under-dispersed. That is a specific, demonstrable failure mode of the most
decision-oriented metric in the recent LV literature — and it is exactly the failure our
cost-savings design avoids, because we calibrate each model's procurement quantile τ to a common
unmet fraction before comparing spend, and report the realised unmet fraction beside every cost. So
the citable claim is not "decision metrics rank differently"; it is "scoring the decision is
necessary but not sufficient, because an uncalibrated threshold metric rewards over-confidence, and
equalising risk before comparing cost is what fixes it". That is a sharper argument for our design
and it is defensible against someone who has read the paper.

### Setup in detail

The dataset is **FeederBW** (Treutlein et al. 2026, arXiv:2602.03521), published as open data: 200
LV feeders in Baden-Württemberg, minutely resolution, 1 April 2023 to 31 March 2025, with feeder
metadata and weather. They resampled to 15 minutes by mean aggregation and used three-phase active
power as the target. Average feeder connects 36.2 housing units and 2.5 commercial or industrial
units. Both consumption and generation are present, with feed-in reaching −50 kW on some feeders.

**Weather is the weak point and they say so — and the two papers describe it differently.** Kaas et
al. state the dataset carries short-term forecasts of one to three hours lead time. The FeederBW
dataset paper describes the same field as hourly output of DWD's **ICON-D2** model, averaged over the
postcode area of each feeder's substation, and says plainly that it **"does not include forecasts"**.
The reconciliation is presumably that ICON-D2 output at one-to-three-hour lead is being used as an
analysis proxy rather than as a forecast anyone could have had four days ahead. Either way the
consequence is the same and Kaas et al. draw it themselves: this weather is not available at forecast
issue time, so they classify their own results as *ex-post*, and their absolute numbers are optimistic
relative to anything we will produce with ECMWF ENS at a genuine 3–10 day lead. Note also that the
weather is postcode-averaged rather than gridded, so nothing in this benchmark speaks to the value of
spatial weather features.

**There is no cross-validation.** The split is a single expanding-window-style arrangement:
feeders 1–160 are trained on 1 April 2023 to 31 March 2024, and everything is evaluated on 1 April
2024 to 31 March 2025. Feeders 161–200 have their first year withheld entirely, giving a
never-trained-on subset. Forecasts are issued every four days for the next four days, so horizon
and stride are both four days — meaning each evaluation point appears once, and there are only
about 91 forecast origins per feeder per year. They aggregate by computing each metric per feeder
and then taking the median across the 200 feeders, which is more robust than a mean but hides the
spread.

The nine models are: WeekNaive (mean and min/max of the same weekday over the last four weeks);
XGBoost trained on metadata and covariates only, with no lagged power at all; XGBoost+ with four
days (384 steps) of context, forecasting autoregressively with a 12-step three-hour gap masked out
because that improved quality; a PyTorch MLP; TFT and PatchTST from AutoGluon; and the three
foundation models with a 1344-step (two-week) context. Chronos-Bolt gets covariates through a
CatBoost covariate regressor rather than natively. TSFM selection is documented properly — they
tabulate 26 foundation models against open weights, probabilistic output, covariate support and
fine-tuning code, and explain why Moirai and Moirai-MoE were dropped after pre-studies. No
fine-tuning was performed on any TSFM.

The application-oriented metric works like this. Choose a window and a threshold inspired by the
fuse's melting time and continuous current. Compute a rolling mean over that window for both the
observation and the forecast, and classify every timestamp as TP/FP/TN/FN by whether each exceeds
the threshold. Build the confusion matrix and report precision, recall and F1. Their reading is
explicit and directly comparable to our unmet-fraction framing: `1 − precision` is the risk of
overreacting, and `1 − recall` is the risk of taking insufficient action; for long-term planning
horizons those become the risk of over- and under-investment. In their experiments the window is
one hour and the threshold is 40% of each feeder's own absolute maximum, evaluated separately for
consumer peaks (using p95) and producer peaks (using p05). On average this selects 5,470 consumer
peaks and 691 producer peaks per feeder out of 34,944 timestamps.

Runtimes are reported and are relevant to our laptop-scale constraint: XGBoost+ took over four
hours to train and over two hours to infer across 200 feeders, TabPFN-TS took over eight hours to
infer, and Chronos-2 managed inference in under 25 minutes with no training at all.

### Read against Flexpectation

**Their XGBoost is not our XGBoost, and the gap is almost exactly our roadmap.** XGBoost+ receives
lagged power, weather, time and metadata covariates, and that is all. There are no linearised
physics features — no clear-sky index, no PV power proxy, no wind power curve — despite the target
being net load on feeders with heavy PV feed-in. There are no monotone constraints. There is no
effective-temperature or degree-day feature. There are no holiday features beyond a binary flag.
There is no per-feeder-type feature list. Every one of those is a Tier-1 or Tier-2 item on our
XGBoost improvements page. The paper's own covariate ablation shows that the models that exploit
irradiance well are the ones that win, which is an argument that the physics features are where the
XGBoost headroom is — and it means their "foundation models beat XGBoost" headline is really
"foundation models beat a lightly-featurised XGBoost".

Two of their design choices deserve to be arms in our leaderboard rather than assumptions.
XGBoost+ forecasts **autoregressively** and found that masking the most recent three hours of
context improved quality, which is a different answer to the same problem our `_nullify_leaky_lags`
convention addresses; it is worth knowing that someone independently found a gap helped. And their
plain XGBoost variant, trained on **metadata only with no time-series context at all**, is a
genuinely useful ablation we do not currently have — it isolates how much of the forecast is "what
kind of feeder is this, and what is the weather" versus "what has this feeder been doing lately".
Given our per-series boosters and the global-model ambition, that decomposition is worth measuring.

**Switching events are not modelled — but, importantly, they were partly filtered out of the
dataset before anyone modelled anything.** The FeederBW paper (Treutlein et al., arXiv:2602.03521,
read in full) is explicit: over the two-year period, switching events altered the grid topology,
often due to construction work; these events are not documented in the data and normally lead to
concept drifts; obvious cases were removed, and some may still be present. The dataset's selection
criteria include an individual data check that blacklists feeders with uncorrectable data quality
issues **"such as topology changes"**, alongside filters for radial (non-meshed) topology, ≥90% data
completeness, and minimum capacity utilisation. The authors acknowledge that these criteria introduce
bias while defending them on data-quality grounds.

That is a sharper and more useful point than "nobody handles switching". **Both KIT papers are
evaluated on a population from which the hardest cases have been partly removed.** Feeders that were
reconfigured are exactly the ones our detector exists for, and they are precisely what the benchmark
excludes. So the published model rankings — including the one where a foundation model beats a
featurised booster — are rankings on cleaned, radial, well-instrumented feeders. Neither paper
overclaims about this; the filtering is documented in the dataset paper rather than the modelling
papers, which is why it is easy to miss. For our document, the claim to make is that the closest
published LV benchmark **selects against** the failure mode that dominates our residuals, not that it
overlooks it.

Their horizon overlaps ours only at the start. Four days sits inside our 3–10 day band but misses
the part that matters most, and their Figure 3 shows error growing visibly from day one to day
four even with ex-post weather. At a real 10-day lead with ENS, the weather uncertainty term
dominates in a way this paper cannot see. Their finding that Chronos-2's accuracy improves with
context length up to about a month, with little gain beyond, is nonetheless worth noting for the
init-time-anchored features work.

### Does it measure what NGED care about?

Closer than anything else I found, and worth citing as precedent for our cost-savings metrics
design. The fuse-derived F1 is structurally the same object as our flexibility-procurement metric
minus the prices: a threshold, a forecast quantile, and a confusion matrix whose two error types
map onto over-procurement and unmet need. Their explicit gloss — over-reaction risk versus
insufficient-action risk — is the sentence we could quote when explaining why the leaderboard
carries a pounds column.

Four limitations to note, the last of which is the important one. They use p95 and p05 only, never
p99, so the deepest part of the tail is untouched. Their threshold is 40% of each feeder's own
maximum, chosen because real fuse ratings are exceeded too rarely to evaluate on — the same
compromise we make with `historical_p99`, and useful cover for that choice when NGED ask why the
limit is synthetic. Their metric is unpriced, so it cannot express the asymmetry that availability
payments are cheap relative to a breach; a single F1 weights the two error types equally, which is
wrong for a risk-averse DSO.

And, as the summary above sets out, **their models are scored at a fixed nominal quantile without
being calibrated to a common realised risk**, which lets an under-dispersed model win by generating
fewer alarms rather than by forecasting better. Our τ-calibration step is the fix, so this is the
clearest place in the whole review where we can point at a published method, name a concrete defect,
and show that our design already addresses it. It is worth being careful about tone: this is a good
paper doing something nobody else has done, and the defect only becomes visible because they
reported empirical coverage alongside the F1. The claim to make is that we extend their approach,
not that we correct it.

---

## 3. Hertel, Pütz, Kolar, Schäfer, Mikut & Hagenmeyer (2026)

*A Benchmark for Electrical Load Forecasting Across Grid Levels: Time-Series Transformers Outperform
Established Methods.* arXiv:2607.15705, 17 July 2026. Code and preprocessed data:
github.com/KIT-IAI/load-forecasting-benchmark.

### Summary

A companion paper to Kaas et al. from the same institute, this one is deterministic and broader:
ten forecasting methods evaluated on three datasets representing a TSO control area, LV grid
feeders, and individual end consumers. The headline is that Transformer-based approaches beat
established methods consistently, reducing forecast error by 6.6% to 10.7%, but that a **standard**
encoder-decoder Transformer beats their purpose-built flexible architecture — so the architectural
modifications proposed across the time-series Transformer literature (patching, CNN layers, LSTM
layers, max pooling, sparse attention) did not help.

They also evaluate Chronos-2 zero-shot. It is competitive at the feeder and client levels but has
large errors on the TSO data, concentrated around holidays: it fails to bring load back to normal
after the winter break, misreads Good Friday, and does not anticipate long weekends or the
pre-Christmas week. Adding binary features for long weekends and school holidays improved it but
did not close the gap to the trained models. Chronos-2 is also strongest at short horizons — best
of everything for the first six hours on the feeder data — and loses to trained models as the
horizon lengthens.

The ablations are the most transferable part. Covariates help on all three datasets, with the
biggest gains on the two most affected by behind-the-meter PV. Monthly retraining beats a single
static annual training on both datasets where it was tested. And Chronos-2's accuracy rises with
context length: one week of context barely beats the baseline, one month is close to optimal, and
the full 8192 steps adds little more.

**One finding cuts against the project's premises and should not be buried.** The improvement of
the best model over the naive baseline is 59.6% on the TSO control area, 42.3% on LV feeders, and
23.3% on individual clients — the authors' own gloss being that it is easier to outperform a simple
approach on highly aggregated data than on volatile feeder and client data. Read as a statement
about headroom, that says the more disaggregated the series, the less sophisticated modelling buys
you over a well-chosen naive method. Our target sits at the feeder-to-primary end of that scale.
It does not follow that modelling is not worth doing — 42% is a large number in absolute terms —
but it does mean the marginal return per experiment is likely to be smaller than at system level,
and it makes the persistence and climatology baselines on our metrics roadmap load-bearing rather
than decorative.

**The runtime numbers, in absolute terms.** Measured on one NVIDIA A100 (40 GB) with 19 Xeon cores
and 128 GB RAM, forecasting 96 hourly steps. Their reported "inference time" covers one forecast for
*all* series in a dataset, so the per-series column is arithmetic on their figures, not theirs.

| Model | 200 feeders, all | Per series | 287 clients, all | Per series | 1 series (TSO) |
|---|---|---|---|---|---|
| Chronos-2 | 11,160 ms | **55.8 ms** | 16,016 ms | **55.8 ms** | 78.3 ms |
| Transformer | 72.5 ms | 0.36 ms | 60.6 ms | 0.21 ms | 6.4 ms |
| TFT | 46.1 ms | 0.23 ms | 75.2 ms | 0.26 ms | 10.8 ms |
| CNN | 28.1 ms | 0.14 ms | 35.8 ms | 0.12 ms | 5.4 ms |
| LSTM | 74.6 ms | 0.37 ms | 55.2 ms | 0.19 ms | 15.3 ms |
| LightGBM | 269.7 ms | 1.35 ms | 239.3 ms | 0.83 ms | 1.2 ms |
| Persistence | 266.3 ms | 1.33 ms | 373.4 ms | 1.30 ms | 1.4 ms |

Chronos-2 costs **55.8 ms per series per forecast** on both multi-series datasets — the same figure to
three significant figures, which suggests it is throughput-bound at its batch size of 32 and scaling
linearly in series count. That is roughly **150–270× slower per series** than the trained neural
models and about 40× slower than LightGBM. Kaas et al. corroborate the order of magnitude on much
weaker hardware: a Tesla T4, 200 feeders, 91 forecast origins of 384 quarter-hourly steps gives
18,200 series-forecasts, so their "under 25 minutes" for Chronos-2 works out at about **82 ms per
series-forecast**, TabPFN-TS's "over eight hours" at about **1.6 s**, autoregressive XGBoost+ at about
**400 ms**, and plain XGBoost at about **49 ms**.

**Read their annualised column carefully — it measures different things for different models.** They
annualise across twelve monthly retrainings *and 8,760 hourly forecasts*: LightGBM costs 121 hours,
the Transformer 19, Chronos-2 27, the persistence baseline 39 minutes. But for the trained models
that total is almost entirely training — the Transformer spends 18 h 53 m training and only about 10
minutes on a whole year of inference — whereas Chronos-2's 27 hours is all inference and no training.
Comparing "27 hours" against a per-forecast millisecond figure is meaningless, and on the
single-series TSO dataset Chronos-2 is the *fastest* model overall precisely because it needs no
training.

**What this means at our scale — extrapolation, clearly labelled.** Our cadence is daily, not hourly,
since ENS gives one run at midnight. At 55.8 ms per series per forecast: 2,500 series with the control
member only is about **2.3 minutes** per daily run, which is nothing. The same 2,500 series times 51
ensemble members is about **2 hours**. Our horizon is 672 half-hourly steps against their 96, so if
cost scales with output length the full-ensemble case becomes roughly **14 hours a day** — a real
constraint on one machine. Two of those three multipliers are ours to choose: collapsing the member
axis into ensemble statistics, already a fork on the roadmap, takes the full-ensemble case back to
about 2.3 minutes. The scaling-with-horizon assumption is untested and should be measured rather than
believed.

### Setup in detail

The three datasets are TransnetBW (one TSO control-area series, 2015–2025, from the ENTSO-E
transparency platform, enriched with **ERA5** reanalysis for temperature, irradiance, precipitation
and 10 m wind speed plus a holiday flag); FeederBW (the same 200 LV feeders as Kaas et al., hourly,
with irradiance, temperature, precipitation and a holiday flag); and Electricity-287, a cleaned
287-series subset of the UCI ElectricityLoadDiagrams set from Portugal, combined with ERA5 — they
believe they are the first to attach weather to that dataset.

The task is 96-hour-ahead hourly forecasting with a **stride of one hour**, which is far denser
than Kaas et al.'s four-day stride and gives much more evaluation mass. Deep learning models use a
global training strategy across series; LightGBM is trained locally per series with one model per
horizon step. Hyper-parameters for every model, including the comparison methods, are optimised
with Bayesian optimisation in Weights & Biases over 100 configurations, which is more even-handed
than most benchmark papers manage. Statistical comparison uses one-sided Diebold–Mariano tests per
series, reported as the fraction of series on which each model significantly beats each other — a
practice worth copying for our leaderboard, since it distinguishes "wins on average" from "wins
consistently".

Retraining is monthly on the two datasets with a year-long test period, using the previous month
as validation and everything before as training. There is no k-fold cross-validation; the design
is forward-chained, which is correct for this problem.

The DM tests reveal something the headline hides. On FeederBW, the Transformer significantly beats
Chronos-2 on 46% of series while Chronos-2 beats it on 20%, with no significant difference on the
rest. On Electricity-287, Chronos-2 has the better aggregate nMAE but only beats the Transformer on
14% of series, losing on 39% — it wins the average by being good on a few difficult, high-error
series. Aggregate metrics can be carried by a minority of series, which matters for us given the
heterogeneity of the 32 trial-area series and the 2,500 to come.

### Read against Flexpectation

Their honesty about optimism is the most useful thing here, and it is quotable: they state that
their achieved errors might be unrealistically low for real applications, because real-time
measurements may only arrive after hours or days but are assumed immediately available, and because
they used weather data not available at forecast time. Our six-hourly NGED meter readings and our
publication-time availability cut are addressing exactly the two problems they name and set aside.
That is a good, non-defensive way to explain why our absolute numbers will look worse than
published ones while being more honest.

Two findings argue for things already on our roadmap. Monthly retraining beating static training
is the same drift argument as our recency sample weights and init-time-anchored features, measured
on 200 feeders. And the context-length result — one month of history capturing nearly all the
value — is a useful prior for how far back the init-time anchors need to reach.

One finding argues against reaching for a Transformer too soon in our setting. Their Transformer
wins by 6.6% on FeederBW over the best non-Transformer, which is real but not transformative, and
it costs monthly global retraining with a 100-configuration HPO behind it. Against that, our
XGBoost improvements page lists a dozen unexploited feature-engineering wins, several of which are
config-level. The ordering implied is clear: exhaust the cheap featurisation first, since their
LightGBM — like Kaas et al.'s XGBoost — carries none of it.

The heaviest caveat is that this paper is **entirely deterministic**. It optimises and scores MAE
and nMAE, mentions probabilistic forecasting only in the discussion as future work, and notes that
extending their deep learning models to quantiles would just mean swapping the output layer and
training with pinball. So it can tell us about architectures, covariates, context and retraining —
but it cannot tell us anything about interval calibration, and nothing whatsoever about tails.

### Does it measure what NGED care about?

No. MAE and nMAE only, and the paper says so. Its value to us is as evidence about *inputs and
training regimes* rather than about *outputs*. If we cite it, it should be for the covariate,
retraining and context-length ablations, and for the negative result that fancy Transformer
variants do not beat a plain one — not for any claim about which model is best, since the metric
that produced that ranking is not the metric NGED operate on. Worth pairing with the Kaas result
above, since the two papers use the same 200 feeders and reach different model rankings under
different metrics: a compact illustration of the whole problem.

---

## 4. Kleinebrahm, Berrisch, Eiser, Fichtner, Hagenmeyer, Hertel, Koster, Lerch, Mikut, Priesmann, Schienle, Schaefer, Weinand & Ziel (2026) — Energy-Arena

*Energy-Arena: A Dynamic Benchmark for Operational Energy Forecasting.* arXiv:2604.24705, 27 April
2026; six pages, presented at EEM 2026 in Trondheim. Platform live at energy-arena.org.
**Provenance note:** the arXiv PDF would not yield machine-readable text to my fetcher, so this
section is built from the abstract, a search-index excerpt of the challenge-design section, and the
live platform itself — which turns out to answer most of what the paper would have.

### Summary

This is the paper to cite for the opening caveat, because it makes our argument in its own abstract.
It names the problem as a persistent comparability gap that makes it difficult to measure consistent
progress over time, attributing it to models being evaluated under study-specific datasets, time
periods, information sets and scoring setups — and adds a point worth borrowing for the NGED
document: widely used benchmarks and competition datasets are tied to fixed historical windows, so
they go stale as energy systems evolve.

Their remedy is an open, API-based submission platform with standardised challenge definitions and
submission deadlines aligned to operational constraints, reporting performance on rolling evaluation
windows via persistent leaderboards. The framing is a move from retrospective backtesting to
forward-looking benchmarking, which enforces ex-ante submission and ex-post evaluation and thereby
prevents both information leakage and retroactive tuning.

The author list matters as much as the content. It spans KIT, Duisburg-Essen, Heidelberg, Marburg
and Jülich, and includes Florian Ziel and Sebastian Lerch — the energy forecasting and statistical
post-processing communities jointly. Hertel is an author here and on the benchmark paper above, and
that paper states an intention to integrate its best approaches into Energy-Arena for continuous
comparison under realistic operational conditions. This is not an isolated proposal; it is where at
least one active group is heading. It is also already in teaching use: KIT runs a summer-semester
seminar in which students submit day-ahead forecasts through the platform and are scored on the
public leaderboard.

### Setup in detail

**Challenges are YAML configuration files**, specifying the target variable, geographic areas,
reference timezone, forecast horizon, submission deadlines and frequency, evaluation metrics, and
leaderboard aggregation settings. New challenges are added by extending a challenge repository
rather than modifying platform code — and there is a public **"propose challenge" workflow with an
active review queue**, which matters for us below.

**The live catalogue, as of this writing, is 24 challenges** on a clean 4 × 2 × 3 grid:

- **Targets:** day-ahead prices, solar generation, wind generation, total load.
- **Areas:** Austria, and Germany–Luxembourg.
- **Formats and their headline metrics:** point forecast scored by **RMSE**; quantile forecast scored
  by **Winkler interval score**; ensemble forecast scored by **CRPS**.

Ground truth is ingested from ENTSO-E, a worker pipeline scores pending submissions once truth
arrives, and browser-visible realised-value charts come from SMARD. Participation is an account plus
API keys, with optional approach metadata and a choice of forecast visibility. Rolling evaluation
horizons are described as 1, 7, 30 and 365 days **[from the authors' own promotional posts rather
than the paper — treat as indicative]**.

**A sister platform is worth knowing about, though I have not verified its authorship.** TS-Arena
(arXiv:2512.20761; Kaas et al. cite a "TS-Arena Technical Report" that may or may not be the same
document) runs 186 live series from
SMARD, gridstatus and FINGRID across 14 challenges, and enforces leakage-freedom in an interesting
way: hosted foundation models only begin participating after their public release date, regardless of
their training cutoff, so a model cannot be credited for a period it might have memorised. It
backtested 2025 to populate the platform, producing over 5,000 challenge rounds, and publishes a
quarterly archive of context, ground truth and forecasts on HuggingFace. Kaas et al. cite it as the
way to see which foundation models are currently best.

### Read against Flexpectation

This is the clearest external validation of the leaderboard strategy, and it also exposes its weakest
point. Their argument is that a leaderboard is only a contribution to the field if it is open,
standardised and forward-looking. Ours is forward-looking — forward-chained folds, and a genuinely
live shadow deployment — but computed on private NGED data with metrics we chose. Structurally, that
is another team playing in its own stadium, which is precisely the failure the opening of the
document diagnoses.

**But we cannot simply submit to it, and it is worth being precise about why.** Every target on the
platform is a national or zonal aggregate from ENTSO-E: prices, total load, and wind and solar
generation for Austria or Germany–Luxembourg. There is no distribution-level target, no net demand,
and no substation anything. Submitting our forecasts is not an option; the platform does not host
our problem.

**The realistic routes, in ascending order of ambition.** Publish our evaluation protocol and metric
code, so the comparison is reproducible in method even where the data cannot be shared — nearly free.
Submit to an existing Energy-Arena challenge with a general-purpose model, purely as a competence
signal that our pipeline works against outside scrutiny — cheap, but it tests almost nothing about
the substation problem. Or use the **propose-challenge workflow** to argue for a distribution-level
challenge. The last is the only one that would move the field, and it is blocked on the same thing
everything else is: somebody has to make suitable data public. NGED already publish some substation
data on their Connected Data Portal (<https://connecteddata.nationalgrid.co.uk/>) and have said
repeatedly that they intend to publish the more rigorously collected telemetry this project uses —
but that is their announcement to make, not ours.

### Does it measure what NGED care about?

Now answerable rather than open, and the answer is a partial yes. **The platform does support
probabilistic submission** — quantile forecasts scored by the Winkler interval score, ensemble
forecasts scored by CRPS — so the earlier worry that even the community's best answer to the
comparability problem might only score point forecasts was unfounded.

What it does not have is any threshold, exceedance, asymmetric cost, or tail-weighted metric. RMSE,
WIS and CRPS are all whole-distribution or whole-series scores; none of them asks whether a network
limit was crossed, and none can express that over-procurement and under-procurement cost different
amounts. So the standardisation problem is being solved while the decision-relevance problem is not
— which is a fair summary of where our contribution sits, and a more precise one than we could make
before seeing the challenge catalogue.

---

## 5. Hong, Xie & Black (2019) — GEFCom2017

*Global energy forecasting competition 2017: Hierarchical probabilistic load forecasting.*
International Journal of Forecasting 35(4), 1389–1399. Competition data and instructions on
blog.drhongtao.com. **The paper itself is paywalled; this section is built from the organisers'
own instruction pages, the abstract, and competitors' method papers — notably Ziel (arXiv:1809.03561),
which is open.**

### Summary

The third and most demanding of the Global Energy Forecasting Competitions, and — for two reasons
that are easy to miss — a closer analogue to Flexpectation than it first appears. It attracted more than 300 students and professionals from over 30 countries, with 177 teams
enrolled, and was the first GEFCom to have a qualifying match, the first to use hierarchical data
with more than two levels, the first to allow external data sources, the first to require real-time
ex-ante forecasts, and the longest, running more than seven months in total.

**First, the horizon is medium-term, not short-term.** The qualifying match ran six
rounds, each asking for the whole of the following month. Round 1 was due 15 December 2016 for a
forecast period of 1–31 January 2017, and because ISO New England publishes each month's data by
around the 15th of the next month, the most recent load available for round 1 was November's. So
contestants were forecasting between roughly two and six weeks ahead with no usable weather forecast
at that range. **That makes GEFCom2017 the longest-horizon competition in this set and the only one
whose horizon brackets our 3–10 day band from above.** It also explains a recurring feature of the
method papers that otherwise looks eccentric: temperature *scenarios*, built by resampling
historical temperature paths, appear in solution after solution, because at a month's lead a
deterministic forecast does not exist. That is the same problem our ensemble members solve, reached
from the opposite direction.

**Second, the forecasts were nine quantiles, the 10th to 90th percentiles**, in a fixed template —
not the 99 quantiles of GEFCom2014, which do not carry over. The upper
tail is therefore absent from the scoring entirely, exactly as in HEFTCom.

The winning methods span the familiar families: quantile regression with lasso estimation (Ziel,
team 'simple_but_good', second in the open track and fourth in the defined-data track), generalised
additive models, ensembles of gradient boosting, quantile random forests and neural networks (team
Orbuculum, third in qualifying and fourth in the final), neural networks driven by temperature
scenarios (team 4C), quantile regression with clustered weather stations and heavy data cleaning
(team QUINKAN), and data visualisation plus forecast combination (team Cassandra, IBM Research
Australia and the University of Melbourne). The final match was won by Kanda and Quintana of the
Japan Meteorological Corporation, with Tangent Works second.

### Setup in detail

The hierarchy has three levels and ten series: eight bottom zones (Maine, New Hampshire, Vermont,
Connecticut, Rhode Island, and three Massachusetts sub-zones), Massachusetts as the sum of its
three, and a system total as the sum of the eight. Forecasts were required for all ten.

The defined-data track restricted inputs to specific columns of ISO New England's published hourly
data — load, dry-bulb temperature and dew-point temperature — plus US federal holidays and general
calendar knowledge. The open track permitted anything. Ziel's winning entry in the open track is
worth noting for how little it used: the load is log-transformed and split into a long-term trend
and a remainder, quantile regression handles the remainder with weekly and annual seasonalities and
their interactions, temperature is used *only* to stabilise the trend component, and public holidays
are ignored entirely. It still placed second in its track. That is a useful counterweight to the
assumption that more features and more experiments are what wins.

Evaluation was by quantile score, with each team compared against a benchmark and the relative
improvement used to rank. The "DEMAND" column published by ISO New England was the target, with the
organisers noting it may be revised during settlement — a data-vintage problem we have our own
version of.

**Still unverified:** the final match, in May 2017, is described only as "a large-scale problem".
Competitor papers refer to individual "load meters", some "of an industrial nature", and to weather
stations clustered into eleven groups, which suggests a different and more granular dataset than the
ISO-NE zones — but I could not confirm what it was, what the horizon was, or how many series were
involved. **[unverified]**

### Read against Flexpectation

**The medium-term horizon makes this a useful precedent, though not the closest one.** Everything
else here is day-ahead or four-day; GEFCom2017's
qualifying match sits at two to six weeks. But that is a qualitatively different regime from ours
rather than a superset of it: at a month's lead there is no usable weather forecast at all, so the
problem collapses to seasonality, trend and calendar with weather entering only as scenarios drawn
from history. At 3–10 days the ENS still carries real skill. So GEFCom2017 brackets our band from
above without covering it, and the honest reading is that our band is bracketed but unstudied —
day-ahead below, month-ahead above, and nothing in between. That is a better framing for the NGED
document than either "nobody forecasts as far ahead as us" (false) or "this is a close precedent"
(overstated). It also supports the per-horizon-window model on our Tier-4 list: if the day-1 regime
and the day-14 regime are this different, one model spanning both is a strong assumption.

The temperature-scenario technique is the specific transferable idea. At a month ahead, teams built
distributions over weather rather than conditioning on a forecast. We do the same thing with 51 ENS
members out to day 15, which is strictly better information — but the scenario literature is where
the methodology for turning weather distributions into load distributions was worked out, and it is
worth reading before we finalise how members are aggregated into quantiles.

The hierarchy analogy is looser than the horizon one. GEFCom2017 aggregates ISO zones to a system
total — hundreds of megawatts to gigawatts, smooth, weather- and calendar-driven. Our hierarchy is
primaries and secondaries with embedded generation, ten megawatts and below, dominated by a handful
of large customers and by PV nobody meters. The statistical character differs enough that the
winning methods may not transfer downward, which is Haben et al.'s standing complaint about the LV
literature borrowing from system level.

Two findings do transfer. Forecast combination appears in nearly every top solution, usually as
plain quantile averaging, with Cassandra reporting that simple averaging beat more sophisticated
weighting — an argument for a cheap combination arm on our leaderboard. And the dataset deliberately
shipped with gaps, outliers and incomplete series, with several teams reporting that cleaning
mattered as much as modelling; that supports pulling our Tier-2 training-data hygiene item forward,
since stuck meters and false zeros affect over 10% of some of our series and currently teach the
model wrong targets.

### Does it measure what NGED care about?

No, and now for a confirmed reason rather than a guessed one. The scoring grid is the nine deciles,
so p95 and p99 are outside it altogether — the same blind spot as HEFTCom, not the diluted-but-present
tail one might assume from GEFCom2014's grid. There is no threshold, no exceedance metric, no asymmetric cost and no
decision layer.

One practice is worth borrowing: ranking by relative improvement over a common benchmark rather than
by raw score, which normalises across series of very different scale. We achieve something similar
with NMAE and capacity normalisation, but a published skill-score-against-benchmark column would be
more legible to NGED than a raw error, and it is what makes cross-series leaderboards comparable at
all.

---

## 6. Shukla & Hong (2024) — BigDEAL Challenge 2022

*BigDEAL Challenge 2022: Forecasting peak timing of electricity demand.* IET Smart Grid 7(4),
442–459. **Open access (CC BY-NC-ND); read in full.** Companion analysis: Donaldson, Browell &
Gilbert, IET Smart Grid, 2024 (doi 10.1049/stg2.12152).

### Summary

The competition that took peaks seriously, and the only one in this review whose organisers set out
an explicit research agenda for metric design. It attracted 121 contestants from 27 countries in 78
teams, built on the observation that the literature has many studies on the magnitude of peak load
and few on the timing.

Hong's framing of *why* competitions matter is also the cleanest statement of the argument the NGED
document opens with, and it comes from the man who has run more energy forecasting competitions than
anyone: annual publications on load forecasting have exceeded 200 since 2019 and total over 1,000
since 2018, yet "most of these articles have not been found useful in practice yet", because the
performance of a proposed model is usually established on a private dataset nobody else can access.
That is worth quoting directly rather than paraphrasing.

**Structure.** A qualifying match — one-year-ahead **ex-post** forecasting of 2007 from five years
(2002–2006) of hourly load for a single US load zone, with three tracks: hourly load (H, scored by
MAPE), daily peak magnitude (M, MAPE) and peak hour (T, plain MAE in hours). Four "virtual" weather
stations were supplied, being the mean, median, minimum and maximum of a group of real stations.
Tracks H, M and T were completed by 43, 41 and 40 teams; 14 qualified. Then a final match — **ex-ante**
short-term peak forecasting across three neighbouring local distribution companies, using 2015–2017
history, run in six rounds covering the calendar months of 2018, with **day-ahead temperature
forecasts released each round** and the previous round's observed temperatures also released.

**The headline finding, stated by the organisers.** Track M's ranks are the *least* correlated with
the team's ranks in Tracks T and S, while T and S are highly correlated: "the top teams in Tracks T
and S are not the top teams in Track M, and vice versa". A model that performs well on peak magnitude
may not be the best for peak timing. Nine of the thirteen finalists, including four of the top five,
built separate models per track rather than one model scored three ways. **[The paper is internally
inconsistent here — the next sentence refers to "seven of these ten teams" where nine were just
described. Quote the finding, not the arithmetic.]**

### Setup in detail

**The two new metrics, now with their actual definitions.** Prior to this competition there were no
standard error measures for peak timing or peak shape, so the organisers proposed two.

*WMAE* (Track T, final match) weights the absolute displacement in hours between actual and forecast
peak hour: with z the displacement, g(z) = z for z ≤ 1, g(z) = 2z for 2 ≤ z ≤ 4, and g(z) = 10
otherwise. Two rationales are given: capping at 10 prevents unnecessary quantification of errors
beyond five hours, since missing by more than five hours is as bad as missing by five; and without a
cap, one bad day would overshadow good forecasts on all the others. Note the qualifying match used
*plain* MAE in hours for the same track — WMAE was introduced for the final.

*Peak Shape Error* (Track S) normalises each hour's actual load by the **actual** daily peak, and
each hour's forecast by the **forecast** daily peak, then sums the absolute differences over the
five-hour peak period — the actual peak hour plus the two hours either side — and averages over days.
Normalising the forecast by its own peak is what makes it a pure shape metric, independent of whether
the level was right. My earlier description of this, taken from a competitor's paper, had the window
right and the normalisation wrong.

**The benchmarks are worth copying, and one detail is easy to miss.** Tao's Vanilla Benchmark is an
MLR with a linear trend, hour-of-day, day-of-week and month classification variables, an
hour×day-type interaction, and temperature polynomials to third order interacted with hour and month.
Shreyashi's Recency Benchmark adds data cleansing, weather-station selection, a recency effect, and
holiday handling. The recency effect enumerates the number of lagged hourly temperatures *h* (1–24)
and lagged daily moving-average temperatures *d* (0–4) — 100 candidate models — and **selects the best
d–h pair separately for each track, using that track's own error measure**, validated over three
years. So even the benchmark is task-specific.

**Their data-cleansing recipe is concrete and cheap**, which is worth noting given our own
training-data hygiene item: fit the Vanilla model, evaluate the in-sample fit by MAPE, mark
observations with MAPE above 40% as outliers, verify them visually, and replace them with the Vanilla
model's own forecast values. Notably, other than one team, **no contestant in the qualifying match
reported any data-cleansing effort at all** — and the two teams that did clean in the final match
were among the better performers.

**Ex-ante weather handling is the most transferable part.** Temperature forecast errors were within
2°F most of the time for five of the six stations, occasionally exceeding 5°F. After the first few
rounds many teams adjusted their models to account for this, and **the top five teams all did it the
same way: they modified the actual temperature series of the *training* period to simulate forecast
error** — adding noise, shifting the series by ±1 hour, or explicitly predicting the temperature
forecast error. One team used the temperature *forecasts* for the validation period for model
selection and hyper-parameter tuning rather than observations.

Rounds 5 and 6 were the hardest for all three LDCs, for two reasons the organisers name: a hurricane
outage depressed load for several days in mid-September 2018, and the autumn and holiday seasons
bring more variation in human activity, with winter's twin morning and evening peaks harder to place
than summer's single evening peak.

**From the precursor competition, a temperature result worth knowing.** In BFCom2018, five of the
thirteen reporting teams used no temperature data at all — including the 3rd and 4th ranked teams —
formulating peak-hour probability as a classification problem on calendar variables only. Their Brier
scores were very close to the second-ranked team, which generated temperature scenarios. The
organisers' conclusion: a model without temperature variables can forecast next year's daily peak
hours nearly as well as one with them.

### Read against Flexpectation

**Simulating weather-forecast error during training is a technique we should adopt, and it
generalises past what we currently plan.** Our roadmap has training with NWP-missing examples so the
model survives a feed outage. The BigDEAL winners did something broader and more routine: they
degraded the *training* weather to match the error distribution of the weather the model will see in
operation. We have a much richer version available — 51 ENS members give the forecast error
distribution directly rather than requiring it to be simulated — so the arm is closer to "train on
members, not the control" than to "add noise", but the principle is the same and it was what
separated the top five from the rest.

**Three metric ideas transfer directly.** WMAE's capping is a design choice we will face: an
uncapped timing error lets one catastrophic day dominate a leaderboard, which is the same pathology
the cost metric's unmet-fraction pooling has. Peak Shape Error's double normalisation — forecast by
forecast peak, actual by actual peak — is how you separate profile error from level error, which
NMAE cannot do. And selecting the *model* per track using that track's own error measure is
something our leaderboard architecture would currently discourage, since we promote a single champion
on a headline metric.

**That last point is the uncomfortable one.** Nine of thirteen finalists, and four of the top five,
concluded that one model scored three ways was the wrong design. Our leaderboard promotes one champion
per series on NMAE. If magnitude, timing and shape genuinely want different models — and this
competition is the strongest evidence that they do — then either the leaderboard needs multiple
champion slots, or we accept that we are optimising for the wrong attribute.

The limits are the usual ones and should be stated. It is US utility load, temperature-driven, with
no embedded generation and no probabilistic component at all. The final match's horizon is also not
what I previously described: contestants forecast whole calendar months on a rolling basis, but were
given day-ahead-quality temperature forecasts, so it is a hybrid — operationally ex-ante on weather,
but months long on the target. That is closer to our 3–10 day band than day-ahead would be, but the
weather information is far better than we will have at day 10.

### Does it measure what NGED care about?

Closer than the GEFCom family, and its authors say explicitly where it stops. Their conclusion sets
out a research agenda that reads as an invitation:

- **"The operational requirements of the peak forecasting problem have not been formally studied."**
- **New error measures are needed, and they suggest "adapting existing error metrics from binary
  event detection to assess the specific forms of peak forecasting problems", adding that "ideally,
  these metrics should be practical and simple to compute and communicate."** That is a precise
  description of Kaas et al.'s fuse-derived confusion matrix — and of our threshold-and-price
  metric, whose whole justification is that pounds are communicable to a board.
- **"None of the teams reported using track-specific features in their models, which could be a
  research opportunity."** Separate models per track, yes; separate *features* per track, nobody.
  Our per-`time_series_type` feature lists are the same idea one axis over.
- Future competitions should be "tailored to operational requirements" — they name calling demand
  response events, scheduling EV charging, forecasting peak periods to design tariffs, and peak
  timing at the meter level.

It remains deterministic throughout, so it cannot express a procurement quantile, and it has no
threshold, no exceedance and no cost. The gap between BigDEAL's peak attributes and our cost metric
is the gap between "how well did you predict the peak" and "what would it have cost to be safe given
your prediction" — but they have asked for the second, in print, which is a better position for us
than asserting the gap ourselves.

---

## 7. Haben, Arora, Giasemidis, Voss & Vukadinović Greetham (2021), and companions

*Review of low voltage load forecasting: Methods, applications, and recommendations.* Applied
Energy 304, 117798. **Open preprint: arXiv:2106.00006, read in full.** Dataset index
at low-voltage-loadforecasting.github.io. Book-length companion: Haben, Voss & Holderbaum (2023),
*Core Concepts and Methods in Load Forecasting: With Applications in Distribution Networks*,
Springer.

### Summary

The standard reference for why LV is a different problem from system-level load forecasting, and the
source of the single most useful statistic in this whole review for the NGED document.

**Of 221 LV forecasting papers reviewed, three used weather forecasts rather than weather
observations, and none used weather ensemble predictions.** The authors put it plainly: the vast
majority of reviewed papers employ actual weather observations, so the corresponding forecast
accuracy can be expected to be over-optimistic; to the best of their knowledge only Haben et al.
(2019) and two others use weather forecast inputs, and those three employ point estimates rather
than ensembles, thereby ignoring the underlying uncertainty. If one sentence has to justify why
Flexpectation is doing something new, that is it — and it comes from the field's own review rather
than from us.

The horizon distribution is nearly as useful. Of the 221 papers, 80 forecast day-ahead, twelve
forecast shorter than an hour, **sixteen sit between two days and a week, and thirteen at a month or
more** — with 80 papers whose horizon could not be identified at all. So the 3–10 day band we care
about is served by something under sixteen papers in the entire reviewed LV literature, and that
number is an upper bound.

Its central methodological complaints are the familiar ones, and they hold up: very few papers
address the secondary or primary substation level; those that do usually construct pseudo-substations
by aggregating smart meters, which misrepresents real LV networks because a feeder carries street
lighting, cameras and other street furniture as well as households; and studies covering a handful
of substations cannot generalise. The review states that Haben et al. (2019), with 100 real feeders,
is to the authors' knowledge the only study of a large number of real feeders — a claim that as of 2026 is
comfortably superseded — by FeederBW's 200 open feeders, and by two orders of magnitude more by
Pinheiro et al.'s 96,989 Portuguese secondary substations (§9), published in 2023 and so post-dating
the review's search. Real substation data at scale now exists; what remains scarce is *open* real
substation data.

### Setup in detail

The review is a Scopus survey: 1,487 manuscripts matched the query, filtered by date, venue and
citation count to 492, of which 221 were read. Two findings from the survey are structural rather
than bibliographic and matter to us directly.

**The power-law relationship between feeder size and relative error.** Relative error scales as a
power law in the size of the feeder — average daily demand, peak demand, or number of connected
customers — so it becomes exponentially harder to forecast smaller feeders. LV feeders average about
50 households; beyond about 100 households profiles are relatively smooth, below that they become
increasingly irregular and spiky. The review also notes that this power law does *not* hold for some
feeders, such as those with large numbers of overnight storage heaters or high commercial or lighting
load, which is the argument for reporting metrics per series rather than pooled.

**The double-penalty effect.** A forecast that gets a peak's amplitude and duration right but places
it half an hour early is penalised twice — once for the peak that did not happen, once for the peak
it missed — so under absolute error a flat forecast can beat a better-informed peaky one. Haben et
al. (2014) proposed an adjusted error measure based on the *p*-norm with *p* = 4 to counter this,
and the review notes the effect has not been investigated for small feeders specifically, only for
individual households.

On open data: only 52 of the 221 papers used any open dataset, and of those, 56% used one of just
four. The review argues that overuse of a single dataset makes a rigorous experiment impossible, and
that trial datasets carry selection bias because participants were subject to tariffs or other
interventions. Their dataset index is at low-voltage-loadforecasting.github.io; the British entries
include the SSEPD NTVV secondary substation data on the CEDA archive and SP Energy Networks'
Flexible Networks data share.

Of the 221 papers, 44 included any probabilistic element. The review notes that bottom-up aggregation
of lower-level forecasts to produce higher-level ones has been shown to produce poor results, and
points to Ben Taieb et al.'s empirical-copula approach for coherent hierarchical probabilistic
forecasts.

### Read against Flexpectation

**The temperature finding is genuinely inconvenient and needs to be met head-on.** Haben et al.
(2019) tested several models with and without temperature — both
forecast and actual — across 100 real feeders and found no effect, or a negative effect, on
short-term accuracy for both point and probabilistic forecasts. The review reports two other papers
reaching the same conclusion: Fidalgo and Lopez discarded temperature features after experiments
showed no strong effect, and Lusis et al. found calendar effects lose predictive power when combined
with weather and seasonality. A fourth (Bennett et al.) found the opposite, with temperature
accounting for about half the variation in day-ahead LV transformer load. So the literature is split,
with the largest real-feeder study on the negative side.

Three things distinguish our case, and the NGED document should state them rather than ignore the
finding. Our horizon is 3–10 days, where there is no useful recent-power anchor and weather is
carrying much more of the forecast than at day-ahead. Our target is net demand with material embedded
PV, where the mechanism is irradiance rather than heating response — and Kaas et al.'s 2026 ablation,
on 200 modern feeders, finds weather covariates worth about 1 kW of MAE concentrated at midday, which
is the PV signature. And this is an empirical question we can settle on our own data, which is what
the ablation ladder on the XGBoost page exists for. If effective temperature loses on our
leaderboard, that is a result with respectable precedent rather than a failure.

**Topology reconfiguration is mentioned, so it is not absent from the literature.** The review notes
in passing that LV connections may be reconfigured over time, citing Mirowski et al. It is a single
clause with no methods attached, so the substantive point stands — no paper in this set models,
detects or evaluates reconfiguration — but the stronger claim that the field has never noticed the
problem would be wrong.

The power-law finding argues for something our metrics design already does and should keep doing:
report per-series and per-`time_series_type` alongside the pooled number, because a pooled metric
across series of very different size is dominated by the large ones and says nothing about the small
feeders where the forecasting problem is actually hard.

**Gilbert, Browell & Stephen (2023)** (arXiv:2206.11745, read in full) is the closest methodological
companion in the review, and its headline result is the single cleanest demonstration anywhere here
that average-error metrics and peak metrics rank models differently.

They forecast day-ahead across a four-level hierarchy — primary substation, secondary substations,
feeders, households — using GAMLSS: Gaussian at the aggregate levels, and the **Generalised Beta
Prime** distribution at household level, chosen because demand there is non-negative and
right-skewed, which rules the Gaussian out. Their contribution is *forecast fusion*: a linear pool of
the half-hourly forecast CDF with a bespoke daily-peak-intensity forecast CDF, where the **weights are
a probabilistic forecast of the peak's timing**, obtained by framing time-of-peak as a discrete
time-to-event (survival) problem with a logit-link hazard in a GAM. Scoring is CRPS, RPS for the
discrete timing forecast, PIT histograms for calibration, skill scores against benchmarks, with
bootstrap resampling for significance.

**The numbers are the point.** Averaged over all periods, fusion beats the advanced half-hourly model
by 0.0–0.4% in CRPS — indistinguishable. Restricted to the periods containing the daily peak, the same
comparison gives 5.7% at the primary substation, 9.0% at secondary, 8.2% at feeder and 6.0% at
household level. Their own summary is that sophisticated conventional forecasts provide skill overall
against competitive benchmarks but "add little value during peaks". Worse, at household level *during
peaks* both the simple and the advanced GAMLSS models are significantly **worse than a plain
time-of-day KDE benchmark** by roughly 1–5% — a direct, measured demonstration of the double-penalty
effect that the Haben review describes. Fusion is the only method that is good on average *and* at
peaks, and it improves 80% of households and every aggregate node.

Three further findings transfer. Peak *timing* skill collapses with disaggregation: better than 20%
over seasonal climatology at the primary substation, 0% at four of the feeders, and under 0.5% at
household level — so at low aggregation, time-of-peak is essentially climatological. There is no
relationship between a node's coefficient of variation and forecast skill at aggregate levels, but a
negative one at household level. And the work is fully reproducible, with data and R code on Zenodo
(10.5281/zenodo.7064279) via their ProbCast package.

Two caveats. The hierarchy is constructed by sampling 742 Low Carbon London smart meters into a
hypothetical network — exactly the pseudo-substation practice the Haben review criticises, applied by
close collaborators of its authors, and they say so: it cannot represent street furniture, embedded
generation, losses or inter-node correlation. And the data is anonymised, so **there is no weather in
the study at all** — the models are autoregressive plus calendar. They also state that net demand,
"demand less embedded generation, is reserved for future work". So this is peak methodology developed
in the absence of the two things that define our problem.

**Their closing future-work list reads like our roadmap**, which is worth noticing: embedded
generation, storage and demand response; *global* models applicable to feeders they were not trained
on, via transfer learning, because networks have tens of thousands of feeders; models that are
adaptive to structural breaks in load behaviour; and exploiting the hierarchy. They also note that the
right tail of their predictive distributions "could be improved at most of the levels to account for
large peaks in demand", reserved for future work. That is four or five of our roadmap items named as
open problems by the people best placed to know.

### Does it measure what NGED care about?

The review's recommendations point at the right target — probabilistic, peak-focused, weather-aware —
without proposing a decision or cost metric, though its discussion of the double-penalty effect and
of application-specific error measures is the closest the LV literature comes to arguing that the
metric should follow the downstream decision. It cites work showing that the optimal choice of error
measure depends on the downstream optimisation objective, and calls for more studies in specific
downstream applications. That is a direct invitation to the kind of metric our cost-savings page
defines, and a good line to cite when justifying it.

---

## 8. Browell & Fasiolo (2021)

*Probabilistic forecasting of regional net-load with conditional extremes and gridded NWP.* IEEE
Transactions on Smart Grid 12(6), 5011–5019. **Open accepted manuscript at
eprints.gla.ac.uk/250372/1/250372.pdf, read in full; preprint arXiv:2103.10335.**
Supplementary data and code at Zenodo 10.5281/zenodo.4618056.

### Summary

This is the most important paper in the review for our purposes, and the easiest to under-rate from
its abstract. It is British, it is net load with embedded renewables, it
uses gridded ECMWF NWP, it models the tails explicitly with a conditional Generalised Pareto
Distribution, and **it evaluates a decision**, in the form of reserve volume procured at a fixed
risk level. That is the same "hold risk constant, compare the
spend" structure our cost-savings metric uses, published five years ago by the person who ran
HEFTCom.

The headline results are three. First, quantile regression tails are **not calibrated beyond about
the 1st and 99th percentiles at any of the 14 GB regions**, so those quantiles are unsuitable for
decision-making; cross-validation identified the 2.5th and 97.5th percentiles (5th and 95th for four
regions) as the last reliable quantile-regression levels. Second, both a static GPD and a conditional
GPD produce calibrated tails, but the conditional version — with its scale parameter varying smoothly
with wind speed, irradiance and expected net load — is materially sharper. Third, in the reserve
use case and at the same risk level, the conditional GPD reduces **upward** reserve volume by up to
24.6% against the static GPD and by 16.5–18.0% against a naive historical-error approach. That
24.6% is the *most* extreme upward level (0.01%); the same column falls to 13.9%, 9.1% and 3.2%
at 0.05%, 0.1% and 0.25%, so 3.2% is the saving at the least extreme upward level tested.
Downward reserve behaves differently: only 0.8–10.8% against the static GPD, but 19.8–25.8% against naive. Do
not quote a single headline percentage — four probability levels times two directions gives eight
numbers spanning 0.8% to 25.8%.

**The finding that should change our leaderboard design** is a methodological aside rather than a
headline. The authors state that the pinball score places greater weight on quantiles near the median
and less on those in the tails, and that pinball scores for individual tail quantiles suffer from high
variance due to the sparsity of observations and are therefore **not suitable for discriminating
between forecasting systems**. Our metrics roadmap plans to score tails; this says that scoring them
with per-quantile pinball will not work. What they use instead is interval width at fixed coverage,
with calibration verified separately by worm plots — a Q-Q variant chosen because it accentuates tail
behaviour — with consistency intervals computed allowing for serial correlation, since the usual
independence assumption fails for load.

The fourth result is the one our users would care about most. The reduction in reserve is **not
uniform in time**: the conditional GPD needs less **upward** reserve than the static one in 56–81% of
settlement periods and *more* in the remaining 19–44%. Those are the periods where the simpler benchmarks were
silently exposing the operator to excess risk. A method that lowers average cost while raising the
requirement exactly when uncertainty is genuinely high is precisely the argument NGED's flexibility
team would find persuasive, and it is already in the literature.

### Setup in detail

The target is GB net load at Grid Supply Point Group level: 14 regions, half-hourly, 2014–2018, with
embedded solar growing from 3 GW to 13 GW across the period. Regions range from 6.3 GW peak (East
England) down to 0.5 GW (North Scotland), the latter exporting 1.2 GW at times because embedded wind
capacity exceeds peak load. All series are standardised to z-scores before modelling, which is how
they compare across regions.

**Weather is ECMWF-HRES — deterministic, not ensemble.** Forecasts with a 00:00 UTC base time,
available around 06:00, extracted on a 0.1° grid: 10 m and 100 m wind speed, surface temperature,
cloud cover, solar irradiation and total precipitation. These are genuine day-ahead ex-ante forecasts,
not reanalysis, which puts this paper ahead of both KIT papers on operational realism.

**Validation is three-fold cross-validation on 2014–2017 with final testing on 2018, and the GAMs are
retrained every two weeks in an expanding window during the test phase.** That is close to our own
expanding-window scheme, and it is worth noting they treat cross-validation and final test as separate
exercises rather than reporting CV numbers as results.

The model family is a progression designed as an ablation: Tao's Vanilla Benchmark and a GAM, each
with (a) calendar and point temperature only, (b) plus spatially-averaged wind speed and irradiance
scaled by embedded capacity, and (c) plus statistics derived from the gridded NWP — spatial standard
deviation, min, max. Central quantiles come from linear quantile regression on the residuals of the
GAM, since fitting quantile GAMs directly was too computationally demanding.

Four results from that ablation are directly relevant to our feature roadmap:

- **Adding wind and irradiance features cut pinball by 40% overall**, but by only 10% in Greater
  London, where there is little embedded generation, and 60% in North Scotland, where embedded wind
  exceeds peak load. Weather features pay in proportion to embedded capacity — which is an argument
  for our per-`time_series_type` feature lists and for the capacity-estimation work, since the
  benefit is conditional on knowing how much generation sits behind the meter.
- **Gridded NWP statistics added no significant value.** GAM-Grid beat GAM-Point significantly in two
  of 14 regions, *lost* significantly in three, and was indistinguishable in the remaining nine. The
  authors conclude gridded NWP does not appear to add significant value in their framework, while
  allowing that other methods might extract it with different features. This is the clearest negative
  result in the review and it lands on our neighbouring-H3-cell item.
- **Autoregressive terms were detrimental**, which the authors attribute to embedded generation
  polluting the seasonal cycles. They replaced them with a two-week rolling mean of net load to
  capture level changes with less sensitivity to individual lagged observations.
- Model comparison used Diebold–Mariano tests per region, with the >10% improvement of GAM over
  Vanilla significant at p<0.001 consistently across all 14.

The conditional GPD itself is modest in construction: log link on the scale parameter, linear effects
for 100 m wind speed and surface irradiation, a smooth effect on expected net load with four basis
functions, and a constant shape parameter. The reserve use case sets requirement as q(0.5) − q(0.0005)
for upward reserve, the 0.0005 level corresponding to a "sufficient in all but four hours per year"
risk appetite; the paper notes the economically relevant probability is the ratio of marginal reserve
cost to value of lost load, typically 0.01%–0.25%.

### Read against Flexpectation

**The GPD recommendation is now much better supported than when I made it, and it comes with a
specific warning attached.** Our plan is to obtain quantiles by training the booster with a pinball
objective at chosen levels and pooling over ensemble members. This paper says two things about that.
It says quantile regression stops being calibrated somewhere around the 1st/99th percentile even with
five years of half-hourly data across 14 large regions — and we have 10⁴–10⁵ rows per series on
individual substations, so our reliable range will be narrower, not wider. And it says per-quantile
pinball in the tail is too noisy to rank systems by, which means the leaderboard cannot simply add a
p99 pinball column and call the tail scored. The replacement they use — interval width at fixed
nominal coverage, with calibration checked separately and serial correlation accounted for — is
directly implementable and should probably be the shape of our tail metrics.

**Three of their negative results are cheap tests we should run rather than assume past.** Gridded
NWP statistics adding nothing is a direct challenge to the neighbouring-H3-cell weather context item
and, more weakly, to the ensemble-statistics-as-features fork; their framework is a GAM rather than a
booster and trees may extract what splines could not, but the prior is now negative and the item
should be ranked accordingly. Autoregressive terms being actively harmful is striking given how
central power lags are to our feature set — but do not read it alone: Pinheiro et al. (§9) found the
opposite on Portuguese national load, where 24-hour and 1-week lags cut MAPE by nearly 40%. The
difference is plausibly net load versus gross demand, which would put us on this paper's side. Note
also what they did *not* try: having diagnosed embedded generation as the cause, they deleted the lags
rather than supplying the weather at the lagged time so the model could judge how normal that
observation was. That intermediate remedy — our aligned-lagged-weather and weather-delta rungs — is
unexplored in this literature and is a defensible thing to claim as a contribution, since the delta
is linear in the capacity-factor change and so is exactly the comparison their diagnosis implies. And weather features paying in proportion to embedded capacity supports doing
capacity estimation before ML tuning, which is the ordering the roadmap already asserts on other
grounds.

The mismatch is scale and horizon. Fourteen GSP Groups at gigawatt scale, day-ahead, is not 32
primaries at ten megawatts, 3–10 days ahead. Their smallest region is still larger than our largest
series. Whether quantile regression's calibration limit sits at the same percentile on a noisier,
smaller series is an open question, and probably the first thing to measure if we pursue this.

The adaptive/state-space family remains the live competitor to the switching detector: de Vilmarest,
Browell, Fasiolo, Goude & Wintenberger apply online Kalman-style updating to net load, and that family
won the post-COVID demand competition, which was designed around abrupt behavioural change. Our case
for an explicit detector — inspectable artefact, feed-forward signal, topology step change rather than
gradual drift — is reasonable and untested, and the leaderboard should carry a "just adapt" arm.

### Does it measure what NGED care about?

**Closer than anything else in this review.** It does not
use pounds, but it does the harder half of the job: fix a risk level, compute the volume each forecast
would require to hold it, and compare. That is our cost-savings structure minus the price vector, and
their reporting of *when* the volumes differ — not just how much — is a refinement we should copy,
because a metric that only reports total volume hides exactly the periods where a benchmark is
under-procuring.

What is left for us to add is genuinely additive rather than corrective: two prices rather than a
volume, so that over-procurement is charged at the availability rate rather than the utilisation rate;
a network threshold rather than a system-wide reserve requirement; and the distribution level, where
the series are small enough that the tail estimation problem they describe gets harder. Framing our
contribution as extending their reserve-setting evaluation to distribution-level flexibility
procurement is both accurate and more credible than claiming the decision-metric ground is unoccupied.

---

## 9. Pinheiro, Madeira & Francisco (2023)

*Short-term electricity load forecasting — A systematic approach from system level to secondary
substations.* Applied Energy 332, 120493. **Gold open access under CC BY; read in full via
escholarship.org/content/qt0s14445q/qt0s14445q.pdf.** Marco Pinheiro is at Instituto Superior
Técnico and at EDP, the Portuguese utility, and declares that as a competing interest; the work was
part of an internal EDP project.

### Summary

The largest real-substation study in this review by two orders of magnitude, and the only entry that
is a production system rather than an experiment. It forecasts day-ahead load for **96,989 secondary
substations covering the whole of mainland Portugal** — 70,510 owned by the DSO and 26,479 owned by
individual large clients — and the resulting system, PREDIS, runs daily in production and is used to
anticipate load peaks and network constraints.

Its evaluation framework is the thing to take from it. The authors score models on four criteria —
**applicability, interpretability, reproducibility and accuracy** — and argue for the first three on
explicitly regulatory grounds: the forecaster has to be approved by managers, understood by
operators, and defensible before the regulator. That framing is much closer to the NGED handover
requirement than anything else in this review, and it is one of the few papers that treats "can a
non-expert operate this" as a first-class criterion rather than an afterthought.

Two headline numbers, and they are not the same number. At **system level**, a GAM with synthetic
explanatory variables cut MAE, MAPE, RMSE and NRMSE by **42–47% against Tao Hong's GLM benchmark**,
and a regime-based ensemble cut RMSE further from 203 MW to 154 MW. At **substation level**, the
results are far more sober: measured by MASE against a 24-hour-naive forecast, the GAM beats
persistence on roughly 83–87% of DSO-owned substations and only **66–70% of client-owned ones**. In
other words, on the largest real deployment in the literature, something like a third of
client-owned secondary substations are not forecast better than by repeating yesterday. The paper
reports these percentages inconsistently — §3.6 says 82.8% and 66.0%, the Figure 14 caption says
86.5% and 70.0%, and the conclusion says 82.1% overall — so quote the range, not a figure.

**The negative result that matters most to us: XGBoost lost to the GAM.** They fitted a gradient
boosting model on identical features with an exhaustive hyper-parameter grid search, temperature
unfolded to polynomial terms, one-hot encoded categoricals, scaled numerics and three families of
base learner. It reached 199 MW RMSE against the GAM's 191 MW. **Corrected 2026-08-22:** the paper's own words are "no improvements in accuracy" and "it achieves the same accuracy, but with two disadvantages" — the two being the cost of hyper-parameter optimisation and the loss of interpretability. Accuracy is not one of the grounds for rejection. It notes that a GBM reaching comparable accuracy needs thousands of weighted trees
or linear functions and so loses the domain-level interpretability that individually-interpretable
base learners promise.

### Setup in detail

**Weather is numerical weather prediction, deterministic, on a grid** — which makes this a fourth
paper using genuine weather forecasts at LV level, beyond the three the Haben review counted, and
unsurprising given it was published after that review's August 2020 search cutoff. The fields are
3-hourly on a 0.125° grid over mainland Portugal, updated at 00 and 12 UTC for the next 72 hours,
and **centrally available only seven to eight hours after base time** — a publication-delay
constraint they state explicitly. Each substation is assigned to its nearest Euclidean grid point.
The variables are mean sea level pressure, 2 m temperature, surface solar radiation downwards
(accumulated), total precipitation (accumulated), and **100 m U and V wind components**.

Three details there are worth pausing on, because they corroborate design decisions on our roadmap
independently. They store and consume wind as **u/v components rather than speed and direction**,
which is the storage change our NWP-conventions work argues for. Their radiation and precipitation
fields are **accumulated**, i.e. the period-ending convention whose mishandling our resample fix
addresses. And they upscale a 3-hourly temperature to 30 minutes by linear interpolation **only when
the gap between consecutive points is no greater than 3 hours** — a bounded-gap rule, which is
exactly the discipline our null-filling item wants to impose on an unbounded `interpolate()`.

The horizon is **24 hours (48 half-hours)**, day-ahead. Load is 15-minute data downsampled to 30
minutes. Validation is **time-series cross-validation with a fixed three-year training window** and
update cycles of one day, one week, two weeks and one year, over 2016–2019 — forward-chained, with
the number of folds varying from 1 to 365 accordingly. Substation results are tested on the whole of
2019, on the 84,663 assets with enough data, the authors noting that a full year is the minimum
acceptable test for a target with annual seasonality.

**They use a peak-aware error measure at substation level, and say why.** Their words: the standard
accuracy measures reward smoothness, and at the distribution level it is the peak that matters for
many use cases. So for the substation models they adopt **Haben's adjusted p-norm error** with
*p* = 4 to penalise missed peaks heavily, and *w* = 3, meaning a forecast may be displaced by up to
three half-hours either side before being charged for the displacement — plus normalised variants
(MAPN, NMAPN) for cross-asset comparison. Median NMAPN is 0.222 for DSO-owned substations; only
14.3% of client-owned ones do that well.

The regime ensemble is the other transferable idea. Eight disjoint regimes — Christmas and New
Year, Carnival, Easter, other public holidays, weekends, August, other spring/summer days, other
autumn/winter days — each get a specialist GAM trained on that regime's data, and a continuous
weighted-majority algorithm combines them online with a delay parameter so that the true label
arrives 24 hours after the prediction. All experts share the same regression technique and the same
explanatory variables; only the training subset differs. That alone took RMSE from 203 MW to 154 MW.

On compute (the paper rounds 96,989 to "100,000" throughout this section): the individual models are
re-inferred daily on a 22-server Hadoop cluster, taking
5 h 42 m on 100 vcores, equivalent to about 24 days of sequential vcore time. Scaled linearly, that
is roughly 20 vcore-seconds per model per day — so our 2,500 series would need on the order of
14 vcore-hours a day, which is reassuring for the single-machine principle. Data is not shared: the
authors state they do not have permission.

### Read against Flexpectation

**This is the closest thing in the literature to the system we are building, and the comparison is
not entirely comfortable.** It is real substations, at national scale, with NWP inputs, deployed in
production for a DSO, evaluated on peak-aware metrics, with interpretability and operability treated
as first-class criteria. Where it differs is horizon (24 hours against our 3–10 days), uncertainty
(deterministic point forecasts throughout — no quantiles, no intervals, no ensemble), and embedded
generation, which does not feature in their model structure at all.

Those three differences are exactly our contribution, and stating them against this paper is a much
stronger claim than stating them against a vacuum. It also means the NGED document should not say
that substation-level forecasting is unstudied. It should say that substation-level *probabilistic*
forecasting, at days-to-weeks horizons, on net demand with embedded generation, is unstudied — which
is both true and narrower.

**The XGBoost result deserves a leaderboard arm rather than a shrug.** A GAM with well-chosen smooth
terms and interactions beat a thoroughly-tuned gradient booster on the same features. Our entire
baseline is XGBoost. The available rebuttals are that theirs is national aggregate load, which is
smooth and where spline structure is a natural fit, and that our targets are volatile net demand at
ten megawatts where trees may do better — but that is a hypothesis, and a GAM arm is cheap given
`mgcv` exists and the paper prints its exact model formula in a footnote.

**Their lag result contradicts Browell & Fasiolo's, and the contradiction is informative.** Here,
adding 24-hour and 1-week lagged load to the GAM cut MAPE from 4.09% to 2.53% — a large gain.
Browell & Fasiolo found autoregressive terms *detrimental* and replaced them with a two-week rolling
mean. The likely reconciliation is embedded generation: Browell & Fasiolo forecast net load where
embedded wind and solar pollute the seasonal cycles that a lag would otherwise carry, while Pinheiro
et al. forecast gross demand. Our targets are net demand, which puts us on the Browell & Fasiolo side
of that line — but the mechanism is a hypothesis, and the pair of results is a good argument for
running the lag ablation rather than assuming either way.

**Two-thirds is the number to remember.** Only about 66–70% of client-owned secondary substations
beat a 24-hour-naive forecast. Those are single-customer sites — one large building or industrial
consumer — and the closest analogue in our population is the small, spiky, few-customer end of the
2,500-series scale-up. It is a strong argument for the persistence and climatology baselines on our
metrics roadmap being load-bearing rather than decorative, and for reporting the *fraction of series
beating baseline* as a headline number alongside pooled error.

### Does it measure what NGED care about?

Partly, and in a way no other entry does. It is the only paper in this review whose accuracy metric
was chosen because standard metrics reward smoothness and a DSO cares about peaks — and it is a
production system, so that choice was made under real operational pressure rather than in a paper.
Haben's adjusted p-norm with a displacement window is a concrete, published, deployed answer to the
double-penalty problem, and it belongs on our candidate metric list.

What it does not do is quantify uncertainty at all. There are no quantiles, no intervals, no
threshold, no exceedance, no risk level and no cost. A DSO running peak-anticipation off point
forecasts has no way to express how confident it is, which is precisely the gap probabilistic
forecasting fills and the reason our p10/p50/p90 outputs are a step beyond the current production
state of the art in a comparable utility.

---

## 10. Adjacent work: three additions, and what was rejected

A targeted search on DER disaggregation, generator capacity estimation, differentiable physics and
switching-event detection, plus UK innovation projects. The bar for inclusion was deliberately high
given the length of this review: a paper had to bear on a specific roadmap item *and* change a claim
made elsewhere in the document. Three cleared it.

### 10a. Artificial Forecasting (Northern Powergrid SIF) — the most important thing in this review that is not a paper

**Reference 10145998 (Beta), preceded by NPG_SIF_003 (Discovery) and NPG_SIF_006 (Alpha).** Beta ran
from February 2025 for 24 months, ending February 2027. Total across all three phases £3,892,646.
Partners: Faculty Science (technical lead and lead project manager, £1.69m of the £2.02m spent in
Beta's first year), EV.energy, Oaktree Power. **Read in full: the Alpha and Beta registration forms;
the 52-page Beta Annual Progress Report of March 2026 including its four learning tables and project
plan; and the two Alpha WP2-D2 technical reports of early 2024 — Scope Item 1.2, customer export and
net demand at EHV-HV (82 slides), and Scope Item 2, active power at HV-LV (48 slides).** Appendix 4
of the Beta report, the post-Beta roadmap, is an image with no extractable text, so anything below
attributed to the roadmap comes from §6.3 instead.

**This is a concurrent UK DNO project doing much of what Flexpectation does, and it is further
along.** As of March 2026 an EHV-HV prototype is deployed on Azure, has passed NPg's Architecture
Review Board, data governance and InfoSec, and was **used operationally by NPg's System Forecasting
team through the Winter 2025-26 flexibility procurement cycle** to support week-ahead dispatch
decisions. They put it at Commercial Readiness Level 5.

#### Where the project actually is, March 2026

They are one year into a two-year Beta, all milestones delivered to plan, with an unconditional pass
at the first Stage Gate. What exists and what does not:

- **EHV-HV is done and live.** Gross demand and net demand endpoints are deployed; the refined gross
  model completed August 2025 and the refined customer-export model February 2026. Forecasts are a
  standing input to NPg's weekly forecasting governance meeting, alongside the in-house growth and
  persistence methods.
- **EHV-HV *net* demand has not yet been through a procurement season.** Only the gross model was
  used in Winter 2025-26; §9.4 says net models "start being used in the winter procurement cycle of
  2026-27". So the disaggregation half of their EHV-HV story is deployed but not yet operationally
  proven.
- **HV-LV is the live workstream.** The refined net model is due 27 June 2026 and the endpoint 3
  August 2026, with MVP sign-off 3 November 2026. The delay to live HV-LV ingestion — caused by
  cloud-environment readiness, security configuration and partner access, not by modelling — **has
  been resolved and live ingestion is in place**; modelling continued on historical data throughout.
- **Cold-start is a dated deliverable, not an aspiration.** WP2-M4, "substation grouping utility
  complete for unmonitored and recently monitored HV-LV substations", is due 3 November 2026. That is
  the transfer-learning question their per-instance TCN leaves open (below), and they have committed
  to answering it before we will.
- **Handover, not research, is what remains.** The rest of the plan is an integration supplier, an
  AI/ML governance framework, a platform security sign-off, and a three-month supported period ending
  30 January 2027. Two risks are still open: authentication currently runs through Faculty's
  environment pending NPg's security process, and the integration-supplier handover is rated medium
  likelihood / high impact.

**Their value case, which is the part we have no equivalent of.** Whole-life NPV around **£60m** for
NPg alone, rising to **£250m** if three further GB DNOs adopt it. The components are worth knowing
because our cost-savings page has to build something similar: a 3% reduction in EHV/HV and HV/LV
reinforcement cost in ED2 rising to 6% in ED3; around **25% improvement in the cost-effectiveness of
DNO-contracted flexibility**; ~£0.5m a year of avoided manual analysis; ~£1m of additional FSP
revenue; and ~700 tCO₂e avoided to 2033. These are Application-stage estimates that Beta has not yet
revalidated — §9.1's evidence column offers a deployment case study, not a measured saving — but they
are a published DNO's own arithmetic for the benefit we are also claiming, and the 25% figure in
particular is the one an NGED reader will compare us against.

**Read the Alpha technical reports before the Beta progress report.** The Beta report is
funder-facing: headline percentages, no sample sizes, no test protocol. The two Alpha WP2-D2 decks
are Faculty's own engineering write-ups, and they carry what the Beta report omits — the evaluation
design, the population sizes, the baseline definitions, the runtimes, the negative results and the
methods that lost. Almost every correction below comes from them rather than from Beta.

**They are public, and the "CONFIDENTIAL" stamp on every slide is stale.** Both decks are published
on the ENA Smarter Networks Portal under project NPG_SIF_006, alongside the WP1-M1 user research
report and the DFQM technical review, and download without registration. Open publication of
deliverables is a condition of SIF funding, so the Faculty boilerplate has been overtaken by the
funder. Cite them normally.

**The horizon depends on the voltage level, and only the EHV-HV one overlaps ours.** The Beta scope
is "operationally usable, day-ahead to week-ahead forecasts", half-hourly, and §9.1 reports that
performance "did not materially degrade across the **11-day horizon** on average". But Alpha Scope
Item 2 defines the HV-LV task as **week-ahead to month-ahead peak active power**, which is longer
than our 3–10 day band and is the reason they could not use weather forecasts there at all (below).
So the horizon overlap is at primary substations; at secondary substations they are forecasting a
different problem from us.

#### Their methods, by voltage level

The Alpha registration named Temporal Convolutional Neural Networks as the novel technique, and I
previously wrote that what shipped was far simpler. That was half right, and the half that was wrong
matters. There are two separate model families, one per voltage level:

**EHV-HV gross demand (Scope 1.1), in production at Beta: a stepwise Bayesian Linear Regression**,
forecasting half-hourly settlement periods sequentially and re-using its own predictions as inputs,
which "allows shorter lags and better performance at the start of the week-ahead horizon".

**EHV-HV customer export (Scope 1.2): a per-substation Random Forest**, in three versions — v0 with
one hyper-parameter configuration per fuel type, v1 adding national demand and operational-margin
features, v2 covering multi-fuel substations using the majority fuel type's configuration. The
baseline is an extrapolation of NPg's DFES installed-capacity projection, calibrated by the fraction
of projected capacity actually generated in 2021–22 and linearly interpolated between mid-year
points. By Beta this had become a Bayesian Ridge model with autocorrelation-based clustering, and
**boosted decision trees were tested and rejected**: they "helped some substations but harmed
others", giving "only modest or inconsistent performance gains" while adding "complexity and new
failure modes".

**HV-LV active power (Scope 2): the TCN was built, and it won.** Four approaches were run over all
729 substations — a heuristic baseline (the mean of the last four weeks' values at the same weekly
position), Prophet, a TCN, and a two-stage Hierarchical Bayesian Linear Regression. The TCN took
11.38% Peak MAPE against the baseline's 12.46%, Hierarchical BLR's 11.74% and Prophet's 15.67%, and
was recommended for Beta "potentially with the baseline method as a supplement for substations where
TCN struggled". So the TCN was not abandoned; it is sitting in the unfinished HV-LV workstream, and
whether it survived into production is still open.

**But the TCN's win is thin, and that is the finding worth carrying.** One point of Peak MAPE over a
method with no fitted parameters. On daily-peak error the baseline actually edges it, 11.39% against
11.47%. In the worst of their three data-quality bands the baseline beats it on both daily- and
weekly-peak error. Prophet — the only one of the four with an explicit trend-plus-seasonality
structure — was worst by four points, which Faculty attribute to its hyper-parameters being tuned on
sample substations and then applied to all of them. Their own reading is that HV-LV demand carries
"irregular short-term variations" that reward adaptivity over structure. This is the fourth instance
in this review of a simple model matching a complex one on substation load, after Browell & Fasiolo,
Pinheiro et al. and Beta's own rejection of boosted trees — and the only one where the complex model
wins on the headline number. It wins by an amount no significance test is reported for, which is
exactly the situation recommendation 4 exists to resolve.

**Every model they built is fitted per substation, and what a category shares is only the
configuration.** This is worth stating plainly because it is easy to assume the opposite. In Scope
1.2 the fuel-type group determines the feature list and the hyper-parameters — grid search produces
"one final model configuration per group" — and then "model was then trained per substation". In
Scope 2, "a separate copy of the model type was trained for each time series instance", so there are
729 × 5 separately fitted TCNs, which is what the 74 s per instance buys. Only the Hierarchical BLR
pools anything: its upper layer is trained across a K-means group and its lower layer is "organised
into many subsets trained respectively on the data of each individual substation".

**Their two biggest modelling failures are both this configuration-sharing, not the model class.**
Prophet's hyper-parameters were tuned on a cross-validation set of **10 randomly chosen substations**
and then applied to all 729, and Faculty name that as why it came last: for Hierarchical BLR the
trend-versus-recent-lag balance was "adjusted by model parameters trained specifically for each
substation; whereas for Prophet they were decided by model hyper-parameters tuned on sample
substations and then applied to all substations, which made Prophet the least adaptive method for
irregular short-term variations". Scope 1.2's central complaint is the same mechanism one level
across: fuel-type groups were not homogeneous, so a group-tuned configuration did not fit its
members. Two independent scope items, one failure mode — a category used to choose settings for
substations that do not actually behave alike.

**And their TCN gets no cross-series learning at all.** Because each one sees a single
substation-instance, the usual argument for a neural network on this problem — pooling thousands of
series so the model learns shared structure and transfers to series with little history — is untested
in either report. They gesture at it (their clustering "could potentially support demand forecasting
for new substations with limited history by transferring models developed for substations with
similar load profiles") and called it "large commercial impact if successful" — and it is now a
dated Beta milestone, WP2-M4, due 3 November 2026. So a global model with series identity or
metadata as features is an open question at the most operationally advanced UK project, one we
already have on the roadmap, and one they have committed to answering on a date we know. Their per-instance
TCN beating a four-week average by one point is a weak result for a neural network precisely because
it was denied the thing neural networks are good at.

#### They dropped weather entirely at HV-LV

This is the largest correction the Alpha reports make, and it widens rather than narrows our
differentiator.

Their weather is **OpenWeather forecasts at three point locations** — Darlington, Kingston upon Hull
and Leeds — with each substation mapped to the nearest. At HV-LV the historical forecast archive
reached only 16 days ahead, which cannot support a month-ahead target, so they substituted the
previous year's observations at the same timestamp. That had "trivial or net negative effects on the
performance of every type of models", so **every final HV-LV result in the Alpha report was produced
with no weather features at all**, and the temperature-augmented baseline lost to the plain one.
Their next-steps list asks for "high quality weather forecast data with at least 1-month-ahead
forecast timespan".

At EHV-HV they do use weather, but the same three-city limitation shows. Wind-connected substations
were the worst-performing group, so they bought postcode-level forecasts for two of them,
Meadowfield and Hazlehead — and it "did not notably improve model performance". Their suggested next
step is to try Open-Meteo.

So the differentiator is no longer just "their uncertainty appears to come from the model rather
than from weather ensemble spread". At secondary substations, on the only results they have
produced, there is no weather input at all; at primary substations the weather is three point
locations from a commercial API, deterministic, and their own tests could not make a finer version
of it pay. A 51-member ECMWF ensemble on an H3 grid is a different class of input, and they name the
gap themselves. The fair caveat is that Beta is two years later and does show live weather feeds —
this is a statement about what has been demonstrated, not about what they will eventually have.

#### Two Alpha results that change what we should test

**Generator availability beat wind speed for wind export.** The single most useful feature added at
v1 was NESO's daily operational-margin "generator availability", which was "almost universally
heavily used as a feature in the model and almost universally substantially improved results" for
wind-connected substations, cutting the group's FC-MAPE from 6.82% to 5.53%. Faculty call this
surprising, because v0 already carried a forecast wind speed. The likelier explanation is that a
wind-speed forecast interpolated from three cities is a bad wind feature rather than that a national
availability signal beats NWP — but it is a real result and it qualifies the Beta lesson about
applying "a high bar for integrating new data sources". One national feature was decisive for one
behaviour class. (Price data was rejected at Alpha on cost alone: a Bloomberg quote came to about
$1,000 a month.)

**Fuel type is the wrong way to group substations for export; behaviour is.** This is Scope 1.2's
central negative result. Of 551 substations with export data, 171 had both non-zero export and a
matching Embedded Capacity Register entry; those were grouped into nine fuel and technology classes,
and within-group variation swamped between-group variation. What emerged instead were behaviour
classes that cut across fuel type: **periodic**, **spike** (irregular spikes against a flat, usually
zero, background), **state** (on/off, or low/medium/high), **irregular**, plus **empty** and
**mixed** at v2. Their recommendation is to assign substations to behaviour groups from the data and
model each group differently — spikes and states as classification problems, periodic as an on/off
classifier then a peak-magnitude model, irregular as regression — which also drops the dependency on
the ECR and its naming and mapping problems. Given that our own capacity-estimation and
disaggregation items lean on NGED's ECR, a live DNO finding that ECR-derived fuel type is not a
usable modelling grouping is worth having before we build on it.

#### A bad export model damages net demand less than you would expect

At the four near-capacity substations used to demonstrate net forecasting, customer-export peak MAPE
was 116% (Southgate), 41.3% (Faraday Street) and 50.9% (Thirsk), while the net-demand peak MAPE at
the same sites was 9.72%, 9.51% and 12.5% against gross-only figures of 8.40%, 6.19% and 11.4%. The
export error is enormous in percentage terms and small in magnitude relative to demand, so it adds
1.3, 3.3 and 1.1 points to the net peak error. That is mild support for disaggregation surviving a
weak component model — but it is four substations and five 8-day instances, and it says nothing
about whether modelling net directly would have done better.

#### They do price the decision, in kilowatt-hours

This is the correction that most affects our claimed novelty. I wrote that they score a threshold
crossing with symmetric TPR/FPR and do not calibrate to a risk level or attach prices. The first
half holds — Beta's key metrics include exceedance true and false positive rates, and Alpha 1.2
already used precision and recall for timestamps at or above 90% utilisation of firm capacity. The
second half is too strong. Alpha 1.2 works out the flexibility volume that forecast error causes NPg
to procure:

- Take the net forecast and inflate it by the peak MAE, to cover under-prediction.
- Compute the flexibility required against firm capacity under that inflated forecast.
- Compute the flexibility actually required from the outturn.
- The difference is the extra procurement caused by forecast error.

Worked at Wheatacre Road across the two 8-day instances that exceeded firm capacity: 5,495 kWh with
a perfect forecast, 20,536 kWh with the risk-aware one, so **15,041 kWh of the procurement is
forecast error**. An appendix repeats it under a deliberately more conservative posture — inflate by
the largest under-prediction seen over any over-capacity period (278 kW), procure for the 69
forecast exceedance periods plus the 27 more the margin brings in, and add the over-prediction total
— giving 22,721 kWh of error cost against the same 5,495 kWh of real need.

So they measure the procurement cost of forecast error, and they vary the risk posture explicitly.
What they do not do is calibrate to a stated risk level rather than a hand-picked safety margin,
attach prices to convert kilowatt-hours into pounds, or cost the false-negative side at all — and
they name that last gap themselves: "One key further point to address is the risk associated with
not forecasting a substation to be over capacity when it is." **Our claim narrows from "nobody
prices the decision" to "the pricing that exists is in energy, under an ad-hoc margin, with the miss
cost unquantified".** That is still a defensible claim, and it is now a claim about a specific
published method rather than about an absence.

#### They are publishing openly, including historical accuracy

Their Open Data Portal work will make public: substation-level EHV-HV gross and net demand forecasts
day- and week-ahead; HV-LV net demand forecasts as they become available; exceedance alerts; model
performance metrics and feature-importance summaries at substation level; and **historical
forecasts, alerts and model performance "to support independent validation and research"**. There is
an open API with five endpoint groups, Dublin Core-aligned metadata, and postcode/outcode search. If
this lands, a UK DNO will be publishing exactly the kind of substation-level forecast-and-accuracy
record whose absence this review has been treating as a structural gap.

#### Switching events: handled, but only as contamination

**The "switching events are absent" claim was wrong, and it was wrong because I read only the Beta
report.** Alpha Scope 2 lists "Network configuration changes" as the fifth of five data-quality
issues, notes that NPg "do not keep a distinct and easily accessible record of these but can take
hypothesised load change timestamps to confirm with control room", and builds detection into the ETL
pipeline:

- Mark candidate step-change boundaries where the deviation between the previous and following
  weeks' average magnitude at a timestamp exceeds 50% — around 100 of the 729 substations.
- Confirm a block if its median falls outside the 10th–90th percentiles of the latest block, then
  rescale it onto that block's median — around 50 substations, most with one block rescaled.
- Record the fraction of points rescaled as one of the two primary data-quality metrics, from which
  the three data-quality bands are built.

They state that the EHV-HV version of this in Scope 1.1 is more sophisticated, that step changes of
this magnitude "cannot be directly handled even by powerful nonlinear models like neural networks"
so must be dealt with before training, and that refining it is a Beta priority — while noting it
affects few HV-LV substations.

**What survives is a narrower and better-evidenced claim.** They treat a switching event as
contamination to be normalised out of the training history. There is no attempt to detect a
reconfiguration at forecast time, to forecast through one, or to quantify what one costs a forecast;
§6.3 still lists "incorporation of additional contextual data (e.g. planned outages)" and "use of AF
for outage risk management and planned works" as *future* use cases, and Faculty are asking NPg for a
configuration-change record that does not exist. Our item is
**quantifying the forecast cost of a switching event, and keeping the affected history usable rather
than deleting it from the training set** — which is now a gap named by a live DNO project rather than
inferred from silence. Note what our item is *not*: we are not trying to predict when a cable will
fault. From v1 the aim is to feed recent observations to the model as residuals against a
switching-independent baseline, so a lagged reading taken during an abnormal running arrangement
still carries information instead of being discarded; in v2 the aim is to reconstruct demand under
the substation's normal running arrangement directly. See
[`docs/roadmap/switching-events.md`](../docs/roadmap/switching-events.md).

One Alpha result argues directly for the item: in data-quality band 3, the substations with more
than 10% of points rescaled or imputed, the four-week-average baseline beats the TCN on both daily-
and weekly-peak error. Faculty's own reading is that "the historical data was too noisy and/or
contained significant pattern changes that made longer-term past information unhelpful". That is
switching contamination measurably degrading a model that leans on long history — and it is the
second instance in this review of the field neutralising these events rather than modelling them,
after FeederBW blacklists feeders with topology changes.

#### LV results exist, and they are already public

I wrote that nobody has published substation-level LV forecasting results from this project. That is
simply wrong: Alpha published them in April 2024, on the ENA portal, and the evaluation is far more
serious than the Beta report's "8 out of 8 near-capacity substations":

- **729 HV-LV substations**, every one with 27 months of complete 10-minute data (from up to 873 NPg
  supplied), each contributing five 13-month instances.
- Each instance is 12 months of training, **a deliberate 2-day gap standing in for data acquisition
  and pre-processing in production**, then 28 days of evaluation.
- Hyper-parameters and features were selected on a separate cross-validation set of 10 randomly
  chosen substations, with evaluation months drawn from a disjoint period.
- Results reported for all 729, and again separately for each of the three data-quality bands (626 /
  79 / 24 substations), with min/quartile/max distributions as well as means.

Headline: TCN at 11.38% Peak MAPE, 11.47% daily-peak MAPE, or 2.77% and 3.63% when normalised by
transformer rating instead of by outturn. Not directly comparable to us — different horizon, no
weather, peak-focused resampling — but it is a real number from a real DNO population at our voltage
level, and the normalised-by-rating figures are the ones closest in spirit to our capacity
normalisation. What is still open at LV is the Beta workstream, whose dates are above.

#### Things worth stealing

From Beta:

- **Normalised MAE against firm capacity over a rolling six-week window**, with thresholds that flag
  degraded models automatically. Decision-relevant and cheap.
- **Stepwise iterative forecasting**, re-using predictions as inputs so that early-horizon lags can
  be shorter. Relevant to our lag ablation: it is a third position between raw lags and no lags.
- **Their evaluation framing**: point forecast, plus a 95th-percentile upper bound, plus an
  "adjusted forecast", scored separately, because the three serve different decisions.

From Alpha, and mostly cheaper:

- **The sign-inversion check.** If more than half of a series' readings between 18:00 and 06:00 are
  negative, its sign has been recorded backwards; flip the whole series. Restricting to night-time
  avoids false positives from daytime solar. Seven of 729 substations were caught. This is an hour
  of work and we have metering of unknown provenance.
- **A timestamp-agnostic daily-peak error.** Compare the day's forecast maximum against the day's
  actual maximum without requiring them to fall in the same settlement period, on the explicit
  argument that flexibility procurement does not need the exact timing; plus a weekly-maximum
  variant. This is a cheap member of the peak-metric family Shukla & Hong ask for, and it separates
  cleanly from our timing metrics.
- **A deliberate gap between training and evaluation windows**, sized to production data logistics.
  Their two days is the same concern as our `power_fcst_init_time` / `nwp_init_time` split, applied
  to the power side rather than the weather side. Worth checking our CV splits make the equivalent
  concession.
- **Data-quality bands as a reporting axis.** Two metrics — fraction of points rescaled, fraction
  imputed — three bands, and every results table repeated per band. It makes "the model is fine, the
  data is not" visible without cherry-picking, and it is what turned their band-3 result into
  evidence. Their band thresholds are 5% and 10%, which is where the Beta data-quality trigger comes
  from.
- **Published runtime per model per instance.** Baseline 0.12 s, Hierarchical BLR 0.5–13 s, Prophet
  15.06 s, TCN 74.08 s, all on one AWS c5.4xlarge. Exactly the cost reporting our operational-cost
  item promises, and a reminder that the number is only meaningful with the hardware attached.
- **The flexibility-cost calculation above**, as the thing our priced metric should be an
  improvement on rather than a replacement for. Their two risk postures are a good template; the
  calibrated quantile and the two-price vector are what we add.
- **Behaviour-based grouping instead of asset-type grouping**, and its consequence — that spike and
  state behaviour should be modelled as classification, not regression. Cheaper than our per-series
  capacity estimation and possibly complementary to it.

Two more we should note and not steal. Their **uneven peak-focused downsampling** — twelve 2-hour
slots per day, full resolution kept in the substation's own peak slots and sparse sampling
elsewhere, 48 points a day total, realigned to a uniform grid for modelling and reversed afterwards
— is elegant, but our data is natively half-hourly so there is nothing to spend. And their
**Bayesian hierarchical model with joint training across substations failed on scale**: roughly 12
million training points against a state of the art they put in the hundreds of thousands, forcing
the two-stage sequential fallback. If we ever reach for partial pooling across 2,500 series, that is
the wall.

#### Two practical consequences

**What the flexibility market says it needs, which is not what we are building for.** Their FSP
engagement — ev.energy, Oaktree Power, Piclo, AMP Clean Energy, GridBeyond — converges on three
requirements: **outcode-level** rather than substation-level addressing, because an aggregator's
assets map to postcodes and not to asset IDs; **30-minute granularity**, repeatedly called "the
unlock" for revenue stacking across markets; and **published historical accuracy**, with model UUIDs,
refresh times and bulk downloads, as the precondition for trusting a DNO signal at all. The first is
a presentation problem we could solve cheaply and have not thought about. The third is the same
argument this review makes for publishing an accuracy record, arriving from the market rather than
from academia, which is the more persuasive direction for an NGED audience.

**Their open historical record has a retention caveat worth noting before we cite it as the gold
standard.** §5.4: ODP storage limits mean older datasets get removed from the portal, remaining
reachable through the API, with removals documented in the catalogue. So the "historical forecasts,
alerts and model performance" commitment is real but API-mediated, not a permanent public archive.

**NGED is already in the conversation.** The Beta report states that "engagement with SSEN, SPEN and
NGED during the Beta Phase has confirmed strong interest in short-term substation-level forecasting
and in using NPg's Open Data Portal and API specifications as a blueprint for interoperable forecast
sharing", with further engagements scheduled once the full product is deployed. Faculty's foreground
IPR is available to other DNOs **royalty-free**, with charges only for integration and support. So
the NGED document should assume its readers may already know this project, and possibly be
considering adopting it.

**Their white paper lands November 2026** (WP1-M6/D6: "details on data, ETL, modelling and
results"), before our ML results are due. Worth watching, and worth citing rather than being
scooped by.

#### Source-quality note

The three sources are of very different quality and should be weighted accordingly. The **Alpha
WP2-D2 decks** are the best of them: real test protocols, stated population sizes, negative results
reported alongside positive ones, and per-substation metric spreadsheets referenced (those are on a
Faculty SharePoint we cannot reach, unlike the decks themselves). Their own caveats are honest —
Faculty describe the export models as "a respectable baseline on which various improvements can be
made rather than a final product", warn
that FC-MAPE divides by firm capacity and so is comparable only within a fuel group, and note that
the v2 table covers different substations from v0 and v1 and is therefore not comparable with them.
The weakest part is the net-forecasting and flexibility-cost work, which rests on four substations.

The **Beta progress report** is funder-facing, and its performance figures carry no confidence
intervals, sample sizes or described test protocol; read them as claims. It also contradicts itself
on which winter the tool was used: §1.3, §2.6, §9, §9.1 and §12.1 say **Winter 2025-26**; §1.5,
§2.2, §2.3, §2.4, §3.2 and §6.1 say **Winter 2024-25**. Since Beta began in February 2025, only 2025-26 is possible. Treat the
2024-25 references as errors.

The **registration forms** turned out to be the least reliable of the three. Three of the seven
claims this section has had to correct — the horizon, the modelling technique and the absence of
weather — came from reading them as though they described what would be built.

**All of it is citable.** The Alpha deliverables sit on the ENA Smarter Networks Portal under project
NPG_SIF_006 and download without registration; the Beta registration is under 10145998. The only
material we cannot reach is the per-substation metric spreadsheets the decks link to, which are on a
Faculty SharePoint. An eighth claim this section had to correct was my own: I read the "CONFIDENTIAL"
stamp on the Alpha slides as a restriction without checking whether the funder had published them.
Open publication of deliverables is a condition of SIF funding, and they had.

### 10b. Installed PV capacity detection on LV substations

*Comparison of Data-Driven and Model-Based methods.* International Journal of Electrical Power &
Energy Systems, 2026 (doi 10.1016/j.ijepes.2026.111848; ScienceDirect S0142061526002905). **Abstract
and preview only — not read in full.**

Included because it is a direct benchmark of the exact problem our capacity-estimation item poses:
estimating installed PV capacity aggregated at the **LV substation level**, using only net load
measured at the substation plus indirect irradiance — the data a DSO actually has. It benchmarks
model-based against data-driven methods, proposes two new model-based methods exploiting the linear
relationship between net load and irradiance, and — most usefully — runs a sensitivity analysis over
PV penetration, aggregation level and data quality, which is precisely the question of *when* each
method stops working. It also frames capacity detection as an intermediate task on the way to
disaggregating the PV profile, which is our sequencing too.

The linear net-load-versus-irradiance relationship they exploit is the same algebra as our
weather-delta feature. Worth reading properly before building anything.

### 10c. Bouman, Schmeitz, Buise, Heres, Shapovalova & Heskes — switch event detection at a real DSO

*Acquiring Better Load Estimates by Combining Anomaly and Change-point Detection in Power Grid
Time-series Measurements.* Sustainable Energy, Grids and Networks; arXiv:2405.16164. Radboud
University with **Alliander N.V.**, a Dutch DSO. **Read in full** (abstract, introduction, data and
methods; not the full results tables).

**The setup.** 180 **primary** substation load measurements, most spanning a full year at 15-minute
resolution — so roughly 35,000 points per series — with average loads ranging from hundreds to tens of
thousands of kW. The target is apparent power with a sign: S = sign(P)·√(P² + Q²) where both are
measured, and S = √3·V·I where they are not. Every point is expert-labelled 0 (normal), 1 (anomaly or
switch event), or **5 (the labeller is unsure whether it is 0 or 1)** — an honest annotation design
worth copying. Their purpose is not forecasting but **capacity planning**: anomalies and switch events
inflate or deflate the annual minimum and maximum, which distorts the estimate of unused capacity and
hence where to reinforce.

**Their switch event is exactly ours.** A cable fault or planned maintenance reroutes part of a
subgrid to a different primary, producing a step up in apparent power at one substation and a step
down at the other. They note the duration range explicitly: **from a few minutes to multiple months**,
depending on how fast the underlying problem is resolved.

**Four things here are directly usable, and none of them is the headline result.**

**1. They detect on a residual, not on the raw series.** Alliander maintains a *bottom-up* load
estimate for each substation — a reconstruction from bulk-consumer telemetry, aggregated smaller
measurements, and modelled profiles for consumers without smart meters or without consent. They then
fit a linear model s = m·b + c on a **quantile-trimmed subset** of the series, so that anomalies and
switch events cannot steer the fit, rescale the bottom-up accordingly, and run detection on the
difference vector δ = s − b. That is a much stronger signal than the raw load, because normal
seasonal and diurnal variation is largely cancelled by the reference. We have no bottom-up estimate,
but we have something adjacent — our own forecast, and eventually our disaggregated components — and
the principle that **the detector should run on a residual against an independent reconstruction, not
on the load itself**, is the single most transferable idea in the paper.

**2. They solve our MVA problem, and the solution is cheap.** Some Alliander substations measure only
absolute current, so the sign is missing — the same defect as our ten MVA-metered trial sites, where
reverse flow appears as a rise rather than a sign change. Their fix: the bottom-up estimate is built
from P measurements and therefore **always** carries a sign, so where the measured series has a
non-negative minimum but the scaled bottom-up goes negative, they take the sign from the bottom-up.
Any independently-signed reference would do the same job. This is worth an hour of thought before we
write those ten sites off.

**3. Event-length stratification, because pooled metrics would be meaningless.** Anomalies are short
and frequent; switch events are long and rare but account for the majority of flagged data. Pooling
per-timepoint metrics would let the long events dominate. So they bucket events into four length
categories — **15 minutes to 6 hours, 6 hours to 3 days, 3 to 42 days, and 42 days or longer** — and
compute every metric within each. Our own event-detection evaluation will have exactly this problem,
and this is a ready-made answer.

**4. Their bottom-up failures are topology failures.** They note that when the bottom-up estimate is
wrong, the cause is usually not the algorithm but **incorrect grid-topology data, wrongly including or
excluding consumers**. That is a warning about the metadata underpinning any disaggregation work,
ours included.

**The method and result.** A sequential ensemble of **binary segmentation** for change-point detection
and **statistical process control** for anomaly detection, selected by comparison and built from
unsupervised methods with supervised hyper-parameter optimisation, explicitly because interpretability
matters when reinforcement decisions rest on the output. Roughly 90% of resulting load estimates fall
within a 10% error margin, with a single significant failure across 60 measurements in the test set.
They also report the "clear wasted potential when filtering is not applied" — what it costs to leave
switch events in.

**Prior art for our detector, and a claim of novelty worth noting.** Binary segmentation is cheap and
well understood; if it works at Alliander it is the obvious first thing to try. Their own claimed gap
is narrow and precise: for series of a year or longer where event lengths vary substantially, they
know of no prior study of load estimation through automated segmentation. They also cite Thomas et al.
(2020), who detect load switch events with *k*-means plus empirical mode decomposition, as the nearest
existing method. Their framing is filtering-for-capacity-planning rather than feeding a detection
forward into a demand forecast, which remains genuinely open — but the detection half of our problem
has been solved by a DSO, and we should start from their answer rather than a blank page.

### What was searched and rejected

For auditability, since the value of a selective review depends on knowing what it excluded:

- **Behind-the-meter PV disaggregation** is a large and active literature — Bayesian dictionary
  learning, mixed hidden Markov models, contrastive optimisation, Bayesian structural time series,
  and more, mostly on US smart-meter data at customer level. Rejected as a body: it is a different
  aggregation level from ours, and adding three or four entries would have doubled the review without
  changing a recommendation. Anchor citations if needed: Mahdavi, Weeraddana & Guo (IEEE TSG 14(3),
  2023) for probabilistic PV estimation at customer *and feeder* level, and the NREL Bayesian
  structural time series report for a probabilistic treatment.
- **Distribution network topology detection** from PMU or high-resolution measurements exists
  (Cavraro, Arghandeh & von Meier and successors), but assumes synchrophasor-grade data that is not
  available to this project, whatever NGED holds internally. One datum worth keeping from it, with a
  caveat: a utility survey put switching actions at **five to ten per urban distribution
  substation**, but Cavraro et al. state **no time period**, and the later journal version of the
  same work phrases it as how many *switches* a substation has. Do not quote it as an annual rate —
  we have no dependable external estimate of the event rate.
- **Concept-drift detection in load forecasting** generally — PELT-driven drift detection,
  detect-then-adapt, drift-adaptive LSTMs — is a large generic literature. Rejected: it is about
  gradual statistical drift rather than topology step changes, and §8's adaptive-methods discussion
  already carries the argument.
- **Differentiable physics applied to energy forecasting** produced no strong hit. There is plenty on
  physics-informed neural networks for power systems generally, but nothing I found that applies
  differentiable physical models to substation demand forecasting in a way that would change a
  recommendation here. Either the search terms were wrong or the intersection is genuinely thin;
  worth someone else's search before concluding the latter.

---

## Cross-cutting: what nobody in this set measures

Collecting the gaps. Each is narrowed to what the sources actually support, since the tempting
stronger version of several of them is not defensible.

**Almost nobody evaluates beyond four days ahead, and the exception is instructive.** The KIT papers
stop at four days; HEFTCom, BigDEAL's final match and Browell & Fasiolo are day-ahead. GEFCom2017's
qualifying match is the outlier at two to six weeks, which brackets our band from above rather than
matching it. The Haben review's own count puts the LV literature at sixteen papers between two days
and a week and thirteen at a month or more, out of 221. The closest architectural antecedent is
Taylor & Buizza (2002), *Neural network load forecasting with weather ensemble predictions*, IEEE
Transactions on Power Systems 17(3), 626–632, with a companion in the International Journal of
Forecasting the following year: they pushed all 51 ECMWF ensemble members through a neural network
to generate 51 load scenarios at one to ten days ahead, and found the scenario mean beat substituting
a single point weather forecast. That is our architecture, twenty-four years old. I have read the
abstract and the closing sections, not the full paper.

**Nobody at LV level uses ensemble NWP, and this is now a counted fact rather than an impression —
with one gap in the count.** The Haben review found three of 221 LV papers using weather *forecasts*
at all, and states that those three use point estimates rather than ensembles, thereby ignoring the
underlying uncertainty. That count closed in August 2020, and at least one paper published since —
Pinheiro et al. (§9), on 96,989 Portuguese secondary substations — is a fourth, using gridded
deterministic NWP with a stated 7–8 hour publication delay. Its inputs are point forecasts, not
ensembles, so the ensemble claim survives it; the claim should be made as "as of the field's own 2021
review, and still true of the largest deployment since" rather than as an unqualified statement about
today. Notably, Pinheiro et al. name this gap themselves, writing that the significant advances in NWP
have not yet been translated into improved LV load forecasts. Above
LV, Browell & Fasiolo use gridded ECMWF-HRES — deterministic. HEFTCom supplied deterministic ICON-EU
and GFS. The KIT papers use short-range forecasts or ERA5 reanalysis. GEFCom and BigDEAL use
temperature only, with GEFCom's month-ahead horizon forcing teams into temperature *scenarios* built
from history rather than from an ensemble. The decomposition our probabilistic design rests on —
weather uncertainty carried by member spread, model uncertainty by the conditional distribution — does
not appear anywhere in this literature.

**Several papers use weather that would not have been available at forecast time, and say so.** Kaas
et al. classify their own results as ex-post; Hertel et al. state their errors may be unrealistically
low; the Haben review says the same of the majority of the 221 papers it reviewed. Browell & Fasiolo
are the exception, using genuine day-ahead HRES forecasts. This is not a criticism of any individual
paper — all are explicit — but it means published absolute numbers are not a bar we should expect to
clear.

**Reconfiguration is acknowledged, filtered out, and not modelled.** The Haben review notes in
passing that LV connections may be reconfigured over time. The FeederBW dataset paper goes further:
it states that switching events altered topology over the two years, that they are undocumented in
the data, that they cause concept drift, and that feeders with topology changes were blacklisted from
the published set. Gilbert, Browell & Stephen name adaptive handling of structural breaks as future
work. So the phenomenon is recognised across the literature. **It is also detected: Bouman et
al. (§10c), working with the Dutch DSO Alliander, detect switch events with binary segmentation plus
statistical process control and use them to filter load estimates.** What remains genuinely open is
the forecasting side — no paper in this set feeds a detected switching event forward into a demand
forecast or evaluates the forecast cost of missing one, and the one dataset built for LV benchmarking
removes affected feeders by construction. A utility survey cited in the topology-detection
literature reports five to ten switching actions at an urban distribution substation, but states no
time period, so it does not give us an event rate.

**The community's standardisation effort scores distributions, not decisions.** Energy-Arena's 24
live challenges use RMSE, the Winkler interval score and CRPS — all whole-distribution or
whole-series scores. None asks whether a network limit was crossed; none can express that
over-procurement and under-procurement cost different amounts. So the comparability problem is being
solved while the decision-relevance problem is not.

**Three papers score something other than average error, and none combines threshold, risk and
price.** Pinheiro et al. adopt Haben's adjusted *p*-norm error at substation level precisely because
standard metrics reward smoothness while a DSO cares about peaks — the only deployed system in this
review to make that choice — but it is a point-forecast metric with no uncertainty in it. Kaas et al.'s fuse-derived F1
is built from the operator's action, but weights over- and under-reaction equally and is evaluated at
a fixed nominal quantile without risk calibration. Browell & Fasiolo's reserve evaluation fixes the
risk level and compares volumes, which is structurally what our cost metric does, but at transmission
scale and without prices. Nobody in this set combines a network threshold, a calibrated risk target
and a price vector, which is a fair statement of where our contribution sits.

**And the result to build the argument around, stated correctly:** in Kaas et al., the two models
that win the p95 overload metric are the two worst-calibrated in the study, at 62% and 58% empirical
coverage against a nominal 90%. A threshold metric evaluated at a fixed nominal quantile, with
symmetric error costs and no risk calibration, is winnable by over-confidence. Browell & Fasiolo
supply the complementary warning from the other direction: per-quantile pinball in the tail is too
noisy to discriminate between systems. Between them, the two papers rule out both of the obvious
ways of scoring a tail, which is a strong argument for the calibrate-then-compare structure.

---

## Counter-evidence: findings that cut against this project's assumptions

Every paper above is read, in its own section, as supporting something on the Flexpectation roadmap.
That should be suspicious, and it is partly an artefact of how these notes were assembled — I went
looking for relevance. Collected here are the findings that point the other way, so that the NGED
document can meet them rather than route around them. Most of them come from papers read in full and are
invisible in the corresponding abstracts, which is a caution about how much second-hand reading was
flattering us.

**Gridded NWP statistics added nothing.** Browell & Fasiolo compared GAM-Point against GAM-Grid across
14 GB regions with Diebold–Mariano tests: significantly better in two, significantly *worse* in three,
indistinguishable in nine. Their conclusion is that gridded NWP does not appear to add significant
value in their framework. Our neighbouring-H3-cell weather context item, and to a lesser extent the
ensemble-statistics-as-features fork, now carry a negative prior. The available rebuttal — that a GAM
with splines and a gradient booster with axis-parallel splits may extract different things from the
same spatial statistics — is plausible and untested.

**A thoroughly-tuned gradient booster lost to a GAM.** Pinheiro et al. fitted XGBoost on identical
features to their GAM, with an exhaustive hyper-parameter grid search, polynomial temperature terms,
one-hot encoding, scaling, and three families of base learner. It reached 199 MW RMSE against the GAM's 191 MW, and they rejected it on tuning cost and interpretability, saying it achieved the same accuracy. Our entire baseline is XGBoost. The
rebuttal is that their target is smooth national aggregate load where spline structure fits
naturally, and ours is volatile net demand at ten megawatts — but that is a hypothesis, and a GAM arm
is cheap.

**The LV benchmark population has been filtered to remove our hardest case.** FeederBW's selection
criteria exclude meshed feeders, feeders below 90% data completeness, feeders with rare producer types,
low-utilisation feeders, and — via an individual data check — feeders with "uncorrectable data quality
issues, such as topology changes". The authors acknowledge the resulting bias. So the model rankings
in both KIT papers, including "foundation models beat XGBoost", are rankings on cleaned, radial,
well-instrumented feeders with reconfigured ones partly removed. That does not invalidate their
results, but it does mean a result obtained there should not be assumed to transfer to a population
that still contains switching events.

**On the largest real deployment, a third of client-owned substations do not beat persistence.**
Pinheiro et al. report MASE against a 24-hour-naive forecast: roughly 83–87% of DSO-owned substations
beat it, but only 66–70% of client-owned single-customer sites do. Those are the closest analogue in
their population to the small, spiky end of our 2,500-series scale-up.

**Lagged load is the one design choice where two full-text papers directly contradict each other.**
Browell & Fasiolo found autoregressive terms *detrimental* on GB regional net load, attributed it to
embedded generation polluting the seasonal cycles, and replaced them with a two-week rolling mean.
Pinheiro et al. found 24-hour and 1-week lags *strongly helpful* on Portuguese national load, cutting
MAPE from 4.09% to 2.53% — and they arrived at those lags deliberately, by running the
autocorrelation and partial autocorrelation functions on the *residuals* of the lag-free model rather
than on the load series, precisely to check that the lags added something the calendar and weather
terms had not already captured. That is a cleaner test than either of us would get from a bare
ablation, and it is worth copying.

Power lags are central to our feature set, so the contradiction matters. The plausible reconciliation
is the target: Browell & Fasiolo forecast **net** load with substantial embedded wind and solar, while
Pinheiro et al. forecast **gross** demand with no embedded generation in the model at all. If that is
the mechanism, we are on the Browell & Fasiolo side of the line — our targets are net demand — and it
would also predict that the harm scales with embedded capacity, which is testable against our
per-series capacity estimates.

**The gap worth claiming: nobody has tried the obvious middle option.** Browell & Fasiolo *diagnosed*
the mechanism — embedded generation polluting the seasonal cycles a lag would otherwise carry — and
then responded by deleting the lags and substituting a two-week rolling mean. What they did not try
is telling the model what the weather was *at the lagged time*, so that it can judge how normal that
lagged observation was. Our XGBoost roadmap has that as an explicit ablation ladder: (a) aligned
lagged weather, a config-only change feeding the weather at each lag's timestamp beside the lag
itself; (b) weather-delta compensation, a single precomputed column giving the change in a linearised
generation proxy between the lagged time and the target time; and (c) full two-stage residual lags
against a fitted baseline.

Rung (b) is the one that maps onto their diagnosis exactly. A substation with unmetered PV or wind
meters roughly `demand − C·cf(weather)`, so the correction mapping a lagged observation onto target
conditions is `≈ −C·(cf_valid − cf_lag)` — **linear in the capacity-factor delta**, with a per-series
constant scale the booster discovers as an ordinary split relationship without ever needing to know
`C`. That is a difference of two continuous inputs, which is exactly the structure trees are bad at
building for themselves and exactly what a precomputed column supplies. If the mechanism is what
Browell & Fasiolo say it is, rung (b) should recover the value that made them abandon lags.

Two honest caveats belong with the claim. Their model class partly explains the omission: in a GAM a
lagged-weather term enters additively and the comparison needed is a difference, so the natural fix is
less available than it is for a booster with a derived column — they had bivariate smooths available
and could in principle have tried it, but it is not the obvious move from where they sat. And the
harder limit is horizon: aligned weather and the delta only carry signal where the power lag itself
survives nullification, so most of the benefit lands at day 0–2, outside our headline 3–10 day band.
Carrying it into that band means the init-time-anchored variant, which is a separate experiment.

Both results are also day-ahead, where a 24-hour lag is available and informative. This is a genuine
open question rather than a warning, and the ablation should be run with Pinheiro et al.'s
residual-ACF diagnostic alongside it.

**Temperature has repeatedly been found less useful than expected.** Haben et al. (2019) on 100
real LV feeders found no effect or a negative effect, for point and probabilistic forecasts, with
both forecast and actual temperature; the Haben review reports two further papers agreeing and one
disagreeing. Independently, in BFCom2018 five of thirteen reporting teams used no temperature at all
— including the 3rd and 4th ranked — treating peak-hour probability as a calendar-only classification
problem, and scored close to the temperature-scenario teams; the organisers concluded a model without
temperature can forecast next year's daily peak hours nearly as well as one with it. Gilbert, Browell
& Stephen had no weather at all and still produced calibrated forecasts across four aggregation
levels. That is four independent settings in which weather bought less than expected. See §7 for why our horizon and our PV exposure may differ, and why
this remains an open question on our data rather than a settled one.

**Diminishing headroom with disaggregation, twice over.** Hertel et al.'s improvement over a naive
baseline falls from 59.6% at TSO level to 42.3% at LV feeder level to 23.3% at client level. The
Haben review's power law says the same thing more sharply: relative error scales as a power law in
feeder size, so smaller feeders are exponentially harder. Our population sits at the disaggregated
end and moves further there at v2.

**Zero-shot models are awkward for the experiment-throughput thesis.** The intro paragraph argues
success is largely a function of how many ideas you test. Two 2026 papers find a model trained on
none of the data is competitive with, and on LV feeders better than, tuned dataset-specific models.
The rebuttals are real — lightly-featurised tree baselines, ex-post weather, inference cost that grows
with series count, failure on special events — but they are rebuttals, and a reader familiar with the
foundation-model literature will raise the objection unprompted.

**Two winning entries won by being simple, not by searching hard.** HEFTCom's SVK tuned exactly one
hyper-parameter and left the rest at defaults; what won was structure. GEFCom2017's Ziel placed second
in the open track with a quantile regression that used temperature only to stabilise a trend component
and ignored public holidays entirely. The honest synthesis is that experiments are how you find the
structure, not a substitute for having one — which is a defensible position, but not the position the
intro currently takes.

**A better-funded UK project is already doing several of the things we claim as novel.** Northern
Powergrid's Artificial Forecasting (§10a) — £3.9m across three SIF phases, Beta running to early 2027
— covers primary and secondary substations, models gross demand and distributed generation separately
rather than treating net demand as one series, and has an LV probabilistic workstream. Their Beta
report shows they use weather, produce 5th–95th intervals, score exceedance with true/false positive
rates, and are preparing to publish substation-level forecasts *and* historical accuracy openly. Their
Alpha technical reports add three more: they detect and normalise switching-induced step changes in
ETL, they compute the flexibility volume that forecast error causes NPg to procure, and they have
already evaluated four model families across 729 secondary substations. What they still do not have is
ensemble NWP — at HV-LV, no weather input at all — tails beyond p95, a risk-calibrated metric priced
in pounds, any use of the network's connectivity map to forecast a point top-down and bottom-up,
any estimate of *unmetered* generation behind the net flow, any tracking of metered generators'
effective capacity as it changes, or any treatment of switching contamination beyond data cleaning.
Any claim of novelty on disaggregation, on substation-level probabilistic forecasting, or on open
publication needs qualifying in front of a DNO audience — and NGED has already been engaged by them.

**And our own leaderboard reproduces the structure we are criticising.** Private data, self-chosen
metrics, self-reported results. The opening argument of the NGED document applies to us until we
publish the protocol or anchor a result on open data. This is the objection most likely to come from a
CIRED reviewer, and it should be answered in the document rather than left for them.

---

## How Flexpectation could contribute to this literature

Several of these papers end by naming what they could not do. Read together, the list of stated open
questions overlaps our roadmap more closely than any of us would have guessed, and answering someone
else's explicitly-posed question is a far stronger contribution claim than asserting a gap ourselves.
This section collects the asks, matches them to what we could plausibly deliver, and — importantly —
says where we cannot help.

| Paper | What they ask for | What we could contribute | Confidence |
|---|---|---|---|
| Haben et al. 2021 | Probabilistic methods, peak forecasting and weather information at LV; more than a handful of real substations | Probabilistic net-demand forecasts with NWP inputs on real network assets — all three at once, which no paper here does | High on the combination; **the "more than a handful" part is only satisfied after the scale-up**, since 32 series is a handful and the population is a mix of primaries, generation sites and batteries rather than 2,500 substations |
| Haben et al. 2021 | The double-penalty effect is established for households but "has not been investigated" for small feeders | Test whether the power law and the double-penalty effect hold across our size range | **Medium, and note the direction**: their open question is about series *smaller* than households-in-aggregate, whereas ours sit at the larger end. We would extend the evidence upward, not answer their question |
| Haben et al. 2021 | Error measures that follow the downstream optimisation objective; "more studies are needed in these specific downstream applications" | A priced, risk-calibrated flexibility-procurement metric — a named downstream application | High on the method; **conditional on NGED supplying availability, utilisation and curtailment prices**, which are open questions on our cost-savings page |
| Pinheiro et al. 2023 | NWP advances "have not yet been translated into improved LV load forecasts" | Quantify what NWP buys at LV, by horizon slice and by embedded capacity | High |
| Pinheiro et al. 2023 | Computationally cheap, explainable models able to forecast thousands of LV assets in useful time | Published runtime and cost figures per series per model class, at 2,500 series on modest hardware | High — we will measure this anyway |
| Browell & Fasiolo 2021 | Gridded NWP added nothing in their GAM framework, but "other forecasting methods would be able to extract value from this data by constructing different features" | Test the same question with a gradient booster and H3-derived spatial features — a direct answer to a stated open question | High |
| Browell & Fasiolo 2021 | Predicting extremes by "decomposing net-load into its constituent parts", which "would require careful modelling of potential tail dependency structures" | Our disaggregation work is exactly that decomposition; if we get anywhere with tails on the parts, it answers their closing paragraph | Medium — depends on disaggregation landing |
| Browell & Fasiolo 2021 (implicit) | They diagnosed embedded generation as the reason lags hurt, then deleted the lags | The aligned-weather and weather-delta rungs test the remedy they did not try | Medium-high — cheap to run, clean result either way |
| Artificial Forecasting (§10a) | They detect switching-induced step changes and rescale them out of the training history, state that such steps "cannot be directly handled even by powerful nonlinear models like neural networks", and are asking NPg for a network-reconfiguration record that does not exist; nothing detects or forecasts through an event at forecast time | Handling a switching event at forecast time rather than deleting it from the training set, and quantifying what one costs a forecast | High that we can address it; the comparison is with a live production system, and their own band-3 result — where a four-week-average baseline beats their TCN on the substations with most rescaling — is evidence the problem is real |
| Artificial Forecasting (§10a) | Their flexibility-cost calculation inflates the forecast by a hand-picked margin (peak MAE, or the largest observed under-prediction), reports the result in kWh, and leaves the miss cost unquantified — "the risk associated with not forecasting a substation to be over capacity when it is" is named as an open point | Calibrating to a stated risk level instead of a margin, converting kWh to pounds with a two-price vector, and costing the false-negative side | High; their method is a published template to improve on rather than an absence to fill, which is a stronger paper either way |
| Gilbert, Browell & Stephen 2023 | Their closing future work names embedded generation, global models transferable to unseen feeders, models adaptive to structural breaks, exploiting hierarchy, and improving the right tail | Four or five roadmap items named as open problems by the authors themselves — disaggregation, the global model, switching-event handling, and tail modelling | High that we address them; medium that we will have results worth publishing by 2027 |
| Shukla & Hong 2024 | "The operational requirements of the peak forecasting problem have not been formally studied"; new metrics should adapt "existing error metrics from binary event detection" and be "practical and simple to compute and communicate" | A threshold-crossing metric priced in pounds, calibrated to a common risk level — binary event detection with an operational cost attached, communicable to a board | High; this is the closest thing in the review to a printed request for the metric we are building |
| Shukla & Hong 2024 | "None of the teams reported using track-specific features in their models, which could be a research opportunity" | Per-`time_series_type` feature lists are the same idea on a different axis; per-horizon-window feature sets would be closer still | Medium — cheap to run, but our axis is series type rather than peak attribute |
| Treutlein et al. 2026 (implicit) | FeederBW blacklists feeders with "uncorrectable data quality issues, such as topology changes", so the LV benchmark population has reconfigured feeders partly removed | Report results on a population that *includes* switching events, and quantify what they cost a **forecast** — Bouman et al. (§10c) detect such events for load estimation, but nobody has priced their forecasting impact, and the reference LV dataset excludes them | Medium-high; the data is ours either way, so this costs only the willingness to report the ugly slice |
| Browell et al. 2025 (HEFTCom) | Intraday and medium-term, days-to-weeks horizons "have received relatively little attention" and would benefit from competition-style scrutiny | A 3–10 day operational band with ensemble NWP, evaluated live | High |
| Browell et al. 2025 | Robustness to missing data and unexpected events is "critical in forecasting practice but often overlooked by academic studies" | Published incident log, publication-time availability cuts, forecast behaviour through NWP outages and switching events | High — the live service generates this as a by-product |
| Hertel et al. 2026 | Extend their benchmark to probabilistic models; integrate results into a rolling platform | Probabilistic results on a comparable task shape — but genuinely answering them means running our model family on *their* data and splits, which is a separate piece of work we have not costed | Low-medium |
| Kaas et al. 2026 | Peak-focused loss and metric design is "a viable research topic"; their own metric is unpriced and uncalibrated | Risk-calibrated, two-price extension of their fuse-style confusion matrix | High |
| Kleinebrahm et al. 2026 (Energy-Arena) | Open, forward-looking benchmarking with ex-ante submission; challenges can be proposed through a public review queue | We cannot submit — every target is a national or zonal ENTSO-E aggregate, so the platform does not host our problem. What we can do is publish the protocol and metric code, and argue for a distribution-level challenge through the proposal workflow | Low on submission; medium on the proposal route, which is blocked on somebody making suitable data public |
| Hong, Xie & Black 2019 | Their own list of candidate future competition topics includes net load forecasting with rooftop solar **[from a snippet of the paywalled paper]** | Net demand with unmetered embedded PV is our target, at substation rather than zonal level | Medium |

**Where we cannot help, and should say so.** Publishing the underlying NGED data is NGED's decision
rather than ours, so we cannot ourselves promise the open real-substation dataset the field most
needs — Pinheiro et al. could not either. What we can commit to unilaterally is the evaluation
protocol, the metric code and the leaderboard. Our trial area is 32 series, which is a case study rather than a population until the
scale-up lands. Ten trial-area sites are metered in MVA, so reverse flow appears as a rise
rather than a sign change — though Bouman et al. (§10c) recover the sign at Alliander from an
independently-signed reference series, which may mean those ten are not lost after all. And we are one project with one person on the ML, so the breadth of a competition
or a systematic benchmark is not available to us.

**The three claims I would actually make in the NGED document**, in decreasing order of how safely
they can be made now:

1. **We are measuring the combination the field has not measured** — ensemble NWP, 3–10 day horizon,
   probabilistic, real substations, net demand with embedded generation. Each element exists
   somewhere; the intersection does not, and the Haben count plus Pinheiro et al.'s own stated gap
   are the evidence. **State this carefully in front of an NGED audience**: Northern Powergrid's
   Artificial Forecasting (§10a) is running an 11-day-horizon probabilistic substation forecast in
   live operational use, with weather inputs, exceedance detection and demand/export disaggregation at
   EHV-HV — and NGED has already engaged with them. The horizon claim in particular no longer
   separates us. What remains distinctively ours, after the Alpha reports narrowed three of these:
   **ensemble NWP as the source of uncertainty** — the strongest of the five, since their weather is
   three point locations from a commercial API and their published HV-LV results use none at all;
   **tails beyond p95**; **a metric calibrated to a risk level and priced in pounds**, theirs being a
   hand-picked margin measured in kWh with the miss cost unquantified; **switching events at forecast
   time**, theirs being normalised out of the training history in ETL; and **disaggregation at
   secondary rather than primary substations**. Two of these are now claims about the limits of a
   published method rather than about an absence, which is a better position to argue from.
2. **We are scoring the decision, with risk held constant.** Others score decisions too — Browell &
   Fasiolo in reserve volume at transmission scale, Kaas et al. in an unpriced and uncalibrated form
   at LV, Pinheiro et al. with a peak-aware error at substation level. What none of them combines is
   a network threshold, a calibrated risk target and a two-price cost vector at distribution level.
3. **We are reporting what operating a forecast actually costs** — runtimes per series, failure
   modes, intervention counts, behaviour through outages. HEFTCom names this as the thing academic
   studies overlook, and a live service produces it for free.

The first is defensible today. The second is defensible once the cost metric ships. The third accrues
automatically and is the one least likely to be undercut by an experiment going the wrong way — worth
remembering when choosing what to promise in a CIRED abstract written before the ML results exist.

---

## Conclusions for Flexpectation

**The literature does not identify a state of the art, and the cleanest demonstration is close to
hand.** Two papers published within a fortnight of each other in July 2026, by
overlapping author groups at the same institute, using the same 200 low-voltage feeders, reach
different conclusions about which model is best — because one scores MAE deterministically and the
other scores probabilistic quantiles and an overload decision. That makes the point about
incommensurable evaluation better than any assertion about incommensurable datasets, and it does so
in the strongest possible case, where dataset, institution and publication date are all held constant.

**The single most quotable statistic in the review is the Haben count.** Of 221 low-voltage load
forecasting papers, three used weather forecasts rather than observations, and none used weather
ensemble predictions. Sixteen forecast between two days and a week ahead; thirteen at a month or more.
That is the field's own review saying, in 2021, that the combination Flexpectation is built on —
ensemble NWP, days-to-weeks horizon, distribution level — was essentially unoccupied. It is stronger
evidence than anything I could assemble by argument, and it should probably be the factual spine of
the NGED document's opening — but cited as of 2021, since the survey closed in August 2020 and
Pinheiro et al. (§9) is a fourth NWP-using paper published after it. That paper does not weaken the
ensemble claim, since its inputs are deterministic point forecasts, and it names the same gap itself:
the significant advances in NWP have not yet been translated into improved LV load forecasts.

**Two papers between them rule out both obvious ways of scoring a tail, which is why the
calibrate-then-compare structure matters.** Kaas et al. show that a threshold metric at a fixed
nominal quantile with symmetric costs is winnable by under-dispersion: the two models topping their
p95 overload F1 have empirical coverages of 62% and 58% against a nominal 90%. Browell & Fasiolo show
that the alternative — per-quantile pinball in the tail — has variance too high to discriminate
between forecasting systems, because the observations are sparse by construction. Our cost metric
calibrates each model's procurement quantile to a common unmet fraction before comparing spend and
reports the realised fraction alongside; that is the structure both failure modes point towards, and
it is a more specific contribution claim than "we score the decision".

**The decision-metric ground is not unoccupied, and we should say so.** Browell & Fasiolo fix a risk
appetite, compute the reserve volume each forecast would need to hold it, and compare — at
transmission scale, in megawatt-hours rather than pounds. Pinheiro et al. independently adopt a
peak-aware error measure at substation level because standard metrics reward smoothness. What nobody
has combined is a network threshold, a calibrated risk target and a two-price cost vector at
distribution level, where the series are small enough that tail estimation is materially harder. One
refinement to copy directly: Browell & Fasiolo report *when* the volumes differ, not just how much —
the conditional model needs more upward reserve in 19–44% of periods, which is exactly where the
benchmarks were silently under-procuring.

**Substation-level forecasting is not unstudied, and the NGED document should not say it is.**
Pinheiro et al. run a production system over 96,989 Portuguese secondary substations with NWP inputs,
peak-aware metrics, forward-chained cross-validation, and applicability and interpretability treated
as first-class criteria. The accurate claim is narrower and still ours: substation-level
*probabilistic* forecasting, at days-to-weeks horizons, on net demand with embedded generation, is
unstudied. Stating our gaps against a named production system is a stronger move than stating them
against a vacuum, and it removes the easiest way for a reviewer to embarrass the document. The
section above sets out the full list of stated open questions we could answer.

**The strongest single argument for peak-specific metrics is now Gilbert, Browell & Stephen's table.**
Averaged over all periods, their fused forecast beats the advanced model by 0.0–0.4% in CRPS —
nothing. Restricted to the periods containing the daily peak, the same comparison gives 5.7–9.0%
depending on aggregation level. And at household level during peaks, both their simple and advanced
models are *worse* than a plain time-of-day KDE. If a leaderboard scores only pooled error, a method
that is 9% better at the moments a DNO acts on is invisible, and a method that is worse at those
moments can win. That is a measured result on a four-level hierarchy, not an argument.

**Four cheap metric additions, all with published definitions.** BigDEAL's WMAE weights peak-timing
displacement — g(z) = z for z ≤ 1, 2z for 2 ≤ z ≤ 4, capped at 10 beyond — the cap justified so that
one catastrophic day cannot overshadow good forecasts on all the others, which is the same pathology
our pooled unmet fraction has. Their Peak Shape Error normalises actuals by the actual peak and
forecasts by the *forecast* peak before summing absolute differences over the peak hour ±2, separating
profile error from level error in a way NMAE cannot. Per-series Diebold–Mariano tests distinguish
"wins on average" from "wins consistently", which matters given how heterogeneous our series are.
And reliability diagnostics belong beside every headline number — HEFTCom's diagrams caught a team
achieving competitive pinball with substantial bias, and Browell & Fasiolo use worm plots specifically
because they accentuate tail behaviour, with consistency intervals allowing for serial correlation.
None of these needs inventing; all four are published and simple.

**The published comparisons are not evidence against a well-featurised gradient booster, though this
is a weaker defence than it sounds.** The tree baselines in both KIT papers carry none of the physics,
calendar or per-type featurisation our roadmap plans, on net load at feeders dominated by PV. So
"foundation models beat XGBoost" is really "foundation models beat a lightly-featurised XGBoost", and
the sequencing implication is to exhaust the cheap featurisation first. But the counter-evidence
section should be read alongside this: a model trained on none of the data was competitive, and if our
featurised booster does not clearly beat a zero-shot arm, that is information about the value of our
whole experimental programme. Running the arm is how we find out. Two constraints shape that work: Chronos-2
needs covariates and about a month of context, and its inference cost grows with series count — 55.8 ms
per series per forecast, so about 2.3 minutes a day for 2,500 control-member forecasts but roughly two
hours for all 51 members, before any adjustment for our longer horizon (§3 has the full table). It is
cheap at the control-member level and expensive only when multiplied by the ensemble and the horizon,
both of which are our design choices rather than the model's.

**Our single-champion leaderboard design is challenged by the strongest task-specificity evidence in
the review.** Nine of thirteen BigDEAL finalists, and four of the top five, built separate models per
track; the organisers report that magnitude ranks are least correlated with timing and shape ranks,
and that the top teams on the latter two are not the top teams on the first. We promote one champion
per series on NMAE. If the three attributes want different models, either the leaderboard needs
multiple champion slots or we are optimising for the wrong one — worth deciding deliberately.

**Move effort towards a parametric tail — and note that adding a p99 pinball column will not give us
a tail metric.** Browell & Fasiolo found quantile regression tails uncalibrated beyond roughly the
1st and 99th percentiles
across 14 GB regions with five years of half-hourly data; cross-validation put the last reliable levels
at 2.5% and 97.5%, or 5% and 95% for four regions. We have 10⁴–10⁵ rows per *substation*, so our
reliable range will be narrower than theirs, not wider. If the cost metric's procurement quantile
calibrates above p95 — which for a risk-averse DSO it plausibly will — the leaderboard ranking then
depends on quantiles our method cannot estimate reliably. Their conditional GPD, with a log-linked
scale parameter varying with wind speed, irradiance and expected load and a constant shape, is a
modest construction and a well-posed estimation problem. The first thing to measure is whether the
calibration limit sits at the same percentile on our smaller, noisier series.

**Five published negative results land on design choices we have already made**, each set out in the
counter-evidence section above: gridded NWP statistics added nothing in Browell & Fasiolo's framework,
which hits our neighbouring-H3-cell item; a tuned XGBoost lost to a GAM on identical features in a
production study, which hits our entire baseline; roughly a third of client-owned substations in that
study failed to beat a 24-hour-naive forecast, which hits our baseline reporting; temperature has
repeatedly been found less useful than expected at LV level; and power lags helped substantially in
one full-text paper while hurting in another. Each has a plausible reason our case may differ — trees
versus splines, gross versus net demand, our horizon, our PV exposure — and each is cheap to test. The
pattern is worth noticing: every one comes from a paper read in full, and none is visible in the
corresponding abstract.

**The lag contradiction is the one worth pursuing rather than merely testing.** Browell & Fasiolo
diagnosed embedded generation as the reason lags hurt, then responded by deleting the lags — never
trying the middle option of supplying the weather at the lagged time so the model can judge how normal
that observation was. Our aligned-weather and weather-delta rungs test precisely that, and the delta
is linear in the capacity-factor change, which is the comparison their own diagnosis implies. Pinheiro
et al. also supply a better diagnostic than a bare ablation: run the autocorrelation functions on the
*lag-free model's residuals* rather than on the load series, since load ACF always shows daily and
weekly peaks whether or not the calendar terms have already captured them. A small but real
contribution to claim.

**Three structural ideas are cheap arms we are not running.** Simple quantile averaging across model
families keeps winning, with one GEFCom2017 team reporting plain averaging beat sophisticated
weighting and the Haben review reporting constrained quantile regression averaging as best of nine
schemes tested. HEFTCom's winner obtained robustness to missing inputs structurally — one model per
NWP source, the missing source's quantiles filled from the survivors — rather than by teaching one
model to tolerate absence. And Pinheiro et al. cut system-level RMSE from 203 MW to 154 MW purely by
training the *same* model on eight calendar regimes — Christmas, Carnival, Easter, other public
holidays, weekends, August, and the two remaining seasons — and combining the experts online with a
weighted-majority rule. No new features, no new technique, only different training subsets. That is
close to free given our per-`time_series_type` machinery.

**Expect our absolute numbers to look worse than published ones, and explain why first.** Most of this
literature uses weather unavailable at forecast time, and the papers say so. Browell & Fasiolo are the
exception. The claim to make is that our numbers are measured under operational constraints and most
of theirs are not — a statement about setup, not about anyone's integrity.

**The leaderboard only helps the industry if we do one more thing.** As designed it is computed on
private data with metrics we chose, which is structurally another team playing in its own stadium.
Publishing the evaluation protocol and metric code is nearly free and makes the comparison reproducible
in method. Anchoring a number on open data is the stronger move but the candidates are imperfect:
FeederBW is German LV feeders at four days with ex-post weather, so a good result there is weak
evidence about our system; Browell & Fasiolo's GB net-load dataset on Zenodo is closer in geography and
in problem but is at GSP-Group scale. Either is mostly a competence signal rather than a validation.
The limits of what we can offer the field, scale included, are set out in the section above.

### The recommendations, ranked and costed

Twenty-three suggestions are scattered through the sections above — far more than can be absorbed by a
roadmap already carrying around forty items and roughly one person. Ordered by expected value per
unit effort. Only the first five are things I would argue for unprompted.

| # | Recommendation | Cost | Why this rank |
|---|---|---|---|
| 1 | Report empirical coverage and tail-focused calibration diagnostics (worm plots with serial-correlation-aware consistency intervals) beside every headline number | Hours to a day | Both published ways of scoring a tail fail without it — Kaas et al.'s metric is winnable by under-dispersion, and Browell & Fasiolo show tail pinball cannot discriminate |
| 2 | Peak-aware error: Haben's adjusted *p*-norm (p=4, displacement window w), plus BigDEAL's WMAE for timing and Peak Shape Error for profile, all reported separately from magnitude | ~1–2 days; all three definitions are published in full | The strongest-evidenced item here. Pinheiro et al. scored the substation models of a live 96,989-substation DSO system with the adjusted *p*-norm *because* standard metrics reward smoothness (the paper does not say the DSO adopted it as an operational metric); BigDEAL found magnitude ranks least correlated with timing and shape ranks, and nine of thirteen finalists built separate models per track |
| 3 | Report the fraction of series beating a persistence baseline as a headline, beside pooled error | Hours | Pinheiro et al.: only 66–70% of client-owned substations beat a 24-hour-naive forecast. A pooled error hides that entirely, and our small-series population is the analogue |
| 4 | Per-series Diebold–Mariano tests on the leaderboard | ~1 day | Distinguishes wins-on-average from wins-consistently; used by Browell & Fasiolo, Hertel et al. and others |
| 5 | Publish the evaluation protocol and metric code | Low, needs an NGED decision | Answers the strongest objection to the whole framing, and costs almost nothing |
| 6 | Conditional GPD tail model | Weeks | Quantile regression loses calibration around p99 on series far larger than ours, and our τ may sit above p95 |
| 7 | A GAM arm alongside XGBoost | Days — `mgcv` exists and Pinheiro et al. print their model formula | A tuned XGBoost lost to a GAM on identical features in a production study; our baseline is XGBoost and we have never tested the alternative |
| 8 | A "just adapt" arm — recency weighting plus frequent retraining, no detector | Days; overlaps existing items | The control that tells us whether the switching-event investment is needed |
| 9 | Train on degraded/perturbed weather to match operational forecast error — for us, train on ENS members rather than the control | Days; the loader already accepts a member list | The top five BigDEAL teams all did a cruder version of this (noise, ±1 h shifts, predicting the forecast error) and the organisers name it as what separated them. We have the error distribution directly from 51 members rather than needing to simulate it |
| 10 | Power-lag ablation ladder: raw lags → aligned lagged weather → weather-delta compensation → two-stage residuals, against a rolling-mean or init-time anchor | Days for the first two rungs (config-level and one derived column); weeks for the last | The one design choice where two full-text papers directly contradict each other — and the middle rungs test a remedy neither tried. Browell & Fasiolo diagnosed embedded generation as the cause and deleted the lags; the delta is linear in the capacity-factor change, so it precomputes the comparison their diagnosis implies. Pair with the residual-ACF diagnostic. Caveat: most of the signal sits at day 0–2 unless anchored to init time |
| 11 | Cross-family quantile averaging, and regime-specific experts combined online | Days, once the quantile family exists | Strong prior from every competition; Pinheiro et al.'s eight-regime weighted-majority ensemble cut RMSE by 24% using the same technique and features, only different training subsets |
| 12 | Remaining ablation arms: gridded/neighbouring-cell weather, effective temperature | Days each | Published negative results on design choices we have made; cheap to test |
| 13 | Switching detector: binary segmentation plus statistical process control, run on a **residual** against an independent reconstruction rather than on raw load, with event-length-stratified evaluation | Days | Bouman et al. (§10c) do exactly this at a Dutch DSO on 180 primary substations, chosen for interpretability, with ~90% of load estimates within 10%. Their four length buckets (15 min–6 h, 6 h–3 d, 3–42 d, 42 d+) exist because pooled metrics let long rare events dominate — our evaluation has the same problem |
| 14 | Adopt Artificial Forecasting's cheap operational instrumentation: normalised MAE against **firm capacity** on a rolling six-week window, a data-quality trigger at >10% of recent input requiring correction, and their data-quality bands as a standing reporting axis | Hours | All from a live DNO deployment (§10a). The first is decision-relevant normalisation; the second is a concrete threshold for the hygiene problem our Tier-2 item addresses; the third is what turned their own worst band into evidence for item 13 rather than a caveat |
| 15 | Run their sign-inversion check — flip any series with more than half its 18:00–06:00 readings negative — and add a timestamp-agnostic daily-peak error to the metric set | Hours each | Both from Alpha (§10a). The sign check caught 7 of 729 NPg substations and restricting it to night-time avoids false positives from daytime solar; the daily-peak metric drops the timing requirement on the explicit argument that flexibility procurement does not need the exact settlement period, which separates it cleanly from item 2's timing metrics |
| 16 | Watch for the Artificial Forecasting white paper, due November 2026 | Free | It will cover their data, ETL, modelling and results, and lands before our ML results are due (§10a). Their Alpha deliverables are already public on the ENA portal and citable now |
| 17 | Recover the sign at the ten MVA-metered sites using an independently-signed reference | Hours to test | Bouman et al. hit the identical defect — equipment measuring only absolute current — and fix it by taking the sign from a reference series built from signed P measurements. Worth an hour before writing those sites off. A different defect from item 15: there the sign exists and is inverted, here it was never recorded |
| 18 | Read the LV PV capacity benchmark (§10b) before building capacity estimation | Hours to read | It benchmarks model-based against data-driven capacity estimation on substation net load plus irradiance, with a sensitivity analysis over PV penetration and data quality — the question of when each method breaks |
| 19 | Chronos-2 zero-shot arm | Days to set up; ~2.3 min/day for 2,500 control-member forecasts, ~2 h for all 51 members before any horizon adjustment | Worth knowing rather than assuming. Cheap on the control member; the ensemble-times-horizon multiplication is what makes it expensive, and that is our choice not the model's |
| 20 | Metadata-only ablation (no time-series context) | ~1 day | A decomposition we lack; diagnostic rather than skill-improving |
| 21 | Temperature-scenario methods from GEFCom2017 for the long end of the horizon | Days to read, weeks to build | Relevant precedent for turning weather distributions into load distributions, but our ENS members are strictly better information |
| 22 | Forecast fusion (Gilbert et al.) and temporal hierarchies | Weeks; their code and data are open (ProbCast, Zenodo 10.5281/zenodo.7064279) | Better evidenced than its rank suggests: fusion is worth 5.7–9.0% CRPS at peaks across all four levels of their hierarchy while costing nothing on average. Ranked here only because items 1–3 must exist first to judge it, and because their study has no weather and no embedded generation |
| 23 | Benchmark our model family on an open dataset | Weeks, plus a different pipeline | Weakest evidential return; do it for credibility if at all, after 1–5 |

Item 12 still bundles two ablations, ordered among themselves by how central the design choice is.
Items 1–5 would change what the leaderboard tells us within a fortnight, and items 2 and 3 are the
two whose evidence comes from a system actually running in production at a DSO. Items 6, 7 and 9 are
the ones most likely to change a conclusion. Items 20–22 should probably not happen this year.

---

## Sources found by the second-round coverage review

These were missed by the first search and are all cited in the Milestone 2 section. Every DOI below
was verified against Crossref, and every portal record was read directly.

**GB network-innovation projects.** The first search looked for UK innovation projects and recorded
no results; it missed all of these.

- **SSEN TRANSITION** (Network Innovation Competition, Oxfordshire, load-forecasting dissemination
  report V3, November 2021). Net load at 13 primary substations, their bulk supply points and 11 kV
  feeders, 30 minutes to 10 days ahead. **40-member ICON-EU-EPS ensemble to H+120, then MOSMIX as a
  single deterministic scenario to H+240** — so 41 weather scenarios in total. Disaggregates net
  load into demand and generation and recomposes it. Uses the connectivity map, and lists
  "historical network connectivity data availability is just as important as historical net demand
  and generation measurements" as a headline learning. Calibration: all primary substation models
  below 10% MAPE except two (13.4% and 19.7%); 94% of 11 kV feeders below 20%. The live PDF now
  redirects to the project landing page; read the Internet Archive snapshot of
  `ssen-transition.com/wp-content/uploads/2021/11/TRANSITION-Load-Forecasting-Dissemination-Report-Final-V3.pdf`.
  **This is the closest precedent for our method and it weakens gaps 1, 4, 5 and 6 as originally
  written.** What survives: the ensemble stops at four days, and the trial was 13 substations.

- **EFFS** (Western Power Distribution, now NGED; NIC, Jan 2018 – Nov 2021, £3,338,896;
  <https://smarter.energynetworks.org/projects/wpden03/>). GSPs, BSPs, primary substation
  transformers and generation sites, hour-ahead to six months. Independently selected XGBoost as the
  best accuracy-versus-effort trade-off. Deterministic only. This is NGED's own predecessor project
  and the review should not omit it.

- **NIA_UKPN0104**, "AI for Visibility and Forecasting of Renewable Generation" (UK Power Networks,
  Oct 2024 – Jul 2026, £389,444, third parties Open Climate Fix and Sheffield Solar;
  <https://smarter.energynetworks.org/projects/nia_ukpn0104/>). Infers unmetered solar capacity
  behind substations, and the portal record states the capacity estimates "feed into a solar forecast
  algorithm to produce forecasts of unmetered solar generation at primary substations".
  **This refutes the unqualified form of gap 5, and OCF is a partner.**

- **SSEN FastTrack** (SIF Round 4 Alpha, 2025–2026, £554,998, with Faculty;
  <https://smarter.energynetworks.org/projects/10166254/>) — probabilistic load forecast per
  substation rolled up to GSP. **SP Energy Networks Predict4Resilience** (SIF Beta, £5,020,674;
  <https://smarter.energynetworks.org/projects/10061710/>) — ensemble NWP driving a probability
  distribution of network faults per district to 7 days, in a control room. **Energy Systems
  Catapult DNO Forecasting Forum** — NGED participates and wrote its forecasting taxonomy.

**Papers.**

- **Paredes & Vargas (2017)**, doi 10.1049/iet-gtd.2017.0129, and **Paredes, Vargas & Maldonado
  (2020)**, doi 10.1049/iet-gtd.2018.7127. Four methods that *adjust* reconfiguration-affected
  history rather than deleting it, on six years of hourly data across 169 real feeders; the 2020
  follow-up forecasts future reconfiguration events. **Refutes "nobody keeps the history".**
- **Huyghues-Beaufond et al. (2020)**, doi 10.1016/j.apenergy.2019.114405. 342 UK MV feeders.
  **Corrected 2026-08-22 after reading the full text** (TU Delft green open-access copy): this is
  *not* a detect-and-delete citation. Binary segmentation detects change-points, but only to bound
  the piecewise-stationary segments inside which Tukey's rule removes **outliers** — "detect and
  remove outliers from piecewise stationary segments". Every segment is kept and the level shifts
  stay in training and test data. The paper reports that they bias the fitted parameters and hurt
  the forecast, while also concluding the forecasters "handle level-shifts well by adapting quickly
  to changes". So it is a counter-example to any claim that nobody trains on switching-contaminated
  history — what survives is that nobody exploits that contamination deliberately.
- **Ludwig, Arora & Taylor (2023)**, doi 10.1080/01605682.2022.2115411. GB national load, 51-member
  ECMWF ENS, 1–6 days, with **EMOS post-processing plus ensemble copula coupling before the load
  model** because raw ensembles are biased and under-dispersed. Names LV-hierarchy application as
  future work. Arora co-authored Haben et al. 2021, whose own future-work list asks for the same
  thing. **This is the missing link between Taylor & Buizza 2002 and us.**
- **Nespoli et al. (2020)**, doi 10.1016/j.epsr.2020.106755 (arXiv 1910.03976), and **Ben Taieb,
  Taylor & Hyndman (2021)**, doi 10.1080/01621459.2020.1736081. Hierarchical reconciliation on a
  real Swiss distribution grid, and the probabilistic version. Narrows gap 6 to: reconciliation uses
  only the summation constraint, which carries no adjacency information and which an abnormal
  running arrangement invalidates by construction.
- **Kara et al. (2018)**, doi 10.1016/j.segan.2017.11.001; **Li et al. (2021)**, doi
  10.1109/TPWRS.2020.3035639; **Stratman et al. (2023)**, doi 10.1109/TIA.2023.3276356. Solar
  disaggregation at feeder-head and substation level — above household level, so the
  behind-the-meter exclusion does not cover them.
- **Göçmen et al. (2021)**, doi 10.5194/wes-6-111-2021. Available-power estimation for a
  down-regulated turbine, from the turbine's own instrumentation. The technique behind gap 7, one
  level down.
- **Haben, Giasemidis, Ziel & Arora (2019)**, doi 10.1016/j.ijforecast.2018.10.007. 100 real LV
  feeders, forecast *and* observed temperature, no effect or a negative one. **Counter-evidence.**
- **Haben, Voss & Holderbaum (2023)**, doi 10.1007/978-3-031-27852-5. Open-access book-length update
  from the lead author of the 2021 review; better entry point than the review for post-2020 work.
- **Browell et al. (2025)** is now journal-published: doi 10.1016/j.ijforecast.2025.10.005.

**Counter-evidence found in this round.** Browell & Fasiolo compared GAM-Point against GAM-Grid
across 14 GB regions: significantly better in two, significantly **worse in three**, indistinguishable
in nine — gridded NWP statistics added no value in their framework. Artificial Forecasting bought
postcode-level forecasts for Meadowfield and Hazlehead and reported no notable improvement. Both bear
directly on our ensemble-on-a-spatial-grid bet and are now in the report.

**Now audited.** See "The CIRED audit" below. The lead about a 2000 CIRED paper on load forecasting under MV network reconfiguration was chased and not found; the nearest thing is Yasuoka (2001), whose load-transfer module was described as future work and never followed up.

## Sources found by the third-round wide-net coverage review

This round searched four areas the earlier rounds had not: estimating the effective capacity of
metered generators, machine-learning operations and reproducibility practice, forecast verification
methodology from meteorology, and a free-ranging sweep. Everything below was verified against a
primary source (Crossref metadata, the paper itself, or the tool's own documentation) before being
used in the report.

### Effective capacity of metered generators — the report's absence claim was too strong

The report previously said the technique was standard one level down "but always from the
generator's own instrumentation". That is false, and the counter-example is by an author the report
already cites three times.

- **Dantas & Browell (2026)**, "Seamless Short- to Mid-Term Probabilistic Wind Power Forecasting",
  *Wind Energy* 29(2) e70079, <https://doi.org/10.1002/we.70079>; preprint
  <https://arxiv.org/abs/2502.11960>. 73 GB wind farms (34 onshore, 39 offshore), 2019–2023, lead
  times 6–162 h, driven by the ECMWF ensemble. Verified verbatim from the paper: "The dataset
  provided by BMRA does not include information on the farms' available capacity over time", so
  "the method described in the supplementary material was applied to estimate the available
  capacity at each time stamp (i.e., a time series of available capacity)". They also exclude
  curtailed half-hours: "Estimated curtailment volumes are contained in BAV data. Thus, periods with
  non-zero BAVs were excluded from modelling and forecast evaluation." Also verbatim: "The wind power
  time series were normalised by each wind farm's available (or nominal) capacity." **Two caveats
  that keep our gap alive.** The estimation algorithm is in the Wiley supplementary material, not the
  free preprint, and two independent attempts failed to retrieve it — so we cannot claim they work
  "from the metered output alone"; the method could draw on a capacity register or REMIT outage
  messages. And bid-acceptance volumes exist for transmission-connected wind farms, not for NGED's
  embedded generators. **Getting that supplementary material is the single highest-value follow-up
  for the effective-capacity work.**

- **RdTools** (NREL), concept DOI <https://doi.org/10.5281/zenodo.1210316>, MIT licence. Estimates
  year-on-year degradation, soiling loss and availability. Important correction to what the
  coverage reviewer reported: the clear-sky workflow is *not* free of site irradiance. Its own
  documentation says "site irradiance data is still required to identify clear-sky conditions to be
  analyzed", and warns that "Satellite and clear-sky analyses tend to provide less stable results
  than sensor-based analysis". The availability analysis needs inverter-level data. So RdTools
  removes the need for plant *internals*, not for irradiance.

- **Severiano et al. (2026)**, *Solar Energy* 308, <https://doi.org/10.1016/j.solener.2026.114382>.
  Rule-based detection and classification of photovoltaic underperformance across 1,089 systems
  (2,213 inverter monitors, Australia) from inverter data — the title says "using inverter data", and
  a reviewer reading the paper found no irradiance input at all, contradicting the coverage
  reviewer's "plus satellite irradiance". 92% and 88% accuracy on two case types but only 56% on
  generation clipping. **Cite the first author as Mendonça Severiano**, which is how Crossref indexes
  the name.

- **Jordan et al. (2022)**, "Photovoltaic fleet degradation insights", *Prog. Photovolt.* 30(10)
  1166–1175, <https://doi.org/10.1002/pip.3566>. 1,700 sites, 7.2 GW. Overall −0.75%/year; verified
  from the abstract, "cooler climates exhibit a median −0.48%/year loss, which increases to
  −0.88%/year in hotter climates". The −0.48% figure is the right prior for GB.

- **Staffell & Green (2014)**, *Renewable Energy* 66, 775–786,
  <https://doi.org/10.1016/j.renene.2013.10.041>. UK onshore load factors decline 0.44 ± 0.04
  percentage points a year, inferred from Ofgem regulatory meter readings alone. Monthly and
  fleet-averaged, so it proves the signal is present in meter data rather than supplying a method.

- Supporting methodological citations behind RdTools, all verified: Jordan et al. (2018)
  <https://doi.org/10.1109/JPHOTOV.2017.2779779> (the year-on-year estimator); Lindig, Theristis &
  Moser (2022) <https://doi.org/10.1088/2516-1083/ac655f> (open access; the same data yields
  materially different degradation rates depending on filter and estimator choice).

- **Validation data with ground truth already exists.** Cubico's Kelmarsh
  (<https://doi.org/10.5281/zenodo.8252025>, 6 turbines, 2016–2022) and Penmanshiel
  (<https://doi.org/10.5281/zenodo.5946808>, 14 turbines, 2016–mid-2021) releases each carry
  10-minute turbine SCADA *with alarm and events data* **and** site substation and fiscal grid meter
  data for the same period, CC-BY-4.0. Verified from the Zenodo records. That is a ready-made rig:
  build the estimator from the meter alone, score it against the turbine records.

- **What survives as a genuine gap:** nothing estimates effective capacity from a *substation's*
  net flow, where generation is mixed with demand; there is no wind equivalent of RdTools working
  from a revenue meter alone (OpenOA's meter-only path still expects a reported availability and
  curtailment table); and much distribution-connected GB curtailment is instructed by the network
  operator under active network management, so that component is a data join inside NGED rather
  than an estimation problem.

### The field's own account of why its results cannot be compared

- **Hong, Pinson, Wang, Weron, Yang & Zareipour (2020)**, "Energy Forecasting: A Review and
  Outlook", *IEEE OAJPE* 7, 376–388, <https://doi.org/10.1109/OAJPE.2020.3029979>. Open access.
  Verified verbatim from the PDF: "Unfortunately, most papers can never be replicated, because the
  data have never been published"; "Sometimes authors tend to pick the error measures in favor of
  their proposed method but hide the results from other error measures"; "When the obtained
  differences in errors are close to zero, the statistical significance tests are seldom
  performed"; "many papers avoid direct comparisons with classic, established, and state-of-the-art
  models. Some even skip comparisons with naive models. Many papers draw a small circle in the case
  study section by only comparing with the models within the immediate family."

- **Tawn & Browell (2022)**, *RSER* 153, 111758, <https://doi.org/10.1016/j.rser.2021.111758>. Open
  copy at <https://eprints.gla.ac.uk/320301/1/320301.pdf>. **Attribution correction:** the "8 of
  42" figure is *not* theirs. Their sentence reads: "we found both solar and wind papers which only
  compare models to their own variations [11 references]: this is in line with a survey by
  Doubleday et al [11], who find that 8 of 42 solar forecasting papers surveyed did not include a
  benchmark other than variants of the same model." Cite Doubleday, Van Scyoc Hernandez & Hodge
  (2020), *Solar Energy* 206, 52–67, <https://doi.org/10.1016/j.solener.2020.05.051> for the
  number, and Tawn & Browell for the eleven papers they found themselves.

- **Nguyen & Müsgens (2026)**, "A meta-analysis of solar forecasting based on skill score", *J.
  Renewable and Sustainable Energy* 18(2), <https://doi.org/10.1063/5.0300682>. The partial
  counter-example to the report's opening thesis: a defensible ranking *can* be recovered from this
  literature, by screening 1,447 studies, reading 320 in full and hand-extracting 4,687 skill scores
  from the 188 that reported one, then regressing out ten other factors. **Count correction:** 320 is
  the number of full texts assessed, not the number that yielded data; the body says "Our work is
  based on 4,687 data points extracted from 188 papers". Verified from the *published* 2026 abstract:
  ensemble–hybrid models raise skill score by 7–27 percentage points over time-series models, "while
  many advanced machine learning methods show inconsistent gains"; day-ahead forecasts do best with
  NWP data (+11.5 pp). Note the 2023 arXiv preprint (v2 of 2208.10536) carries a different Table 3
  whose largest ensemble–hybrid coefficient is +21.2 pp — cite the published version's figures, not
  the preprint's.

### Forecast verification methodology from meteorology

- **Weigel, Liniger & Appenzeller (2007)**, *MWR* 135(1), 118–124,
  <https://doi.org/10.1175/MWR3280.1>. The ranked probability skill score is negatively biased for
  small ensembles; they derive an analytical bias correction for any ensemble size. Directly
  relevant: we run 51 members and will be compared against studies running far fewer.

- **Lerch, Thorarinsdottir, Ravazzolo & Gneiting (2017)**, "Forecaster's Dilemma: Extreme Events and
  Forecast Evaluation", *Statistical Science* 32(1), 106–127, <https://doi.org/10.1214/16-STS588>.
  Scoring only the cases where an extreme occurred is *biased*, not merely noisy: conditioning the
  evaluation on the outcome rewards a forecaster who over-predicts extremes. No proper scoring rule
  stays proper on an outcome-conditioned subset. This constrains how our tail metric may be built.

- **Gneiting & Ranjan (2011)**, *JBES* 29(3), 411–422, <https://doi.org/10.1198/jbes.2010.08110>.
  The threshold-weighted CRPS: put a weight function on the outcome axis (for us,
  `w(z) = 1{z > firm capacity}` per substation) and the score stays proper. Implemented in the
  Python `scoringrules` package. The same paper gives the negative result: multiplying a proper
  score by an outcome-dependent weight destroys propriety.

- **Foygel Barber, Candès, Ramdas & Tibshirani (2020)**, *Information and Inference* 10(2),
  455–482, <https://doi.org/10.1093/imaiai/iaaa017>. Distribution-free *conditional* coverage is
  impossible. Conformal prediction therefore cannot promise 90% coverage at the peaks, only on
  average — which is why finding 5 in the report commits to *stratified* coverage.

- **Hamill (2001)**, *MWR* 129(3), 550–560, and **Allen, Koh, Segers & Ziegel (2025)**, *JASA*
  120(552), 2796–2808, <https://doi.org/10.1080/01621459.2025.2506194>. A flat rank histogram, and
  equally an on-target coverage figure, can be produced by a forecast that is over-dispersed on
  some days and under-dispersed on others; and a forecast can be calibrated in the standard sense
  and still be unreliable for extremes.

- **Gilleland, Ahijevych, Brown, Casati & Ebert (2009)**, *Weather and Forecasting* 24(5),
  1416–1430, <https://doi.org/10.1175/2009WAF2222269.1>. The double penalty and the four families of fix. **Two caveats:** the double penalty they name is
  *spatial* ("a forecast feature with the correct size and structure might yield very poor
  verification scores if the feature is displaced slightly in space"), not temporal, and the paper
  is an intercomparison of spatial verification methods for gridded fields. And it does **not**
  discuss propriety; its nearest passage says the opposite of a confident claim — "it is unlikely
  (but not impossible) that one would tune a model to obtain the best performance by hedging…
  Nevertheless, it will be worth investigating how each method could be hedged, if at all." The
  statement that most displacement-tolerant scores are not proper is ours, not theirs.

- **Richardson (2000)**, *QJRMS* 126(563), 649–667, <https://doi.org/10.1002/qj.49712656313>.
  Relative economic value of the ECMWF ensemble across the full range of user cost-loss ratios.
  Meteorology has priced forecast decisions since 2000; the report's "the price is missing" claim
  only holds with an explicit "at distribution level" qualifier.

- **Buizza & Leutbecher (2015)**, *QJRMS* 141(693), 3366–3382, <https://doi.org/10.1002/qj.2619>.
  The forecast skill horizon — the lead time at which the ensemble stops beating a climatological
  distribution, scored by CRPS — is 16 to 23 days for instantaneous grid-point fields. **Read the
  caveat before quoting:** the study's variables are all free-atmosphere (Z500, T850, T200, U850,
  V850). It assesses no 2-metre temperature and no irradiance, and makes no claim that surface
  variables have shorter horizons. Its only contrast is instantaneous versus time- and space-averaged
  fields, where averaging makes horizons *longer*. Any "shorter at the surface" statement is our
  inference and must be worded as one.

- **Vannitsem et al. (2021)**, *BAMS* 102(3), E681–E699, <https://doi.org/10.1175/BAMS-D-19-0308.1>.
  The current review of statistical post-processing, including ensemble copula coupling and the
  Schaake shuffle, which restore space-time correlation after per-variable calibration. Relevant
  because we push a raw 51-member ensemble through a load model.

### Machine-learning weather models

- **Price et al. (2025)**, "Probabilistic weather forecasting with machine learning", *Nature* 637,
  84–90, <https://doi.org/10.1038/s41586-024-08252-9>. GenCast: 15-day global ensembles at 0.25°
  and 12-hour steps in 8 minutes; verified from the abstract, "greater skill than ENS on 97.2% of
  1,320 targets we evaluated and better predicts extreme weather, tropical cyclone tracks and wind
  power production". (The arXiv preprint says 97.4%; cite the *Nature* figure.) **Scope correction:**
  the 1,320 targets are combinations of variable, lead time and vertical level. Wind power is a
  *separate* downstream experiment in Fig. 4, interpolating 10 m wind speed to 5,344 wind farm
  locations from the Global Power Plant Database and applying an idealised power curve — it is not
  one of the 1,320.

- **Lang et al. (2026)**, "AIFS-CRPS", *npj Artificial Intelligence* 2(1),
  <https://doi.org/10.1038/s44387-026-00073-7>. ECMWF's machine-learned ensemble outperforms the
  physics IFS ensemble for the majority of variables and lead times in the medium range. Deployment
  facts verified against ECMWF's own announcement: operational **1 July 2025**, 51 members (50
  perturbed plus one control), 31 km resolution against the physics ensemble's 9 km, 6-hourly steps
  to 15 days.

### Differentiable physics for demand — the report had the physics wrong

For demand at a substation the dominant physics is the thermal response of a few thousand
buildings, not a panel and a turbine, and that has a mature differentiable literature.

- **Di Natale, Svetozarevic, Heer & Jones (2022)**, "Physically Consistent Neural Networks for
  building thermal modeling", *Applied Energy* 325, 119806,
  <https://doi.org/10.1016/j.apenergy.2022.119806>.
- **Jiang, Wang, Li, Hong & You (2025)**, "Physics-informed machine learning for building
  performance simulation", *Advances in Applied Energy* 18, 100223,
  <https://doi.org/10.1016/j.adapen.2025.100223>.

The gap the report describes still stands in narrowed form: nobody has aggregated building-thermal
physics to a substation and put it inside a probabilistic forecast.

### Machine-learning operations — a real gap that citations cannot close

- **Sculley et al. (2015)**, "Hidden Technical Debt in Machine Learning Systems", NeurIPS 28,
  2503–2511, <https://papers.nips.cc/paper/5656-hidden-technical-debt-in-machine-learning-systems>
  (NIPS 2015 is not DOI-registered). **Terminology correction:** the phrases "skew" and "serving" appear **zero** times in this paper —
  "training-serving skew" is later Google TFX vocabulary, not Sculley's, and must not be attributed
  here. The terms the paper actually uses are configuration debt, glue code, pipeline jungles, dead
  experimental codepaths, data dependencies, CACE, correction cascades and undeclared consumers. Two
  further cautions: their hedge is "Glue code and pipeline jungles are symptomatic of integration
  issues that **may have** a root cause in overly separated 'research' and 'engineering' roles", and
  their prescribed remedy is different from ours ("Pipeline jungles can only be avoided by thinking
  holistically about data collection and feature extraction"). Their next named debt, dead
  experimental codepaths, warns against experimenting via "conditional branches within the main
  production code" — adjacent enough that a reader who knows the paper could misread us.
- **Kreuzberger, Kühl & Hirschl (2023)**, *IEEE Access*,
  <https://doi.org/10.1109/ACCESS.2023.3262138>. The reference architecture that licenses "current
  industry best practice".
- **The honest limit:** there is no citable empirical study measuring whether config-driven,
  automatically-tracked, production-path experimentation makes research *faster*. The one
  MLOps-for-energy-forecasting paper, Zhao, Ma & Jørgensen (2026),
  <https://doi.org/10.3390/info17040328>, is a capability mapping of 13 platforms, not a velocity
  measurement. The report therefore states the speed claim as our own experience, not as a finding
  from the literature. Pre-registration in forecasting is essentially an unwritten literature, so
  advocating it is an original argument rather than a cited one.

### Other verified sources not yet used, or used once

- **Ruhhütl, Schmaranz & Dietrichsteiner (2023)**, "Load and generation forecast on substation
  level", *CIRED 2023*, IET Conference Proceedings 2023(6), 706–710,
  <https://doi.org/10.1049/icp.2023.0476>. Kärnten Netz, Austria. Abstract read via OpenAlex; full text paywalled on the IET Digital Library and on IEEE Xplore, and absent from the open CIRED repository. Day-ahead, deterministic, prefers linear or Gaussian regression, MAPE 3–8% varying with how industrial and how large the supplied area is. Weather input not mentioned. Corroborates findings 1 and 2; bears on none of the seven gaps.
- **Akhtar, Mohammadi-Ivatloo & Lassila (2026)**, "Datasets for wind energy forecasting
  applications", *RSER* 236, 116941, <https://doi.org/10.1016/j.rser.2026.116941>. Over 1,400
  articles; strong geographic bias; proposes a benchmarking protocol. Published August 2026,
  independent 2026 support for the report's central thesis.
- **Roelofs et al. (2019)**, "A Meta-Analysis of Overfitting in Machine Learning", NeurIPS 32. Over
  a hundred Kaggle competitions; public-leaderboard rank tracks private-test rank with little
  evidence of substantial overfitting. Useful if the report's leaderboard proposal is challenged.
- **Bracher, Ray, Gneiting & Reich (2021)**, *PLOS Comp. Biol.* 17(2), e1008618,
  <https://doi.org/10.1371/journal.pcbi.1008618>. The weighted interval score: proper, in the units
  of the target, decomposes into dispersion plus under-prediction plus over-prediction, and it is
  the operational scoring scheme of a large multi-team public forecast hub — a direct precedent for
  the leaderboard we propose.
- **Tools:** `scoringrules` (Python) implements the threshold-weighted CRPS directly; `scores`
  (Bureau of Meteorology, <https://doi.org/10.21105/joss.06889>) has 50+ peer-reviewed metrics
  including isotonic reliability diagrams and rank histograms, though it is xarray-native.
- **Standards correction:** IEC 61400-26-1:2019 cancels and replaces TS 61400-26-1:2011, -26-2:2014
  and -26-3:2016. Cite the consolidated 2019 standard, not `-26-2`. It says how to *account for*
  lost production once known, not how to *estimate* it without turbine state signals.
- **NGED curtailment analysis and Curtailment Estimator** for flexible and curtailable connections:
  <https://dso.nationalgrid.co.uk/planning-our-future-network/curtailment-analysis>.

### Freshness check

Nothing published since the earlier rounds supersedes the report's load-forecasting sources. Hertel
et al. (2026) and Kaas et al. (2026) remain the current benchmarks; Paredes & Vargas remains the
canonical "rewrite the history rather than delete it" citation. One earlier attribution was
re-checked and confirmed correct: the claim that per-quantile pinball loss in the tail is too noisy
to rank forecasting systems is near-verbatim Browell & Fasiolo (2021), Section V-A.

## The CIRED audit

Method: Crossref cursor dump of ISSN 2732-4494 and 2515-0855 for 2020–2026, filtered to the CIRED
volume blocks, plus ISSN-level enumeration for CIRED 2017; abstracts pulled from OpenAlex. The
resulting corpus is 4,151 records with 4,148 abstracts, covering CIRED 2017 Glasgow (695 papers),
2020 Berlin Workshop (228), 2021 online (675), 2022 Porto Workshop (243), 2023 Rome (833), 2024
Vienna Workshop (257) and 2025 Geneva (681) — about 3,600 CIRED papers, with roughly 540 adjacent
non-CIRED IET conference records left in as a safety margin, which only strengthens the negatives.
CIRED 2018 Ljubljana and 2019 Madrid are not in Crossref under the 10.1049 prefix; they were
searched by query against the full-text-indexed DSpace API at `cired-repository.org`. Pre-2017 was sampled by targeted Crossref queries, not enumerated. **CIRED 2026 Brussels Workshop, 9–10 June 2026**, is not yet in Crossref — volume 2026 of IET Conference Proceedings holds 11 issues and none is CIRED — so it was searched from the organisers' accepted-papers list (<https://2026brussels.cired.net/>, saved to `literature/papers/`), 265 papers, titles and authors only. Past deposit lag runs 3 to 7 months after the event (Rome June 2023 appeared that September, Geneva June 2025 that October, Vienna June 2024 the following January), so the Brussels abstracts should be searchable from about September 2026 and are worth re-running then. Of the 265 titles, 19 name forecasting or prediction; zero name an ensemble; the only horizon named is day-ahead; switching, capacity availability and disaggregation appear in no forecasting title. Two apply time-series foundation models (1383 grid-edge energy management, 1537 residential consumption), and 17 titles concern electrification — the topic our own review does not cover. The next main conference is **CIRED 2027 Stockholm, 14–17 June 2027**. Abstracts were read, not full texts: CIRED
2023 and 2025 full texts are paywalled on the IET Digital Library and are not on the open
repository, which stops at 2019.

**No CIRED paper undercuts any of the seven absence claims.** The verified negatives, each
re-derived independently rather than taken from the search agent:

- **Weather ensembles.** Nine papers mention "ensemble" at all; one co-occurs with any weather word,
  and it is about climate-change temperature extremes. Two name an NWP system. No CIRED paper drives
  a load or generation forecast from a weather ensemble.
- **Horizon.** 443 papers mention forecasting. Ten-day and week-ahead: zero. Forty-eight-hour: one.
  Day-ahead: 22. The only 14-day forecast predicts feeder faults, not load
  (<https://doi.org/10.1049/icp.2025.1612>).
- **Tail.** Three papers use tail vocabulary, all irrelevant (IEC 61850, power-line communications,
  rooftop PV profitability). One mentions pinball or CRPS, in a blockchain reserve paper.
- **Probabilistic forecasting.** Fourteen papers, of which one is at MV substation scale.
- **Switching-contaminated history.** Thirty-two papers on reconfiguration, four of which mention
  forecasting; none is about what a reconfiguration does to a training set.
- **Topology as adjacency.** Eight papers use graph or adjacency vocabulary; only Jung et al. (2024)
  puts grid adjacency inside a forecasting model, and it forecasts voltage in simulation.
- **Effective capacity.** 136 papers use capacity-availability vocabulary; all are hosting capacity,
  inverter reliability or EV charging headroom.

Added to the report: Wade et al. (2024), Cordier et al. (2024), Mesarcik et al. (2025), Jung et al. (2024), Pfeifer et al. (2021), Fox et al. (2018), and the LianderPower open dataset.

**Fox, Plećaš, Neilson, Cannon & Parr (2018)**, "Analysis of local demand trends and forecasting through weather correction and benefit to DSO transition and microgrids", CIRED 2018 Ljubljana Workshop, paper 0415, <https://doi.org/10.34890/134>. SP Energy Networks with Digital Engineering. **Full text read** — the open CIRED repository serves it through its DSpace bitstream API, which is how to reach anything from CIRED 2018 or 2019. They run a numerical weather prediction model over Scotland at 1 km squares and 30-minute steps for ten years, map it to each primary substation weighted by customer density, and fit machine learning against measured demand to compute a Weather Correction Factor per substation. Thirteen 33/11 kV primaries in the proof of concept, almost 400 in production. Results worth having: customer sensitivity to effective temperature runs from 1.4% to 4.8% demand change per degree Celsius across the 13 sites, varying with customer mix (industrial customers are less weather-sensitive than domestic); weather correction moves an annual peak by 5 to 10%; effective temperature, global horizontal irradiance and the cooling power of wind carried the explanatory power, while relative humidity carried none. Their motivating finding is that weather masks the underlying trend — raw flows looked flat from 2007/08 to 2010/11 while the corrected trend was falling, and the mild winters from 2013/14 to 2016/17 hid a trend that was higher than the raw flows suggested. Cited in the report's GB-networks section, and used to qualify the claim that no measurement of weather sensitivity exists at GB primary substation level.

**Audited, on-topic, and deliberately left out** — real and relevant, but changing no conclusion.
Recorded here so the absence claims have a citable evidence base if challenged: Wiss & Ruwaida
(2025) `icp.2025.2257`, five winters of congestion forecasts driving Swedish flexibility markets;
Peppanen & Alvarez-Fernandez (2025) `icp.2025.2297`, an EPRI survey of DSO forecasting gaps;
Gonçalves et al. (2021) `icp.2021.1696` and `icp.2021.2191`, E-REDES PREDIS forecasting 100k HV/MV
connection points — the same operator as Pinheiro et al. (2023); Camal et al. (2023) `icp.2023.1212`,
hierarchical reconciliation to the TSO/DSO interface; Howorth et al. (2023) `icp.2023.0471` and
Plana Ollé et al. (2023) `icp.2023.1009`, the two closest CIRED papers to gap 3, both pricing
decisions without holding risk constant; Lusis et al. (2025) `icp.2025.2219`, dynamic safety
thresholds bolted onto a point forecast — the practice gap 2 argues against; Jayan et al. (2025)
`icp.2025.1502`, propagating a *known* topology change into congestion calculations; Gérossier et al.
(2017) `oap-cired.2017.0625`, the oldest genuinely probabilistic distribution forecast in CIRED;
Heres et al. (2023) `icp.2023.0814` and Kloibhofer et al. (2017) `oap-cired.2017.1333`, gap-5
adjacent but below our aggregation level; and Yasuoka (2001) `cp:20010890`, whose promised load-transfer module is CIRED's only ever gesture at gap 4.

## The Dantas and Browell available-capacity algorithm

Obtained. The Wiley supporting information is unreachable, but the paper's data-availability
statement names nine Zenodo deposits, and DOI 10.5281/zenodo.13309890 ("Pre-processed BMRA data and
scripts") contains `Pre-processing_AuxiliaryMaterial.pdf` — "Observational Data Preprocessing",
Dantas & Browell, 6 pages — together with the Python that implements it. Both are saved to
`literature/papers/` (which is gitignored, so they live on disk only). Note for future searches:
`export.arxiv.org` is reachable from this machine even though `arxiv.org` is blocked.

**They use the metered output alone, plus ERA5 reanalysis wind speed. No capacity register, no
REMIT, no turbine-model data — they state that the BMRA database carries none.** Two stages:

1. Find timestamps at rated power: a moving window of L = 4 half-hours where the spread of metered
   output is under ΔP\* = 0.5 MW *and* wind speed is at or above rated throughout. At those
   timestamps available capacity equals metered power. Rated wind speed is inferred per site as the
   upper limit of the bin containing `argmax(f_w · w³)`, w binned at 1 m/s — entirely from the site's
   own wind-speed distribution, because no turbine model is known. Wind speed is ERA5 at 100 m,
   bilinearly interpolated to the farm's coordinates.
2. Between two rated-power timestamps, capacity holds the earlier value; if metered power ever
   exceeds the assigned capacity, that timestamp itself becomes a rated-power timestamp. So the
   estimate is a step function that only ratchets up when the meter forces it.

Two reliability filters, both from the metered output: drop timestamps where estimated availability
is under 40% of the series maximum, and drop each farm's first six months of operation. Curtailment
is handled by discarding half-hours with non-zero bid-acceptance volumes — the one input
Flexpectation will not have for embedded generators.

**Consequence for the report.** Two gap-7 claims were false and are now fixed: that nobody estimates
this from a revenue meter alone, and that we could not say how Dantas and Browell did it. What
survives is solar (the equivalent plateau test needs irradiance the site does not measure) and the
absent outage register.

## The Browell corpus audit

Every load-bearing figure the report attributes to Browell's five papers was checked against the
source text and **all are correct**: the 24.6%/3.2% reserve savings (Table IV, upward, 0.01% and
0.25% levels), the 40%/60%/10% pinball-loss gains, the two-helped/three-hurt/nine-neutral split on
spatial statistics, the 1st-and-99th-percentile calibration limit, Gilbert's 0.0–0.4% and
5.7/9.0/8.2/6.0% figures and the peak-timing range, and the HEFTCom scoreboard (22.18, 23.18, 24.64,
25.38, 53.58) with the Hornsea 1 episode.

Two papers earned a place and are now cited: **Angus, Browell, Greenwood & Deakin (2027)**, "Risk-
based dynamic thermal rating in distribution transformers via probabilistic forecasting", *EPSR*
262, 113545, <https://doi.org/10.1016/j.epsr.2026.113545> (preprint arXiv:2603.11905), which holds
risk constant at 644 individual LV transformers and so narrows gap 3; and **de Vilmarest, Browell,
Fasiolo, Goude & Wintenberger (2024)**, "Adaptive Probabilistic Forecasting of Electricity
(Net-)Load", *IEEE TPWRS* 39(2), 4154–4163, <https://doi.org/10.1109/TPWRS.2023.3310280>, which
narrows gap 4 and supplies gap 7's strongest counter-argument — a Kalman-adapted GAM absorbed the
removal of embedded generation capacities entirely (offline error up more than 10%, adaptive error
down 0.4%).

Considered and left out: Gioia et al. (2024) additive covariance matrices, and Browell, Gilbert &
Fasiolo (2022) covariance structures — both share information statistically rather than
topologically, which is the distinction gap 6 already draws with the hierarchical Bayesian example.
Telford et al. (2021) Dirichlet-sampled LV capacity and loss estimation — network headroom from
partial metering, not generation availability, but the right prior if the scale-up ever needs one
for unobserved customers. Browell, Stock & McMillan (2019), "Recommendation for the Evaluation of
Wind Farm Power Available Signal Accuracy" — scores an availability signal a farm already produces
from its own telemetry, so it is the opposite problem, but it is the obvious yardstick if
Flexpectation ever needs to score its own effective-capacity estimates. Donaldson, Browell & Gilbert
(2023) BigDEAL peak timing — already covered by Shukla & Hong (2024) and Gilbert et al. (2023); its
numbers were not verified. Tawn, Browell & Dinwoodie (2020), Messner et al. (2020), Möhrlen et al.
(2022) and ProbCast — all say what Hong et al. (2020) and Tawn & Browell (2022) already say in the
report.

## Full reference list

- Browell, J., van der Meer, D., Kälvegren, H., Haglund, S., Simioni, E., Bessa, R. J. & Wang, Y.
  (2025). The hybrid renewable energy forecasting and trading competition 2024. *International
  Journal of Forecasting*, 42(3), 709–723. arXiv:2507.01579.
- Browell, J. (2024). Hybrid Energy Forecasting and Trading Competition Data. Zenodo,
  doi:10.5281/zenodo.13950764.
- Pu, C., Fan, F., Tai, N., Liu, S. & Yu, J. (2025). A hybrid strategy for probabilistic forecasting
  and trading of aggregated wind–solar power: design and analysis in HEFTCom2024. *IJF*.
  arXiv:2505.10367. (Team GEB's method paper; code at github.com/BigdogManLuo/HEFTcom24.)
- Kaas, B., Treutlein, M., Gerber, H. B., Neumann, O., Phatthanakhuha, C., Resch, O., Mikut, R. &
  Hagenmeyer, V. (2026). Probabilistic Low-Voltage Peak Load Forecasting with Time Series Foundation
  Models Evaluated on Application-Oriented Metrics. arXiv:2607.01966.
- Treutlein, M., Bothe, P., Schmidt, M., Hahn, R., Neumann, O., Mikut, R. & Hagenmeyer, V. (2026).
  Real-world energy data of 200 feeders from low-voltage grids with metadata in Germany over two
  years. arXiv:2602.03521. (The FeederBW dataset.)
- Hertel, M., Pütz, S., Kolar, J., Schäfer, B., Mikut, R. & Hagenmeyer, V. (2026). A Benchmark for
  Electrical Load Forecasting Across Grid Levels. arXiv:2607.15705.
  github.com/KIT-IAI/load-forecasting-benchmark.
- Kleinebrahm, M., Berrisch, J., Eiser, P., Fichtner, W., Hagenmeyer, V., Hertel, M., Koster, N.,
  Lerch, S., Mikut, R., Priesmann, J., Schienle, M., Schaefer, B., Weinand, J. & Ziel, F. (2026).
  Energy-Arena: A Dynamic Benchmark for Operational Energy Forecasting. arXiv:2604.24705.
  energy-arena.org.
- TS-Arena — A Live Forecast Pre-Registration Platform. arXiv:2512.20761. Archive:
  huggingface.co/datasets/DAG-UPB/TS-Arena-Archive. **[Authors and year not verified — Kaas et al.
  cite a "TS-Arena Technical Report" as Meyer et al. 2025, and I have not confirmed that this is the
  same document.]** Sister live platform: 186 series from SMARD, gridstatus and FINGRID across 14
  challenges, leakage-free by gating foundation models on their public release date.
- Hong, T., Xie, J. & Black, J. (2019). Global energy forecasting competition 2017: Hierarchical
  probabilistic load forecasting. *IJF*, 35(4), 1389–1399.
- Hong, T., Pinson, P., Fan, S., Zareipour, H., Troccoli, A. & Hyndman, R. J. (2016). Probabilistic
  energy forecasting: GEFCom2014 and beyond. *IJF*, 32(3), 896–913.
- Hong, T., Pinson, P., Wang, Y., Weron, R., Yang, D. & Zareipour, H. (2020). Energy Forecasting: A
  Review and Outlook. *IEEE Open Access Journal of Power and Energy*, 7, 376–388.
- Shukla, S. & Hong, T. (2024). BigDEAL Challenge 2022: Forecasting peak timing of electricity
  demand. *IET Smart Grid*, 7(4), 442–459.
- Donaldson, D. L., Browell, J. & Gilbert, C. (2024). Predicting the magnitude and timing of peak
  electricity demand: a competition case study. *IET Smart Grid*. doi:10.1049/stg2.12152.
- Haben, S., Arora, S., Giasemidis, G., Voss, M. & Vukadinović Greetham, D. (2021). Review of low
  voltage load forecasting: Methods, applications, and recommendations. *Applied Energy*, 304,
  117798. arXiv:2106.00006.
- Pinheiro, M. G., Madeira, S. C. & Francisco, A. P. (2023). Short-term electricity load
  forecasting — A systematic approach from system level to secondary substations. *Applied Energy*,
  332, 120493. Open access (CC BY): escholarship.org/content/qt0s14445q/qt0s14445q.pdf.
- Haben, S., Ward, J., Vukadinović Greetham, D., Singleton, C. & Grindrod, P. (2014). A new error
  measure for forecasts of household-level, high resolution electrical energy consumption. *IJF*,
  30(2), 246–256. (The adjusted *p*-norm error used by Pinheiro et al. at substation level.)
- Haben, S., Voss, M. & Holderbaum, W. (2023). *Core Concepts and Methods in Load Forecasting: With
  Applications in Distribution Networks*. Springer.
- Gilbert, C., Browell, J. & Stephen, B. (2023). Probabilistic load forecasting for the low voltage
  network: forecast fusion and daily peaks. *Sustainable Energy, Grids and Networks*.
  arXiv:2206.11745. Data and R code: doi:10.5281/zenodo.7064279; ProbCast package.
- Browell, J. & Gilbert, C. (2020). ProbCast: Open-source production, evaluation and visualisation of
  probabilistic forecasts. PMAPS.
- Browell, J. & Fasiolo, M. (2021). Probabilistic forecasting of regional net-load with conditional
  extremes and gridded NWP. *IEEE Transactions on Smart Grid*, 12(6), 5011–5019. Open accepted
  manuscript: eprints.gla.ac.uk/250372. Preprint: arXiv:2103.10335. Data and code:
  doi:10.5281/zenodo.4618056.
- Ziel, F. (2019). Quantile regression for the qualifying match of GEFCom2017 probabilistic load
  forecasting. *IJF*, 35(4), 1400–1408. Open preprint: arXiv:1809.03561.
- Hong, T. (2016). Instructions for GEFCom2017 Qualifying Match. blog.drhongtao.com — the source for
  the nine-quantile format, the ten-zone hierarchy and the monthly round schedule.
- Rubattu, N., Maroni, G. & Corani, G. (2023). Electricity Load and Peak Forecasting: Feature
  Engineering, Probabilistic LightGBM and Temporal Hierarchies. arXiv:2305.05575.
- de Vilmarest, J., Browell, J., Fasiolo, M., Goude, Y. & Wintenberger, O. (2024). Adaptive
  Probabilistic Forecasting of Electricity (Net-)Load. *IEEE Transactions on Power Systems*, 39(2),
  4154–4163.
- Farrokhabadi, M., Browell, J., Wang, Y., Makonin, S., Su, W. & Zareipour, H. (2022). Day-Ahead
  Electricity Demand Forecasting Competition: Post-COVID Paradigm. *IEEE Open Access Journal of
  Power and Energy*.
- Taylor, J. W. & Buizza, R. (2002). Neural network load forecasting with weather ensemble
  predictions. *IEEE Transactions on Power Systems*, 17(3), 626–632. doi:10.1109/TPWRS.2002.800906.
- Taylor, J. W. & Buizza, R. (2003). Using weather ensemble predictions in electricity demand
  forecasting. *International Journal of Forecasting*, 19(1), 57–70.
- Möhrlen, C., Zack, J. W. & Giebel, G. (2023). *IEA Wind Recommended Practice for the
  Implementation of Renewable Energy Forecasting Solutions*. Elsevier.
- Ansari, A. F. et al. (2025). Chronos-2: From Univariate to Universal Forecasting.
  arXiv:2510.15821.
- Hewamalage, H., Ackermann, K. & Bergmeir, C. (2023). Forecast evaluation for data scientists:
  common pitfalls and best practices. *Data Mining and Knowledge Discovery*, 37(2), 788–832.
- Northern Powergrid & Faculty Science (2024). *Artificial Forecasting Alpha (SIF), WP2-D2 — Results
  for Scope Item 1.2: Forecasting of Customer Export Power and Net Demand* (82 slides), and *WP2-D2 —
  Results for Scope Item 2: Forecasting of Active Power at HV-LV Substations* (48 slides). Published
  15 April 2024 on the ENA Smarter Networks Portal, project NPG_SIF_006, together with the WP1-M1
  user research report and the DFQM technical review:
  <https://smarter.energynetworks.org/projects/npg_sif_006-1/>. Free download, no registration; the
  "CONFIDENTIAL" stamp on each slide is stale Faculty boilerplate (§10a).
- Northern Powergrid, Faculty Science, EV.energy & Oaktree Power (2026). *Artificial Forecasting Beta
  (SIF) — Annual Progress Report, March 2026* (52 pages). SIF reference 10145998 (§10a).

## Sources found by the 2026-08-22 search agents

Three literature-scout agents were run against the eight-problem structure. Everything below was
absent from this file before that run. Each entry says whether the full text was read.

### Effective capacity — the section's central absence claim was wrong

- **Meyers, Deceglie, Deline & Jordan (2020)**, "Signal Processing on PV Time-Series Data: Robust
  Degradation Analysis Without Physical Models", *IEEE J. Photovoltaics* 10(2) 546–553,
  <https://doi.org/10.1109/JPHOTOV.2019.2957646>. Abstract verified verbatim via OpenAlex: "This
  approach only requires a measured power signal as an input-no irradiance data, temperature data,
  or system configuration information are required", validated against RdTools on the same NREL
  dataset. **This kills the claim that solar capacity estimation needs irradiance the site does not
  measure.** It is now the open-source Solar Data Tools (NREL/Stanford), whose pipeline does
  capacity-change detection, clipping detection and a Monte Carlo degradation estimate — a
  distribution, not a point value. Abstract only; the package documentation was read.
- **Viotti, Arnqvist & Olauson (2026)**, "Estimating Wind-Power Capacity Time Series From
  Production Data Using a Power Curve Model and Quadratic Optimization", *Wind Energy* 29(8)
  e70136, <https://doi.org/10.1002/we.70136>. **Full text read** (CC-BY, Uppsala DiVA). Verified
  verbatim: the cumulative maximum "requires monotonically increasing capacity and relies on
  frequent high wind events"; 27.2% lower normalised MAE quantifying capacity after a step change;
  a forecasting model normalised their way scored 2.0% lower MAE and 2.3% lower RMSE day-ahead.
  Directly criticises the ratchet Dantas & Browell use, on the ground that matters to us — our
  effective capacity goes *down* when a turbine is out.
- **Pierrot & Pinson (2024)**, "On Tracking Varying Bounds When Forecasting Bounded Time Series",
  *Technometrics* 66(4) 651–661, <https://doi.org/10.1080/00401706.2024.2350421>, preprint
  arXiv:2306.13428. **Preprint read.** Capacity as the time-varying upper bound of a generalized
  logit-normal distribution, tracked online by normalised gradient descent jointly with the
  forecast. Anholt offshore, 14 months, 10-minute: CRPS 34.22% better than probabilistic
  persistence and 17.89% better than the same model with a fixed bound. Their motivation is ours
  verbatim — the bound "may change over time, while being unknown, for example in case of
  curtailment actions for which information is not available or not reliable". **This is the
  published precedent for the joint approach we described as our own.**
- **Perry, Muller & Anderson (2021)**, IEEE PVSC 48, <https://doi.org/10.1109/PVSC43889.2021.9518733>.
  Clipping detection scored against expert labels on 36 systems from AC power alone: F-score 85.0
  for a logic-based detector, 56.4 for RdTools; detector choice moves the degradation estimate by
  up to 0.6%/year. Abstract only.
- **Cronin et al. (2014)**, <https://doi.org/10.1002/pip.2310> — fleet-relative degradation without
  irradiance. **Peratikou & Charalambides (2022)**, <https://doi.org/10.1016/j.seja.2022.100015> —
  clear-sky output from photovoltaic data alone. Both abstract only.

### Substation forecasting and topology

- **Bernecker et al. (2025)**, *IJEPES* 168 110713, <https://doi.org/10.1016/j.ijepes.2025.110713>.
  **Numbers verified from the full PDF** (open access via d-nb.info when ScienceDirect 403s).
  Congestion-management cost held at a 95% confidence level: 3,102 EUR with standard load profiles
  against 86 EUR with a smart-meter forecast; a 1% cut in forecast-error standard deviation is
  worth ~1.4% of cost. **Refutes "nobody prices a risk-constant decision metric at distribution
  level"** — but on a modified IEEE 33-node *test* system, comparing two information levels rather
  than two forecasting models.
- **Campagne et al. (2025)**, arXiv:2507.03690. **Full text read.** GNN load forecasting on French
  regions and on the GB DNOs' open smart-meter feed (SSEN and NGED, ~2M meters, 50,000
  substations). Graph-aware models beat the baselines, but "for the UK data, data-driven graphs
  proved more suitable since that dataset exhibits finer spatial granularity and noisier
  correlations", and they are explicit that "the objective in forecasting is not to reproduce the
  transmission network itself". Their graphs are geodesic or correlation-based, never electrical
  adjacency — so the specific connectivity-map question survives, narrowly.
- **Bian et al. (2024)**, *IEEE TSG* 15(2) 1608–1619, <https://doi.org/10.1109/TSG.2023.3303469>.
  Recovers a price-taking storage operator's optimisation parameters by gradient descent on prices
  and observed dispatch, with a convergence proof. Abstract verified. The method to borrow for the
  trial area's battery.

### Disaggregation — and a production system this review had missed entirely

- **Teng et al. (2023)**, *RSER* 186 113662, <https://doi.org/10.1016/j.rser.2023.113662>. Abstract
  verified via OpenAlex. Trains on 10 fully-metered Dutch substations, then predicts solar and wind
  *separately* at substations with no renewable metering, from weather plus geospatial position and
  known facility capacity, at 15 minutes; RMSEP 0.07 against 0.70 for default transfer learning.
  **Refutes "nothing estimates unmetered wind behind a substation."** Qualifications that keep a
  narrower gap alive: it is a nowcast, it needs a metered population to transfer from, and it is
  *given* each site's capacity rather than estimating it.
- **OpenSTEF** (<https://lfenergy.org/projects/openstef/>) — Alliander's open-source forecasting
  stack under LF Energy, in production across thousands of grid connection points to 48 hours, of
  which Teng et al.'s method is the `split_energy` component. A second live production comparator
  alongside Pinheiro et al., and the only one whose code can be read.

### Leads not yet closed

- **Giamarelos & Zois (2024)**, *SEGAN* 38 101304, <https://doi.org/10.1016/j.segan.2024.101304> —
  a case study on one real HV/MV substation, our voltage level, but Elsevier blocked every route
  and Crossref carries no abstract. Worth a table row if the numbers can be got.
- **Li, Li & Wang (2026)**, *RSER* 226 116383, <https://doi.org/10.1016/j.rser.2025.116383> — a
  review of customer-baseline-load estimation, the mature literature on counterfactual load that
  was never metered. Metadata only; the abstract could not be retrieved. Relevant to problem 5, and
  NGED procures flexibility against such baselines.
- **Wang & Samworth (2018)**, *JRSS-B* 80(1) 57–83, <https://doi.org/10.1111/rssb.12243> — sparse
  multivariate change-point detection by projection. No energy application, which is the point: the
  fitted projection direction would have opposite signs on a donor substation and its recipients,
  which is exactly the partial multi-recipient transfer problem 4 describes.
- **Bloomfield et al. (2021)**, <https://doi.org/10.5194/essd-13-2259-2021>; **Lindas et al.
  (2026)**, <https://doi.org/10.1088/2753-3751/ae5ca6> — calibrated ensemble-to-energy forecasts
  past day 10, at country scale. The reference implementations for the 14-day ambition.

### Tool notes for the next search

- `arxiv.org` is blocked; `export.arxiv.org` works but **301-redirects to https, so curl needs
  `-L`** or it silently returns nothing. Quoted multi-term `all:` boolean queries return nothing;
  use `abs:"phrase" AND abs:word`.
- `mdpi.com`, `sciencedirect.com` and `linkinghub.elsevier.com` return 403. Elsevier open-access
  articles are sometimes served whole by the German National Library mirror at `d-nb.info`.
- OpenAlex has a daily free budget that two of the three agents exhausted; Semantic Scholar
  rate-limits without a key. Citation-graph walks are therefore the least-explored avenue.
