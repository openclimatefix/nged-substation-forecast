# Plan: a substation-limit curtailment-cost metric for many nodes

## Take-home

**The metric asks one question per half-hour: will a bulk supply point (BSP) or grid supply point
(GSP) export more than its firm limit, and if so, which generators beneath it must be capped, and
what does that cost in pounds?** A generator is curtailed only when the forecast says the
substation's export would exceed the substation's limit. The metric is a generic function over a
node table and a generator table, so the same code scores the 4-node pilot today and the ~2,500
NGED series due in January 2027. Primaries are out of scope until NGED supplies the topology, and
experiment 83's single-series cost metric is deleted first.

**Every result is reported twice, as two co-equal headlines.** (a) At the real firm limit the pilot
has about 0 to 1 events, so the figure demonstrates the arithmetic only. (b) At a scaled limit,
defined by one rule that works for every node, including the ~2,500 series where no real limit is
known, the pilot has enough events to exercise the arithmetic. Scaled-limit figures are a
calibration to history, not a saving. Both are headlines because the real limit is what NGED
enforces but almost never binds, and the scaled limit is the only one that exists for every node.
A ranking of forecasts waits for the January 2,500-series delivery unless the event counts in
section 1 are large enough.

## 0. Stage 0 — delete the single-series metric (its own PR, before any new code)

**`studies.curtailment_cost` scores each meter alone against a percentile of that meter's own
history, and curtailment does not bind at a single meter, so the module and its worked example are
removed.** The stage is a deletion with one move, so it ships as its own small PR off `main`.

- **Delete** `packages/studies/src/studies/curtailment_cost.py` and
  `docs/ml-experiments/00083-curtailment-cost-and-tail-calibration-worked-example/figure_1_cost.svg`.
- **Move** `quantile_levels`, `align_arms`, and `read_arms` (with `ROW_KEYS` and
  `QUANTILE_COLUMN_PATTERN`) into `packages/studies/src/studies/tail_calibration.py`, which already
  owns the per-arm tail table those helpers feed; no new module. Their existing tests in
  `packages/studies/tests/test_curtailment_cost.py` (`test_quantile_levels_sorts_and_requires_two_columns`,
  `test_align_arms_keeps_shared_rows_and_drops_a_nan_actual`, and the `align_arms` half of
  `test_a_nan_actual_in_a_later_arm_is_dropped_by_align_and_refused_without_it`) move into
  `packages/studies/tests/test_tail_calibration.py`. `read_arms` has no test: add one that writes two
  small parquet files and reads them back. Then delete `test_curtailment_cost.py`.
- **Trim** `ml-experiments/00083-.../e00083_report.py` to section D (the tail table), importing from
  `studies.tail_calibration`, and drop `draw_figure`.
- **Rewrite experiment 83's record** to the tail-calibration result, keeping the folder slug so no
  link breaks. `result.md` keeps verdict `inconclusive` (the ml-experiment skill's vocabulary has no
  "withdrawn"; the tail result stands); `verdict_note` says the single-series cost half was
  withdrawn because curtailment binds at a substation, and `headline`, `best_arm`, and the `advice`
  item naming `studies.curtailment_cost` are rewritten. Also the `mkdocs.yml` nav title and the
  research-log row and entry (the figure link goes).
- **Docs:** delete the `curtailment_cost` bullet in `packages/studies/README.md` and extend the
  `tail_calibration` bullet. In `docs/roadmap/cost-savings-metrics.md`, delete the sentence in
  "Choosing the limit" describing the deleted code. In `docs/background/requirements.md` (about
  line 64), replace "nets at one primary before summing up the substation hierarchy" with a
  description of scoring against BSP and GSP limits, keeping the `#metric-2-curtailment-cost` link.
  In `docs/roadmap/metrics-and-leaderboard.md` (about line 770) and
  `docs/techniques/evaluation-metrics.md` (around lines 487 and 551), remove the claim that the
  curtailment metric uses the leaderboard's `historical_p99` rung or holds every model to equal
  risk; the leaderboard's own `historical_p99` stays. The "Choosing the limit" heading text stays
  unchanged so the inbound `#choosing-the-limit` anchor survives.
- **Check:** `grep -rn "curtailment_cost" --include=*.py --include=*.md --include=*.yml .` finds
  only the roadmap's `METRIC_NAMES` line (a future `Metrics` column name, which stays); the
  `studies` tests, the study-boundary test, `mkdocs build --strict`, and `pymarkdown` pass.
- **Memory:** the maintainer's memory note "Curtailment and tail calibration" still asks for the
  simple metric; flag it for the maintainer to update once this plan is decided.

## 1. The node and generator tables: the metric's only input about the electricity network

**Every node-specific fact lives in two Polars frames, validated on entry, so 2,500 series need
longer tables, not new code.**

- `nodes`: `node_id`, `node_type` (`BSP` or `GSP`), `parent_node_id` (nullable, from NGED's
  private `Substations.csv`), `export_sign` (`+1`, `-1`, or null), `sign_source` (text), and one
  row per `(node_id, season)` of `export_limit_mw` and `import_limit_mw` with `limit_source`.
- `generators`: `generator_id`, `node_id`, `capacity_mw`, `connection_date` (nullable, null in the
  pilot), and `is_forecast` (bool, from the metadata and experiment 48's manifests).

**Our own power data is authoritative for flows and signs; NGED's public portal is used only for
published limits and season boundaries.** The portal's flows are multi-meter sums with wrong signs,
so no sign argument rests on them and no portal flow is loaded. Net export is `export_sign x flow`.
A node with a null `export_sign` is excluded and counted as `excluded_unknown_sign`, never guessed.

| Node | `export_sign` | Evidence, all from our own data |
|---|---|---|
| BOSTON BSP, SKEGNESS BSP | `-1` (positive flow = import) | BSP ≈ sum of our Primaries beneath it minus our generators beneath it (correlation 0.977 at BOSTON, 0.984 at SKEGNESS); generators correlate -0.80 and -0.72 with the BSP; a June–July midday dip |
| BICKER FEN GSP | null until NGED confirms; `+1` suspected | Positive looks like export: correlation +0.71 with the generators, -0.29 with the Primaries, -0.51 with its child SKEGNESS, and a positive midday mean |
| WALPOLE GSP | null until NGED confirms | Leans positive = import (-0.51 with the generators, +0.30 with BOSTON), but its published export limit is crossed on 53% of half-hours under that reading |

**Seasons are NGED's, as whole months:** Winter December–February, Intermediate Cool March, April
and November, Intermediate Warm May, September and October, and Summer June–August.

**The limit is the firm (N-1) limit: the smaller of the transformer pair's per-transformer pre-event
limits (PELs).** Each BSP's two transformers are redundant, so one carries the full load when the
other is out. Pilot export limits, the same in all four seasons: SKEGNESS 29.8 MW, BOSTON 45.0 MW.
The import limit follows the same rule and is used for the import census only. BOSTON's all-history
export maximum, 45.08 MW, sits on the 45.0 MW limit, so 45.0 MW may be an enforced operating ceiling
or a clipped meter rather than an exceedance. BICKER FEN's export Technical Limit (263.5 MW) is
crossed on 2 half-hours if its sign is `+1`; a GSP limit is used only once NGED confirms the sign.

**Two limits, both reported.** The real limit is the firm PEL above, kept as its own row. The scaled
limit is `L = f x` the node's maximum observed export (the negative tail for import-positive nodes),
computed only from history before the scored year (before 2025-07-02), with `f` = 0.9 by default and
0.75 and 0.5 swept. "p90" here means 0.9 x the maximum, not the 90th percentile of the distribution,
because a percentile of a node's own flow is exceeded on 10% of half-hours by construction. The rule
needs no external limit, so it applies to every future series.

Half-hours in the scored window (2025-07-02 to 2026-06-30, 17,471 half-hours for BOSTON and
SKEGNESS, 15,995 for BICKER FEN) with export above `f x` the pre-scored maximum, from cleaned
unflagged power:

| Node (pre-scored maximum export) | Real limit | f = 0.9 | f = 0.75 | f = 0.5 |
|---|---|---|---|---|
| BOSTON (39.15 MW) | 1 at 45.0 MW | 17 | 36 | 215 |
| SKEGNESS (20.02 MW) | 0 at 29.8 MW | 2 | 10 | 108 |
| BICKER FEN, sign `+1`, unconfirmed (284.03 MW) | not used | 0 | 84 | 734 |

**The counts show the pilot cannot rank forecasts at the default.** At f = 0.9 there are 19
half-hours across the two nodes used in scoring, which is too few to separate arms, and BICKER FEN
has none. At f = 0.75 there are 46, and only at f = 0.5 (323) are there hundreds, at limits that
are half the observed peak and so not an operating condition. The pilot therefore proves the
arithmetic and reports counts and GBP at all three `f`; a ranking waits for the January delivery.

## 2. The rule

**Rows.** Forecast leads of at least 12 hours and under 36 hours, as in experiment 83.

**Planned node curtailment.** `v_node = max(0, q_tau(export) - L) x 0.5 h` in MWh, where `L` is the
limit in MW (real or scaled) and `q_tau(export)` is the tau quantile of net export in MW (the `1 - tau`
flow quantile when `export_sign` is `-1`). `tau = 1 - 1/r` with the breach ratio `r` fixed at 20
(tau 0.95, the highest level experiment 48's stored sets hold). The manual heuristic is scored at
the same tau.

**Nesting is deferred to stage 3.** The pilot's only parent-child pair is BICKER FEN over SKEGNESS,
and BICKER FEN's sign is unconfirmed, so a nesting pass would change no pilot number.

**Generators: who is curtailable.** Metadata holds 11 EHV-customer generator series (BOSTON 6,
SKEGNESS 5); experiment 48 forecasts 9 of them (BOSTON 4, SKEGNESS 5). The two unforecast BOSTON
series, one of them a battery, are not curtailable under this rule, because capping them from
their actual output would leak the outcome.

**Allocation order, a stated assumption: pro rata by capacity.** Generator `g`'s planned cut `v_g`
in MWh is its capacity share of `v_node`. A share larger than the generator's forecast energy
`f_g x 0.5 h` is clamped to that forecast energy, and the excess is redistributed to the unclamped
generators by capacity, repeating until nothing is clamped or every generator is at its forecast.
Any remainder is unallocated and shows up as residual breach. Last-in-first-out (LIFO) by
connection date is a sensitivity only where dates exist, so it is not built for the pilot.

**Caps and what they cost the generator.** `cap_g = max(0, f_g - v_g / 0.5 h)` in MW, from the
forecast median `f_g` in MW. The energy actually lost is `c_g = max(0, a_g - cap_g) x 0.5 h` in MWh
against metered output `a_g` in MW. This is where the generator forecasts enter the score. The
node's residual breach in MWh is `max(0, actual export - sum_g c_g / 0.5 h - L) x 0.5 h`.

**Import-side exceedance is counted, not priced.** Curtailing generators cannot relieve an import
constraint; Metric 1 of the
[cost-savings roadmap](https://openclimatefix.github.io/nged-substation-forecast/roadmap/cost-savings-metrics/)
prices that. The output reports import-exceedance half-hours per node and season.

**Node-only mode is the simple variant:** the same nodes, limits, and tau, no generators, with
`v_node` assumed delivered exactly.

## 3. Price: a pound-sterling headline

**The headline is an energy proxy for NGED's payment rule, in GBP: `price x (sum_g c_g + r x
residual breach)`.** DCUSA Schedule 2D v15.3, clause 3.2, measures Full Export Curtailment Hours
as `sum(duration x Curtailment Instruction Value) / Curtailable Export Capacity`, and the
Curtailment Instruction Value is the instructed reduction of the generator's Maximum Export
Capacity, not the energy actually lost. Clause 3.4 pays `(hours - Export Curtailment Limit) x
capacity x Exceeded Curtailment Price`. The headline prices energy actually lost instead, and is
labelled "energy proxy" wherever it appears, for two reasons: tau = 1 - 1/r minimises it, and the
DCUSA figure depends on how NGED's active network management expresses a setpoint against Maximum
Export Capacity, which is unknown.

**The DCUSA-form figure is reported beside the headline.** With instruction value `capacity_g -
cap_g` (clamped to `[0, capacity_g]`) on every half-hour where `v_g > 0`, Full Export Curtailment
Hours are `sum(0.5 h x (capacity_g - cap_g)) / capacity_g`, and the payment is that minus the
Curtailment Limit, times `capacity_g x price`. Under this form an instruction to a generator running
well below capacity is charged for the whole headroom, so the two figures can rank arms differently.
Megavolt-amperes (MVA) are taken as MW at unity power factor, and all capacity as curtailable.

**Three prices, one headline:** GBP 380, 2,000 (headline), and 11,100 per MWh, the ends and the
geometric middle of the GBP 380 to 11,100 per MVAh range reconstructed from April 2025 contract data
(not a published figure). The breach is priced at `r x price`. Results are reported as total GBP
over the scored window and as GBP per MW of curtailable generator capacity per year.

**Ranking is done only at a Curtailment Limit of 0.** There tau = 1 - 1/r is the cost-minimising
quantile for the energy proxy, so the score is proper (exactly so when the generator forecasts are
perfect; otherwise it is the operating rule every arm shares). At a Curtailment Limit of 0 the price
scales every arm alike, so the ranking is price-free but the GBP figure is not. A Curtailment Limit
of 100 or 500 hours a year is shown, if at all, as a restatement of the same plans, never ranked.

**Avoidable cost** is arm cost minus perfect-forecast cost on the same rows; the perfect forecast
uses actual node flow and actual generator output. **Avoided reinforcement** is named in the
write-up as the alternative framing and not built.

## 4. Code (stage 1)

**One module, `packages/studies/src/studies/node_curtailment.py`, with
`packages/studies/tests/test_node_curtailment.py`.**

```python
SEASONS: Final[Mapping[int, SeasonType]]   # NGED's whole-month seasons
def node_table_pilot(*, metadata, substations, portal_limits) -> NodeTable
def node_curtailment(*, table: NodeTable, node_forecasts: Mapping[str, pl.DataFrame],
                     generator_forecasts: Mapping[str, pl.DataFrame],
                     assumptions: NodeCurtailmentAssumptions) -> NodeCurtailmentResult
    # per_arm (energy proxy and DCUSA form, GBP and MWh), per_node_season, event_census
    # (export and import), month_totals (for bootstrap_month_totals), excluded nodes with reasons
```

`NodeCurtailmentAssumptions` holds the price, `r`, the Curtailment Limit, the lead window, the limit
scale, and `use_generators`. The arithmetic runs in Polars over `(node, valid_time)`, linear in
rows, with no per-node Python loop. Input sets are fractions of capacity, converted to MW with each
series' `effective_capacity_mw`, an estimate whose error carries into every cap.

**Tests, each worked by hand in its docstring, units stated:**

- per-season limits picked by month on NGED's seasons;
- a node with `export_sign` `-1` and flow -70 MW against BOSTON's 45.0 MW limit is a 25.0 MW
  excess, 12.5 MWh in the half-hour;
- a null sign is excluded and counted;
- perfect forecasts cost exactly the needed curtailment (MWh) and no breach, **when** the
  curtailable generators' actual output covers the excess; otherwise the uncovered MWh is breach;
- a generator forecast too high leaves a loose cap and the breach (MWh, GBP) is charged exactly;
  one too low over-cuts by an exact MWh;
- pro rata by capacity for two unequal generators, including a share clamped at one generator's
  forecast and redistributed to the other, and a remainder that becomes breach;
- the DCUSA form charges `capacity - cap` (MW) x 0.5 h on an instructed half-hour, and the excess
  hours rule at Curtailment Limits of 0 and above the hours used;
- an import exceedance is counted and not priced;
- node-only mode equals full mode with one generator forecast perfectly, **when** that generator is
  the only one and its actual output is at least `v_node / 0.5 h` MW.

**Roadmap rewrite, in the same PR:** `docs/roadmap/cost-savings-metrics.md` Metric 2 loses Tier 1
(per-Primary netting); Tier 2 becomes the node metric described here, with the firm limits, the
explicit sign, and the DCUSA price structure; "Curtailment price basis" and "Choosing the limit"
are rewritten to match, the latter keeping its heading text.

## 5. The pilot worked example

**An ml-experiment, figure-led and exploratory, with no verdict on the forecasts.** It re-scores
experiment 48's stored sets on `fold_a_daily_gap_v1` (2025-07-02 to 2026-06-30, seed 0) as paired
node and generator arms: best guess (`weekly` + `gen_monthly`), best-guess stack (`frozen_cs` +
`gen_frozen`), manual heuristic at tau 0.95, climatology, the perfect forecast, and a diagnostic arm
of best-guess nodes with actual generator output. Set names are checked against experiment 48's
manifests first.

**The first output is the event census, and it decides what the rest may claim.** For each node and
season, over all history (SKEGNESS from 2019-09, BOSTON from 2022-07) and over the scored year,
count export and import exceedances at the real limit and at each `f`. The write-up says plainly
whether the counts are large enough to rank forecasts, and labels every GBP figure at the real limit
a demonstration of the arithmetic and every scaled-limit figure a calibration to history.

**Four figures:** (1) the event census; (2) each node's net-export distribution by season with the
real limit and the three scaled limits marked; (3) a hand-checkable known-answer example through the
whole chain beside the hand-computed answer; (4) the headline: avoidable cost per arm in GBP (energy
proxy, headline price, Curtailment Limit 0) with month-resampled 95% intervals, drawn twice, at the
real limit and at f = 0.9, with the other two prices and the DCUSA form as marks beside it.

## 6. Licence, attribution, and anonymisation

**NGED's portal limits are public, but the generator joins are not.** Every page or figure using a
portal limit carries the licence attribution "Supported by NGED Open Data". `Substations.csv` and
every table joining a generator series to a node stay in the private data store. Nodes may be named;
no generator appears singly, in MW, or with an identifier in any published output or reviewer
brief, only summed over at least 4 generators and normalised.

## 7. Sizing and reviews

**The five triggers:** stored data are a private experiment folder only, with no production table or
contract; the production serving path is untouched; no degradation rule is touched; more than one
design is defensible (allocation order, energy proxy versus DCUSA form); and every caller can be
named, since stage 0 removes the only caller of the deleted module. **Stage 0 is simple** (deletion
and a move): no plan review, verified by the checks in section 0. **Stage 1 is medium-to-complex:**
Sonnet implements tests first, one Opus review of the code before its first run (a wrong metric
silently mis-ranks every later experiment), and a Sonnet first-stumble and `prose-review` pass on
the write-up. Both PRs touch `packages/`, so neither merges on its own.

## 8. Questions still open for the maintainer or NGED

1. **The sign and definition of the WALPOLE and BICKER FEN GSP series (awaiting NGED),** and whether
   WALPOLE's published export limit applies to what our series meters.
2. **The October 2026 Exceeded Curtailment Prices** for the East Midlands licence area.
3. **NGED's confirmation of how active network management expresses an instruction.** The one
   on-disk ANM record (`data/studies/anm/README.md`, one trial-area PV generator, 8 Feb 2024 to
   20 Sep 2026) stores the cap in force as an absolute level, not an instruction value or a cut from
   output: for example 0, 0.25, 10, and 18 MW against an 18.799 MW connection limit, at the limit
   81% of the time since go-live and at 0 MW 12%. That weakly suggests NGED expresses an instruction
   as a new cap on Maximum Export Capacity, which matches DCUSA Schedule 2D clause 3.2. The record
   is not under the SKEGNESS or BOSTON nodes, so it cannot validate the node-level metric. A mostly
   binary cap (full or zero) makes "curtailed or not" the natural cost unit. Still wanted: what
   share of each connection is curtailable.
4. **Curtailment Limits in hours and firm or non-firm status** for the 9 forecast generators.
   Connection dates and Embedded Capacity Register matching are deferred by the maintainer, so
   LIFO runs only as a sensitivity once dates exist.
5. **Are the firm PELs real limits?** NGED floors a PEL at 30% of rating, and SKEGNESS's 29.8 MW
   sits near that floor; is BOSTON held at 45.0 MW by an operating ceiling or a clipped meter?
6. **The format of the ~2,500-series January delivery:** does it carry the hierarchy, each series'
   sign convention, and generator capacities, so the tables can be built from it directly?

## 9. Build order

- **Stage 0 (own PR):** section 0.
- **Stage 1:** build the pilot tables from the portal limits and `Substations.csv` (stored privately
  with their lineage), the tests, `node_curtailment.py`, and the roadmap rewrite; commit `result.md`
  as `planned`; print the event census; run the Opus code review; run the report; draw, review, and
  write up.
- **Stage 2, when NGED supplies the topology:** Primaries, with the generators placed under them,
  and a check that the needed curtailment above the one generator under active network management
  coincides with its real cap events.
- **Stage 3, the January 2027 delivery:** build the tables for ~2,500 series, add nesting and LIFO
  where dates exist, and run the same function; this is where a ranking becomes possible.
