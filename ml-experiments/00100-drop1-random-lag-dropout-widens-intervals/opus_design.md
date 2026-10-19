# Experiment 100 (DROP1): design

Opus design review of `design_brief.md`, with the maintainer's addition of 2026-10-19 (valid truth,
interval behaviour as the primary read, injected switches for what real data cannot validate, no
point-accuracy metric during a real ARA) folded in. Nothing has been fitted.

## Facts checked before designing

- **Labelled set.** Experiment 41's `labelled_set.parquet` holds 82 visible events, which are the
  kinds `real_operation`, `real_short`, `real_edge_close`, `real_edge_open`, and `alarm`, with the
  visibility `conserved` or `one_sided`. Of the 82, 66 are `real_operation` (34 of them longer than
  1 day, median 28 h), 9 are `alarm`, 4 are `real_edge_open`, 2 are `real_short`, and 1 is
  `real_edge_close`. In total 42 last longer than 1 day. `e00086_labels.load_labels` builds
  `in_event` from exactly these 82 events and `closed` from the logged closures. Use that function
  unchanged; the label set is the same as experiment 86's.
- **How many rows the mask touches.** In experiment 86's `plain` metrics (3,199,243 cases), 2.08% of
  all rows have one of the three lags in an event, and 37,394 (1.17%) are valid-truth rows with such
  a lag. Those 37,394 rows are concentrated: 81% come from three series (D18 12,732, D15 11,287,
  D16 6,253), five series have none, and two valid-time months (2025-09 and 2026-05) have none.
  Five months (Jul, Oct, Nov, Jan, Jun) hold 89%. Coverage of the 5-95% band on these rows falls
  with lead time, from 0.83 at 0-24 h to 0.77 beyond 6 days.
- **Lag columns.** `common.LAG_COLUMNS` is `lag_origin`, `lag_same_time_latest_day`, and
  `lag_same_time_latest_week`. All three lie within the 7 days before the newest usable value, and
  the first two within its last 24 hours. An event masks them jointly: a run whose newest value is
  inside an event loses `lag_origin` for every valid time, and usually the day lag as well.
- **Experiment 89's `M` arm did see missing lags in training.** `e00089_fit.train_matrix` applies
  `for_arm` to the training rows, so about 13% of training rows had masked lags. The brief's
  explanation ("never saw missing lags") is therefore incomplete. The missingness in experiment 89
  was confined to rows just after a switching interval, and was absent at series with few labels.
  The ARA marked 9 to 16% of half-hours, mostly RO-NC detections NGED did not log, so masking also
  discarded many good lags. Random dropout changes the first two conditions only. The `mask` arm
  below (logged events only) tests the third condition.
- **Stored `plain` has nine levels (q05 to q95).** It cannot give the 1-99% band. Experiment 96's
  `e00096_common` already carries 11 levels (q01 to q99) and the matching `case_metrics`.

## Valid truth (pre-registered, maintainer item 1)

- **Real rows with valid truth:** the target's valid time is outside every labelled event and every
  logged closure of the series (experiment 86's `nra` flag). Among them, the **masked rows** have at
  least one lag inside a labelled event (`nra & lag_oracle`, the 37,394 rows). The **unmasked rows**
  are `nra & ~lag_oracle`.
- **Real rows whose target lies inside an event or a closure have no valid truth for the normal
  running arrangement.** They are never scored, for any arm or metric.
- **Injected rows:** experiment 88's family A cases (2,452 held-out injected cases, shapes S2, S3, S4,
  and S6, newest usable lag inside the injected transfer, leads up to 240 h). Truth is the
  unmodified series, so the rows are valid both when the target lies after the transfer and when the
  target lies inside it (the target-in-ARA case real data cannot score). The test transfers keep
  experiment 88's 7-day quiet rule, so unlogged real switching contaminating the truth is
  unlikely, though not excluded.
- **Not a metric:** point accuracy of real rows during a real ARA (maintainer item 4).

## Arms (four, seed 0, all refitted at 11 levels)

All arms use experiment 89's `r` recipe: 365-day rolling window, control member, monthly P-roll
refits, 150 rounds, eta 0.15, half-life 0.5 years, `common.QUANTILE_PARAMETERS` with the 11 levels
of `e00096_common.LEVELS`, a 51-member mixture at predict time, and the spread factor 1.15 applied
to every arm.

1. **`plain11`**: experiment 86's `plain` recipe refitted at 11 levels. All targets kept, no lag
   masked. It replaces the stored nine-level `plain` as the benchmark, because the 1-99% band is a
   primary read. Guard: its q05 to q95 quantile score must reproduce experiment 86's stored `plain`
   on all `nra` rows to within 0.01 points (experiment 96 reproduced to 0.000).
2. **`mask`**: drop training targets inside a labelled event or a logged closure (the `nra`
   complement). Set each lag inside a labelled event to missing, in training and at test time. No
   dropout.
3. **`drop10`** (primary arm): `mask`, plus pseudo-event dropout in training only, at a 10% rate.
4. **`drop30`**: the same pseudo events, at a 30% rate (dose response, descriptive).

**Pseudo-event dropout.** For each series, draw an ordered list of random intervals on the dense
grid. Draw each interval's onset uniformly over the training period and its duration by bootstrap
from the 82 labelled events' durations. Seed the draw with `hash((SEED, site))` so that it is fixed
across refits. A training row's lag is masked if the lag time lies inside a labelled event (as in
`mask`) or inside a pseudo event. `drop10` uses the first k10 intervals of the list and `drop30` the
first k30, so the masks are nested. Choose k so that, over each series' training rows across all
refits, the share with at least one pseudo-masked lag is 10% (30%) within 1.5 points. Print the
achieved share per series and refit. Pseudo events never drop a target, because their targets are
ordinary.

## Answers to the brief's questions

### Question 1: the dropout mechanism

- **Mask in place (replacement), not by adding copies.** Replacement keeps the rows, the targets,
  and the recency weights identical to `mask`. `drop10` minus `mask` then differs only in which lags
  are missing, and no duplicate-row control arm is needed. Augmentation would give the selected
  targets double weight, or else need a split-weight scheme, and it would change the matrix's row
  count under `subsample=0.8`. Replacement does cost the lag on 10% (30%) of training rows. That
  cost is the sharpness cost the no-harm read measures, and production would pay it too.
- **Mask intervals, not independent lags.** Pseudo events reproduce the joint masking patterns real
  events produce: origin and day lag together for a whole run, the week lag alone, and every
  pattern in between. Independent per-lag dropout would mostly teach the patterns real events rarely
  produce, such as the day lag missing with `lag_origin` present. The dropout is correlated within a
  run, as real masking is. Each masked run still holds hundreds of valid times with varied weather
  and targets.
- **Rate as a share of rows, not of lag cells.** Independent 10% per lag would mask about 27% of
  rows. Matching the 1.2% share of rows with a lag in an event would give too few masked rows per
  tree node to learn from. A 10% row share is the primary rate, and 30% is the dose check.
- **Recency weight.** A masked row keeps its own weight. This follows automatically from
  replacement.
- **XGBoost's default branch.** At each split, XGBoost sends a missing value to a default child.
  The default is learnt from the training rows in that node that had the value missing. Where a node
  held no such rows, the default comes from the order of split enumeration and is not learnt: a
  missing lag then behaves like an arbitrary extreme lag. That is a biased point forecast, which is
  the failure the idea targets. With rows masked only around events at three series, most nodes at
  most series have no masked rows. Spreading masked rows over every series and period is what
  dropout fixes. **Remaining risk:** the default child pools the missing rows with the low-lag or
  the high-lag rows of that node, not in a leaf of their own. The quantile leaves for a missing lag
  are therefore shared with one side, and widening is partial. Missingness indicator columns would
  allow a dedicated branch, but they are a separate design question, left out to keep four arms (see
  "Next").

### Question 2: tails when a lag is missing

- `reg:quantileerror` with a vector `quantile_alpha` fits each level's trees separately, so each
  level learns its own default directions. No monotone constraint is needed: lags are not monotone
  in the target once the trees use weather.
- Quantiles can cross. Keep the existing fix (sort after the mixture, `mixture_quantiles` then
  `np.sort`). Report the rate of crossing before the sort per arm, on masked and on unmasked rows.
  A higher rate on masked rows in the dropout arms would show the levels' default branches
  disagreeing.
- The spread factor of 1.15 was tuned on the plain model. Keep it for every arm, as pre-registered.
  Report the coverage without it (factor 1.0) as a descriptive read, because a dropout arm could
  look over-wide only because of the factor.

### Question 3: training rows and masks

Use `load_labels` (82 events, from 2022-01). In training, drop targets that lie inside an event or a
closure, and mask lags that lie inside an event, as in `e00089_fit.add_lags` and `for_arm` with
`ara = in_event | closed` for targets and `in_event` for lags. The labels use hindsight: an event's
end is known afterwards, as in experiment 86. Print, per series, the share of training rows with a
lag in a real event, so the write-up can compare it with the dropout rate. The expected share is
0 to about 8%, since 2.1% of all scored rows have such a lag.

### Question 4: scoring, contrasts, and decision rule

**Primary reads are interval behaviour (maintainer item 2), paired against `plain11` on the same
rows**, with whole valid-time months resampled (injected rows: the transfer's onset month), and
20,000 resamples.

- Coverage of the 5-95% band (nominal 0.90) and of the 1-99% band (0.98). The share below q01 and
  q05 and above q95 and q99. Pinball loss at q01, q05, q95, and q99.
- **Width:** the mean of q95 minus q05, and of q99 minus q01, in points of capacity. **Widening
  ratio:** an arm's mean width divided by `plain11`'s mean width on the same rows, on masked and on
  unmasked rows separately. Do not compare masked rows with unmasked rows inside one arm as the
  headline: their series, months, and leads differ.
- The quantile score is the proper score and guards against widening that is merely wide.
  **Coverage is not a proper score.** A model reaches 0.90 by widening everywhere, which is why each
  coverage gain must come with a quantile score that does not worsen.

**Planned contrasts (98.75%, Bonferroni across four). Arm is `drop10` minus `plain11`.**

- **P1:** change in 5-95% band coverage on the real masked rows (37,394). Prediction: +0.03 to
  +0.10, from 0.78 towards 0.90.
- **P2:** change in 5-95% band coverage on the injected family A rows, with lags masked by the
  injected interval (oracle masking). Prediction: +0.02 to +0.08.
- **G1:** change in quantile score on the injected family A rows. Prediction: -0.05 to -0.40 points
  of capacity (masking removes the contaminated anchor). Guard: upper end at most +0.02.
- **G2:** change in quantile score on the real unmasked rows. Guard: upper end at most +0.02 points
  (experiment 86's margin, 0.37% of the score). The widening ratio on those rows must be at most
  1.03; its 98.75% interval is reported, and only the point value is in the rule.

**Verdict.**

- `supported`: P1 and P2 have lower ends above zero, G1 and G2 pass, and `drop10`'s 5-95% band
  coverage on the masked rows of both kinds is no higher than 0.95 (no overshoot).
- `small gain`: P2 holds, G1 and G2 pass, and P1's interval includes zero (real data too thin to
  confirm).
- `refuted`: P2's upper end is below zero, or G1's lower end is above +0.02. Either would mean the
  widening is absent or in the wrong place.
- `harmful elsewhere` (reported with `not-for-production`): G2 fails, whatever the others show.
- `inconclusive`: otherwise.

**Threshold.** Dropout lives in the training loop and adds nothing to serving, so its threshold is
zero (an interval excluding zero). Test-time masking needs an event label on the serving path,
which is a later decision. All masked-row results here use hindsight labels and are upper bounds.

**Pre-registered descriptive reads (95%):**

- `drop10` minus `mask` on every read above (attribution: does dropout add to masking alone?).
- `mask` minus `plain11`.
- `drop30` minus `drop10` (dose).
- Every read by lead band.
- Injected family A split by target after the transfer versus target inside it.
- Injected cases with RO-NC causal masks in place of oracle masks, which come cheaply because
  experiment 96's harness already runs the causal detector per origin.
- Real masked rows split into the three dominant series and the rest.
- Quantile score on the real masked rows (expect an interval of about 2.7 points, as experiment
  86's C2 had).
- The climatology and `manual_heuristic` rows of experiment 48 on the real rows.

### Question 5: pitfalls

- **Sample size versus dropout.** Replacement removes this confound (Question 1). With
  augmentation, an arm adding unmasked duplicates would have been needed.
- **Power.** The real masked rows are about 15 to 20 events at three series in roughly 10 months.
  P1 may well be inconclusive, which is why P2 and G1 exist. State the effective counts (events,
  series, months) beside every real-row number.
- **Leakage.** Pseudo events and the dropout draw must not read the target or the test period. The
  real masks use hindsight labels in training and test alike, as experiment 86 did. A refit must
  see no row with a valid time later than `refit - REFIT_CUTOFF_AGE` (copy `known_rows`). The
  pseudo-event list covers the training period only and is never applied at test time.
- **Natural missing lags.** Telemetry gaps already make some lags NaN in every arm, including
  `plain11`. Print their share. If it exceeds about 1% of training rows, it partly explains why the
  default branches in `plain11` behave sensibly.
- **Experiment 89's harm.** That harm came from over-marking (9 to 16% of half-hours, mostly
  detector alarms) and from correlated missingness. Training exposure alone does not explain it,
  since `M` trained with masked lags. If `mask` here also loses to `plain11` on quantile score, the
  lost anchor is the cause, and dropout cannot fix it.
- **Spread factor.** The factor is shared by every arm (Question 2).
- **Injected and real scoring use different ensemble sizes.** The injected scoring uses 11 members
  (experiment 96's `MEMBERS`) and the real scoring uses 51. Compare arms only within one row kind.
- **Stored `plain` versus `plain11`.** Use `plain11` everywhere. The stored nine-level `plain` is
  only for the reproduction guard.

### Question 6: injected-case scoring

Copy `e00096_cases.py` and change only the variants and the prediction. The variants are `plain11`
on contaminated lags (`plain`), `plain11` on unmodified lags (`ceiling`, descriptive), and `mask`,
`drop10`, and `drop30` with lags masked inside the injected interval (`inj`). Add the same three
arms masked by the causal intervals (`caus`, descriptive). `predict` builds the lags from the
injected power with `add_lags`, then sets to missing each lag whose time lies inside the mask array,
using the `for_arm` logic, instead of calling `corrected`. The boosters take the plain column set
(`input_columns(with_mw=False)`), so the feature-count check must expect that set. Keep the
`classify` families, the shapes, the 7-day quiet rule, the members, and the even onset months. The
score is against the unmodified series, with `case_metrics` at 11 levels, and the cells are kept per
lead band. Add the width columns (q95 minus q05 and q99 minus q01) to `case_metrics`' output in the
copy.

## Compute

768 fits (4 arms, 12 refits, 16 series) at about 25 s each, run in parallel within the slot, plus
predictions on 292 runs. Injected scoring took about 3 CPU hours in experiment 96, here with 5 to 8
variants. Smoke-test 2 series and 2 refits first.

## Checklist for the code reviewer

- [ ] Labels come from `e00086_labels.load_labels` unchanged. Targets are dropped by
  `in_event | closed`, and lags are masked by `in_event` only.
- [ ] `mask`, `drop10`, and `drop30` apply the same real-event masking at test time. Pseudo events
  appear only in `train_matrix`.
- [ ] Dropout is in place: the dropout arms have the same row count and weights as `mask` for each
  (series, refit). Assert this.
- [ ] Pseudo-event lists are seeded by (seed, site), nested between `drop10` and `drop30`, and the
  achieved shares are printed within 1.5 points of 10% and 30%.
- [ ] Every arm has 11 levels, and the `plain11` reproduction guard passes at 0.01 points.
- [ ] Every arm gets the spread factor 1.15, and the reads without the factor are clearly labelled.
- [ ] Scored real rows are `nra` only. No row with its target inside an event or a closure enters
  any metric. Masked and unmasked subsets use the same `lag_oracle` definition as experiment 86.
- [ ] `known_rows` cutoff copied; no valid time on or after 2026-07-01 scored.
- [ ] Bootstrap: months resampled, 20,000 resamples, a 98.75% level for P1, P2, G1, and G2, and 95%
  for the descriptive reads. Paired on (series, run, valid time).
- [ ] The injected scoring masks by the injected interval for `inj` and by causal intervals for
  `caus`, never by the real labels. The `ceiling` reads unmodified power.
- [ ] Crossing rate before the sort is reported per arm for masked and unmasked rows.

## Next (not in this experiment)

If the widening is partial, test missingness indicator columns (one per lag) on top of `drop10`, so
that trees can give a missing lag its own quantile leaves. If `mask` loses to `plain11` on the
quantile score, the next step is to replace the masked lag with a normal-arrangement estimate, not
to leave it missing.
