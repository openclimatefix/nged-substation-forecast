# Evaluating data sources with limited history

How to assess a **new input data source whose history is shorter than the canonical CV folds** —
the motivating case is adding ICON-EU NWP (from Dynamical.org), whose archive starts later than
the leaderboard folds. Such a source **cannot** enter the canonical folds (there is no
overlapping history), so its evaluation lives entirely in the `metrics` asset's
`evaluation_scope="ad_hoc"` and **never** feeds the leaderboard.

Three patterns, answering three different questions.

## 1. Controlled ablation — "does the source add skill?"

The principled comparison: hold *everything* constant except the source under test. Because the
new source only exists from (say) 2026, the shared window must live within its availability:

- Pick an evaluation window bounded by the new source's history; split it into train/validation
  within that window.
- **Baseline experiment:** existing features only (e.g. `weather_source = "ecmwf"`).
- **Treatment experiment:** existing + new-source features (e.g. `weather_source = "ecmwf_icon"`).
- Both train on the **identical rows** and are scored on the **identical rows** — same
  `time_series_id` population, same `power_fcst_init_time` grid — differing *only* in the
  feature set. Score both with `evaluation_scope="ad_hoc"` over the same `PopulationFilter`.

To inherit the leaderboard's same-population guarantee for this off-leaderboard window,
materialise a **frozen ad-hoc eligibility set** for the window that both experiments read,
rather than letting each pick its own population. (`trained_time_series_ids` forces
train == predict *per model*, but does not by itself force the *two* experiments to share a
population — the frozen set does.)

## 2. Confound warning — do NOT read this as the ablation

The tempting shortcut — take the canonical leaderboard champion, run it on the new source's
window, and compare against a new-source model on that window — is **statistically confounded**
and must not be read as evidence about the source. The two models differ in **two** variables
at once: the feature set *and* the training window (the champion trained on the full archive;
the new-source model is forced onto the short sliver). A win or loss cannot be attributed to
the source rather than to the training data.

This comparison is legitimate only as a **deployment** question — "which forecast is better to
ship *today*?" — where the confound is irrelevant because we only care which is better now,
not why.

## 3. Epoch path — the eventual leaderboard-quality answer

Once the new source has accumulated enough history (roughly 1–2 complete years), promote it via
a **new leaderboard epoch**: a fold set over source-era complete years in which the new source
is canonically available, with every experiment re-scored against that fold set for
apples-to-apples comparison (see
[Cross-validation folds](cross-validation-folds.md)). The ad-hoc ablation is the **interim**
signal obtained before enough history exists to do this properly; it should never be presented
with leaderboard rigour.

---

Note: this page concerns only *evaluation*. Actually **ingesting** a second NWP source (a
second downloader, NWP contract changes, source-aware weather-feature parsing, a dual-source
join in feature engineering) is separate engineering work — see the
[roadmap](../roadmap/index.md).
