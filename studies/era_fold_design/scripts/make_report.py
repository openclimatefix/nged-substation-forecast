"""Build out/report.md from the saved tables. Nothing is fitted here."""
import json
import polars as pl
from common import OUT

def fmt(r, wide=True):
    if r is None:
        return "n/a"
    star = "" if (r["lo"] <= 0 <= r["hi"]) else "*"
    return f"{r['diff_pp']:+.3f} [{r['lo']:+.3f}, {r['hi']:+.3f}]{star}"

def index(df, keys):
    return {tuple(r[k] for k in keys): r for r in df.iter_rows(named=True)}

A = json.loads((OUT / "partA.json").read_text())
Bc = pl.read_parquet(OUT / "partB_contrasts.parquet")
Ba = pl.read_parquet(OUT / "partB_absolute.parquet")
Bj = json.loads((OUT / "partB.json").read_text())
Cc = pl.read_parquet(OUT / "partC_contrasts_both.parquet")
Ca = pl.read_parquet(OUT / "partC_absolute_both.parquet")
L = []
w = L.append

w("# Era-wise fold design: how much it moves the published study results\n")
w("Interval marks: `*` means the 95% interval excludes zero. All contrasts are in percentage points of capacity (mean absolute error, capped, as a fraction of each row's own capacity). "
  "Positive means the first arm has the larger error. Every interval resamples whole months and one of three fitting seeds, 2,000 times (`studies.bootstrap`). "
  "A 'change' interval is the paired bootstrap of (design's per-row difference minus the comparison design's per-row difference) on shared rows.\n")
w("Designs. **D0** today's folds: `with_eras`, folds cut within each UKV era, era fold numbers both 0-4. **D1** era fold numbers rotated (post-upgrade era by 2, "
  "the first rotation `search_fold_offsets` returns) so every calendar month is covered. **D2** folds cut by site over the whole record, no era cut, era feature kept. "
  "Part C: **D0trim** D0 with the IFS 49r1 straddle rows (2024-11-12 to 2024-11-30) removed; **I** three eras (IFS 49r1 from 2024-12, UKV upgrade from 2026-02) cut by `cut_eras` with an "
  "`era_code` feature and searched offsets, on the D0trim rows; **II** rows from 2024-12-01 with today's UKV-era design.\n")

# ---------------- summary
w("## Summary tables\n")
for setting, title in (("pooled", "primary hyperparameter setting"), ("sensitivity", "second hyperparameter setting")):
    w(f"### Part B, {title}: planned contrasts\n")
    w("| Study | Contrast | D0 | D1 | D2 | D1 − D0 | D2 − D0 |\n|---|---|---|---|---|---|---|")
    ix = index(Bc.filter(pl.col("setting") == setting), ["study", "treatment", "reference", "design"])
    seen = []
    for r in Bc.filter(pl.col("setting") == setting, pl.col("design") == "D0").iter_rows(named=True):
        key = (r["study"], r["treatment"], r["reference"])
        cells = [fmt(ix.get((*key, d))) for d in ("D0", "D1", "D2", "D1-D0", "D2-D0")]
        tag = "" if r["planned"] else " (exploratory)"
        w(f"| {r['study']} | {r['treatment']} − {r['reference']}{tag} | " + " | ".join(cells) + " |")
    w("")
for setting, title in (("pooled", "primary setting"), ("sensitivity", "second setting")):
    w(f"### Part C, {title}: planned contrasts and headline references\n")
    w("| Domain | Contrast | D0 (published) | D0trim | I (era cut, offsets A) | I2 (era cut, offsets B) | II (uncovered folds, superseded) | IIr (II with covering rotation 2) | I − D0trim | I2 − D0trim | IIr − D0 |\n|---|---|---|---|---|---|---|---|---|---|---|")
    sub = Cc.filter(pl.col("setting") == setting, pl.col("kind").is_in(["planned", "reference", "members"]))
    ix = index(sub, ["domain", "treatment", "reference", "design"])
    for dom in ("wind", "solar"):
        for r in sub.filter(pl.col("domain") == dom, pl.col("design") == "D0").iter_rows(named=True):
            key = (dom, r["treatment"], r["reference"])
            cells = [fmt(ix.get((*key, d))) for d in ("D0", "D0trim", "I", "I2", "II", "IIr", "I-D0trim", "I2-D0trim", "IIr-D0")]
            tag = " (planned)" if r["planned"] else " (exploratory)"
            w(f"| {dom} | {r['treatment']} − {r['reference']}{tag} | " + " | ".join(cells) + " |")
    w("")

# ---------------- Part A
w("## Part A: coverage arithmetic (no fits)\n")
w("A cell is a (site, fold, calendar month) that `uncovered_months` flags: no training row carries that calendar month and the month occurs in more than one year. "
  "'Scored hours' are site-hours (one row each, not row x seed).\n")
for study, o in A.items():
    label = {"wind": "Wind study (`wind_products.py` main frame, three wind farms W1-W3)", "solar": "Solar study (`weather_products.py` `long` panel frame, six solar farms A-F, eight products)"}[study]
    w(f"### {label}\n")
    w(f"- `cut_eras` with zero offsets equals `with_eras` fold for fold on every (site, time): **{o['cut_eras_zero_equals_with_eras']}**.")
    w(f"- `search_fold_offsets(first_months=[UKV_UPGRADE_MONTH])` finds these covering offsets (era 1's rotation): {[list(x.values())[1] for x in o['search_found']]}; uncovered hours by rotation of era 1: {o['uncovered_hours_by_rotation']}. "
      f"Design D1 uses rotation {o['best_rotation']} (fewest non-zero rotations, then smallest sum).\n")
    w("| Design | Scored hours | Uncovered hours | Share | Calendar months uncovered | Post-upgrade hours scored by a model with no post-upgrade training row |\n|---|---|---|---|---|---|")
    for name in ("D0", "D1_rot0", "D1_rot1", "D1_rot2", "D1_rot3", "D1_rot4", "D2"):
        d = o["designs"][name]
        w(f"| {name} | {d['hours']:,} | {d['uncovered']:,} | {100 * d['uncovered'] / d['hours']:.2f}% | {d['months'] or 'none'} | {d['post_scored_with_no_post_training']:,} of {d['post_hours']:,} |")
    w("\nPer site (D0 and D2):\n")
    w("| Site | Hours | D0 uncovered hours (share) | D0 months | D2 uncovered hours |\n|---|---|---|---|---|")
    for s, v in o["designs"]["D0"]["per_site"].items():
        v2 = o["designs"]["D2"]["per_site"][s]
        w(f"| {s} | {v['hours']:,} | {v['uncovered']:,} ({100 * v['uncovered'] / v['hours']:.1f}%) | {v['months'] or 'none'} | {v2['uncovered']:,} |")
    w("")

# ---------------- Part B details
w("## Part B: reproduction and details\n")
w("Reproduction of the published per-row losses under D0 (same arms, same setting, joined on arm, site, time, seed):\n")
w("| Study / setting | Rows (new, published) | Unmatched | Share bit-identical | Max abs difference (fraction of capacity) |\n|---|---|---|---|---|")
for k, v in Bj.items():
    w(f"| {k} | {v['rows_new']:,}, {v['rows_pub']:,} | {v['unmatched']} | {v['exact_equal_share']:.6f} | {v['max_abs_diff']:.3g} |")
w("")
w("Absolute mean absolute error (percent of capacity) of every fitted arm, with 95% interval:\n")
w("| Study | Setting | Arm | D0 | D1 | D2 |\n|---|---|---|---|---|---|")
ix = index(Ba, ["study", "setting", "arm", "design"])
for study in ("wind", "solar"):
    for setting in ("pooled", "sensitivity"):
        for arm in sorted(Ba.filter(pl.col("study") == study, pl.col("setting") == setting)["arm"].unique()):
            cells = []
            for d in ("D0", "D1", "D2"):
                r = ix[(study, setting, arm, d)]
                cells.append(f"{r['value']:.3f} [{r['lo']:.2f}, {r['hi']:.2f}]")
            w(f"| {study} | {setting} | {arm} | " + " | ".join(cells) + " |")
w("")

# ---------------- Part C details
w("## Part C: absolute error of the fitted arms\n")
w("| Domain | Setting | Arm | D0 | D0trim | I | I2 | II (uncovered) | IIr |\n|---|---|---|---|---|---|---|---|---|")
ix = index(Ca, ["domain", "setting", "arm", "design"])
for dom in ("wind", "solar"):
    for setting in ("pooled", "sensitivity"):
        for arm in list(dict.fromkeys(Ca.filter(pl.col("domain") == dom, pl.col("setting") == setting)["arm"].to_list())):
            cells = []
            for d in ("D0", "D0trim", "I", "I2", "II", "IIr"):
                r = ix.get((dom, setting, arm, d))
                cells.append("n/a" if r is None else f"{r['value']:.3f} [{r['lo']:.2f}, {r['hi']:.2f}]")
            w(f"| {dom} | {setting} | {arm} | " + " | ".join(cells) + " |")
w("")
w("## Part C: every contrast, primary setting\n")
for dom in ("wind", "solar"):
    for kind in list(dict.fromkeys(Cc.filter(pl.col("domain") == dom)["kind"].to_list())):
        w(f"### {dom}: {kind}\n")
        w("| Contrast | D0 | D0trim | I | I2 | II (uncovered) | IIr | I − D0trim | I2 − D0trim | IIr − D0 |\n|---|---|---|---|---|---|---|---|---|---|")
        sub = Cc.filter(pl.col("domain") == dom, pl.col("kind") == kind, pl.col("setting") == "pooled")
        ix = index(sub, ["treatment", "reference", "design"])
        for r in sub.filter(pl.col("design") == "D0").iter_rows(named=True):
            key = (r["treatment"], r["reference"])
            cells = [fmt(ix.get((*key, d))) for d in ("D0", "D0trim", "I", "I2", "II", "IIr", "I-D0trim", "I2-D0trim", "IIr-D0")]
            w(f"| {r['treatment']} − {r['reference']} | " + " | ".join(cells) + " |")
        w("")
w("## Part C: second-setting contrasts (planned, references, and those near the 5% line)\n")
for dom in ("wind", "solar"):
    w(f"### {dom}\n")
    w("| Contrast | D0 | D0trim | I | I2 | II (uncovered) | IIr | I − D0trim | I2 − D0trim | IIr − D0 |\n|---|---|---|---|---|---|---|---|---|---|")
    sub = Cc.filter(pl.col("domain") == dom, pl.col("setting") == "sensitivity")
    ix = index(sub, ["treatment", "reference", "design"])
    for r in sub.filter(pl.col("design") == "D0").iter_rows(named=True):
        key = (r["treatment"], r["reference"])
        cells = [fmt(ix.get((*key, d))) for d in ("D0", "D0trim", "I", "I2", "II", "IIr", "I-D0trim", "I2-D0trim", "IIr-D0")]
        w(f"| {r['treatment']} − {r['reference']} | " + " | ".join(cells) + " |")
    w("")
(OUT / "report_tables.md").write_text("\n".join(L))
print("written", len(L))
