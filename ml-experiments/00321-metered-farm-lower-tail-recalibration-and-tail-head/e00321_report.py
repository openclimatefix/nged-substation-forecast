# ruff: noqa
"""Print the tables whose numbers `result.md` quotes, from the score tables of experiment 321.

    PYTHONPATH=ml-experiments/00321-metered-farm-lower-tail-recalibration-and-tail-head \
        uv run --no-sync python ml-experiments/00321-*/e00321_report.py
"""

import e00321_common as common
import polars as pl

D = str(common.DATA_DIR) + "/"
s = pl.read_csv(D + "scores.csv")
c = pl.read_csv(D + "contrasts.csv")
order = ["d1", "d2", "d3", "d4_5", "d6_7", "d8_10"]
pl.Config.set_tbl_rows(100)
pl.Config.set_tbl_cols(12)
pl.Config.set_tbl_width_chars(200)
arms = [
    "recal",
    "recal_shrunk",
    "recal_stale7",
    "recal_swapped",
    "zi_alone",
    "zi_blend",
    "tail",
    "tail_shuffled",
    "pure_lead_pooled",
    "rd1_5seed",
]
for setting in ["own_history", "zero_history"]:
    print("#####", setting)
    for m in ["score", "pin01", "pin99"]:
        t = c.filter(
            (pl.col("setting") == setting)
            & (pl.col("half") == "all")
            & (pl.col("metric") == m)
            & (pl.col("reference") == "rd1")
            & pl.col("arm").is_in(arms)
        )
        t = t.with_columns(
            txt=pl.format(
                "{}/{}/{}",
                pl.col("points").round(3) if m == "score" else pl.col("percent").round(1),
                pl.col("points_lo").round(3) if m == "score" else pl.col("lo").round(1),
                pl.col("points_hi").round(3) if m == "score" else pl.col("hi").round(1),
            )
        )
        print("##", m, "vs rd1 (score: points/lo/hi ; pin: percent/lo/hi)")
        print(t.pivot(on="band", index="arm", values="txt").select("arm", *order))
    for m in ["strict01", "cover01", "zero_actual"]:
        t = s.filter(
            (pl.col("setting") == setting)
            & (pl.col("half") == "all")
            & (pl.col("metric") == m)
            & pl.col("arm").is_in(
                arms + ["rd1", "rd1_mdn", "rd1_gam", "rd1_xgb", "floor_climatology"]
            )
        ).with_columns(pl.col("value").round(4))
        print("##", m)
        print(t.pivot(on="band", index="arm", values="value").select("arm", *order))

D = str(common.DATA_DIR) + "/"
s = pl.read_csv(D + "scores.csv")
c = pl.read_csv(D + "contrasts.csv")
order = ["d1", "d2", "d3", "d4_5", "d6_7", "d8_10"]
pl.Config.set_tbl_rows(100)
pl.Config.set_tbl_cols(12)
pl.Config.set_tbl_width_chars(200)


def show(title, df):
    print("##", title)
    print(df.select("arm", *order) if "arm" in df.columns else df)


# halves
for setting, arm, m in [
    ("zero_history", "recal_shrunk", "score"),
    ("own_history", "tail", "pin99"),
    ("own_history", "pure_lead_pooled", "pin99"),
    ("zero_history", "tail", "pin99"),
    ("zero_history", "pure_lead_pooled", "pin99"),
    ("own_history", "zi_blend", "score"),
    ("own_history", "zi_blend", "pin99"),
    ("own_history", "tail", "score"),
]:
    t = c.filter(
        (pl.col("setting") == setting)
        & (pl.col("arm") == arm)
        & (pl.col("reference") == "rd1")
        & (pl.col("metric") == m)
        & pl.col("half").is_in(["h1", "h2"])
    )
    t = t.with_columns(
        txt=pl.format(
            "{}/{}/{}",
            (pl.col("points") if m == "score" else pl.col("percent")).round(
                3 if m == "score" else 1
            ),
            (pl.col("points_lo") if m == "score" else pl.col("lo")).round(3 if m == "score" else 1),
            (pl.col("points_hi") if m == "score" else pl.col("hi")).round(3 if m == "score" else 1),
        )
    )
    print("##", setting, arm, m, "halves")
    print(t.pivot(on="band", index="half", values="txt").select("half", *order))
# ae_mean contrasts
for setting in ["own_history", "zero_history"]:
    t = c.filter(
        (pl.col("setting") == setting)
        & (pl.col("half") == "all")
        & (pl.col("metric") == "ae_mean")
        & (pl.col("reference") == "rd1")
        & pl.col("arm").is_in(["recal_shrunk", "tail", "zi_blend", "pure_lead_pooled", "recal"])
    ).with_columns(
        txt=pl.format(
            "{}/{}/{}", pl.col("percent").round(1), pl.col("lo").round(1), pl.col("hi").round(1)
        )
    )
    print("## ae_mean % vs rd1", setting)
    print(t.pivot(on="band", index="arm", values="txt").select("arm", *order))
# median minus mean
for setting in ["own_history"]:
    t = s.filter(
        (pl.col("setting") == setting)
        & (pl.col("half") == "all")
        & (pl.col("metric") == "median_minus_mean")
        & pl.col("arm").is_in(
            ["rd1", "recal_shrunk", "tail", "zi_blend", "pure_lead_pooled", "rd1_mdn"]
        )
    ).with_columns(pl.col("value").round(2))
    print("## median minus mean", setting)
    print(t.pivot(on="band", index="arm", values="value").select("arm", *order))
# extras
for a, r, m in [
    ("tail", "tail_shuffled", "score"),
    ("tail", "tail_shuffled", "pin99"),
    ("zi_blend", "rd1_5seed", "score"),
    ("zi_blend", "rd1_5seed", "pin01"),
    ("zi_blend", "rd1_5seed", "pin99"),
    ("zi_alone", "mdn5_318", "pin01"),
    ("rd1_mdn", "rd1", "score"),
    ("rd1", "floor_climatology", "score"),
    ("rd1", "floor_heuristic", "score"),
]:
    t = c.filter(
        (pl.col("setting") == "own_history")
        & (pl.col("half") == "all")
        & (pl.col("metric") == m)
        & (pl.col("arm") == a)
        & (pl.col("reference") == r)
    ).with_columns(
        txt=pl.format(
            "{}/{}/{}",
            (pl.col("points") if m == "score" else pl.col("percent")).round(
                3 if m == "score" else 1
            ),
            (pl.col("points_lo") if m == "score" else pl.col("lo")).round(3 if m == "score" else 1),
            (pl.col("points_hi") if m == "score" else pl.col("hi")).round(3 if m == "score" else 1),
        )
    )
    print("##", a, "vs", r, m)
    print(t.select("band", "txt").transpose())
# tail coverage 05 95 strict
for m in ["strict05", "strict95", "strict99", "pin05", "pin95"]:
    t = s.filter(
        (pl.col("setting") == "own_history")
        & (pl.col("half") == "all")
        & (pl.col("metric") == m)
        & pl.col("arm").is_in(["rd1", "recal_shrunk", "tail", "zi_blend", "pure_lead_pooled"])
    ).with_columns(pl.col("value").round(4))
    print("##", m)
    print(t.pivot(on="band", index="arm", values="value").select("arm", *order))

D = str(common.DATA_DIR) + "/"
s = pl.read_csv(D + "scores.csv")
order = ["d1", "d2", "d3", "d4_5", "d6_7", "d8_10"]
pl.Config.set_tbl_rows(100)
pl.Config.set_tbl_cols(12)
pl.Config.set_tbl_width_chars(200)
for st in ["own_history", "zero_history"]:
    for m in ["score", "pin99"]:
        t = s.filter(
            (pl.col("setting") == st)
            & (pl.col("half") == "all")
            & (pl.col("metric") == m)
            & pl.col("arm").is_in(
                [
                    "rd1",
                    "rd1_mdn",
                    "rd1_gam",
                    "rd1_xgb",
                    "tail",
                    "recal_shrunk",
                    "zi_blend",
                    "pure_lead_pooled",
                ]
            )
        ).with_columns(pl.col("value").round(3))
        print("##", st, m)
        print(t.pivot(on="band", index="arm", values="value").select("arm", *order))
