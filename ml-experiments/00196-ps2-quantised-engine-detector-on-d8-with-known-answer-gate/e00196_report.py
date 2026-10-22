"""Print every number the write-up quotes and draw its figures as SVG.

Afterwards optimise each SVG with `npx svgo@4 --multipass --precision=1 --final-newline`.
G2's output is drawn on a days-1-to-7 axis with no calendar dates. D8 is a substation.

    OMP_NUM_THREADS=1 uv run --no-sync python ml-experiments/00196-*/e00196_report.py
"""

import json
import re
import textwrap
from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import altair as alt
import numpy as np
import plotting.ocf_theme as ocf
import polars as pl
from e00196_common import (
    CLEAN_END,
    DATA_DIR,
    FIT_END,
    FIT_START,
    G2_FIRST,
    G2_LAST,
    POST_START,
    g2_output,
    stamp,
)
from e00196_hmm import rolling_offset
from e00196_inject import build
from e00196_pipeline import load_series

OUT_DIR: Final[Path] = (
    Path(__file__).resolve().parents[2]
    / "docs"
    / "ml-experiments"
    / "00196-ps2-quantised-engine-detector-on-d8-with-known-answer-gate"
)
SOLAR_MONTH: Final[datetime] = datetime(2025, 3, 1, tzinfo=UTC)
GREY: Final[str] = "#888888"


def title(text: str, sub: str) -> alt.TitleParams:
    """A wrapped title and subtitle."""
    return alt.TitleParams(
        text=textwrap.wrap(text, 70),
        subtitle=textwrap.wrap(sub, 90),
        anchor="start",
        fontSize=14,
        subtitleFontSize=11,
    )


def load(variant: str) -> dict:
    """Gate results of one variant."""
    return json.loads((DATA_DIR / f"gate_results_{variant}.json").read_text())["wide"]


def overall_recall(*, sc: dict) -> float:
    """Recall over all running half-hours from the 1-unit and 2-3-unit recalls."""
    n = sc["n_units"]
    n1, n23 = n["1"], n["2"] + n["3"]
    return (sc["recall_1_unit"]["value"] * n1 + sc["recall_2_or_3_units"]["value"] * n23) / (
        n1 + n23
    )


def positive_spells() -> None:
    """Runs of 4 or more half-hours more than 1.5 MW above a 30-day rolling median, per 100 days."""
    s = load_series(label="D8")
    off = rolling_offset(y=s.y, load=s.load, g_unused=0.0, q=0.14)
    dev = s.resid - off
    up = np.nan_to_num(dev) > 1.5
    edges = np.flatnonzero(np.diff(np.concatenate([[0], up.astype(int), [0]])))
    starts, ends = edges[::2], edges[1::2]
    keep = (ends - starts) >= 4
    starts = starts[keep]
    for name, lo, hi in (
        ("clean history", 0, stamp(CLEAN_END)),
        (
            "2023-09 to 2025-02",
            stamp(FIT_START),
            stamp(SOLAR_MONTH),
        ),
        (
            "2025-03 to 2025-12",
            stamp(SOLAR_MONTH),
            stamp(FIT_END),
        ),
        ("2026-04 to 2026-06", stamp(POST_START), len(s.y)),
    ):
        n = int(((starts >= lo) & (starts < hi)).sum())
        print(
            f"positive spells (>=4 half-hours above +1.5 MW) {name}: {n} in "
            f"{(hi - lo) / 48:.0f} days = {100 * n / ((hi - lo) / 48):.1f} per 100 days"
        )


def main() -> None:  # noqa: PLR0912, PLR0915
    """Print and draw."""
    base = load("fixed")
    local = load("local")
    print(
        "unit size (gate fit):",
        base["g"],
        base["g_path"],
        "local-noise:",
        local["g"],
        local["g_path"],
    )
    for variant, r in (("fixed noise (pre-registered)", base), ("local noise (post hoc)", local)):
        print("==", variant)
        for arm in ("K1", "K2", "synthetic_x0.5", "synthetic_x2.0", "q_0.0", "q_0.5", "q_1.0"):
            sc = r[arm]
            print(
                arm,
                "units",
                sc["n_units"],
                {
                    k: [round(sc[k][x], 3) for x in ("value", "lo", "hi")]
                    for k in (
                        "recall_1_unit",
                        "recall_2_or_3_units",
                        "precision",
                        "energy_error",
                        "flagged_share",
                    )
                },
                "filtered recall",
                round(sc["recall_filtered_any_running"], 3),
                "exact-k",
                round(sc["exact_k_accuracy"], 3),
                "pass",
                all(sc["passes"].values()),
                sc["passes"],
            )
        print("false alarms clean even months", r["false_alarms_clean_even_months"])
        print("false alarms clean all", r["false_alarms_clean_all"])
        for k in r:
            if k.startswith(("flagged_share_", "left_tail_share_")):
                print(k, round(r[k], 4))
    for variant in ("fixed", "local"):
        prim = json.loads((DATA_DIR / f"primaries_results_{variant}.json").read_text())
        for label, d in prim.items():
            print(
                variant,
                label,
                round(d["wide"]["flagged_share"], 4),
                round(d["wide"]["false_active_days_per_month"], 2),
                d["wide"]["flagged_share_by_year"],
            )
    baselines = json.loads((DATA_DIR / "baseline_results.json").read_text())
    for arm, d in baselines.items():
        for det, c in d.items():
            print(
                "baseline",
                arm,
                det,
                {k: (round(v, 3) if isinstance(v, float) else v) for k, v in c.items()},
            )
    positive_spells()

    # Figure 1: the engine detector against the step detectors
    rows = []
    for arm in ("K1", "K2"):
        sc = base[arm]
        rows.append(
            {
                "detector": "Engine detector (this experiment)",
                "arm": arm,
                "recall": overall_recall(sc=sc),
            }
        )
        for det, c in baselines[arm].items():
            rows.append({"detector": det, "arm": arm, "recall": c["half_hour_recall_found_spells"]})
    frame = pl.DataFrame(rows)
    chart = (
        alt.Chart(frame)  # ty: ignore[unresolved-attribute]
        .mark_bar()
        .encode(
            y=alt.Y("detector:N", title=None, sort=None, axis=alt.Axis(labelLimit=300)),
            x=alt.X(
                "recall:Q",
                title="Share of injected running half-hours found (larger is better)",
                scale=alt.Scale(domain=[0, 1]),
            ),
            yOffset="arm:N",
            color=alt.Color(
                "arm:N",
                scale=alt.Scale(domain=["K1", "K2"], range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE]),
                legend=alt.Legend(title=None, orient="top"),
            ),
        )
        .properties(
            width=520,
            height=240,
            title=title(
                (
                    "The engine detector finds the injected engine; step detectors find "
                    "a quarter or "
                    "less"
                ),
                (
                    "Share of the half-hours in which the injected engine ran that each detector "
                    "flags. Step detectors get credit for the whole spell when an alarm lies "
                    "within 2 "
                    "hours of its start or end. K1: exact replay at D8. K2: nine shifted copies."
                ),
            ),
        )
    )
    chart.save(OUT_DIR / "figure_1_against_step_detectors.svg")

    # Figure 2: profile likelihood of the unit size
    log = (DATA_DIR / "fit_wide.log").read_text()
    prof = [
        (int(i), float(g), float(v))
        for i, (g, v) in enumerate(
            [(m[0], m[1]) for m in re.findall(r"g=(\d\.\d+) MW, objective (-?\d+)", log)]
        )
    ]
    per_round = 11
    rounds = [prof[i : i + per_round] for i in range(0, len(prof), per_round)]
    pf = pl.DataFrame(
        [
            {"round": f"fit {r + 1}", "g": g, "ll": v - max(x[2] for x in rr)}
            for r, rr in enumerate(rounds)
            for _, g, v in rr
        ]
    )
    (
        alt.Chart(pf)  # ty: ignore[unresolved-attribute]
        .mark_line(point=True)
        .encode(
            x=alt.X("g:Q", title="Unit size (MW)", scale=alt.Scale(domain=[1, 3.5])),
            y=alt.Y("ll:Q", title="Log-likelihood minus the best (nats)"),
            color=alt.Color(
                "round:N",
                scale=alt.Scale(range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE, ocf.DATA_GREEN]),
                legend=alt.Legend(title=None, orient="top"),
            ),
        )
        .properties(
            width=520,
            height=220,
            title=title(
                "The data prefer a unit of 1.5 to 1.75 MW, and a 2 MW unit fits worse",
                (
                    "Log-likelihood of D8's residual from 2023-09 to 2025-12 at each unit size, "
                    "transitions refitted at each size. Fit 1 is the first detection; fits 2 and 3 "
                    "follow each re-centring."
                ),
            ),
        )
    ).save(OUT_DIR / "figure_2_unit_size_profile.svg")

    # Figure 3: flagged share and left-tail share by month
    mrows = []
    for variant, r in (("Fixed noise", base), ("Local noise (post hoc)", local)):
        for m in r["monthly"]:
            y, mo = divmod(m["month"] - 1, 12)
            if y + 0 >= 2022 * 0 and m["month"] >= 2022 * 12 + 1:
                mrows.append(
                    {
                        "month": f"{y}-{mo + 1:02d}-01",
                        "series": f"Flagged running: {variant}",
                        "share": m["flag"],
                    }
                )
    for m in base["monthly"]:
        y, mo = divmod(m["month"] - 1, 12)
        mrows.append(
            {
                "month": f"{y}-{mo + 1:02d}-01",
                "series": "Residual below -2.5 MW",
                "share": m["left"],
            }
        )
    mf = pl.DataFrame(mrows).with_columns(pl.col("month").str.to_date())
    marks = pl.DataFrame(
        {
            "when": ["2023-08-16", "2025-03-24", "2026-02-01"],
            "label": ["G2 readings stop", "11 kV solar connected", "left tail falls"],
        }
    ).with_columns(pl.col("when").str.to_date())
    line = (
        alt.Chart(mf)  # ty: ignore[unresolved-attribute]
        .mark_line()
        .encode(
            x=alt.X("month:T", title="Month"),
            y=alt.Y("share:Q", title="Share of half-hours", scale=alt.Scale(domain=[0, 0.7])),
            color=alt.Color(
                "series:N",
                scale=alt.Scale(
                    domain=[
                        "Flagged running: Fixed noise",
                        "Flagged running: Local noise (post hoc)",
                        "Residual below -2.5 MW",
                    ],
                    range=[ocf.DATA_BLUE, ocf.DATA_SKY, ocf.BRAND_ORANGE],
                ),
                legend=alt.Legend(title=None, orient="top", columns=1, labelLimit=400),
            ),
        )
    )
    rules = alt.Chart(marks).mark_rule(color=GREY, strokeDash=[4, 3]).encode(x="when:T")  # ty: ignore[unresolved-attribute]
    text = (
        alt.Chart(marks)  # ty: ignore[unresolved-attribute]
        .mark_text(angle=270, align="left", dx=-4, dy=-4, fontSize=10, color=GREY)
        .encode(x="when:T", text="label:N", y=alt.value(0))
    )
    (line + rules + text).properties(
        width=520,
        height=240,
        title=title(
            "The detector over-flags after solar was connected, when the noise is larger",
            (
                "D8, share of half-hours per month. Before 2023-06 the engine is absent and the "
                "detector flags almost nothing. From 2025-03 it flags far more than the left tail "
                "explains."
            ),
        ),
    ).save(OUT_DIR / "figure_3_flagged_share_by_month.svg")

    # Figure 4: one known-answer week (days 1 to 7, no dates)
    s = load_series(label="D8")
    g2 = g2_output(n=len(s.y))
    inj = build(y=s.y, g2=g2, with_k2=False)
    np.load(DATA_DIR / "smoothed_wide_fixed.npy")  # uninjected; replaced below by the K1 run
    import pickle

    from e00196_pipeline import run_loop

    fits = pickle.loads((DATA_DIR / "fits_fixed.pkl").read_bytes())
    _, post = run_loop(
        s=s.with_reading(inj.y), noise=fits["noise"], params=fits["fits"]["wide"]["params"]
    )
    ek = (post.smoothed * np.arange(4)).sum(axis=1) * fits["fits"]["wide"]["params"].g
    out = np.nan_to_num(inj.output)
    weeks = [
        (w, out[w * 336 : (w + 1) * 336].sum())
        for w in range(
            stamp(G2_FIRST) // 336,
            stamp(G2_LAST) // 336,
        )
    ]
    w = max(weeks, key=lambda x: x[1])[0]
    sl = slice(w * 336, (w + 1) * 336)
    wk = pl.DataFrame(
        {
            "day": np.arange(336) / 48 + 1,
            "Injected output (recorded)": out[sl],
            "Detector's estimate": ek[sl],
        }
    ).unpivot(index="day", variable_name="series", value_name="mw")
    (
        alt.Chart(wk)  # ty: ignore[unresolved-attribute]
        .mark_line(interpolate="step-after")
        .encode(
            x=alt.X(
                "day:Q",
                title="Day of the week (1 to 7)",
                axis=alt.Axis(values=[1, 2, 3, 4, 5, 6, 7]),
                scale=alt.Scale(domain=[1, 8]),
            ),
            y=alt.Y("mw:Q", title="Engine output (MW)"),
            color=alt.Color(
                "series:N",
                scale=alt.Scale(
                    domain=["Detector's estimate", "Injected output (recorded)"],
                    range=[ocf.DATA_BLUE, ocf.BRAND_ORANGE],
                ),
                legend=alt.Legend(title=None, orient="top"),
            ),
        )
        .properties(
            width=520,
            height=200,
            title=title(
                "The detector's estimate follows the injected engine's output",
                (
                    "One week in summer 2023 with the most running. Blue: expected number of units "
                    "running times the fitted unit size. Orange: the recorded output of the "
                    "metered "
                    "engine that was subtracted from D8's reading."
                ),
            ),
        )
    ).save(OUT_DIR / "figure_4_known_answer_week.svg")
    print("figures written")


if __name__ == "__main__":
    main()
