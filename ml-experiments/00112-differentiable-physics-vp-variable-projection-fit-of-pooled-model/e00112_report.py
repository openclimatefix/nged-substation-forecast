"""Score every arm and print the numbers the write-up of experiment 112 quotes.

Reads the prediction sets that `e00112_forecast.py`, `e00112_onboard.py`, `e00112_ladder.py`,
`e00112_convex.py`, and `e00112_refs.py` wrote, experiment 105's stored sets, and Part 2's stored
convex and reference sets. It prints one report, also written to `work/report.md`: absolute scores,
the four planned contrasts (Bonferroni, 98.75% intervals) for the scored year and the replicate
year, the exploratory contrasts, each contrast split by season and with each site dropped in turn,
D10 with its November 2025 jump masked, the all-rows score, tails, the mean-forecast error beside
every score, the median minus the mean forecast, and the decision rule's conditions.

The score of a row is twice the mean pinball loss over the 11 levels, in points of effective
capacity. Headline rows are those outside the hand-filtered switching events at the scored
substation. The decision set is D5, D10, D11, D14, and D16; D9 is reported beside it.

    uv run --no-sync python ml-experiments/00112-*/e00112_report.py
"""

import contextlib
import io
import json
import sys
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Final

import e00112_common as common
import numpy as np
import polars as pl
from studies.bootstrap import (
    bootstrap_difference,
    bootstrap_difference_at_level,
    paired_differences,
)
from studies.dp_gam import QUANTILE_COLUMNS
from studies.experiment_store import losses_for_bootstrap

KEYS: Final[tuple[str, ...]] = ("time_series_id", "power_fcst_init_time", "valid_time", "seed")
N_VERDICT_CONTRASTS: Final[int] = 4
BONFERRONI_LEVEL: Final[float] = 100.0 * (1.0 - 0.05 / N_VERDICT_CONTRASTS)
N_RESAMPLES: Final[int] = 20000
NON_INFERIORITY_MARGIN: Final[float] = 0.10
START_SPREAD_LIMIT: Final[float] = 0.10
TAIL_COLUMNS: Final[tuple[str, ...]] = ("q01", "q05", "q95", "q99")
TAIL_LEVELS: Final[tuple[float, ...]] = (0.01, 0.05, 0.95, 0.99)
D10_JUMP: Final[tuple[datetime, datetime]] = (
    datetime(2025, 11, 17, tzinfo=UTC),
    datetime(2025, 12, 14, tzinfo=UTC),
)
"""D10's unmasked jump in the scored year, probably an unlabelled switching event."""
PART_105: Final[Path] = (
    common.ML_EXPERIMENTS_DATA_DIR
    / "00105-differentiable-physics-3-pooled-and-per-site-demand-model"
)
OWN_ARMS: Final[tuple[str, ...]] = (
    "vp_p",
    "vp_p_s0",
    "vp_p_s1",
    "vp_s",
    "vp_s_s0",
    "vp_s_s1",
    "vp_p_nonet",
    "vp_p_nonet_s0",
    "vp_p_nonet_s1",
    "p_map_same",
    "p_map_polish",
    "vp_onb4w",
    "vp_s_4w",
    "vp_short3m",
    "vp_s_3m",
    "vp_l1_equiv_all",
    "g_2y",
    "p_map",
    "s",
)
"""Sets written by this experiment; the replicate year's carry the suffix `_rep`."""
FROM_105: Final[tuple[str, ...]] = (
    "p",
    "p_map",
    "s",
    "s_equiv_all",
    "p_onb4w",
    "s_4w",
    "p_short3m",
    "s_3m",
)
PART_2_ARMS: Final[tuple[str, ...]] = ("g_all", "climatology", "manual_heuristic")
CENTRED_PART_2: Final[tuple[str, ...]] = ("g_all",)
PLANNED: Final[tuple[tuple[str, str, str, str], ...]] = (
    ("V1 fitting method on one objective", "vp_p", "p_map_same", "five"),
    ("C1v pooling on all rows under VP", "vp_p", "vp_s", "five"),
    ("C2v onboarding, 4 weeks", "vp_onb4w", "vp_s_4w", "six"),
    ("C3v short history, 3 months", "vp_short3m", "vp_s_3m", "six"),
)
EXPLORATORY: Final[tuple[tuple[str, str], ...]] = (
    ("vp_p", "p_map"),
    ("vp_p", "p"),
    ("vp_p", "s"),
    ("vp_p", "g_2y"),
    ("vp_s", "s"),
    ("vp_s", "g_2y"),
    ("p_map_polish", "p_map_same"),
    ("vp_onb4w", "p_onb4w"),
    ("vp_s_4w", "s_4w"),
    ("vp_short3m", "p_short3m"),
    ("vp_s_3m", "s_3m"),
    ("vp_l1_equiv_all", "s_equiv_all"),
    ("vp_l1_equiv_all", "g_all"),
    ("s_equiv_all", "g_all"),
    ("vp_p", "g_all"),
    ("vp_p", "manual_heuristic"),
    ("vp_p", "climatology"),
    ("vp_p_nonet", "vp_p"),
    ("vp_p_nonet", "p_map_same"),
    ("vp_p_nonet", "vp_s"),
    ("vp_p_s0", "vp_p_s1"),
    ("vp_s_s0", "vp_s_s1"),
)


def site_of_id() -> dict[int, str]:
    """Return the substation label of each time-series id."""
    table = pl.read_parquet(common.SERIES_TABLE).select("time_series_id", "site")
    return dict(table.iter_rows())


def load_set(*, folder: Path, name: str, ids: dict[int, str]) -> pl.DataFrame | None:
    """Read a prediction set with its per-row score, site label, and error of the mean."""
    path = folder / "predictions" / name / "predictions.parquet"
    if not path.exists():
        return None
    frame = pl.read_parquet(path)
    quantiles = frame.select(QUANTILE_COLUMNS).to_numpy().astype(np.float64)
    actual = frame["actual"].to_numpy().astype(np.float64)
    frame = frame.with_columns(
        score=pl.Series(common.quantile_score(actual=actual, quantiles=quantiles)),
        site=pl.col("time_series_id").replace_strict(ids, return_dtype=pl.String),
    )
    if "mean" in frame.columns:
        frame = frame.with_columns(mean_error=(pl.col("mean") - pl.col("actual")).abs())
    return frame


def load_year(*, year: str, ids: dict[int, str]) -> dict[str, pl.DataFrame]:
    """Return every arm of a year (`scored` or `rep`) under its canonical name, all rows."""
    tag = "_rep" if year == "rep" else ""
    arms: dict[str, pl.DataFrame] = {}
    for name in OWN_ARMS:
        frame = load_set(folder=common.DATA_DIR, name=f"{name}{tag}", ids=ids)
        if frame is not None:
            arms[name] = frame
    if year == "scored":
        for name in FROM_105:
            frame = load_set(folder=PART_105, name=name, ids=ids)
            if frame is not None:
                arms[name] = frame
        for name in PART_2_ARMS:
            stored = f"{name}_centred" if name in CENTRED_PART_2 else name
            frame = load_set(folder=common.PART_2_DIR, name=stored, ids=ids)
            if frame is not None:
                arms[name] = frame
    else:
        for name in ("p_map", "s"):
            frame = load_set(folder=common.DATA_DIR, name=f"{name}_rep", ids=ids)
            if frame is not None:
                arms[name] = frame
    return arms


def headline(frame: pl.DataFrame) -> pl.DataFrame:
    """Keep the rows outside the switching masks."""
    return frame.filter(~pl.col("switching_mask"))


def contrast(
    *,
    arms: dict[str, pl.DataFrame],
    treatment: str,
    reference: str,
    sites: tuple[str, ...],
    level: float | None,
    metric: str = "score",
    where: pl.Expr | None = None,
) -> dict:
    """Return a paired contrast (treatment minus reference) with an interval and site counts."""

    def prepared(name: str) -> pl.DataFrame:
        frame = arms[name].filter(pl.col("site").is_in(sites))
        return frame.filter(where) if where is not None else frame

    losses = losses_for_bootstrap(
        arm_losses={
            name: prepared(name).select(*KEYS, pl.col(metric).alias("score"))
            for name in (treatment, reference)
        }
    )
    differences, _ = paired_differences(
        losses=losses, treatment=treatment, reference=reference, metric="score"
    )
    if level is None:
        interval = bootstrap_difference(
            losses=losses, treatment=treatment, reference=reference, metric="score"
        )
        low, high = interval["lower_95"], interval["upper_95"]
    else:
        low, high = bootstrap_difference_at_level(
            losses=losses,
            treatment=treatment,
            reference=reference,
            metric="score",
            level=level,
            n_resamples=N_RESAMPLES,
        )
    paired = (
        prepared(treatment)
        .select("site", "valid_time", treatment=pl.col(metric))
        .join(
            prepared(reference).select("site", "valid_time", reference=pl.col(metric)),
            on=["site", "valid_time"],
            validate="1:1",
        )
        .group_by("site")
        .agg((pl.col("treatment") - pl.col("reference")).mean().alias("difference"))
    )
    per_site = dict(paired.iter_rows())
    return {
        "estimate": float(differences.mean()),
        "low": float(low),
        "high": float(high),
        "level": level or 95.0,
        "better": sum(v < 0 for v in per_site.values()),
        "worse": sum(v > 0 for v in per_site.values()),
        "per_site": per_site,
    }


def fmt(result: dict) -> str:
    """Format a contrast as `estimate (low to high)` with the sites that improve."""
    return (
        f"{result['estimate']:+.3f} ({result['low']:+.3f} to {result['high']:+.3f}, "
        f"{result['level']:.2f}%); {result['better']} better, {result['worse']} worse"
    )


def print_contrast_block(
    *,
    label: str,
    arms: dict[str, pl.DataFrame],
    all_rows: dict[str, pl.DataFrame],
    treatment: str,
    reference: str,
    sites: tuple[str, ...],
    level: float | None,
    out: dict,
) -> None:
    """Print a contrast with its mean-error twin, season split, drop-one-site, and row views."""
    if treatment not in arms or reference not in arms:
        print(f"{label}: {treatment} - {reference}: arm missing")
        return
    main = contrast(arms=arms, treatment=treatment, reference=reference, sites=sites, level=level)
    out[label] = main
    print(f"{label}: {treatment} - {reference}: {fmt(main)}")
    print(f"   per site: { {k: round(v, 3) for k, v in main['per_site'].items()} }")
    if "mean_error" in arms[treatment].columns and "mean_error" in arms[reference].columns:
        twin = contrast(
            arms=arms,
            treatment=treatment,
            reference=reference,
            sites=sites,
            level=None,
            metric="mean_error",
        )
        print(f"   error of the mean forecast beside it: {fmt(twin)}")
    month = pl.col("valid_time").dt.month()
    for season, where in (
        ("winter Nov-Mar", month.is_in([11, 12, 1, 2, 3])),
        ("summer Apr-Oct", ~month.is_in([11, 12, 1, 2, 3])),
    ):
        part = contrast(
            arms=arms,
            treatment=treatment,
            reference=reference,
            sites=sites,
            level=None,
            where=where,
        )
        print(f"   {season}: {fmt(part)}")
    for dropped in sites:
        kept = tuple(s for s in sites if s != dropped)
        if kept:
            part = contrast(
                arms=arms, treatment=treatment, reference=reference, sites=kept, level=None
            )
            print(f"   without {dropped}: {fmt(part)}")
    if "D10" in sites:
        not_jump = ~(
            (pl.col("site") == "D10")
            & (pl.col("valid_time") >= D10_JUMP[0])
            & (pl.col("valid_time") <= D10_JUMP[1])
        )
        part = contrast(
            arms=arms,
            treatment=treatment,
            reference=reference,
            sites=sites,
            level=None,
            where=not_jump,
        )
        print(f"   D10 jump (2025-11-17 to 2025-12-14) masked: {fmt(part)}")
    if treatment in all_rows and reference in all_rows:
        part = contrast(
            arms=all_rows, treatment=treatment, reference=reference, sites=sites, level=None
        )
        print(f"   all present rows, masks included: {fmt(part)}")


def print_absolute(*, arms: dict[str, pl.DataFrame]) -> None:
    """Print each arm's headline score per site, pooled over the decision set and with D9."""
    order = ["D5", "D10", "D11", "D14", "D16", "D9", "D18"]
    print("\n== Absolute quantile score, points (rows outside the switching masks) ==")
    print(f"{'arm':16}" + "".join(f"{site:>8}" for site in order) + f"{'five':>8}{'six':>8}")
    for name, frame in arms.items():
        by_site = dict(frame.group_by("site").agg(pl.col("score").mean()).iter_rows())
        five = frame.filter(pl.col("site").is_in(common.ELIGIBLE_SITES))["score"].mean()
        six = frame.filter(pl.col("site").is_in(common.SCORED_SITES))["score"].mean()
        cells = "".join(f"{by_site.get(site, float('nan')):8.3f}" for site in order)
        print(f"{name:16}{cells}{five:8.3f}{six:8.3f}")


def print_tails(*, arms: dict[str, pl.DataFrame]) -> None:
    """Print coverage and pinball loss at the four tail levels, decision set pooled."""
    print("\n== Tails: coverage (nominal 0.01, 0.05, 0.95, 0.99) and pinball at each level ==")
    for name, frame in arms.items():
        kept = frame.filter(pl.col("site").is_in(common.ELIGIBLE_SITES))
        actual = kept["actual"].to_numpy().astype(np.float64)
        coverage, pinball = [], []
        for column, level in zip(TAIL_COLUMNS, TAIL_LEVELS, strict=True):
            predicted = kept[column].to_numpy().astype(np.float64)
            coverage.append(float((actual <= predicted).mean()))
            error = actual - predicted
            pinball.append(float(np.maximum(level * error, (level - 1.0) * error).mean()))
        print(
            f"{name:16}"
            + "".join(f"{value:8.3f}" for value in coverage)
            + "  |  "
            + " ".join(f"{value:.3f}" for value in pinball)
        )


def print_checks(*, arms: dict[str, pl.DataFrame]) -> None:
    """Print the median minus the mean forecast per site (a bias check) for the model arms."""
    print("\n== Checks: median minus mean forecast per site, points ==")
    for name, frame in arms.items():
        if "mean" not in frame.columns:
            continue
        gap = (
            frame.with_columns(gap=pl.col("q50") - pl.col("mean"))
            .group_by("site")
            .agg(pl.col("gap").mean())
            .sort("site")
        )
        print(f"{name:16}", {site: round(value, 2) for site, value in gap.iter_rows()})


BLOWUP_POINTS: Final[float] = 30.0
"""A day on which an arm's mean median forecast error over the five sites exceeds this is listed."""
EXCLUDED_DAYS: Final[list[date]] = [
    date(2026, 3, 28),
    date(2026, 3, 29),
    date(2026, 4, 3),
    date(2026, 4, 6),
]
"""The four days of 2026 on which the VP-P forecast was up to 280 points too high.

They are the weekend of the spring clock change (28 and 29 March), Good Friday, and Easter Monday.

The numbers audit found them; excluding them is a post hoc robustness check, not a planned one.
"""
BLOWUP_ARMS: Final[tuple[str, ...]] = (
    "vp_p",
    "vp_p_s1",
    "vp_onb4w",
    "vp_p_nonet",
    "p_map_same",
    "g_2y",
)


def blowup_days(*, arms: dict[str, pl.DataFrame], always: list[date]) -> list:
    """Print the days on which an arm's forecast is far off, and return their dates.

    For each arm, the error of a day is the mean over the five decision substations of the median
    forecast minus the actual, in points of effective capacity. A day is listed when any listed arm
    exceeds `BLOWUP_POINTS` in absolute value, and so is each day in `always`.
    """
    table = {}
    for name in BLOWUP_ARMS:
        if name not in arms:
            continue
        frame = arms[name].filter(pl.col("site").is_in(common.ELIGIBLE_SITES))
        daily = frame.group_by(pl.col("valid_time").dt.date().alias("day")).agg(
            (pl.col("q50") - pl.col("actual")).mean().alias("error")
        )
        table[name] = dict(daily.iter_rows())
    days = sorted(
        {d for errors in table.values() for d, e in errors.items() if abs(e) > BLOWUP_POINTS}
        | set(always)
    )
    print(f"\n== Days with an error over {BLOWUP_POINTS:.0f} points (five-site mean error) ==")
    for day in days:
        cells = {name: round(errors.get(day, float("nan")), 1) for name, errors in table.items()}
        print(day, cells)
    if not days:
        print("none")
    return days


def print_excluding_days(*, arms: dict[str, pl.DataFrame], days: list, out: dict) -> None:
    """Print the planned contrasts and the start spread with the listed days left out (post hoc)."""
    if not days:
        return
    keep = ~pl.col("valid_time").dt.date().is_in(days)
    print(f"\n== Post hoc: the same contrasts without the {len(days)} listed days ==")
    for label, treatment, reference, which in PLANNED[:2]:
        if treatment in arms and reference in arms:
            sites = {"five": common.ELIGIBLE_SITES, "six": common.SCORED_SITES}[which]
            result = contrast(
                arms=arms,
                treatment=treatment,
                reference=reference,
                sites=sites,
                level=BONFERRONI_LEVEL,
                where=keep,
            )
            out[f"{label} without the listed days"] = result
            print(f"{label}: {fmt(result)}")
    if "vp_p_s0" in arms and "vp_p_s1" in arms:
        five = arms["vp_p_s0"].filter(pl.col("site").is_in(common.ELIGIBLE_SITES)).filter(keep)
        other = arms["vp_p_s1"].filter(pl.col("site").is_in(common.ELIGIBLE_SITES)).filter(keep)
        spread = abs(
            float(five["score"].to_numpy().mean()) - float(other["score"].to_numpy().mean())
        )
        print(f"VP-P start spread without the listed days: {spread:.3f}")
    for name in ("vp_p", "p_map_same"):
        if name in arms:
            part = arms[name].filter(pl.col("site").is_in(common.ELIGIBLE_SITES)).filter(keep)
            mean_score = float(part["score"].to_numpy().mean())
            print(f"{name} five-site score without the listed days: {mean_score:.3f}")


def report_year(*, year: str, ids: dict[int, str]) -> dict:
    """Print the report of one year and return its planned contrasts."""
    all_rows = load_year(year=year, ids=ids)
    arms = {name: headline(frame) for name, frame in all_rows.items()}
    title = "Scored year 2025-07-01 to 2026-06-30"
    if year == "rep":
        title = "Replicate year 2024-07-01 to 2025-06-30"
    print(f"\n######## {title} ########")
    print_absolute(arms=arms)
    sets = {"five": common.ELIGIBLE_SITES, "six": common.SCORED_SITES}
    out: dict = {}
    print(f"\n== Planned contrasts (Bonferroni, {N_VERDICT_CONTRASTS} contrasts) ==")
    for label, treatment, reference, which in PLANNED:
        print_contrast_block(
            label=label,
            arms=arms,
            all_rows=all_rows,
            treatment=treatment,
            reference=reference,
            sites=sets[which],
            level=BONFERRONI_LEVEL,
            out=out,
        )
    for label, treatment, reference, _ in PLANNED[2:]:
        print_contrast_block(
            label=f"{label} (five decision sites)",
            arms=arms,
            all_rows=all_rows,
            treatment=treatment,
            reference=reference,
            sites=sets["five"],
            level=BONFERRONI_LEVEL,
            out=out,
        )
    print("\n== Exploratory contrasts (95%) ==")
    for treatment, reference in EXPLORATORY:
        print_contrast_block(
            label=f"{treatment}-{reference}",
            arms=arms,
            all_rows=all_rows,
            treatment=treatment,
            reference=reference,
            sites=sets["five"],
            level=None,
            out=out,
        )
    print_tails(arms=arms)
    print_checks(arms=arms)
    days = blowup_days(arms=arms, always=EXCLUDED_DAYS if year == "scored" else [])
    print_excluding_days(arms=arms, days=EXCLUDED_DAYS if year == "scored" else days, out=out)
    if "g_all" in arms:
        five = arms["g_all"].filter(pl.col("site").is_in(common.ELIGIBLE_SITES))
        score = float(five["score"].to_numpy().mean())
        print(f"\nL0 control: G-all scores {score:.3f} at the five decision sites (expected 3.072)")
    start_spread = None
    if "vp_p_s0" in arms and "vp_p_s1" in arms:
        five = common.ELIGIBLE_SITES
        scores = [
            float(arms[name].filter(pl.col("site").is_in(five))["score"].to_numpy().mean())
            for name in ("vp_p_s0", "vp_p_s1")
        ]
        start_spread = abs(scores[0] - scores[1])
        print(f"\nVP-P score spread between the two starts: {start_spread:.3f} points")
    out["start_spread"] = start_spread
    return out


C1V: Final[str] = "C1v pooling on all rows under VP"
V1: Final[str] = "V1 fitting method on one objective"


def read_json(*, name: str) -> dict:
    """Return a stored result file, or an empty dict when it does not exist."""
    path = common.WORK_DIR / name
    return json.loads(path.read_text()) if path.exists() else {}


def conditions(*, scored: dict, replicate: dict) -> None:
    """Print the decision rule's conditions, the controls, and the verdict they give.

    The rule is the plan's: `supported` needs the gate, C1v passing, V1 below zero in both years
    with 4 of 5 sites better in the scored year, VP-S minus S at most +0.05, and a start spread of
    at most 0.10; `partly supported` is V1 meeting its bar with C1v failing, or V1 below zero at 4
    of 5 sites in the scored year only; `refuted` is V1 at or above zero in the scored year with
    C1v's interval wholly above zero; anything else, or a start spread over 0.10, is
    `inconclusive`.
    """
    print("\n######## Decision rule conditions ########")
    gate = read_json(name="gate.json")
    gate_passed = all(gate.get(t, {}).get("passed", False) for t in ("test_2", "test_3", "test_4"))
    print(f"known-answer gate (tests 2 to 4) passed: {gate_passed}")
    objectives = read_json(name="objectives.json")
    controls_ok = True
    for cut, values in objectives.items():
        if "p_map_polish" in values and "p_map_same" in values:
            ok = values["p_map_polish"] <= values["p_map_same"] + 1.0
            controls_ok &= ok
            print(f"control at {cut}: P-map-polish not above P-map-same on the VP objective: {ok}")
    c1, c1_rep = scored.get(C1V), replicate.get(C1V)
    v1, v1_rep = scored.get(V1), replicate.get(V1)
    vp_s_vs_s = scored.get("vp_s-s")
    spread = scored.get("start_spread")
    if not (c1 and c1_rep and v1 and v1_rep):
        print("verdict: cannot be read, a planned contrast is missing")
        return
    c1_passes = (
        c1["estimate"] <= NON_INFERIORITY_MARGIN
        and c1["low"] <= 0.0
        and c1_rep["estimate"] <= NON_INFERIORITY_MARGIN
    )
    sites = v1["better"] + v1["worse"]
    v1_bar = v1["estimate"] < 0 and v1_rep["estimate"] < 0 and v1["better"] >= sites - 1
    v1_scored_only = v1["estimate"] < 0 and v1["better"] >= sites - 1 and v1_rep["estimate"] >= 0
    s_ok = vp_s_vs_s is not None and vp_s_vs_s["estimate"] <= 0.05
    spread_ok = spread is not None and spread <= START_SPREAD_LIMIT
    print(
        f"C1v passes: {c1_passes} (estimate {c1['estimate']:+.3f}, interval {c1['low']:+.3f} to "
        f"{c1['high']:+.3f}, upper end against +{NON_INFERIORITY_MARGIN}; replicate "
        f"{c1_rep['estimate']:+.3f})"
    )
    print(
        f"V1 below zero in both years with {sites - 1} of {sites} sites better: {v1_bar} "
        f"(scored {v1['estimate']:+.3f}, {v1['better']} better; replicate "
        f"{v1_rep['estimate']:+.3f}, {v1_rep['better']} better)"
    )
    print(f"VP-S minus S at most +0.05 in the scored year: {s_ok}")
    print(f"start spread at most {START_SPREAD_LIMIT}: {spread_ok} ({spread})")
    if not gate_passed or not spread_ok:
        verdict = "inconclusive"
    elif c1_passes and v1_bar and s_ok:
        verdict = "supported"
    elif (v1_bar and not c1_passes) or v1_scored_only:
        verdict = "partly supported"
    elif v1["estimate"] >= 0 and c1["low"] > 0:
        verdict = "refuted"
    else:
        verdict = "inconclusive"
    print(f"verdict by the plan's rule: {verdict}")
    default_fitter = gate_passed and v1_bar
    pooled_continues = c1["low"] <= NON_INFERIORITY_MARGIN
    print(f"handoff: fitter='vp' becomes the pooled default: {default_fitter}")
    print(
        f"handoff: the pooled neural DP continues (C1v not wholly above +0.10): {pooled_continues}"
    )
    print(f"controls held: {controls_ok}")
    profile = read_json(name="profile.json")
    for name, result in profile.items():
        checks = result["checks"]
        print(
            f"capacity profile {name}: starts agree {checks['line_1_starts_agree']}, "
            f"objective {checks['line_2_objective']}, "
            f"capacity near minimum {checks['line_3_capacity_near_minimum']}"
        )


def print_v1_in_nats() -> None:
    """Print V1's content in nats: the final objectives of the fits on one objective function."""
    objectives = read_json(name="objectives.json")
    gate = read_json(name="gate.json")
    passed = all(gate.get(t, {}).get("passed", False) for t in ("test_2", "test_3", "test_4"))
    if not passed:
        print("######## GATE FAILED: the verdict below is capped at inconclusive ########")
    print(
        "\n######## V1 in nats (the VP objective, one function, same rows and noise scale) ########"
    )
    for cut, values in objectives.items():
        if cut.startswith("weights_"):
            print(f"\nSizes of the linear coefficients and the network's last layer ({cut}):")
            for name, sizes in values.items():
                print(name, {k: round(v, 1) for k, v in sizes.items()})
            continue
        print(cut, {name: round(value) for name, value in values.items()})
        if "vp_p_s0" in values and "p_map_same" in values:
            low = min(values.get("vp_p_s0", 1e99), values.get("vp_p_s1", 1e99))
            print(f"   VP-P (lower start) minus P-map-same: {low - values['p_map_same']:+.0f} nats")
        if "p_map_105" in values and "p_map_same" in values:
            gap = values["p_map_105"] - values["p_map_same"]
            print(f"   P-map of experiment 105 minus P-map-same: {gap:+.0f} nats")


def main() -> None:
    """Print the full report, and write it and the contrasts under `work/`."""
    ids = site_of_id()
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        print_v1_in_nats()
        scored = report_year(year="scored", ids=ids)
        replicate = report_year(year="rep", ids=ids)
        conditions(scored=scored, replicate=replicate)
    text = buffer.getvalue()
    sys.stdout.write(text)
    common.WORK_DIR.mkdir(parents=True, exist_ok=True)
    (common.WORK_DIR / "report.md").write_text(text)
    common.write_json(
        name="report_contrasts.json", payload={"scored": scored, "replicate": replicate}
    )


if __name__ == "__main__":
    main()
