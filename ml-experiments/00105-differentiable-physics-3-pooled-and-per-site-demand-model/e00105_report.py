"""Score every arm and print the numbers the write-up quotes (experiment 105).

Reads the prediction sets that `e00105_fit.py`, `e00105_onboard.py`, and `e00105_xgb.py` wrote,
and Part 2's stored convex and reference arms, and prints one report: absolute scores, the planned
contrasts C1 to C3 (Bonferroni, 3 contrasts) and the descriptive C4 and C5, the exploratory
contrasts, each contrast split by winter and summer and with each decision site dropped in turn,
D10 with its November 2025 jump masked, tails, the mean-forecast error beside every score, and the
median minus the mean forecast.

The score of a row is twice the mean pinball loss over the 11 levels, in points of effective
capacity. Headline rows are those outside the hand-filtered switching events at the scored
substation. The decision set is D5, D10, D11, D14, and D16; D9 is reported beside it.

    uv run --no-sync python ml-experiments/00105-*/e00105_report.py
"""

from datetime import UTC, datetime
from pathlib import Path
from typing import Final

import e00105_common as common
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
N_VERDICT_CONTRASTS: Final[int] = 3
BONFERRONI_LEVEL: Final[float] = 100.0 * (1.0 - 0.05 / N_VERDICT_CONTRASTS)
N_RESAMPLES: Final[int] = 20000
TAIL_COLUMNS: Final[tuple[str, ...]] = ("q01", "q05", "q95", "q99")
TAIL_LEVELS: Final[tuple[float, ...]] = (0.01, 0.05, 0.95, 0.99)
D10_JUMP: Final[tuple[datetime, datetime]] = (
    datetime(2025, 11, 17, tzinfo=UTC),
    datetime(2025, 12, 14, tzinfo=UTC),
)
"""D10's unmasked jump in the scored year, probably an unlabelled switching event."""
PART_2_ARMS: Final[tuple[str, ...]] = (
    "g1",
    "g_all",
    "g_all_d1",
    "g_all_d1_steps",
    "climatology",
    "manual_heuristic",
)
OWN_ARMS: Final[tuple[str, ...]] = (
    "p",
    "p_map",
    "p_nonet",
    "s",
    "p_all",
    "s_all",
    "s_equiv_all",
    "p_onb0w",
    "p_onb2w",
    "p_onb4w",
    "p_onb8w",
    "s_4w",
    "p_short3m",
    "s_3m",
    "x_2y",
    "x_2yq",
    "x_2yz",
    "x_2yam",
    "x_2yamq",
    "x_4w",
    "x_3m",
)
XGB_BAR: Final[tuple[str, ...]] = ("x_2y", "x_2yq", "x_2yamq")


def site_of_id() -> dict[int, str]:
    """Return the substation label of each time-series id."""
    table = pl.read_parquet(common.SERIES_TABLE).select("time_series_id", "site")
    return dict(table.iter_rows())


def load_arm(*, name: str, folder: Path, ids: dict[int, str]) -> pl.DataFrame | None:
    """Read an arm's predictions with its per-row score, site label, and error of the mean."""
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
    treatment: str,
    reference: str,
    sites: tuple[str, ...],
    level: float | None,
    out: dict,
) -> None:
    """Print a contrast with its mean-error twin, season split, drop-one-site, and D9/D10 views."""
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
        if len(kept) >= 1:
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


def print_absolute(*, arms: dict[str, pl.DataFrame]) -> None:
    """Print each arm's headline score per site, pooled over the decision set and with D9."""
    order = ["D5", "D10", "D11", "D14", "D16", "D9", "D18"]
    print("\n== 1. Absolute quantile score, points (rows outside the switching masks) ==")
    print(f"{'arm':14}" + "".join(f"{site:>8}" for site in order) + f"{'five':>8}{'six':>8}")
    for name, frame in arms.items():
        by_site = dict(frame.group_by("site").agg(pl.col("score").mean()).iter_rows())
        five = frame.filter(pl.col("site").is_in(common.ELIGIBLE_SITES))["score"].mean()
        six = frame.filter(pl.col("site").is_in(common.SCORED_SITES))["score"].mean()
        cells = "".join(f"{by_site.get(site, float('nan')):8.3f}" for site in order)
        print(f"{name:14}{cells}{five:8.3f}{six:8.3f}")


def print_tails(*, arms: dict[str, pl.DataFrame]) -> None:
    """Print coverage and pinball loss at the four tail levels, decision set pooled."""
    print("\n== 5. Tails: coverage (nominal 0.01, 0.05, 0.95, 0.99) and pinball at each level ==")
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
            f"{name:14}"
            + "".join(f"{value:8.3f}" for value in coverage)
            + "  |  "
            + " ".join(f"{value:.3f}" for value in pinball)
        )


def print_checks(*, arms: dict[str, pl.DataFrame]) -> None:
    """Print the median minus the mean forecast per site (a bias check) for the model arms."""
    print("\n== 7. Checks: median minus mean forecast per site, points ==")
    for name in ("p", "s", "p_all", "p_onb4w", "p_short3m"):
        if name not in arms or "mean" not in arms[name].columns:
            continue
        gap = (
            arms[name]
            .with_columns(gap=pl.col("q50") - pl.col("mean"))
            .group_by("site")
            .agg(pl.col("gap").mean())
            .sort("site")
        )
        print(f"{name:12}", {site: round(value, 2) for site, value in gap.iter_rows()})


def main() -> None:
    """Print the full report and write the contrasts to `work/report_contrasts.json`."""
    ids = site_of_id()
    arms: dict[str, pl.DataFrame] = {}
    for name in OWN_ARMS:
        frame = load_arm(name=name, folder=common.DATA_DIR, ids=ids)
        if frame is not None:
            arms[name] = headline(frame)
    for name in PART_2_ARMS:
        frame = load_arm(name=name, folder=common.PART_2_DIR, ids=ids)
        if frame is not None:
            arms[name] = headline(frame)
    print_absolute(arms=arms)
    five, six = common.ELIGIBLE_SITES, common.SCORED_SITES
    out: dict = {}
    print("\n== 2. Verdict contrasts C1 to C3 (Bonferroni, 3 contrasts) ==")
    for label, treatment, reference, sites in (
        ("C1 pooled vs per-site, all rows", "p", "s", five),
        ("C2 onboarding, 4 weeks", "p_onb4w", "s_4w", six),
        ("C3 short history, 3 months", "p_short3m", "s_3m", six),
    ):
        print_contrast_block(
            label=label,
            arms=arms,
            treatment=treatment,
            reference=reference,
            sites=sites,
            level=BONFERRONI_LEVEL,
            out=out,
        )
    print("\n== 3. C4 against lagless XGBoost and C5 against Part 2's convex models (95%) ==")
    for bar in XGB_BAR:
        print_contrast_block(
            label=f"C4 vs {bar}",
            arms=arms,
            treatment="p",
            reference=bar,
            sites=five,
            level=None,
            out=out,
        )
    for reference in ("g_all_d1", "g1", "g_all", "g_all_d1_steps"):
        print_contrast_block(
            label=f"C5 vs {reference}",
            arms=arms,
            treatment="p_all",
            reference=reference,
            sites=five,
            level=None,
            out=out,
        )
    print("\n== 3b. Pooled history arms against lagless XGBoost on the same history (95%) ==")
    for treatment, reference, sites in (
        ("p_onb4w", "x_4w", six),
        ("p_short3m", "x_3m", six),
        ("s_4w", "x_4w", six),
        ("s_3m", "x_3m", six),
    ):
        print_contrast_block(
            label=f"{treatment}-{reference}",
            arms=arms,
            treatment=treatment,
            reference=reference,
            sites=sites,
            level=None,
            out=out,
        )
    print("\n== 4. Secondary and exploratory contrasts (95%) ==")
    for treatment, reference in (
        ("p_all", "s_all"),
        ("s_equiv_all", "g_all"),
        ("s_equiv_all", "s_all"),
        ("p_map", "p"),
        ("p_nonet", "p"),
        ("p_onb4w", "p"),
        ("p_onb0w", "s_4w"),
        ("p_onb2w", "s_4w"),
        ("p_onb8w", "s_4w"),
        ("p", "manual_heuristic"),
        ("p", "climatology"),
        ("s", "x_2yq"),
        ("x_2yamq", "x_2yq"),
        ("x_2yam", "x_2yz"),
        ("x_2yq", "x_2y"),
    ):
        print_contrast_block(
            label=f"{treatment}-{reference}",
            arms=arms,
            treatment=treatment,
            reference=reference,
            sites=five,
            level=None,
            out=out,
        )
    print_tails(arms=arms)
    print_checks(arms=arms)
    common.write_json(name="report_contrasts.json", payload=out)


if __name__ == "__main__":
    main()
