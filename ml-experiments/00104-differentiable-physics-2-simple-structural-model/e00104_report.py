"""Score every arm and print the numbers the write-up quotes.

Reads the prediction sets that `e00104_gam.py`, `e00104_xgb.py`, and `e00104_references.py` wrote
and prints one report: absolute scores, the three planned contrasts with Bonferroni intervals, the
exploratory contrasts, the prediction checks, tails, staleness by month, capacities, solar weights,
scarce-data rows, the cold block, and the lagged references. Everything the write-up quotes comes
from this report.

The score of a row is twice the mean pinball loss over the 11 levels, in points of effective
capacity. The headline rows are those with a reading and outside the hand-filtered switching events
at the scored substation. The decision set is D5, D10, D11, D14, and D16; D9 holds 9.0 months at
2024-07-01 and is reported beside it.

    uv run python ml-experiments/00104-*/e00104_report.py > ml-experiments/00104-*/draft/report.txt
"""

import json
from datetime import UTC, datetime
from typing import Final

import e00104_common as common
import numpy as np
import polars as pl
from studies.bootstrap import (
    bootstrap_difference,
    bootstrap_difference_at_level,
    paired_differences,
)
from studies.dp_gam import QUANTILE_COLUMNS
from studies.experiment_store import losses_for_bootstrap

SITE_OF_ID: Final[dict[int, str]] = {7: "D9", 8: "D11", 10: "D10", 16: "D14", 17: "D5", 18: "D16"}
GAM_ARMS: Final[tuple[str, ...]] = common.GAM_ARMS
XGB_ARMS: Final[tuple[str, ...]] = (
    "x_12",
    "x_12z",
    "x_12am",
    "x_12amq",
    "x_all",
    "x_allz",
    "x_allam",
    "x_allamq",
)
NO_IRRADIANCE_ARMS: Final[tuple[str, ...]] = ("x_12z_ng", "x_12am_ng")
REFERENCE_ARMS: Final[tuple[str, ...]] = ("climatology", "manual_heuristic")
MAIN_ARMS: Final[tuple[str, ...]] = (*GAM_ARMS, *XGB_ARMS, *NO_IRRADIANCE_ARMS, *REFERENCE_ARMS)
KEYS: Final[tuple[str, ...]] = ("time_series_id", "power_fcst_init_time", "valid_time", "seed")
N_PLANNED_CONTRASTS: Final[int] = 3
BONFERRONI_LEVEL: Final[float] = 100.0 * (1.0 - 0.05 / N_PLANNED_CONTRASTS)
N_RESAMPLES: Final[int] = 20000
TAIL_LEVELS: Final[tuple[float, ...]] = (0.01, 0.05, 0.95, 0.99)
TAIL_COLUMNS: Final[tuple[str, ...]] = ("q01", "q05", "q95", "q99")
ID_OF_SITE: Final[dict[str, int]] = {site: ident for ident, site in SITE_OF_ID.items()}


def load_arm(*, name: str, headline: bool = True, uncentred: bool = False) -> pl.DataFrame:
    """Read an arm's predictions with its per-row score and site label.

    Args:
        name: The prediction set's name.
        headline: True to drop rows inside the switching masks.
        uncentred: True to read a convex-model arm's first-run set, whose spread carries the
            replica year's mean bias, whatever `common.SPREAD` says.

    Returns:
        The predictions with `score` and `site` columns.
    """
    stored = name if uncentred else common.stored_name(name=name)
    frame = pl.read_parquet(common.DATA_DIR / "predictions" / stored / "predictions.parquet")
    if headline:
        frame = frame.filter(~pl.col("switching_mask"))
    quantiles = frame.select(QUANTILE_COLUMNS).to_numpy().astype(np.float64)
    actual = frame["actual"].to_numpy().astype(np.float64)
    return frame.with_columns(
        score=pl.Series(common.quantile_score(actual=actual, quantiles=quantiles)),
        site=pl.col("time_series_id").replace_strict(SITE_OF_ID, return_dtype=pl.String),
    )


def loss_frame(*, arms: dict[str, pl.DataFrame], sites: tuple[str, ...]) -> pl.DataFrame:
    """Build the bootstrap loss frame of the given arms restricted to `sites`."""
    return losses_for_bootstrap(
        arm_losses={
            name: frame.filter(pl.col("site").is_in(sites)).select(*KEYS, "score")
            for name, frame in arms.items()
        }
    )


def contrast(
    *,
    arms: dict[str, pl.DataFrame],
    treatment: str,
    reference: str,
    sites: tuple[str, ...],
    level: float | None,
) -> dict:
    """Return a paired contrast (treatment minus reference) with an interval and site counts.

    Args:
        arms: Every arm's predictions with scores.
        treatment: The arm compared.
        reference: The arm it is compared against.
        sites: The substations pooled.
        level: Coverage in percent, or `None` for the 95% interval.

    Returns:
        The estimate, interval, the number of sites that improve and worsen, and per-site means.
    """
    losses = loss_frame(arms={treatment: arms[treatment], reference: arms[reference]}, sites=sites)
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
        arms[treatment]
        .filter(pl.col("site").is_in(sites))
        .select("site", "valid_time", treatment=pl.col("score"))
        .join(
            arms[reference]
            .filter(pl.col("site").is_in(sites))
            .select("site", "valid_time", reference=pl.col("score")),
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
        f"{result['level']:.2f}%); {result['better']} sites better, {result['worse']} worse"
    )


def print_absolute(*, arms: dict[str, pl.DataFrame]) -> None:
    """Print each arm's headline score per site and pooled over the decision set."""
    print(
        "\n== 1. Absolute quantile score, points of capacity (rows outside the switching masks) =="
    )
    order = ["D5", "D10", "D11", "D14", "D16", "D9"]
    print(f"{'arm':18}" + "".join(f"{site:>8}" for site in order) + f"{'five':>8}{'six':>8}")
    for name, frame in arms.items():
        by_site = dict(frame.group_by("site").agg(pl.col("score").mean()).iter_rows())
        five = frame.filter(pl.col("site").is_in(common.ELIGIBLE_SITES))["score"].mean()
        print(
            f"{name:18}"
            + "".join(f"{by_site[site]:8.3f}" for site in order)
            + f"{five:8.3f}{frame['score'].mean():8.3f}"
        )
    print("\n(five = D5, D10, D11, D14, D16 pooled; six adds D9)")


def print_contrasts(*, arms: dict[str, pl.DataFrame]) -> dict[str, dict]:
    """Print the planned and exploratory contrasts, and return them by name."""
    five, six = common.ELIGIBLE_SITES, common.SCORED_SITES
    results: dict[str, dict] = {}
    print("\n== 2. Planned contrasts (Bonferroni, 3 contrasts), decision set of five ==")
    for name, treatment, reference in (
        ("C1 years", "g_all_d1", "g1"),
        ("C2 GAM vs XGBoost, 12 months", "g1", "x_12am"),
        ("C3 GAM vs XGBoost, all years", "g_all_d1", "x_allam"),
    ):
        results[name] = contrast(
            arms=arms, treatment=treatment, reference=reference, sites=five, level=BONFERRONI_LEVEL
        )
        print(f"{name}: {treatment} - {reference}: {fmt(results[name])}")
        print(f"   per site: { {k: round(v, 3) for k, v in results[name]['per_site'].items()} }")
        with_nine = contrast(
            arms=arms, treatment=treatment, reference=reference, sites=six, level=BONFERRONI_LEVEL
        )
        print(f"   with D9 (six sites): {fmt(with_nine)}")
    print("\n== 3. Exploratory contrasts (95%), decision set of five ==")
    for treatment, reference in (
        ("g_all", "g1"),
        ("g_all_d1_steps", "g_all_d1"),
        ("g_all_d1", "g_all"),
        ("x_12z", "x_12"),
        ("x_allz", "x_all"),
        ("x_12am", "x_12z"),
        ("x_allam", "x_allz"),
        ("x_12amq", "x_12"),
        ("x_allamq", "x_all"),
        ("x_12am_ng", "x_12z_ng"),
        ("x_all", "x_12"),
        ("x_allam", "x_12am"),
        ("g_all", "x_allam"),
        ("g1", "x_12"),
        ("g1", "x_12amq"),
        ("g_all", "x_all"),
        ("g_all", "x_12"),
        ("g_all_d1", "x_12"),
        ("g_all", "x_allamq"),
        ("g_all_d1", "x_all"),
        ("g_all_d1", "x_allamq"),
    ):
        results[f"{treatment}-{reference}"] = contrast(
            arms=arms, treatment=treatment, reference=reference, sites=five, level=None
        )
        print(f"{treatment} - {reference}: {fmt(results[f'{treatment}-{reference}'])}")
    print("\n== 3b. The coastal substations (D5 and D9) ==")
    for treatment, reference in (("g_all_d1", "g1"), ("g1", "x_12am"), ("g_all_d1", "x_allam")):
        coastal = contrast(
            arms=arms,
            treatment=treatment,
            reference=reference,
            sites=common.COASTAL_SITES,
            level=None,
        )
        print(f"{treatment} - {reference}: {fmt(coastal)}; per site {coastal['per_site']}")
    return results


def verdict(*, c1: dict) -> str:
    """Apply the merged Phase A rule to C1 alone: supported, partly supported, or not supported."""
    if c1["high"] < 0 and c1["better"] >= 4:
        return "supported"
    if c1["estimate"] < 0 and c1["better"] >= 4:
        return "partly supported"
    if c1["estimate"] < 0:
        return "partly supported"
    if c1["low"] > 0:
        return "refuted"
    return "not supported"


def print_checks(*, results: dict[str, dict], arms: dict[str, pl.DataFrame]) -> None:
    """Print each pre-registered prediction as held or failed."""
    print("\n== 4. Prediction checks ==")

    def line(label: str, held: bool, detail: str) -> None:
        print(f"{'HELD  ' if held else 'FAILED'} {label}: {detail}")

    g_all = results["g_all-g1"]
    line(
        "old years without drift hurt (G-all - G1 +0.3 to +0.8, interval above zero)",
        0.3 <= g_all["estimate"] <= 0.8 and g_all["low"] > 0,
        fmt(g_all),
    )
    solar = results["x_12am-x_12z"]
    line(
        "solar offset (X-12am - X-12z -0.40 to -0.10, interval below zero)",
        -0.4 <= solar["estimate"] <= -0.1 and solar["high"] < 0,
        fmt(solar),
    )
    ng = results["x_12am_ng-x_12z_ng"]
    line(
        "solar offset without the irradiance input (X-12am-ng - X-12z-ng -0.40 to -0.10)",
        -0.4 <= ng["estimate"] <= -0.1 and ng["high"] < 0,
        fmt(ng),
    )
    zero = results["x_12z-x_12"]
    line(
        "zero start (X-12z - X-12 between -0.10 and +0.02)",
        -0.1 <= zero["estimate"] <= 0.02,
        fmt(zero),
    )
    c1 = results["C1 years"]
    line(
        "C1 +0.0 to +0.4, interval spanning zero",
        0.0 <= c1["estimate"] <= 0.4 and c1["low"] < 0 < c1["high"],
        fmt(c1),
    )
    c2 = results["C2 GAM vs XGBoost, 12 months"]
    line("C2 +0.1 to +0.5", 0.1 <= c2["estimate"] <= 0.5, fmt(c2))
    c3 = results["C3 GAM vs XGBoost, all years"]
    line("C3 +0.0 to +0.5", 0.0 <= c3["estimate"] <= 0.5, fmt(c3))
    steps = results["g_all_d1_steps-g_all_d1"]
    line("steps within +-0.1 of D1", abs(steps["estimate"]) <= 0.1, fmt(steps))
    worse = []
    for name in (*GAM_ARMS, *XGB_ARMS):
        for site, value in arms[name].group_by("site").agg(pl.col("score").mean()).iter_rows():
            for reference in REFERENCE_ARMS:
                base = arms[reference].filter(pl.col("site") == site)["score"].mean()
                if value >= base:
                    worse.append(f"{name} vs {reference} at {site} ({value:.2f} vs {base:.2f})")
    line("baselines worse than every model arm at every site", not worse, "; ".join(worse) or "yes")
    print(
        f"\nC1 alone under the merged Phase A rule: {verdict(c1=c1)}. The plan overrides this with "
        "`inconclusive` when a positive control fails."
    )


def print_tails(*, arms: dict[str, pl.DataFrame]) -> None:
    """Print coverage and pinball loss at the four tail levels, decision set pooled."""
    print("\n== 5. Tails: coverage (nominal 1%, 5%, 95%, 99%) and pinball loss at each level ==")
    print(f"{'arm':18}" + "".join(f"{c:>9}" for c in TAIL_COLUMNS) + "   |   " + "pinball")
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
            f"{name:18}"
            + "".join(f"{value:9.3f}" for value in coverage)
            + "   |   "
            + " ".join(f"{value:.3f}" for value in pinball)
        )


def print_staleness(*, arms: dict[str, pl.DataFrame]) -> None:
    """Print the score and the q05 and q95 coverage by month since the cut."""
    print("\n== 6. Staleness: score and tail coverage by month of the scored year ==")
    names = ("g1", "g_all", "g_all_d1", "x_12am", "x_allam")
    header = f"{'month':>6}" + "".join(f"{name:>22}" for name in names)
    print("score | q05 coverage | q95 coverage\n" + header)
    months: dict[str, dict[int, tuple[float, float, float]]] = {}
    for name in names:
        kept = (
            arms[name]
            .filter(pl.col("site").is_in(common.ELIGIBLE_SITES))
            .with_columns(
                month=(
                    (pl.col("valid_time").dt.year() - 2025) * 12
                    + pl.col("valid_time").dt.month()
                    - 7
                )
            )
        )
        stats = kept.group_by("month").agg(
            pl.col("score").mean(),
            (pl.col("actual") <= pl.col("q05")).mean().alias("c05"),
            (pl.col("actual") <= pl.col("q95")).mean().alias("c95"),
        )
        months[name] = {m: (s, a, b) for m, s, a, b in stats.iter_rows()}
    for month in range(12):
        print(
            f"{month:6d}"
            + "".join("   {:5.2f} {:5.3f} {:5.3f}".format(*months[n][month]) for n in names)
        )


def vintage_and_hindsight(*, site: str) -> pl.DataFrame:
    """Return the season-year capacity in MW from the real-time vintage and the hindsight file."""
    from e00104_xgb import VINTAGES_PATH, vintage_capacity

    cache = common.load_cache()
    table = cache.sites[site].table
    vintage = vintage_capacity(cache=cache, site=site, cut=common.FINAL_CUT)
    weeks = pl.DataFrame(
        {
            "week_start": vintage.weeks,
            "vintage_mw": vintage.points * table.effective_capacity_mw / 100,
        }
    )
    hindsight = (
        pl.read_parquet(VINTAGES_PATH.parent / "capacity_weekly.parquet")
        .filter(
            pl.col("variant") == "season_year_constant",
            pl.col("time_series_id") == table.time_series_id,
        )
        .select("week_start", hindsight_mw=pl.col("capacity_mw"))
    )
    return weeks.join(hindsight, on="week_start", how="inner").with_columns(
        season_year=pl.col("week_start").dt.year()
        - (pl.col("week_start").dt.month() < 7).cast(pl.Int32)
    )


def print_capacity() -> None:
    """Print the fitted capacities beside the vintage and the hindsight constant."""
    print("\n== 7. Capacity per season-year, MW (fitted by the convex model, and experiment 45) ==")
    fitted = pl.read_parquet(common.WORK_DIR / "gam_capacities_scored.parquet")
    for site in common.SCORED_SITES:
        both = vintage_and_hindsight(site=site)
        ratio = both["vintage_mw"] / both["hindsight_mw"]
        print(
            f"{site}: vintage 'frozen' over hindsight season-year constant, {len(both)} weeks: "
            f"min {ratio.min():.4f}, max {ratio.max():.4f}, "
            f"share within 0.1%: {float(np.mean(np.abs(ratio.to_numpy() - 1.0) < 1e-3)):.3f}"
        )
        by_year = both.group_by("season_year").agg(
            pl.col("vintage_mw").mean(), pl.col("hindsight_mw").mean()
        )
        for arm in ("g1", "g_all", "g_all_d1"):
            row = fitted.filter(
                pl.col("site") == site, pl.col("arm") == arm, pl.col("season_year") == 2024
            )
            vintage_2024 = by_year.filter(pl.col("season_year") == 2024)["vintage_mw"]
            if row.height and vintage_2024.len():
                value = row["capacity_mw"][0]
                print(
                    f"   {arm} 2024-25: {value:.2f} MW; vintage {vintage_2024[0]:.2f}; "
                    f"ratio {value / vintage_2024[0]:.2f}"
                )


def print_weights() -> None:
    """Print each substation's fitted solar weight `w` and the capacity rows it needed."""
    weights = pl.read_parquet(common.WORK_DIR / "xgb_weights.parquet").sort("site", "window")
    print("\n== 8. Solar offset weights (fitted on a replica's April-to-September residuals) ==")
    print(weights)


def print_scarce(*, arms: dict[str, pl.DataFrame]) -> None:
    """Print mean score by scarce-data row kind, with no interval."""
    print("\n== 9. Scarce-data rows (descriptive, no interval): mean score by kind ==")
    rows = pl.read_parquet(common.PART_1_ROWS, columns=["site", "time", "scarce_kind"]).rename(
        {"time": "valid_time"}
    )
    names = ("g1", "g_all", "g_all_d1", "x_12am", "x_allam", "manual_heuristic")
    print(f"{'kind':14}{'rows':>8}" + "".join(f"{n:>17}" for n in names))
    joined = {
        name: arms[name]
        .filter(pl.col("site").is_in(common.ELIGIBLE_SITES))
        .join(rows, on=["site", "valid_time"], how="left")
        .with_columns(kind=pl.col("scarce_kind").fill_null("other"))
        for name in names
    }
    for kind in ("bank_holiday", "christmas_run", "cold_day", "other"):
        values = [joined[n].filter(pl.col("kind") == kind)["score"].mean() for n in names]
        count = joined[names[0]].filter(pl.col("kind") == kind).height
        print(f"{kind:14}{count:8d}" + "".join(f"{v:17.3f}" for v in values))


def print_cold() -> None:
    """Print the cold block, November 2022 to February 2023: the central forecast's error."""
    print("\n== 10. Cold block, November 2022 to February 2023 (descriptive) ==")
    print("Mean absolute error of the central forecast (GAM mean; XGBoost median), points:")
    names = ("cold_g1", "cold_g_all", "cold_g_all_d1", "cold_x_12")
    frames = {n: load_arm(name=n) for n in names}
    print(f"{'site':6}" + "".join(f"{n:>16}" for n in names))
    for site in common.COLD_SITES:
        values = []
        for n in names:
            kept = frames[n].filter(pl.col("site") == site)
            centre = kept["mean"] if "mean" in kept.columns else kept["q50"]
            values.append(float(np.mean(np.abs((kept["actual"] - centre).to_numpy()))))
        print(f"{site:6}" + "".join(f"{v:16.3f}" for v in values))
    mean_of = {}
    for n, f in frames.items():
        kept = f.filter(pl.col("site").is_in(common.COLD_SITES))
        centre = kept["mean"] if "mean" in kept.columns else kept["q50"]
        mean_of[n] = float(np.mean(np.abs((kept["actual"] - centre).to_numpy())))
    print("pooled:", {k: round(v, 3) for k, v in mean_of.items()})


def print_lagged(*, arms: dict[str, pl.DataFrame]) -> None:
    """Print the descriptive lagged reference contrasts at the two lead bands."""
    print("\n== 11. Lagged reference (descriptive): lagged XGBoost minus the lagless arm ==")
    for scheme, band in (("d1", "leads 24.5 to 48 h"), ("d6", "leads 144.5 to 168 h")):
        extra = {name: load_arm(name=name) for name in (f"x_12_{scheme}", f"l_12_{scheme}")}
        result = contrast(
            arms=extra,
            treatment=f"l_12_{scheme}",
            reference=f"x_12_{scheme}",
            sites=common.ELIGIBLE_SITES,
            level=None,
        )
        scores = {
            n: round(
                float(
                    np.mean(
                        f.filter(pl.col("site").is_in(common.ELIGIBLE_SITES))["score"].to_numpy()
                    )
                ),
                3,
            )
            for n, f in extra.items()
        }
        print(f"{band}: {fmt(result)}; absolute {scores}")
    del arms


D10_JUMP_START: Final[datetime] = datetime(2025, 11, 17, tzinfo=UTC)
D10_JUMP_END: Final[datetime] = datetime(2025, 12, 15, tzinfo=UTC)
"""D10's scored-year window with a 15 to 20 point rise in residuals and in the night-time floor
(from 17 November to 14 December 2025), probably unlabelled switching. An earlier window of the
same kind starts on 8 November 2024, inside the training data, and is not removed from any fit."""

MEAN_FORECAST_CONTRASTS: Final[tuple[tuple[str, str], ...]] = (
    ("g_all_d1", "g1"),
    ("g_all", "g1"),
    ("g_all_d1", "g_all"),
    ("g1", "x_12am"),
    ("g_all_d1", "x_allam"),
    ("g_all", "x_12"),
)


def print_spread_bias() -> None:
    """Print the score of the uncentred spread against the centred spread, per convex arm.

    The uncentred sets (the first run's) add the replica year's mean bias to every quantile.
    """
    print(
        "\n== 12. The replica's mean bias as its own arm: uncentred minus centred spread =="
        "\n(scores: five sites, rows outside the switching masks; q50 shifts: all rows)"
    )
    sites = common.ELIGIBLE_SITES
    for name in common.GAM_ARMS:
        centred = pl.read_parquet(
            common.DATA_DIR / "predictions" / f"{name}_centred" / "predictions.parquet"
        )
        raw = pl.read_parquet(common.DATA_DIR / "predictions" / name / "predictions.parquet")
        scored = {}
        for label, unmasked in (("centred", centred), ("uncentred", raw)):
            headline = unmasked.filter(~pl.col("switching_mask"))
            score = common.quantile_score(
                actual=headline["actual"].to_numpy().astype(np.float64),
                quantiles=headline.select(QUANTILE_COLUMNS).to_numpy().astype(np.float64),
            )
            scored[label] = headline.with_columns(
                score=pl.Series(score),
                site=pl.col("time_series_id").replace_strict(SITE_OF_ID, return_dtype=pl.String),
            )
        five = {
            k: float(v.filter(pl.col("site").is_in(sites))["score"].mean())  # ty: ignore[invalid-argument-type]
            for k, v in scored.items()
        }
        by_site = {
            label: dict(frame.group_by("site").agg(pl.col("score").mean()).sort("site").iter_rows())
            for label, frame in scored.items()
        }
        shift = (
            raw.join(centred, on=["time_series_id", "valid_time"], suffix="_c", validate="1:1")
            .group_by("time_series_id")
            .agg((pl.col("q50") - pl.col("q50_c")).mean().alias("shift"))
            .with_columns(site=pl.col("time_series_id").replace_strict(SITE_OF_ID))
        )
        print(
            f"{name:16} five-site score: uncentred {five['uncentred']:.3f}, centred "
            f"{five['centred']:.3f}; uncentred minus centred "
            f"{five['uncentred'] - five['centred']:+.3f}"
        )
        print(
            "   per site (uncentred minus centred): "
            + ", ".join(
                f"{site} {by_site['uncentred'][site] - by_site['centred'][site]:+.2f}"
                for site in sorted(by_site["centred"])
            )
        )
        print(
            "   mean shift of q50 (points): "
            + ", ".join(
                f"{s} {v:+.2f}" for s, v in sorted(shift.select("site", "shift").iter_rows())
            )
        )


def print_d10_window(*, arms: dict[str, pl.DataFrame]) -> None:
    """Print the headline contrasts and D10's scores with and without D10's 2025 jump window."""
    print("\n== 13. D10 without its 2025-11-17 to 2025-12-14 window (rows dropped at D10 only) ==")
    inside = (
        (pl.col("site") == "D10")
        & (pl.col("valid_time") >= D10_JUMP_START)
        & (pl.col("valid_time") < D10_JUMP_END)
    )
    trimmed = {name: frame.filter(~inside) for name, frame in arms.items()}
    n_dropped = arms["g1"].filter(inside).height
    print(f"rows dropped: {n_dropped} of {arms['g1'].filter(pl.col('site') == 'D10').height}")
    print(f"{'arm':18}{'D10 with':>10}{'D10 without':>13}{'five with':>11}{'five without':>14}")
    for name in ("g1", "g_all", "g_all_d1", "g_all_d1_steps", "x_12", "x_12am", "x_allam"):
        frame, cut = arms[name], trimmed[name]
        print(
            f"{name:18}"
            f"{frame.filter(pl.col('site') == 'D10')['score'].mean():10.3f}"
            f"{cut.filter(pl.col('site') == 'D10')['score'].mean():13.3f}"
            f"{frame.filter(pl.col('site').is_in(common.ELIGIBLE_SITES))['score'].mean():11.3f}"
            f"{cut.filter(pl.col('site').is_in(common.ELIGIBLE_SITES))['score'].mean():14.3f}"
        )
    print("contrasts, five sites, 95% interval, window dropped:")
    for treatment, reference in MEAN_FORECAST_CONTRASTS:
        result = contrast(
            arms=trimmed,
            treatment=treatment,
            reference=reference,
            sites=common.ELIGIBLE_SITES,
            level=None,
        )
        print(f"  {treatment} - {reference}: {fmt(result)}")


LEAVE_ONE_OUT_CONTRASTS: Final[tuple[tuple[str, str], ...]] = (
    ("g_all_d1", "g1"),
    ("g1", "x_12am"),
    ("g_all_d1", "x_allam"),
    ("g_all", "g1"),
    ("g_all_d1", "g_all"),
)


def print_checks_for_audit(*, arms: dict[str, pl.DataFrame]) -> None:
    """Print the spread-free and leave-one-out checks of the `ml-experiment` skill's audit rule.

    The error of the central forecast is the absolute error of the convex model's mean and of
    XGBoost's median, so it does not depend on the spread. Each contrast is also repeated with one
    decision site left out.
    """
    print("\n== 14. Checks: error of the central forecast, and leaving one site out ==")
    print("median forecast minus mean forecast, headline rows, mean per site (points):")
    for name in common.GAM_ARMS:
        frame = arms[name].filter(pl.col("site").is_in(common.SCORED_SITES))
        shift = dict(
            frame.group_by("site")
            .agg((pl.col("q50") - pl.col("mean")).mean())
            .sort("site")
            .iter_rows()
        )
        print(f"  {name:16}" + ", ".join(f"{k} {v:+.2f}" for k, v in shift.items()))
    central = {}
    for name, frame in arms.items():
        column = "mean" if name in common.GAM_ARMS else "q50"
        central[name] = frame.with_columns(
            score=(pl.col("actual") - pl.col(column)).abs().cast(pl.Float64)
        )
    print("mean absolute error of the central forecast, five sites (GAM mean, XGBoost median):")
    for name in ("g1", "g_all", "g_all_d1", "g_all_d1_steps", "x_12", "x_all", "x_12amq"):
        mae = central[name].filter(pl.col("site").is_in(common.ELIGIBLE_SITES))["score"].mean()
        print(f"  {name:16}{mae:8.3f}")
    print("contrasts on the error of the central forecast (95% interval), five sites:")
    for treatment, reference in LEAVE_ONE_OUT_CONTRASTS:
        error = contrast(
            arms=central,
            treatment=treatment,
            reference=reference,
            sites=common.ELIGIBLE_SITES,
            level=None,
        )
        quantile = contrast(
            arms=arms,
            treatment=treatment,
            reference=reference,
            sites=common.ELIGIBLE_SITES,
            level=None,
        )
        print(
            f"  {treatment} - {reference}: error {error['estimate']:+.3f} "
            f"({error['low']:+.3f} to {error['high']:+.3f}); quantile score "
            f"{quantile['estimate']:+.3f}"
        )
    print("leaving one decision site out, quantile score, point estimate and sites better:")
    for treatment, reference in LEAVE_ONE_OUT_CONTRASTS:
        cells = []
        for left_out in common.ELIGIBLE_SITES:
            kept = tuple(site for site in common.ELIGIBLE_SITES if site != left_out)
            result = contrast(
                arms=arms, treatment=treatment, reference=reference, sites=kept, level=None
            )
            cells.append(f"without {left_out} {result['estimate']:+.3f}")
        print(f"  {treatment} - {reference}: " + "; ".join(cells))


def main() -> None:
    """Print the whole report."""
    arms = {name: load_arm(name=name) for name in MAIN_ARMS}
    print("Experiment 104 report. Rows outside the switching masks; points of effective capacity.")
    tuned = json.loads((common.WORK_DIR / "tuned.json").read_text())
    print(
        "tuned: ridge",
        tuned["ridge"]["chosen"],
        "drift step",
        tuned["drift_step"]["chosen"],
        "step penalty",
        tuned["step_penalty"]["chosen"],
        "steps gain over D1 on the tuning year",
        round(tuned["steps_gain_over_d1"], 3),
    )
    print_absolute(arms=arms)
    results = print_contrasts(arms=arms)
    print_checks(results=results, arms=arms)
    print_tails(arms=arms)
    print_staleness(arms=arms)
    print_capacity()
    print_weights()
    print_scarce(arms=arms)
    print_cold()
    print_lagged(arms=arms)
    print_spread_bias()
    print_d10_window(arms=arms)
    print_checks_for_audit(arms=arms)
    print("\nAll-rows scores (including rows inside the switching masks), decision set of five:")
    for name in MAIN_ARMS:
        frame = load_arm(name=name, headline=False)
        five = frame.filter(pl.col("site").is_in(common.ELIGIBLE_SITES))["score"].mean()
        print(f"  {name:18}{five:8.3f}")


if __name__ == "__main__":
    main()
