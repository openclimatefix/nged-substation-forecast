"""Split the excess loss of experiment 172's unlabelled worst days three ways (experiment 174).

Weather: how much of a day's excess disappears when arm D is driven by ERA5 and CAMS. Shared: the
day is also in the worst 5% of days of another Primary (BG2's ranking). Neither: the excess that
stays with perfect weather on a day no other Primary shares. Prints the report; writes the day
tables under `WORK_DIR`.

    systemd-run --user --scope -p MemoryMax=16G -p OOMPolicy=continue \
        uv run --no-sync python ml-experiments/00174-*/e00174_run.py > printed_report.txt
"""

import e00174_common as common
import e00174_controls as controls
import e00174_others as others
import numpy as np
import polars as pl

pl.Config.set_tbl_rows(100)
pl.Config.set_tbl_cols(30)
pl.Config.set_tbl_width_chars(220)
KEYS = ["site", "local_date"]
DECISION = list(common.DECISION_SITES)


def weather_part(*, e_ens: pl.Expr, e_new: pl.Expr) -> pl.Expr:
    """Return the part of a day's excess that disappears: `E_ens - E_new` within `[0, E_ens]`."""
    return pl.min_horizontal(pl.max_horizontal(e_ens - e_new, pl.lit(0.0)), e_ens)


def joined_days(
    *, ens: pl.DataFrame, era5: pl.DataFrame, spread_a: pl.DataFrame, era5f: pl.DataFrame
) -> pl.DataFrame:
    """Join the three arms' day tables on (site, local_date)."""
    pick = lambda frame, suffix: frame.select(  # noqa: E731
        *KEYS,
        pl.col("score").alias(f"score_{suffix}"),
        pl.col("excess").alias(f"excess_{suffix}"),
        pl.col("mean_abs_error").alias(f"mae_{suffix}"),
        pl.col("excess_mae").alias(f"excess_mae_{suffix}"),
    )
    return (
        pick(ens, "ens")
        .join(pick(era5, "era5"), on=KEYS)
        .join(pick(spread_a, "sa"), on=KEYS)
        .join(pick(era5f, "f"), on=KEYS)
        .with_columns(
            weather_f=weather_part(e_ens=pl.col("excess_ens"), e_new=pl.col("excess_f")),
            weather=weather_part(e_ens=pl.col("excess_ens"), e_new=pl.col("excess_era5")),
            weather_sa=weather_part(e_ens=pl.col("excess_ens"), e_new=pl.col("excess_sa")),
            weather_abs=weather_part(
                e_ens=pl.col("excess_ens"),
                e_new=pl.col("score_era5") - pl.col("score_ens") + pl.col("excess_ens"),
            ),
            weather_mae=weather_part(
                e_ens=pl.col("excess_mae_ens"), e_new=pl.col("excess_mae_era5")
            ),
        )
    )


def shares(*, frame: pl.DataFrame) -> dict[str, float]:
    """Return the weather shares of the frame's excess, by each definition."""
    total = common.total(frame["excess_ens"])
    return {
        "W": common.total(frame["weather"]) / total,
        "W_spread_a": common.total(frame["weather_sa"]) / total,
        "W_filter_on_era5": common.total(frame["weather_f"]) / total,
        "W_absolute_reference": common.total(frame["weather_abs"]) / total,
        "W_mean_error": common.total(frame["weather_mae"]) / common.total(frame["excess_mae_ens"]),
    }


def month_bootstrap(*, frame: pl.DataFrame, rng: np.random.Generator) -> dict[str, np.ndarray]:
    """Resample the calendar months of the frame's days; return draws of W, S, and the split."""
    months = sorted(frame["year_month"].unique().to_list())
    blocks = {
        m: frame.filter(pl.col("year_month") == m)
        .select(
            e=pl.col("excess_ens").sum(),
            w=pl.col("weather").sum(),
            wsh=(pl.col("weather") * pl.col("shared")).sum(),
            sh=(pl.col("excess_ens") * pl.col("shared")).sum(),
            n=pl.len(),
            ns=pl.col("shared").sum(),
        )
        .row(0)
        for m in months
    }
    draws = {"W": [], "S": [], "shared_part": [], "neither_part": []}
    for _ in range(common.N_RESAMPLES):
        pick = rng.integers(0, len(months), len(months))
        e, w, wsh, sh, n, ns = np.sum([blocks[months[i]] for i in pick], axis=0)
        if e <= 0 or n == 0:
            continue
        draws["W"].append(w / e)
        draws["S"].append(ns / n)
        draws["shared_part"].append((sh - wsh) / e)
        draws["neither_part"].append((e - w - (sh - wsh)) / e)
    return {k: np.array(v) for k, v in draws.items()}


def interval(*, draws: np.ndarray) -> str:
    """Return a 95% percentile interval as text."""
    low, high = np.percentile(draws, [2.5, 97.5])
    return f"95% interval {low:.3f} to {high:.3f}"


def main() -> None:  # noqa: PLR0915
    """Run the analysis and print the report."""
    rng = np.random.default_rng(common.SEED)
    common.WORK_DIR.mkdir(parents=True, exist_ok=True)
    tables = {
        arm: common.day_table(rows=common.read_arm_rows(arm=arm))
        for arm in ("e155_d", "ens_d", "era5_d", "era5_d_spread_a", "era5f_d")
    }
    reference = pl.read_parquet(common.E172_WORK_DIR / "days.parquet")
    check = tables["ens_d"].join(reference.select(*KEYS, "score", "excess"), on=KEYS, how="full")
    print(
        "== Reproduction of experiment 172's day table: days",
        tables["ens_d"].height,
        "vs",
        reference.height,
        "max |score difference|",
        (check["score"] - check["score_right"]).abs().max(),
    )
    days = joined_days(
        ens=tables["ens_d"],
        era5=tables["era5_d"],
        spread_a=tables["era5_d_spread_a"],
        era5f=tables["era5f_d"],
    ).join(
        reference.select(
            *KEYS, "year_month", "labelled", "switching", "holiday", "cold", "flagged", "economy7"
        ),
        on=KEYS,
    )
    worst = (
        reference.filter(pl.col("site").is_in(DECISION))
        .with_columns(
            rank=pl.col("excess").rank(method="ordinal", descending=True).over("site"),
            n=pl.len().over("site"),
        )
        .filter(pl.col("rank") <= (pl.col("n") * common.WORST_FRACTION).ceil())
        .select(*KEYS)
    )
    print("worst days (decision sites):", worst.height)
    bg2 = others.worst_flags(days=others.bg2_days())
    links = others.neighbours()
    numbers = others.primary_numbers()
    pair_count = len({frozenset((numbers[a], numbers[b])) for a, v in links.items() for b in v})
    print(
        f"== BG2 series: {len(numbers)} on {len(set(numbers.values()))} Primaries; "
        f"interconnected pairs of Primaries: {pair_count}"
    )
    target = (
        days.join(worst, on=KEYS)
        .pipe(lambda f: others.coincidence(flags=bg2, targets=f, links=links))
        .with_columns(shared=pl.col("n_others") >= 1, near=pl.col("n_neighbours") >= 1)
    )
    target.write_parquet(common.WORK_DIR / "worst_days_split.parquet")
    print("rows without a BG2 day:", target.height - worst.height)
    # headline
    for name, scope in (
        ("unlabelled", target.filter(~pl.col("labelled"))),
        ("labelled", target.filter(pl.col("labelled"))),
        ("all 72", target),
    ):
        w = shares(frame=scope)
        print(f"\n== {name}: {scope.height} days, total excess {scope['excess_ens'].sum():.1f}")
        print("  weather shares:", {k: round(v, 3) for k, v in w.items()})
        print(
            f"  shared (any other Primary): {scope['shared'].mean():.3f}; "
            f"interconnected neighbour: {scope['near'].mean():.3f}; "
            f"mean others {scope['n_others'].mean():.2f}"
        )
    unl = target.filter(~pl.col("labelled"))
    boot = month_bootstrap(frame=unl, rng=rng)
    print("\n== Unlabelled-days bootstrap (months):")
    for key in ("W", "S", "shared_part", "neither_part"):
        print(f"  {key}: point {boot[key].mean():.3f}, {interval(draws=boot[key])}")
    total = common.total(unl["excess_ens"])
    weather = common.total(unl["weather"])
    shared = common.total((unl["excess_ens"] - unl["weather"]) * unl["shared"])
    neither = total - weather - shared
    print(
        f"  split of the unlabelled days' excess: weather {weather / total:.3f}, "
        f"shared {shared / total:.3f}, neither {neither / total:.3f}"
    )
    print(
        "  ... of which remains on days with an interconnected neighbour also worst:",
        round(common.total((unl["excess_ens"] - unl["weather"]) * unl["near"]) / total, 3),
    )
    print("\n== Leave one substation out (unlabelled days): W, S")
    for site in DECISION:
        rest = unl.filter(pl.col("site") != site)
        print(f"  without {site}: W {shares(frame=rest)['W']:.3f}, S {rest['shared'].mean():.3f}")
    print("== Median forecast minus mean forecast, ERA5 arm, per site:")
    rows_era5 = common.read_arm_rows(arm="era5_d")
    print(
        rows_era5.group_by("site")
        .agg(
            median_minus_mean=(pl.col("q50") - pl.col("mean")).mean(),
            max_abs=(pl.col("q50") - pl.col("mean")).abs().max(),
        )
        .sort("site")
    )
    # chance control
    chance = others.shifted_rate(flags=bg2, targets=unl, rng=rng)
    for key, observed in (
        ("any_other", unl["shared"].mean()),
        ("any_neighbour", unl["near"].mean()),
    ):
        draws = chance[key]
        print(
            f"== Circular-shift control, {key}: observed {observed:.3f}; chance mean "
            f"{draws.mean():.3f} ({interval(draws=draws)}); share of draws at or above "
            f"observed {(draws >= observed).mean():.3f}"
        )
    print("\n== Known-answer controls (10 points on 10 random days per decision site):")
    print(
        "  ",
        {
            name: {k: round(v, 3) for k, v in values.items()}
            for name, values in controls.run(days=days, rng=rng).items()
        },
    )
    together = others.shifted_rate(flags=bg2, targets=unl, rng=rng, common_shift=True)
    draws = together["any_other"]
    print(
        "== Circular-shift control with one shared shift for all other Primaries, any_other: "
        f"chance mean {draws.mean():.3f} ({interval(draws=draws)}); share of draws at or above "
        f"observed {(draws >= unl['shared'].mean()).mean():.3f}"
    )
    np.save(common.WORK_DIR / "chance_any_other.npy", chance["any_other"])
    pl.DataFrame(
        {
            "part": ["Weather", "Shared with another Primary", "Neither"],
            "share": [weather / total, shared / total, neither / total],
            "low": [np.percentile(boot[k], 2.5) for k in ("W", "shared_part", "neither_part")],
            "high": [np.percentile(boot[k], 97.5) for k in ("W", "shared_part", "neither_part")],
        }
    ).write_parquet(common.WORK_DIR / "split.parquet")
    # per day table
    print("\n== Unlabelled worst days")
    print(
        unl.sort("excess_ens", descending=True)
        .select(
            *KEYS,
            "excess_ens",
            "excess_era5",
            "weather",
            "n_others",
            "n_neighbours",
        )
        .with_columns(pl.col(pl.Float32, pl.Float64).round(2))
    )
    print("\n== Per site (unlabelled)")
    print(
        unl.group_by("site")
        .agg(
            n=pl.len(),
            excess=pl.col("excess_ens").sum(),
            weather=pl.col("weather").sum(),
            shared_share=pl.col("shared").mean(),
        )
        .with_columns(w=pl.col("weather") / pl.col("excess"))
        .sort("site")
    )
    print("\n== Per month (unlabelled)")
    print(
        unl.group_by("year_month")
        .agg(
            n=pl.len(),
            excess=pl.col("excess_ens").sum(),
            weather=pl.col("weather").sum(),
            shared_share=pl.col("shared").mean(),
        )
        .sort("year_month")
    )
    # all days: weather effect on every day, for context
    print("\n== All days of decision sites: pooled score ENS vs ERA5")
    scope = days.filter(pl.col("site").is_in(DECISION))
    print(
        scope.select(
            ens=pl.col("score_ens").mean(),
            era5=pl.col("score_era5").mean(),
            spread_a=pl.col("score_sa").mean(),
            mae_ens=pl.col("mae_ens").mean(),
            mae_era5=pl.col("mae_era5").mean(),
        )
    )
    days.write_parquet(common.WORK_DIR / "days_joined.parquet")


if __name__ == "__main__":
    main()
