"""Score experiment 167: spreads, quantile scores, the three planned contrasts, and the checks.

The spread of an arm at an origin is experiment 107's baseline (S0): the arm's own out-of-sample
residuals over the four previous scored windows, per hour band and calendar quarter, centred. A
site-origin pair is scored only if the site has 24 months of usable rows before the origin, in all
arms. Every number the write-up quotes is printed by this script, and the whole report is written
to `report.txt` in the experiment's data folder.

    uv run --no-sync python ml-experiments/00167-*/e00167_score.py
"""

import io
import sys
from collections.abc import Sequence
from contextlib import redirect_stdout
from dataclasses import dataclass
from datetime import datetime
from typing import Final

import e00167_common as common
import numpy as np
import polars as pl
from studies.bootstrap import bootstrap_difference_at_level
from studies.dp_gam import QUANTILE_COLUMNS, QUANTILE_LEVELS, ResidualSpread

ARMS: Final[tuple[str, ...]] = ("m24", "all", "all_d1", "m24_d1")
N_WEEK_BLOCKS: Final[int] = 13
N_BLOCK_DRAWS: Final[int] = 4000
N_POOLED_RESAMPLES: Final[int] = 4000
POOLED_LEVEL: Final[float] = 98.33
"""The pooled interval's coverage: 95% Bonferroni-corrected over three planned contrasts."""
THRESHOLD: Final[float] = 0.05
MIN_NEGATIVE_ORIGINS: Final[int] = 8
TAIL_LEVELS: Final[tuple[str, ...]] = ("q01", "q05", "q95", "q99")


@dataclass(frozen=True)
class Contrast:
    """A treatment arm minus a reference arm."""

    name: str
    treatment: str
    reference: str
    planned: bool
    label: str


CONTRASTS: Final[tuple[Contrast, ...]] = (
    Contrast("C1", "all_d1", "m24", True, "all years with drift minus 24 months"),
    Contrast("C2", "all", "m24", True, "all years without drift minus 24 months"),
    Contrast("C3", "all_d1", "all", True, "drift minus no drift, on all years"),
    Contrast("E1", "m24_d1", "m24", False, "drift minus no drift, on 24 months"),
    Contrast(
        "E2", "all_d1", "m24_d1", False, "all years minus 24 months, both with drift (post hoc)"
    ),
)


def previous_cuts(*, cut: datetime) -> list[datetime]:
    """Return the four cuts whose scored windows make the pool for origin `cut`, oldest first."""
    return [
        common.add_months(moment=cut, months=-common.SCORED_MONTHS * step)
        for step in range(common.POOL_MONTHS // common.SCORED_MONTHS, 0, -1)
    ]


def read_means(*, variant: str) -> pl.DataFrame:
    """Return every fit's rows, with D10's unmasked refits swapped in for the `unmasked` variant."""
    folder = common.WORK_DIR / "means"
    files = sorted(folder.glob("*.parquet"))
    if variant == "unmasked":
        files = [file for file in files if not file.name.startswith("D10_")]
        files += sorted((common.WORK_DIR / "means_unmasked").glob("D10_*.parquet"))
    return pl.concat([pl.read_parquet(file) for file in files])


def grid_positions(*, end_times: np.ndarray, frame: pl.DataFrame) -> np.ndarray:
    """Return each row's position on the cache's grid."""
    valid = frame["valid_time"].dt.replace_time_zone(None).to_numpy().astype("datetime64[us]")
    position = np.searchsorted(end_times, valid)
    if not np.array_equal(end_times[position], valid):
        msg = "a valid time is not on the grid"
        raise ValueError(msg)
    return position


def scored_rows(*, means: pl.DataFrame) -> pl.DataFrame:
    """Return the scored rows with their 11 quantiles, one row per arm, site, origin, half-hour.

    Args:
        means: Every fit's rows.

    Returns:
        The scored rows, with `q01` to `q99`.

    Raises:
        ValueError: If a pool row ends after its origin.
    """
    cache = common.d10_masked_cache(cache=common.load_cache())
    positions = grid_positions(end_times=cache.end_times, frame=means)
    means = means.with_columns(
        band=pl.Series(cache.calendar.band[positions].astype(np.int64)),
        quarter=pl.Series(cache.calendar.quarter[positions].astype(np.int64)),
        residual=pl.col("actual") - pl.col("mean"),
    )
    parts = []
    for arm in ARMS:
        for site in common.ALL_SITES:
            rows = means.filter(pl.col("arm") == arm, pl.col("site") == site)
            for cut in common.SCORED_ORIGINS:
                if not common.is_scored(cache=cache, site=site, cut=cut):
                    continue
                scored = rows.filter(pl.col("origin") == common.cut_label(cut=cut))
                labels = [common.cut_label(cut=c) for c in previous_cuts(cut=cut)]
                pool = rows.filter(pl.col("origin").is_in(labels), ~pl.col("switching_mask"))
                if pool.height and pool["valid_time"].max() > cut:  # ty: ignore[unsupported-operator]
                    msg = f"a pool row for {site} origin {cut} ends after the origin"
                    raise ValueError(msg)
                spread = ResidualSpread.fit(
                    residuals=pool["residual"].to_numpy(),
                    band=pool["band"].to_numpy(),
                    quarter=pool["quarter"].to_numpy(),
                    centre=True,
                )
                quantiles = np.sort(
                    spread.quantiles(
                        mean=scored["mean"].to_numpy(),
                        band=scored["band"].to_numpy(),
                        quarter=scored["quarter"].to_numpy(),
                    ),
                    axis=1,
                )
                parts.append(
                    scored.drop("band", "quarter", "residual").with_columns(
                        **{
                            column: pl.Series(quantiles[:, position])
                            for position, column in enumerate(QUANTILE_COLUMNS)
                        }
                    )
                )
    return pl.concat(parts)


def build_wide(*, scored: pl.DataFrame) -> pl.DataFrame:
    """Return one row per half-hour with each arm's score and absolute error as columns."""
    scored = scored.with_columns(
        score=pl.Series(
            common.quantile_score(
                actual=scored["actual"].to_numpy(),
                quantiles=scored.select(QUANTILE_COLUMNS).to_numpy(),
            )
        ),
        mae=(pl.col("actual") - pl.col("mean")).abs(),
    )
    keys = ["site", "origin", "valid_time"]
    wide = scored.filter(pl.col("arm") == ARMS[0]).select(
        *keys, "actual", "switching_mask", "lead_hours"
    )
    for arm in ARMS:
        part = scored.filter(pl.col("arm") == arm).select(
            *keys,
            pl.col("score").alias(f"score__{arm}"),
            pl.col("mae").alias(f"mae__{arm}"),
            pl.col("mean").alias(f"mean__{arm}"),
            pl.col("q50").alias(f"q50__{arm}"),
        )
        wide = wide.join(part, on=keys, how="left")
    for arm in ARMS:
        if scored.filter(pl.col("arm") == arm).height != wide.height:
            msg = f"arm {arm} does not score the same rows as {ARMS[0]}"
            raise ValueError(msg)
    if wide.select(pl.any_horizontal(pl.all().is_null())).to_series().any():
        msg = "an arm lacks a row that another arm has"
        raise ValueError(msg)
    return wide.with_columns(month=pl.col("valid_time").dt.strftime("%Y-%m"))


def week_blocks(*, wide: pl.DataFrame) -> np.ndarray:
    """Return each row's week-block number, 0 to 12, counted from its cut."""
    days = (
        (wide["valid_time"] - wide["origin"].str.to_date().cast(pl.Datetime("us", "UTC")))
        .dt.total_days()
        .to_numpy()
    )
    return np.minimum(days // 7, N_WEEK_BLOCKS - 1).astype(np.int64)


def block_bootstrap(*, values: np.ndarray, blocks: np.ndarray, seed: int) -> tuple[float, float]:
    """Resample whole week blocks; return the estimate and the standard error of the resamples."""
    sums = np.bincount(blocks, weights=values, minlength=N_WEEK_BLOCKS)
    counts = np.bincount(blocks, minlength=N_WEEK_BLOCKS).astype(np.float64)
    draws = np.random.default_rng(seed).integers(
        0, N_WEEK_BLOCKS, size=(N_BLOCK_DRAWS, N_WEEK_BLOCKS)
    )
    resampled = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return float(values.mean()), float(resampled.std())


def usable(*, wide: pl.DataFrame, sites: Sequence[str], origins: Sequence[str]) -> pl.DataFrame:
    """Return the rows of the given sites and origins outside the switching masks."""
    return wide.filter(
        pl.col("site").is_in(list(sites)),
        pl.col("origin").is_in(list(origins)),
        ~pl.col("switching_mask"),
    )


def per_origin(
    *, wide: pl.DataFrame, contrast: Contrast, metric: str, sites: Sequence[str]
) -> pl.DataFrame:
    """Return the contrast's estimate, standard error, and side (-1, 0, +1) at each origin.

    A difference smaller than one week-block standard error counts as a tie (side 0).
    """
    rows = usable(wide=wide, sites=sites, origins=origin_labels())
    blocks = week_blocks(wide=rows)
    difference = (
        rows[f"{metric}__{contrast.treatment}"] - rows[f"{metric}__{contrast.reference}"]
    ).to_numpy()
    origin = rows["origin"].to_numpy()
    out = []
    for position, label in enumerate(origin_labels()):
        pick = origin == label
        if not pick.any():
            continue
        estimate, se = block_bootstrap(values=difference[pick], blocks=blocks[pick], seed=position)
        tie = abs(estimate) <= se
        out.append(
            {
                "origin": label,
                "estimate": estimate,
                "se": se,
                "side": 0 if tie else int(np.sign(estimate)),
                "negative": estimate < 0,
                "n_rows": int(pick.sum()),
            }
        )
    return pl.DataFrame(out)


def origin_labels() -> list[str]:
    """Return the 12 scored origins' labels."""
    return [common.cut_label(cut=cut) for cut in common.SCORED_ORIGINS]


def pooled(
    *,
    wide: pl.DataFrame,
    contrast: Contrast,
    metric: str,
    sites: Sequence[str],
    origins: Sequence[str],
    level: float = POOLED_LEVEL,
) -> dict[str, float]:
    """Return the pooled estimate and its interval, resampling whole calendar months."""
    rows = usable(wide=wide, sites=sites, origins=origins)
    if rows.is_empty():
        return {"estimate": float("nan"), "lower": float("nan"), "upper": float("nan"), "n_rows": 0}
    losses = pl.concat(
        [
            rows.select(
                arm=pl.lit(name),
                site="site",
                time=pl.col("valid_time").cast(pl.String),
                seed=pl.lit(0),
                month="month",
                loss=pl.col(f"{metric}__{name}"),
            )
            for name in (contrast.treatment, contrast.reference)
        ]
    )
    lower, upper = bootstrap_difference_at_level(
        losses=losses,
        treatment=contrast.treatment,
        reference=contrast.reference,
        metric="loss",
        level=level,
        n_resamples=N_POOLED_RESAMPLES,
    )
    estimate = rows.select(
        (
            pl.col(f"{metric}__{contrast.treatment}") - pl.col(f"{metric}__{contrast.reference}")
        ).mean()
    ).item()
    return {"estimate": estimate, "lower": lower, "upper": upper, "n_rows": rows.height}


def verdict_of(*, c1: dict[str, float], n_negative: int) -> str:
    """Apply the pre-registered decision rule to C1."""
    if c1["lower"] > 0:
        return "refuted"
    if c1["upper"] < 0:
        if c1["estimate"] <= -THRESHOLD and n_negative >= MIN_NEGATIVE_ORIGINS:
            return "supported"
        return "partly supported"
    return "inconclusive"


def interval_text(*, value: dict[str, float]) -> str:
    """Format a pooled estimate with its interval."""
    return (
        f"{value['estimate']:+.3f} (98.33% interval {value['lower']:+.3f} to {value['upper']:+.3f})"
    )


def report_absolute(*, wide: pl.DataFrame) -> None:
    """Print each arm's mean score by site and pooled, on the decision sites."""
    print("\n== Absolute quantile score (points), decision sites, rows outside switching masks ==")
    rows = usable(wide=wide, sites=common.DECISION_SITES, origins=origin_labels())
    for arm in ARMS:
        by_site = rows.group_by("site").agg(pl.col(f"score__{arm}").mean()).sort("site")
        text = ", ".join(
            f"{r['site']} {r[f'score__{arm}']:.3f}" for r in by_site.iter_rows(named=True)
        )
        print(f"{arm:7s} pooled {rows[f'score__{arm}'].mean():.3f}  | {text}")
        print(f"        mae {rows[f'mae__{arm}'].mean():.3f}")
    print("\nD9 (descriptive, scored from 2025-10-01):")
    d9 = usable(wide=wide, sites=common.DESCRIPTIVE_SITES, origins=origin_labels())
    for arm in ARMS:
        print(f"  {arm:7s} {d9[f'score__{arm}'].mean():.3f} over {d9.height} rows")


def report_contrasts(*, wide: pl.DataFrame) -> dict[str, dict]:
    """Print each contrast pooled, per origin, by season, and leaving one site out."""
    results: dict[str, dict] = {}
    all_origins = origin_labels()
    winter = [common.cut_label(cut=c) for c in common.SCORED_ORIGINS if c.month in (10, 1)]
    summer = [o for o in all_origins if o not in winter]
    for contrast in CONTRASTS:
        print(
            f"\n== {contrast.name}: {contrast.label} "
            f"({contrast.treatment} - {contrast.reference}) =="
        )
        quantile = pooled(
            wide=wide,
            contrast=contrast,
            metric="score",
            sites=common.DECISION_SITES,
            origins=all_origins,
        )
        mean_error = pooled(
            wide=wide,
            contrast=contrast,
            metric="mae",
            sites=common.DECISION_SITES,
            origins=all_origins,
        )
        table = per_origin(
            wide=wide, contrast=contrast, metric="score", sites=common.DECISION_SITES
        )
        n_negative = int(table["negative"].sum())
        n_neg_tie = int((table["side"] == -1).sum())
        n_pos_tie = int((table["side"] == 1).sum())
        print(
            f"quantile score pooled: {interval_text(value=quantile)}; n_rows {quantile['n_rows']}"
        )
        print(f"mean-absolute-error contrast: {interval_text(value=mean_error)}")
        print(
            f"origins below zero: {n_negative} of {table.height}; "
            f"outside tie band: {n_neg_tie} below, {n_pos_tie} above"
        )
        for row in table.iter_rows(named=True):
            print(
                f"  {row['origin']}: {row['estimate']:+.3f} (se {row['se']:.3f}) "
                f"side {row['side']:+d}"
            )
        for name, group in (("winter-quarter origins", winter), ("summer-quarter origins", summer)):
            seasonal = pooled(
                wide=wide,
                contrast=contrast,
                metric="score",
                sites=common.DECISION_SITES,
                origins=group,
            )
            below = int(table.filter(pl.col("origin").is_in(group))["negative"].sum())
            print(f"{name}: {interval_text(value=seasonal)}; {below} of {len(group)} below zero")
        loso = {}
        for left_out in common.DECISION_SITES:
            kept = [s for s in common.DECISION_SITES if s != left_out]
            sub = pooled(
                wide=wide, contrast=contrast, metric="score", sites=kept, origins=all_origins
            )
            sub_table = per_origin(wide=wide, contrast=contrast, metric="score", sites=kept)
            loso[left_out] = {**sub, "n_negative": int(sub_table["negative"].sum())}
            print(
                f"  without {left_out}: {interval_text(value=sub)}; "
                f"{loso[left_out]['n_negative']} of 12 below zero"
            )
        by_site = {}
        for site in common.DECISION_SITES:
            site_pooled = pooled(
                wide=wide, contrast=contrast, metric="score", sites=[site], origins=all_origins
            )
            by_site[site] = site_pooled["estimate"]
        print("  per site: " + ", ".join(f"{s} {v:+.3f}" for s, v in by_site.items()))
        results[contrast.name] = {"quantile": quantile, "n_negative": n_negative, "loso": loso}
    return results


def report_verdict(*, results: dict[str, dict]) -> None:
    """Print the pre-registered verdict and whether leaving one site out changes it."""
    c1 = results["C1"]
    verdict = verdict_of(c1=c1["quantile"], n_negative=c1["n_negative"])
    print(f"\n== Verdict by the pre-registered rule: {verdict} ==")
    print(f"C1 {interval_text(value=c1['quantile'])}; {c1['n_negative']} of 12 origins below zero")
    flips = {
        site: verdict_of(c1=value, n_negative=value["n_negative"])
        for site, value in c1["loso"].items()
    }
    print("Verdict leaving each site out: " + ", ".join(f"{s} {v}" for s, v in flips.items()))
    if any(v != verdict for v in flips.values()):
        print("The verdict changes when one site is left out, so it is capped at inconclusive.")
        verdict = "inconclusive"
    print(f"Final verdict after the leave-one-site-out cap: {verdict}")


def report_checks(*, wide: pl.DataFrame, scored: pl.DataFrame) -> None:
    """Print median-minus-mean per site and arm, and tail coverage and pinball loss."""
    print("\n== Checks ==")
    rows = usable(wide=wide, sites=common.DECISION_SITES, origins=origin_labels())
    print("Median forecast minus mean forecast, per site (points):")
    for arm in ARMS:
        by_site = (
            rows.group_by("site")
            .agg((pl.col(f"q50__{arm}") - pl.col(f"mean__{arm}")).mean().alias("gap"))
            .sort("site")
        )
        print(
            f"  {arm:7s} "
            + ", ".join(f"{r['site']} {r['gap']:+.3f}" for r in by_site.iter_rows(named=True))
        )
    print(
        "Tail coverage (share of actuals at or below the quantile) and pinball loss, "
        "decision sites:"
    )
    keep = scored.filter(
        pl.col("site").is_in(list(common.DECISION_SITES)), ~pl.col("switching_mask")
    )
    for arm in ARMS:
        group = keep.filter(pl.col("arm") == arm)
        actual = group["actual"].to_numpy()
        cells = []
        for column in TAIL_LEVELS:
            level = QUANTILE_LEVELS[QUANTILE_COLUMNS.index(column)]
            q = group[column].to_numpy()
            coverage = float((actual <= q).mean())
            error = actual - q
            pinball = float(np.maximum(level * error, (level - 1.0) * error).mean())
            cells.append(f"{column} cover {coverage:.3f} pinball {pinball:.3f}")
        print(f"  {arm:7s} " + "; ".join(cells))
    print(
        "Rows scored per arm: "
        + ", ".join(f"{arm} {keep.filter(pl.col('arm') == arm).height}" for arm in ARMS)
    )


def report_rule(*, scored: pl.DataFrame) -> None:
    """Print which site-origin pairs the 24-month history rule kept."""
    print("\n== History rule: scored origins per site ==")
    pairs = scored.select("site", "origin").unique().sort("site", "origin")
    for site in common.ALL_SITES:
        origins = pairs.filter(pl.col("site") == site)["origin"].to_list()
        print(f"  {site}: {len(origins)} origins" + (f", first {origins[0]}" if origins else ""))


def main() -> None:
    """Score everything and write the printed report."""
    variant = "unmasked" if "--unmasked" in sys.argv else "masked"
    suffix = "_unmasked" if variant == "unmasked" else ""
    buffer = io.StringIO()
    with redirect_stdout(buffer):
        scored = scored_rows(means=read_means(variant=variant))
        wide = build_wide(scored=scored)
        scored_path = common.WORK_DIR / f"scored{suffix}.parquet"
        common.write_atomic(frame=scored, path=scored_path)
        report_rule(scored=scored)
        report_absolute(wide=wide)
        results = report_contrasts(wide=wide)
        report_verdict(results=results)
        report_checks(wide=wide, scored=scored)
    text = buffer.getvalue()
    (common.DATA_DIR / f"report{suffix}.txt").write_text(text)
    sys.stdout.write(text)


if __name__ == "__main__":
    main()
