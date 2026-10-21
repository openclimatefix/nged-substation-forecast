"""Assemble and score experiment 149: spreads, per-origin contrasts, pooled intervals, verdict.

Reads the means of `e00149_gam.py` (`work/gam/`) and experiment 107's climatology and manual
heuristic reference rows. Applies experiment 107's baseline spread S0 (a rolling pool of the arm's
own out-of-sample residuals over the 12 months before each origin) to every arm, keeps only the
substation-origin pairs that pass the history rule (`common.passes_history_rule`), and prints the
report that the write-up quotes. It saves its tables to `work/tables/` for the charts.

**Metric.** The quantile score is twice the mean pinball loss over 11 levels, in points of effective
capacity, on rows outside the hand-filtered switching masks. Every contrast is treatment minus
reference, so a negative value favours the treatment.

    uv run --no-sync python ml-experiments/00149-*/e00149_score.py
"""

# ruff: noqa: E501
import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Final

import e00149_common as common
import numpy as np
import polars as pl
from studies.bootstrap import bootstrap_difference_at_level
from studies.dp_features import FeatureCache
from studies.dp_gam import QUANTILE_COLUMNS, ResidualSpread
from studies.sources import ML_EXPERIMENTS_DATA_DIR

ARM_NAMES: Final[tuple[str, ...]] = ("a24", "a24_d1", "aall", "aall_d1")
REFERENCE_NAMES: Final[tuple[str, ...]] = ("climatology__own", "manual_heuristic__own")
SCORED_107: Final = ML_EXPERIMENTS_DATA_DIR.glob(
    "00107-differentiable-physics-rolling-origin-history-and-tails/work/scored"
)
TABLE_DIR: Final = common.WORK_DIR / "tables"
N_WEEK_BLOCKS: Final[int] = 13
N_BLOCK_DRAWS: Final[int] = 1000
N_POOLED_RESAMPLES: Final[int] = 4000
BONFERRONI_LEVEL: Final[float] = 98.75
"""The pooled interval's coverage: 95% with a Bonferroni correction over four planned contrasts."""
TAIL_LEVELS: Final[tuple[float, ...]] = (0.01, 0.05, 0.95, 0.99)
WINTER_ROW_MONTHS: Final[tuple[int, ...]] = (11, 12, 1, 2, 3)
LEAD_BANDS: Final[tuple[tuple[str, float, float], ...]] = (
    ("day 1", 0.0, 24.0),
    ("days 2 to 7", 24.0, 168.0),
    ("after day 7", 168.0, 1e9),
)


@dataclass(frozen=True)
class Contrast:
    """One planned contrast: treatment minus reference."""

    key: str
    treatment: str
    reference: str
    text: str


CONTRASTS: Final[tuple[Contrast, ...]] = (
    Contrast("C1", "aall__s0", "a24__s0", "all history minus 24 months, no drift"),
    Contrast("C2", "aall_d1__s0", "a24_d1__s0", "all history minus 24 months, both with drift D1"),
    Contrast("C3", "aall_d1__s0", "aall__s0", "drift D1 minus no drift, all history"),
    Contrast("C4", "a24_d1__s0", "a24__s0", "drift D1 minus no drift, 24 months"),
)


def grid_positions(*, cache: FeatureCache, frame: pl.DataFrame) -> np.ndarray:
    """Return each row's position on the cache's grid."""
    valid = frame["valid_time"].dt.replace_time_zone(None).to_numpy().astype("datetime64[us]")
    position = np.searchsorted(cache.end_times, valid)
    if not np.array_equal(cache.end_times[position], valid):
        msg = "a valid time is not on the grid"
        raise ValueError(msg)
    return position


def previous_cuts(*, cut: datetime) -> list[datetime]:
    """Return the four cuts whose scored windows make the pool for origin `cut`, oldest first."""
    return [
        common.add_months(moment=cut, months=-common.SCORED_MONTHS * step)
        for step in range(common.POOL_MONTHS // common.SCORED_MONTHS, 0, -1)
    ]


def assemble() -> pl.DataFrame:
    """Return every arm's scored rows with the 11 S0 quantiles, for the passing pairs only."""
    cache = common.load_cache()
    gam = pl.concat(
        [pl.read_parquet(file) for file in sorted((common.WORK_DIR / "gam").glob("*.parquet"))]
    )
    positions = grid_positions(cache=cache, frame=gam)
    gam = gam.with_columns(
        band=pl.Series(cache.calendar.band[positions].astype(np.int64)),
        quarter=pl.Series(cache.calendar.quarter[positions].astype(np.int64)),
        residual=pl.col("actual") - pl.col("mean"),
    )
    parts: list[pl.DataFrame] = []
    for arm in ARM_NAMES:
        for site in common.DECISION_SITES:
            arm_rows = gam.filter(pl.col("arm") == arm, pl.col("site") == site)
            for cut in common.SCORED_ORIGINS:
                if not common.passes_history_rule(site=site, cut=cut):
                    continue
                scored = arm_rows.filter(pl.col("origin") == common.cut_label(cut=cut))
                labels = [common.cut_label(cut=c) for c in previous_cuts(cut=cut)]
                pool = arm_rows.filter(pl.col("origin").is_in(labels), ~pl.col("switching_mask"))
                if pool.select(pl.col("valid_time").max()).item() > cut:
                    msg = f"a pool row for origin {cut} has a valid time after the origin"
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
                    scored.select(
                        "site",
                        "origin",
                        "valid_time",
                        "lead_hours",
                        "actual",
                        "switching_mask",
                        "mean",
                    ).with_columns(
                        arm=pl.lit(f"{arm}__s0"),
                        **{
                            column: pl.Series(quantiles[:, position])
                            for position, column in enumerate(QUANTILE_COLUMNS)
                        },
                    )
                )
    return pl.concat(parts)


def read_references(*, keys: pl.DataFrame) -> pl.DataFrame:
    """Return experiment 107's climatology and manual-heuristic rows for the same pairs."""
    folder = next(iter(SCORED_107))
    parts = []
    for name in REFERENCE_NAMES:
        frame = pl.read_parquet(folder / f"{name}.parquet")
        parts.append(
            frame.join(keys, on=["site", "origin"], how="semi")
            .select(
                "site",
                "origin",
                "valid_time",
                "lead_hours",
                "actual",
                "switching_mask",
                "mean",
                *QUANTILE_COLUMNS,
            )
            .with_columns(arm=pl.lit(name))
        )
    return pl.concat(parts)


def quantile_score(*, frame: pl.DataFrame) -> np.ndarray:
    """Return twice the mean pinball loss over the 11 levels, per row."""
    return common.quantile_score(
        actual=frame["actual"].to_numpy(), quantiles=frame.select(QUANTILE_COLUMNS).to_numpy()
    )


def build_wide(*, rows: pl.DataFrame) -> pl.DataFrame:
    """Return one row per scored half-hour of the passing pairs, one score and error column per arm."""
    base = (
        rows.filter(pl.col("arm") == "a24__s0")
        .select("site", "origin", "valid_time", "lead_hours", "actual", "switching_mask")
        .with_columns(
            month=pl.col("valid_time").dt.strftime("%Y-%m"),
            calendar_month=pl.col("valid_time").dt.month(),
        )
    )
    for name in rows["arm"].unique(maintain_order=True):
        frame = rows.filter(pl.col("arm") == name)
        columns = frame.select("site", "origin", "valid_time").with_columns(
            pl.Series(f"score__{name}", quantile_score(frame=frame)),
            pl.Series(
                f"mae__{name}", np.abs(frame["actual"].to_numpy() - frame["mean"].to_numpy())
            ),
            pl.Series(f"err__{name}", (frame["actual"] - frame["mean"]).to_numpy()),
        )
        base = base.join(columns, on=["site", "origin", "valid_time"], how="left")
    return base.filter(~pl.col("switching_mask"))


def week_blocks(*, wide: pl.DataFrame) -> np.ndarray:
    """Return each row's week-block number, 0 to 12, counted from its cut."""
    days = (
        (wide["valid_time"] - wide["origin"].str.to_date().cast(pl.Datetime("us", "UTC")))
        .dt.total_days()
        .to_numpy()
    )
    return np.minimum(days // 7, N_WEEK_BLOCKS - 1).astype(np.int64)


def block_bootstrap(
    *, values: np.ndarray, blocks: np.ndarray, seed: int
) -> tuple[float, float, float, float]:
    """Return the mean, the standard error, and the 95% interval, resampling whole week blocks."""
    sums = np.bincount(blocks, weights=values, minlength=N_WEEK_BLOCKS)
    counts = np.bincount(blocks, minlength=N_WEEK_BLOCKS).astype(np.float64)
    draws = np.random.default_rng(seed).integers(
        0, N_WEEK_BLOCKS, size=(N_BLOCK_DRAWS, N_WEEK_BLOCKS)
    )
    resampled = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return (
        float(values.mean()),
        float(resampled.std()),
        float(np.percentile(resampled, 2.5)),
        float(np.percentile(resampled, 97.5)),
    )


def per_origin_table(
    *, wide: pl.DataFrame, blocks: np.ndarray, contrast: Contrast, metric: str, sites: Sequence[str]
) -> pl.DataFrame:
    """Return the contrast's estimate, standard error, interval, and tie flag at each origin."""
    difference = (
        wide[f"{metric}__{contrast.treatment}"] - wide[f"{metric}__{contrast.reference}"]
    ).to_numpy()
    keep = wide["site"].is_in(list(sites)).to_numpy() & np.isfinite(difference)
    origin = wide["origin"].to_numpy()
    rows = []
    for position, label in enumerate(sorted(set(origin[keep]))):
        pick = keep & (origin == label)
        estimate, se, lower, upper = block_bootstrap(
            values=difference[pick], blocks=blocks[pick], seed=position
        )
        tie = abs(estimate) <= se
        rows.append(
            {
                "origin": label,
                "estimate": estimate,
                "se": se,
                "lower_95": lower,
                "upper_95": upper,
                "tie": tie,
                "side": 0 if tie else int(np.sign(estimate)),
                "n_rows": int(pick.sum()),
                "n_sites": len(set(wide["site"].to_numpy()[pick])),
            }
        )
    return pl.DataFrame(rows)


def pooled_interval(
    *,
    wide: pl.DataFrame,
    contrast: Contrast,
    metric: str,
    sites: Sequence[str],
    origins: Sequence[str] | None = None,
    level: float = BONFERRONI_LEVEL,
    extra: pl.Expr | None = None,
) -> dict[str, float]:
    """Return the pooled estimate and its interval, resampling whole calendar months."""
    rows = wide.filter(pl.col("site").is_in(list(sites)))
    if origins is not None:
        rows = rows.filter(pl.col("origin").is_in(list(origins)))
    if extra is not None:
        rows = rows.filter(extra)
    treatment, reference = f"{metric}__{contrast.treatment}", f"{metric}__{contrast.reference}"
    losses = pl.concat(
        [
            rows.select(
                arm=pl.lit(name),
                site="site",
                time=pl.col("valid_time").cast(pl.String),
                seed=pl.lit(0),
                month="month",
                loss=pl.col(column),
            )
            for name, column in ((contrast.treatment, treatment), (contrast.reference, reference))
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
    estimate = rows.select((pl.col(treatment) - pl.col(reference)).mean()).item()
    return {"estimate": estimate, "lower": lower, "upper": upper, "n_rows": rows.height}


def sides(*, table: pl.DataFrame, origins: Sequence[str] | None = None) -> tuple[int, int, int]:
    """Return the origins below zero, above zero, and tied (outside the tie band)."""
    rows = table if origins is None else table.filter(pl.col("origin").is_in(list(origins)))
    return (
        int((rows["side"] == -1).sum()),
        int((rows["side"] == 1).sum()),
        int((rows["side"] == 0).sum()),
    )


def origin_groups() -> dict[str, list[str]]:
    """Return the origin labels of each origin year and quarter type."""
    groups: dict[str, list[str]] = {}
    for cut in common.SCORED_ORIGINS:
        label = common.cut_label(cut=cut)
        groups.setdefault(common.origin_year(cut=cut), []).append(label)
        groups.setdefault(common.quarter_type(cut=cut), []).append(label)
    return groups


def verdict_of(*, c2: dict) -> str:
    """Apply the pre-registered decision rule to contrast C2."""
    pooled, table, loso, groups = c2["pooled"], c2["table"], c2["loso"], c2["groups"]
    below, above, _ = sides(table=table)
    if pooled["lower"] > 0:
        return "refuted"
    if pooled["upper"] < 0:
        loso_ok = all(value["estimate"] < 0 for value in loso.values())
        if below >= 9 and above <= 1 and loso_ok:
            return "supported"
        return "partly supported"
    winter, summer = groups["winter"], groups["summer"]
    if winter["upper"] < 0 and summer["upper"] >= 0:
        return "partly supported"
    return "inconclusive"


def month_block_mean(*, values: np.ndarray, months: np.ndarray) -> tuple[float, float, float]:
    """Return a mean and its 95% interval from resampling whole calendar months."""
    unique, inverse = np.unique(months, return_inverse=True)
    sums = np.bincount(inverse, weights=values, minlength=len(unique))
    counts = np.bincount(inverse, minlength=len(unique)).astype(np.float64)
    draws = np.random.default_rng(0).integers(0, len(unique), size=(2000, len(unique)))
    resampled = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    return (
        float(values.mean()),
        float(np.percentile(resampled, 2.5)),
        float(np.percentile(resampled, 97.5)),
    )


def tail_table(*, rows: pl.DataFrame) -> pl.DataFrame:
    """Return coverage and pinball loss at the four tail levels per arm."""
    out = []
    for name in rows["arm"].unique(maintain_order=True):
        frame = rows.filter(pl.col("arm") == name, ~pl.col("switching_mask"))
        months = frame["valid_time"].dt.strftime("%Y-%m").to_numpy()
        actual = frame["actual"].to_numpy()
        for level in TAIL_LEVELS:
            quantile = frame[f"q{round(level * 100):02d}"].to_numpy()
            error = actual - quantile
            coverage, c_low, c_high = month_block_mean(
                values=(actual <= quantile).astype(np.float64), months=months
            )
            pinball, _, _ = month_block_mean(
                values=np.maximum(level * error, (level - 1.0) * error), months=months
            )
            out.append(
                {
                    "arm": name,
                    "level": level,
                    "coverage": coverage,
                    "cov_lo": c_low,
                    "cov_hi": c_high,
                    "pinball": pinball,
                }
            )
    return pl.DataFrame(out)


def reproduction_check(*, cache_rows: pl.DataFrame) -> list[str]:
    """Return lines comparing Aall and Aall+D1 means with experiment 107's G-all and G-all+D1."""
    folder = (
        ML_EXPERIMENTS_DATA_DIR
        / "00107-differentiable-physics-rolling-origin-history-and-tails/work/gam/base"
    )
    old = pl.concat([pl.read_parquet(file) for file in sorted(folder.glob("*.parquet"))])
    lines = []
    for new_name, old_name in (("aall", "g_all"), ("aall_d1", "g_all_d1")):
        new = pl.concat(
            [pl.read_parquet(f) for f in sorted((common.WORK_DIR / "gam").glob("*.parquet"))]
        ).filter(pl.col("arm") == new_name)
        joined = new.join(
            old.filter(pl.col("arm") == old_name),
            on=["site", "origin", "valid_time"],
            suffix="_107",
        )
        difference = (joined["mean"] - joined["mean_107"]).abs().max()
        lines.append(
            f"reproduction {new_name} vs 107 {old_name}: {joined.height} rows, max abs mean difference {difference}"
        )
    return lines


def main() -> None:  # noqa: PLR0915
    """Assemble, score, and print the report."""
    common.WORK_DIR.mkdir(parents=True, exist_ok=True)
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    out: list[str] = []

    def say(text: str = "") -> None:
        out.append(text)
        print(text)

    arms = assemble()
    keys = arms.select("site", "origin").unique()
    references = read_references(keys=keys)
    rows = pl.concat([arms, references.select(arms.columns)])
    wide = build_wide(rows=rows)
    wide.write_parquet(TABLE_DIR / "wide.parquet")
    blocks = week_blocks(wide=wide)
    sites = list(common.DECISION_SITES)
    groups = origin_groups()

    say("== History rule: substation-origin pairs scored ==")
    pairs = (
        wide.group_by("origin")
        .agg(pl.col("site").n_unique().alias("sites"), pl.col("site").unique().sort().alias("list"))
        .sort("origin")
    )
    for row in pairs.iter_rows(named=True):
        say(f"{row['origin']}: {row['sites']} sites {row['list']}")
    say(f"rows scored: {wide.height}")
    for line in reproduction_check(cache_rows=rows):
        say(line)

    say()
    say("== Absolute scores (quantile score, points; mean error = actual minus mean forecast) ==")
    absolute = []
    for name in rows["arm"].unique(maintain_order=True):
        score = wide[f"score__{name}"].to_numpy()
        months = wide["month"].to_numpy()
        mean, low, high = month_block_mean(values=score, months=months)
        absolute.append(
            {
                "arm": name,
                "score": mean,
                "lo": low,
                "hi": high,
                "mae": float(wide.select(pl.col(f"mae__{name}").mean()).item()),
                "mean_error": float(wide.select(pl.col(f"err__{name}").mean()).item()),
            }
        )
        say(
            f"{name}: {mean:.3f} [{low:.3f}, {high:.3f}], MAE {absolute[-1]['mae']:.3f}, mean error {absolute[-1]['mean_error']:+.3f}"
        )
    pl.DataFrame(absolute).write_parquet(TABLE_DIR / "absolute.parquet")
    say("per site (score):")
    per_site = (
        wide.group_by("site")
        .agg(
            [
                pl.col(f"score__{name}").mean().alias(name)
                for name in rows["arm"].unique(maintain_order=True)
            ]
        )
        .sort("site")
    )
    per_site.write_parquet(TABLE_DIR / "per_site.parquet")
    say(str(per_site))
    say(
        "median minus mean forecast per site: exactly 0.000 for every arm (the spread S0 is centred on its median)."
    )

    say()
    say("== Planned contrasts (treatment minus reference; negative favours the treatment) ==")
    results: dict[str, dict] = {}
    for contrast in CONTRASTS:
        table = per_origin_table(
            wide=wide, blocks=blocks, contrast=contrast, metric="score", sites=sites
        )
        table.write_parquet(TABLE_DIR / f"origin_{contrast.key}.parquet")
        mae_table = per_origin_table(
            wide=wide, blocks=blocks, contrast=contrast, metric="mae", sites=sites
        )
        mae_table.write_parquet(TABLE_DIR / f"origin_mae_{contrast.key}.parquet")
        pooled = pooled_interval(wide=wide, contrast=contrast, metric="score", sites=sites)
        pooled_mae = pooled_interval(wide=wide, contrast=contrast, metric="mae", sites=sites)
        loso = {
            site: pooled_interval(
                wide=wide, contrast=contrast, metric="score", sites=[s for s in sites if s != site]
            )
            for site in sites
        }
        by_group = {
            name: pooled_interval(
                wide=wide,
                contrast=contrast,
                metric="score",
                sites=sites,
                origins=groups[name],
                level=95.0,
            )
            for name in ("2023-24", "2024-25", "2025-26", "winter", "summer")
        }
        by_site = {
            site: pooled_interval(
                wide=wide, contrast=contrast, metric="score", sites=[site], level=95.0
            )
            for site in sites
        }
        by_season_rows = {
            "Nov-Mar rows": pooled_interval(
                wide=wide,
                contrast=contrast,
                metric="score",
                sites=sites,
                level=95.0,
                extra=pl.col("calendar_month").is_in(WINTER_ROW_MONTHS),
            ),
            "Apr-Oct rows": pooled_interval(
                wide=wide,
                contrast=contrast,
                metric="score",
                sites=sites,
                level=95.0,
                extra=~pl.col("calendar_month").is_in(WINTER_ROW_MONTHS),
            ),
        }
        by_lead = {
            label: pooled_interval(
                wide=wide,
                contrast=contrast,
                metric="score",
                sites=sites,
                level=95.0,
                extra=(pl.col("lead_hours") > low) & (pl.col("lead_hours") <= high),
            )
            for label, low, high in LEAD_BANDS
        }
        below, above, tied = sides(table=table)
        results[contrast.key] = {"pooled": pooled, "table": table, "loso": loso, "groups": by_group}
        say(f"{contrast.key}: {contrast.text}")
        say(
            f"  pooled {pooled['estimate']:+.3f} [{pooled['lower']:+.3f}, {pooled['upper']:+.3f}] ({BONFERRONI_LEVEL}% month-block), {pooled['n_rows']} rows"
        )
        say(f"  origins below / above / tied: {below} / {above} / {tied} of {table.height}")
        say(
            f"  MAE of the mean forecast: {pooled_mae['estimate']:+.3f} [{pooled_mae['lower']:+.3f}, {pooled_mae['upper']:+.3f}]"
        )
        say(
            "  per origin: "
            + ", ".join(
                f"{r['origin']} {r['estimate']:+.3f}{'~' if r['tie'] else ''}"
                for r in table.iter_rows(named=True)
            )
        )
        say(
            "  leave one site out: "
            + ", ".join(
                f"-{s} {v['estimate']:+.3f} [{v['lower']:+.3f}, {v['upper']:+.3f}]"
                for s, v in loso.items()
            )
        )
        say(
            "  by origin group (95%): "
            + ", ".join(
                f"{k} {v['estimate']:+.3f} [{v['lower']:+.3f}, {v['upper']:+.3f}]"
                for k, v in by_group.items()
            )
        )
        say(
            "  by site alone (95%): "
            + ", ".join(
                f"{k} {v['estimate']:+.3f} [{v['lower']:+.3f}, {v['upper']:+.3f}]"
                for k, v in by_site.items()
            )
        )
        say(
            "  by row season (95%): "
            + ", ".join(
                f"{k} {v['estimate']:+.3f} [{v['lower']:+.3f}, {v['upper']:+.3f}]"
                for k, v in by_season_rows.items()
            )
        )
        say(
            "  by lead band (95%): "
            + ", ".join(
                f"{k} {v['estimate']:+.3f} [{v['lower']:+.3f}, {v['upper']:+.3f}]"
                for k, v in by_lead.items()
            )
        )
        pl.DataFrame(
            [
                {"contrast": contrast.key, "group": k, **v}
                for k, v in {**by_group, **by_site, **by_season_rows, **by_lead}.items()
            ]
        ).write_parquet(TABLE_DIR / f"splits_{contrast.key}.parquet")
        pl.DataFrame(
            [{"contrast": contrast.key, "left_out": k, **v} for k, v in loso.items()]
        ).write_parquet(TABLE_DIR / f"loso_{contrast.key}.parquet")
        pl.DataFrame(
            [
                {"contrast": contrast.key, "metric": "score", **pooled},
                {"contrast": contrast.key, "metric": "mae", **pooled_mae},
            ]
        ).write_parquet(TABLE_DIR / f"pooled_{contrast.key}.parquet")

    verdict = verdict_of(c2={**results["C2"], "groups": results["C2"]["groups"]})
    say()
    say(f"== Verdict on C2 by the pre-registered rule: {verdict} ==")
    say("== Tail calibration (coverage of the quantile; nominal level in column `level`) ==")
    tails = tail_table(rows=rows.filter(pl.col("site").is_in(sites)))
    tails.write_parquet(TABLE_DIR / "tails.parquet")
    say(str(tails))
    (common.WORK_DIR / "report.txt").write_text("\n".join(out) + "\n")
    (common.WORK_DIR / "verdict.json").write_text(json.dumps({"verdict": verdict}))


if __name__ == "__main__":
    main()
