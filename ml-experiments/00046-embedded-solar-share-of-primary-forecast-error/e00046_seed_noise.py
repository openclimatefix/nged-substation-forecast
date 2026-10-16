"""Print how much the seed-to-seed refit changes experiment 00046's scores and contrasts.

Reads the assembled arms through `e00046_stats.load_losses()` and keeps the scored rows (April to
September), the same rows the contrasts of `e00046_stats.py` use. For each series and for the
pooled verdict series it prints the standard deviation across the 3 seeds of:

- the mean quantile score of `d0`;
- C6 (`d1l_ens_shift` minus `d0`), with C6 for each seed;
- C3 (`d1l_ens_shift` minus `d1l`);
- `d0` minus `d1l`.

Every score is in points of capacity (1 point is 1% of the series' effective capacity). The
standard deviation is the sample one (`ddof=1`) across the seeds. The per-series rows cover the six
series of the experiment; the pooled row covers the five verdict series (D5 excluded).

    uv run python ml-experiments/00046-*/e00046_seed_noise.py
"""

import os
from typing import Final

os.environ.setdefault("POLARS_MAX_THREADS", "2")
os.environ.setdefault("OMP_NUM_THREADS", "2")

import e00046_common as common
import e00046_stats as stats
import polars as pl

CONTRASTS: Final[dict[str, tuple[str, str]]] = {
    "C6": ("d1l_ens_shift", "d0"),
    "C3": ("d1l_ens_shift", "d1l"),
    "d0_minus_d1l": ("d0", "d1l"),
}
"""Each printed contrast as (treatment, reference)."""


def seed_scores(*, scored: pl.DataFrame, by: list[str]) -> pl.DataFrame:
    """Return the mean quantile score of each arm, for each seed and each group in `by`."""
    return scored.group_by(*by, "arm", "seed").agg(qs=pl.col("qs").mean())


def noise_table(*, scores: pl.DataFrame, by: list[str]) -> pl.DataFrame:
    """Return per group the standard deviation across seeds of `d0` and of each contrast."""
    wide = scores.pivot(on="arm", index=[*by, "seed"], values="qs")
    wide = wide.with_columns(
        *(
            (pl.col(treatment) - pl.col(reference)).alias(name)
            for name, (treatment, reference) in CONTRASTS.items()
        )
    )
    aggregations = [pl.col("d0").std().alias("d0_sd")] + [
        pl.col(name).std().alias(f"{name}_sd") for name in CONTRASTS
    ]
    if by:
        return wide.group_by(*by, maintain_order=True).agg(*aggregations).sort(*by)
    return wide.select(*aggregations)


def main() -> None:
    """Print the seed-to-seed standard deviations per series, pooled, and C6 per seed."""
    losses = stats.load_losses()
    scored = losses.filter(pl.col("scored") & pl.col("arm").is_in(["d0", "d1l", "d1l_ens_shift"]))
    verdict = scored.filter(stats.VERDICT_SCOPE)
    print(f"seeds {sorted(scored['seed'].unique().to_list())}; sites {list(common.SITES)}")
    rows = verdict.filter((pl.col("arm") == "d0") & (pl.col("seed") == 0)).height
    print(f"scored rows per seed and arm on the verdict series: {rows:,}")
    with pl.Config(tbl_rows=50, tbl_cols=20, fmt_float="full", float_precision=5):
        print("\nSD across seeds of mean quantile score (points of capacity), per series")
        print(
            noise_table(scores=seed_scores(scored=scored, by=["series_label"]), by=["series_label"])
        )
        pooled = noise_table(scores=seed_scores(scored=verdict, by=[]), by=[])
        print("\nSD across seeds, pooled over the 5 verdict series")
        print(pooled)
        per_seed = seed_scores(scored=verdict, by=[]).pivot(on="arm", index="seed", values="qs")
        print("\nPooled verdict-series mean score and C6 by seed")
        print(per_seed.with_columns(C6=pl.col("d1l_ens_shift") - pl.col("d0")).sort("seed"))
        print("\nC6 by seed, per series")
        print(
            seed_scores(scored=scored, by=["series_label"])
            .pivot(on="arm", index=["series_label", "seed"], values="qs")
            .with_columns(C6=pl.col("d1l_ens_shift") - pl.col("d0"))
            .sort("series_label", "seed")
        )


if __name__ == "__main__":
    main()
