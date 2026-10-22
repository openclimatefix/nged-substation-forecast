"""Known-answer controls for the split of experiment 174.

Ten random days per decision substation get 10 points of error, in two ways:

- demand side: 10 points added to the observed demand of the day in both arms, an error that
  past weather cannot remove (the day should keep its excess, so its weather share should be
  below 0.5);
- weather side: 10 points added to the ensemble-driven arm's mean and quantiles only, an error
  that past weather removes (its weather share should be at least 0.5).

A day is recovered when its weather share is on the right side of 0.5. The controls pass when at
least 90% of the injected days are recovered. Experiment 155's offset and profile are not rebuilt
after the injection, so the control tests the day-level split and not the arms' error correction.
"""

import e00174_common as common
import numpy as np
import polars as pl

KEYS = ["site", "local_date"]
QUANTILE_COLUMNS = ["mean", *common.LEVELS]


def injected_days(*, days: pl.DataFrame, rng: np.random.Generator) -> pl.DataFrame:
    """Return `INJECTED_DAYS` random (site, local_date) days per decision site."""
    pieces = []
    for site in common.DECISION_SITES:
        dates = days.filter(pl.col("site") == site)["local_date"].to_list()
        chosen = rng.choice(len(dates), size=common.INJECTED_DAYS, replace=False)
        pieces.append(
            pl.DataFrame({"site": site, "local_date": [dates[i] for i in sorted(chosen)]})
        )
    return pl.concat(pieces)


def run(*, days: pl.DataFrame, rng: np.random.Generator) -> dict[str, dict[str, float]]:
    """Return, per control, the share of injected days recovered and the mean weather share."""
    chosen = injected_days(days=days, rng=rng)
    ens = common.read_arm_rows(arm="ens_d", sites=common.DECISION_SITES)
    era5 = common.read_arm_rows(arm="era5_d", sites=common.DECISION_SITES)
    out = {}
    for name in ("demand_side", "weather_side"):
        ens_c, era5_c = ens, era5
        flag = (
            pl.struct("site", "local_date")
            .is_in(chosen.select(pl.struct("site", "local_date")).to_series())
            .alias("injected")
        )
        if name == "demand_side":
            bump = pl.when(flag).then(common.INJECTED_POINTS).otherwise(0.0)
            ens_c = ens.with_columns(actual=pl.col("actual") + bump).with_columns(
                score=common.pinball_score_expr()
            )
            era5_c = era5.with_columns(actual=pl.col("actual") + bump).with_columns(
                score=common.pinball_score_expr()
            )
        else:
            bump = pl.when(flag).then(common.INJECTED_POINTS).otherwise(0.0)
            ens_c = ens.with_columns(
                *[(pl.col(c) + bump).alias(c) for c in QUANTILE_COLUMNS]
            ).with_columns(score=common.pinball_score_expr())
        d_ens, d_era5 = common.day_table(rows=ens_c), common.day_table(rows=era5_c)
        joined = (
            chosen.join(d_ens.select(*KEYS, e_ens="excess"), on=KEYS)
            .join(d_era5.select(*KEYS, e_era5="excess"), on=KEYS)
            .with_columns(
                share=(
                    pl.min_horizontal(
                        pl.max_horizontal(pl.col("e_ens") - pl.col("e_era5"), pl.lit(0.0)),
                        pl.col("e_ens"),
                    )
                    / pl.col("e_ens")
                )
            )
        )
        right = pl.col("share") >= 0.5 if name == "weather_side" else pl.col("share") < 0.5
        by_site = joined.group_by("site").agg(recovered=right.mean()).sort("site")
        out[name] = {
            "recovered": float(joined.select(right.mean()).item()),
            "mean_weather_share": common.mean_of(joined["share"]),
            "n": joined.height,
            **{
                f"recovered_{row['site']}": float(row["recovered"])
                for row in by_site.iter_rows(named=True)
            },
        }
    return out
