"""Step 3: which property of the ensemble's wind is wrong, one correction at a time.

Each correction is fitted on experiment 00259's 442 daily hindcast runs against the implied wind
of `e00307_implied.py`, applied to every ensemble member of the 53 scored Tuesday runs (multiplying
the 10 m and 100 m speeds alike, so the shear exponent is unchanged), and passed through the
ERA5-fitted physics core, its residual quantiles, and the equal-weight member mixture. An arm is
scored by the share of kept rows whose observed power is below the mixture's median. The gap an arm
closes is that share minus the raw mixture's, in points, out of the gap from the raw mixture to the
ERA5 oracle.

Arms: `raw` (no correction); `a_mean` (one multiplicative factor); `e_lead` (a factor per lead
day); `d_direction` (a factor per 45-degree sector of the ensemble-mean direction); `b_spread`
(member deviations about the case mean multiplied by a factor per lead day); `a_mean_era5` and
`e_lead_era5` (the same factors fitted to ERA5's 100 m speed, as a control);
`c_shift_<hours>` (each member's speeds read from a different valid time of the same run);
`all_edb` (e, then d, then b).

Output, in `work/`: `arms_scored.parquet` (one row per case, a `q50` and `score` per arm) and
`corrections.json` (fitted factors).

    systemd-run --user --scope -p MemoryMax=12G -p OOMPolicy=continue \
      uv run --no-sync python ml-experiments/00307-*/e00307_corrections.py
"""

import json
import os
import pickle
from pathlib import Path
from typing import Final

for _name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_name, "1")

import e00307_common as common  # noqa: E402
import e00307_core as core_module  # noqa: E402
import numpy as np  # noqa: E402
import polars as pl  # noqa: E402
from e00307_implied import direction_sector  # noqa: E402

PRIOR_259: Final[Path] = (
    common.ML_EXPERIMENTS_DATA_DIR / "00259-wind-plant-plus-quantile-offset-head" / "work"
)
SHIFT_HOURS: Final[tuple[int, ...]] = (-3, -2, -1, 1, 2, 3)
MAX_LEAD_DAY: Final[int] = 9
SPREAD_RANGE: Final[tuple[float, float]] = (0.5, 3.0)
ARMS: Final[tuple[str, ...]] = (
    "raw",
    "a_mean",
    "e_lead",
    "d_direction",
    "b_spread",
    "a_mean_era5",
    "e_lead_era5",
    *(f"c_shift_{h:+d}h" for h in SHIFT_HOURS),
    "all_edb",
)


def with_direction(*, name: str) -> pl.DataFrame:
    """Return a set's cases with the ensemble-mean direction sector and the implied wind."""
    cases = pl.read_parquet(common.WORK_DIR / f"cases_{name}.parquet")
    summary = pl.read_parquet(PRIOR_259 / "base" / f"summaries_{name}.parquet").select(
        *common.KEY_COLUMNS, "dir_sin", "dir_cos"
    )
    joined = cases.join(summary, on=list(common.KEY_COLUMNS), how="left", validate="1:1")
    sector = direction_sector(sin=joined["dir_sin"].to_numpy(), cos=joined["dir_cos"].to_numpy())
    return joined.with_columns(
        ens_sector=pl.Series(sector),
        lead_day_capped=pl.col("lead_day").clip(upper_bound=MAX_LEAD_DAY),
    )


def fit_factor(*, fit: pl.DataFrame, by: str | None, target: str = "implied_100m") -> dict:
    """Return mean(`target`) over mean(ensemble mean) overall or per level of `by`."""
    if by is None:
        return {
            "all": float(fit[target].to_numpy().mean() / fit["ens_speed_100m"].to_numpy().mean())
        }
    grouped = fit.group_by(by).agg(
        implied=pl.col(target).mean(), ens=pl.col("ens_speed_100m").mean()
    )
    return {
        int(row[by]): float(row["implied"] / row["ens"]) for row in grouped.iter_rows(named=True)
    }


def fit_corrections(*, hindcast: pl.DataFrame) -> dict:
    """Return the fitted factors of every arm from the hindcast's invertible, unmasked rows."""
    fit = hindcast.filter(pl.col("implied_100m").is_not_null() & ~pl.col("masked"))
    out: dict = {
        "a_mean": fit_factor(fit=fit, by=None),
        "e_lead": fit_factor(fit=fit, by="lead_day_capped"),
        "d_direction": fit_factor(fit=fit, by="ens_sector"),
    }
    everything = hindcast.filter(~pl.col("masked") & pl.col("era5_speed_100m").is_not_null())
    out["a_mean_era5"] = fit_factor(fit=everything, by=None, target="era5_speed_100m")
    out["e_lead_era5"] = fit_factor(fit=everything, by="lead_day_capped", target="era5_speed_100m")
    # Sequential: e, then d on the residual ratio, then b on what is left.
    lead_factor = fit["lead_day_capped"].replace_strict(
        {int(k): v for k, v in out["e_lead"].items()}, default=1.0, return_dtype=pl.Float64
    )
    after_e = fit.with_columns(ens_speed_100m=fit["ens_speed_100m"] * lead_factor)
    out["all_d_after_e"] = fit_factor(fit=after_e, by="ens_sector")
    sector_factor = after_e["ens_sector"].replace_strict(
        {int(k): v for k, v in out["all_d_after_e"].items()}, default=1.0, return_dtype=pl.Float64
    )
    mean_after = after_e["ens_speed_100m"] * sector_factor
    spread_fit = fit.with_columns(
        residual=pl.col("implied_100m") - mean_after,
        member_sd=((pl.col("ens_speed_100m_p90") - pl.col("ens_speed_100m_p10")) / 2.5631),
    )
    out["b_spread"] = spread_gain(
        fit=fit.with_columns(
            residual=pl.col("implied_100m") - pl.col("ens_speed_100m"),
            member_sd=spread_fit["member_sd"],
        )
    )
    out["all_b_spread"] = spread_gain(fit=spread_fit)
    return out


def spread_gain(*, fit: pl.DataFrame) -> dict:
    """Return the per-lead-day factor taking the member spread to the spread of the residual."""
    grouped = fit.group_by("lead_day_capped").agg(
        residual_sd=pl.col("residual").std(), member_sd=pl.col("member_sd").mean()
    )
    return {
        int(row["lead_day_capped"]): float(
            np.clip(row["residual_sd"] / row["member_sd"], *SPREAD_RANGE)
        )
        for row in grouped.iter_rows(named=True)
    }


def lookup(*, values: np.ndarray, table: dict, default: float) -> np.ndarray:
    """Return `table[value]` per element of `values`, `default` for unseen keys."""
    return np.array([table.get(int(v), default) for v in values], dtype=np.float32)


def corrected_members(
    *,
    arrays: dict[str, np.ndarray],
    cases: pl.DataFrame,
    arm: str,
    fitted: dict,
    shift_index: dict[int, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    """Return the (cases, 51) 10 m and 100 m speeds of one arm."""
    s10, s100 = arrays["s10"], arrays["s100"]
    if arm == "raw":
        return s10, s100
    if arm.startswith("c_shift_"):
        index = shift_index[int(arm.rsplit("_", 1)[-1].removesuffix("h"))]
        return s10[index], s100[index]
    lead = cases["lead_day_capped"].to_numpy()
    sector = cases["ens_sector"].to_numpy()
    factor = np.ones(len(cases), dtype=np.float32)
    gain = np.ones(len(cases), dtype=np.float32)
    if arm == "a_mean_era5":
        factor = np.full(len(cases), fitted["a_mean_era5"]["all"], dtype=np.float32)
    if arm == "e_lead_era5":
        factor = lookup(values=lead, table=fitted["e_lead_era5"], default=1.0)
    if arm == "a_mean":
        factor = np.full(len(cases), fitted["a_mean"]["all"], dtype=np.float32)
    if arm == "e_lead":
        factor = lookup(values=lead, table=fitted["e_lead"], default=1.0)
    if arm == "d_direction":
        factor = lookup(values=sector, table=fitted["d_direction"], default=1.0)
    if arm == "b_spread":
        gain = lookup(values=lead, table=fitted["b_spread"], default=1.0)
    if arm == "all_edb":
        factor = lookup(values=lead, table=fitted["e_lead"], default=1.0) * lookup(
            values=sector, table=fitted["all_d_after_e"], default=1.0
        )
        gain = lookup(values=lead, table=fitted["all_b_spread"], default=1.0)
    mean10, mean100 = s10.mean(axis=1, keepdims=True), s100.mean(axis=1, keepdims=True)
    new10 = factor[:, None] * (mean10 + gain[:, None] * (s10 - mean10))
    new100 = factor[:, None] * (mean100 + gain[:, None] * (s100 - mean100))
    return np.maximum(new10, 0.0), np.maximum(new100, 0.0)


def shifted_indices(*, cases: pl.DataFrame) -> dict[int, np.ndarray]:
    """Return, per shift in hours, the row of the case read instead (itself where none exists)."""
    keyed = cases.select(*common.KEY_COLUMNS).with_row_index("row")
    out = {}
    for hours in SHIFT_HOURS:
        moved = keyed.with_columns(valid_time=pl.col("valid_time") + pl.duration(hours=hours))
        joined = moved.join(
            keyed.rename({"row": "other"}),
            on=list(common.KEY_COLUMNS),
            how="left",
            validate="1:1",
        ).sort("row")
        out[hours] = joined["other"].fill_null(joined["row"]).to_numpy().astype(np.int64)
    return out


def main() -> None:
    """Fit each correction on the hindcast and score it on the scored year."""
    sites = common.site_table()
    hindcast = with_direction(name="hindcast")
    scored = with_direction(name="scored")
    fitted = fit_corrections(hindcast=hindcast)
    (common.WORK_DIR / "corrections.json").write_text(json.dumps(fitted, indent=2))
    print(json.dumps(fitted, indent=1))
    with (common.WORK_DIR / "cores.pkl").open("rb") as handle:
        stored = pickle.load(handle)
    members_all = common.ens_member_rows(sites=sites)
    pieces = []
    for row in sites.iter_rows(named=True):
        series, farm = int(row["time_series_id"]), row["site"]
        entry = stored[farm]
        core = core_module.WindCore(
            with_idle_layer=entry["with_idle_layer"], idle_floor=entry["idle_floor"]
        )
        core.load_state_dict(entry["state"])
        core.eval()
        residuals = entry["residuals"]
        farm_cases = scored.filter(pl.col("time_series_id") == series)
        members = (
            members_all.filter(pl.col("time_series_id") == series)
            .join(farm_cases.select(*common.KEY_COLUMNS), on=list(common.KEY_COLUMNS), how="semi")
            .sort(*common.KEY_COLUMNS, "ensemble_member")
        )
        keys = members.gather_every(common.MEMBERS).select(common.KEY_COLUMNS)
        cases = keys.join(farm_cases, on=list(common.KEY_COLUMNS), how="left", validate="1:1")
        arrays = {
            "s10": members["wind_speed_10m"].to_numpy().reshape(-1, common.MEMBERS),
            "s100": members["wind_speed_100m"].to_numpy().reshape(-1, common.MEMBERS),
            "temp": members["temperature_c"].to_numpy().reshape(-1, common.MEMBERS),
        }
        shift_index = shifted_indices(cases=cases)
        table = cases.select(
            *common.KEY_COLUMNS, "site", "lead_day", "lead_hours", "actual_norm", "masked"
        )
        actual = cases["actual_norm"].to_numpy().astype(np.float64)
        for arm in ARMS:
            s10, s100 = corrected_members(
                arrays=arrays, cases=cases, arm=arm, fitted=fitted, shift_index=shift_index
            )
            temp = (
                arrays["temp"][shift_index[int(arm.split("_")[-1].removesuffix("h"))]]
                if arm.startswith("c_shift_")
                else arrays["temp"]
            )
            expected = core_module.predict_expected_power(
                core=core,
                weather=core_module.WeatherArrays(
                    speed_10m=s10.reshape(-1).astype(np.float32),
                    speed_100m=s100.reshape(-1).astype(np.float32),
                    temperature_c=temp.reshape(-1).astype(np.float32),
                ),
            ).reshape(-1, common.MEMBERS)
            member_quantiles = np.stack(
                [residuals.quantiles(expected=expected[:, m]) for m in range(common.MEMBERS)],
                axis=1,
            )
            mixture = common.mixture_quantiles(member_quantiles=member_quantiles)
            score = common.quantile_score(quantiles=mixture, actual=actual)
            table = table.with_columns(
                pl.Series(f"q50_{arm}", mixture[:, common.MEDIAN_INDEX]),
                pl.Series(f"score_{arm}", score.astype(np.float32)),
            )
            print(f"{farm} {arm}: done", flush=True)
        pieces.append(table)
    pl.concat(pieces).write_parquet(common.WORK_DIR / "arms_scored.parquet")


if __name__ == "__main__":
    main()
