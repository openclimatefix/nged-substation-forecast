"""Score the arms of experiment 216 and the stored network, and write `work/score.json`.

Rows are every half-hour of experiment 155's daily runs outside switching events, at leads 0.5 to
168 hours. Arms: `stored` (experiment 155's arm D as stored), `r0` (arm D rebuilt with this
folder's code, the reference of the contrasts), `f1_<ratio>` and `f2_<ratio>` (the tail columns),
and `n` (experiment 201's arm 2: arm D plus the network's correction, scored per seed against the
same rows). Contrasts are paired differences in the per-row quantile score, resampled by calendar
month of the run's origin (20,000 resamples) at the 95% and the 97.5% levels.

    uv run --no-sync python ml-experiments/00216-*/e00216_score.py [--ratio 1]
"""

import argparse
import json
from typing import Final

import e00216_common as common
import e00216_ens as ens
import numpy as np
import polars as pl
from studies.bootstrap import bootstrap_difference_at_level
from studies.dp_features import FeatureCache  # noqa: F401
from studies.sources import ML_EXPERIMENTS_DATA_DIR
from studies.tail_calibration import tail_calibration_table

QUANTILES: Final[tuple[str, ...]] = common.QUANTILES
N_RESAMPLES: Final[int] = 20_000
LEVELS: Final[tuple[float, ...]] = (95.0, 97.5)
E201_PREDS: Final = ML_EXPERIMENTS_DATA_DIR / "00201-dp-gam-residual-mlp" / "work" / "preds"
N_SEEDS: Final[tuple[int, ...]] = (0, 1, 2)
KEYS: Final[tuple[str, ...]] = ("site", "power_fcst_init_time", "valid_time")


def mean_of(series: pl.Series) -> float:
    """Return a series' mean as a float."""
    return float(series.to_numpy().mean())


def read_arm(*, arm: str, period: str) -> pl.DataFrame:
    """Return an arm's rows of one period at the six substations, with a `site` column."""
    frames = []
    for site in common.SCORED_SITES:
        path = (
            common.E155_DP_PREDS / f"d_{period}_{site}.parquet"
            if arm == "stored"
            else common.WORK_DIR / "dp_preds" / f"{arm}_{period}_{site}.parquet"
        )
        frames.append(pl.read_parquet(path).with_columns(site=pl.lit(site)))
    frame = pl.concat(frames).filter(
        ~pl.col("switching_mask") & pl.col("lead_hours").is_between(0.5, 168.0)
    )
    return frame.with_columns(arm=pl.lit(arm))


def read_network(*, rows: pl.DataFrame) -> dict[int, pl.DataFrame]:
    """Return experiment 201's arm 2 per seed, on the keys of `rows` (the scored year)."""
    keys = rows.select(*KEYS).unique()
    out = {}
    for seed in N_SEEDS:
        frame = pl.read_parquet(E201_PREDS / f"arm2_seed{seed}.parquet").select(
            "site",
            "power_fcst_init_time",
            "valid_time",
            "lead_hours",
            "actual",
            *QUANTILES,
            "delta",
        )
        out[seed] = frame.join(keys, on=list(KEYS), how="inner")
    return out


def add_losses(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Add `qs` (quantile score), `abs_mean` and `abs_median` (errors of the mean and median)."""
    actual = frame["actual"].to_numpy().astype(np.float64)
    quantiles = frame.select(QUANTILES).to_numpy().astype(np.float64)
    columns = {"qs": ens.quantile_score(actual=actual, quantiles=quantiles)}
    columns["abs_median"] = np.abs(actual - frame["q50"].to_numpy().astype(np.float64))
    if "mean" in frame.columns:
        columns["abs_mean"] = np.abs(actual - frame["mean"].to_numpy().astype(np.float64))
    return frame.with_columns(**{name: pl.Series(name, values) for name, values in columns.items()})


def bootstrap_frame(*, frames: dict[str, pl.DataFrame]) -> pl.DataFrame:
    """Return the per-row losses of arms in the layout `bootstrap_difference_at_level` reads."""
    pieces = []
    for arm, frame in frames.items():
        seed_col = frame["seed"] if "seed" in frame.columns else pl.Series([0] * frame.height)
        pieces.append(
            frame.select(
                "site",
                time=(
                    pl.col("power_fcst_init_time").dt.epoch("s") * 1000
                    + (pl.col("lead_hours") * 2).round().cast(pl.Int64)
                ),
                month=pl.col("power_fcst_init_time").dt.strftime("%Y-%m"),
                qs=pl.col("qs"),
                abs_mean=pl.col("abs_mean")
                if "abs_mean" in frame.columns
                else pl.col("abs_median"),
                lead_hours=pl.col("lead_hours"),
            ).with_columns(arm=pl.lit(arm), seed=seed_col.cast(pl.Int64))
        )
    return pl.concat(pieces)


def contrast(
    *, losses: pl.DataFrame, treatment: str, reference: str, metric: str = "qs"
) -> dict[str, float]:
    """Return the mean paired difference and its 95% and 97.5% intervals."""
    paired = losses.filter(pl.col("arm").is_in([treatment, reference]))
    ref = paired.filter(pl.col("arm") == reference).select("site", "time", "seed", ref=metric)
    tre = paired.filter(pl.col("arm") == treatment).select("site", "time", "seed", tre=metric)
    mean = float(
        ref.join(tre, on=["site", "time", "seed"])
        .select((pl.col("tre") - pl.col("ref")).mean())
        .item()
    )
    result = {"difference": mean}
    for level in LEVELS:
        lower, upper = bootstrap_difference_at_level(
            losses=paired, treatment=treatment, reference=reference, metric=metric,
            level=level, n_resamples=N_RESAMPLES,
        )  # fmt: skip
        result[f"lower_{level:g}"], result[f"upper_{level:g}"] = lower, upper
    return result


def score_period(*, period: str, ratio: str, with_network: bool) -> dict:
    """Score every arm of one period and return the nested results."""
    names = ("stored", "r0", f"f1_{ratio}", f"f2_{ratio}")
    frames = {name: add_losses(frame=read_arm(arm=name, period=period)) for name in names}
    decision = {
        name: frame.filter(pl.col("site").is_in(common.DECISION_SITES))
        for name, frame in frames.items()
    }
    out: dict = {
        "absolute": {},
        "contrasts": {},
        "by_lead": {},
        "by_site": {},
        "tail": {},
        "checks": {},
    }
    for name, frame in decision.items():
        out["absolute"][name] = {
            "qs": mean_of(frame["qs"]),
            "abs_mean": mean_of(frame["abs_mean"]),
            "abs_median": mean_of(frame["abs_median"]),
            "rows": frame.height,
        }
        table = tail_calibration_table(frame=frame)
        out["tail"][name] = table.to_dicts()
    for name, frame in frames.items():
        out["by_site"][name] = {
            site: mean_of(frame.filter(pl.col("site") == site)["qs"])
            for site in common.SCORED_SITES
        }
        out["checks"][name] = {
            site: mean_of(
                frame.filter(pl.col("site") == site)["q50"]
                - frame.filter(pl.col("site") == site)["mean"]
            )
            for site in common.SCORED_SITES
        }
    losses = bootstrap_frame(frames=decision)
    arms_f = (f"f1_{ratio}", f"f2_{ratio}")
    out["contrasts"]["r0_vs_stored"] = contrast(losses=losses, treatment="r0", reference="stored")
    for arm in arms_f:
        out["contrasts"][f"{arm}_vs_r0"] = contrast(losses=losses, treatment=arm, reference="r0")
        out["contrasts"][f"{arm}_vs_r0_mean_error"] = contrast(
            losses=losses, treatment=arm, reference="r0", metric="abs_mean"
        )
        out["contrasts"][f"{arm}_vs_stored"] = contrast(
            losses=losses, treatment=arm, reference="stored"
        )
        for band, condition in (
            ("under_6h", pl.col("lead_hours") < 6.0),
            ("from_6h", pl.col("lead_hours") >= 6.0),
        ):
            out["by_lead"][f"{arm}_{band}"] = contrast(
                losses=losses.filter(condition), treatment=arm, reference="r0"
            )
        out["by_site"][f"{arm}_minus_r0"] = {
            site: mean_of(frames[arm].filter(pl.col("site") == site)["qs"])
            - mean_of(frames["r0"].filter(pl.col("site") == site)["qs"])
            for site in common.SCORED_SITES
        }
        out["by_site"][f"{arm}_vs_r0_leave_one_out"] = {
            site: contrast(
                losses=losses.filter(pl.col("site") != site), treatment=arm, reference="r0"
            )["difference"]
            for site in common.DECISION_SITES
        }
    if with_network:
        network = read_network(rows=frames["stored"])
        scored = {
            f"n_seed{seed}": add_losses(frame=frame.with_columns(seed=pl.lit(0, dtype=pl.Int64)))
            for seed, frame in network.items()
        }
        gains = {}
        for name, frame in scored.items():
            both = frame.filter(pl.col("site").is_in(common.DECISION_SITES))
            stored_rows = decision["stored"].join(both.select(*KEYS), on=list(KEYS), how="inner")
            sub = bootstrap_frame(frames={"stored": stored_rows, name: both})
            gains[name] = contrast(losses=sub, treatment=name, reference="stored")
            gains[name]["absolute_n"] = mean_of(both["qs"])
            gains[name]["absolute_stored_same_rows"] = mean_of(stored_rows["qs"])
        out["network"] = gains
    return out


def main() -> None:
    """Score the scored year (with the network) and the replicate year, and write the file."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--ratio", default=json.loads((common.WORK_DIR / "tail_ratio.json").read_text())["chosen"]
    )
    args = parser.parse_args()
    result = {
        "ratio": args.ratio,
        "scored": score_period(period="scored", ratio=args.ratio, with_network=True),
        "replicate": score_period(period="replicate", ratio=args.ratio, with_network=False),
        "tuning": score_period(period="tuning", ratio=args.ratio, with_network=False),
    }
    common.write_json_atomic(path=common.WORK_DIR / "score.json", content=result)
    for period in ("scored", "replicate"):
        print(period, json.dumps(result[period]["absolute"], indent=1))
        print(json.dumps(result[period]["contrasts"], indent=1))
    print(json.dumps(result["scored"].get("network"), indent=1))


if __name__ == "__main__":
    main()
