"""Score experiment 119's arms: the planned contrasts, the winter and robustness rows, and tails.

Reads the final fits of `e00119_final.py` and writes `results_<direction>.json`, a markdown report
for the page, and the arms' seed-mean forecasts as named prediction sets. Every interval is a 99%
paired interval (a Bonferroni correction over the five planned contrasts) that resamples whole
calendar months, or 7-day blocks for the hard-day subsets.

    uv run python ml-experiments/00119-*/e00119_score.py [--directions F1 F2]
"""

import argparse
from dataclasses import dataclass

import e00119_bundle as bundles
import e00119_common as common
import e00119_cv as cv
import e00119_data as data
import e00119_final as final
import e00119_stats as stats
import numpy as np
import polars as pl
from studies.dp_calendar import bank_holiday_names
from studies.experiment_store import write_predictions
from studies.tail_calibration import tail_calibration_table

ARMS = ("ENC-c", "ENC-c0", "ENC-cps", "ENC-cpt", "ENC-r", "ENC-r0")
LEAD_BANDS = {
    "0.5-168 h": (0.0, 168.1),
    "D+1 (15.5-39 h)": (15.5, 39.0),
    "0-24 h": (0.0, 24.0),
    "24-48 h": (24.0, 48.0),
    "48-144 h": (48.0, 144.0),
    "144-168 h": (144.0, 168.1),
}
CONTRASTS = {
    "C1": ("ENC-c", "K0"),
    "C2": ("ENC-c", "ENC-c0"),
    "C3": ("ENC-c", "ENC-cpt"),
    "C3b": ("ENC-c", "ENC-cps"),
    "C4": ("ENC-r", "K0"),
}
EVENING_HOURS = (16, 20)


@dataclass
class Scored:
    """One direction's rows, K0, and every arm's seed-mean forecast on the scored year."""

    rows: pl.DataFrame
    quantiles: dict[str, np.ndarray]
    per_seed: dict[str, np.ndarray]
    score: dict[str, np.ndarray]


def load(*, direction: cv.Direction) -> Scored:
    """Load the scored-year rows and every arm's forecasts for a direction."""
    dataset = data.load_dataset()
    rows_all = bundles.rows_with_sensitivities(dataset=dataset)
    first = np.load(final.final_path(direction=direction.name, arm="ENC-c"))
    index = first["index"]
    rows = rows_all[index.tolist()]
    quantiles = {"K0": rows.select(common.QUANTILE_COLUMNS).to_numpy().astype(np.float64)}
    per_seed = {}
    for arm in ARMS:
        stored = np.load(final.final_path(direction=direction.name, arm=arm))
        if not np.array_equal(stored["index"], index):
            msg = f"{arm} covers different rows"
            raise ValueError(msg)
        per_seed[arm] = stored["quantiles"].astype(np.float64)
        quantiles[arm] = np.sort(per_seed[arm].mean(axis=0), axis=1)
    actual = rows["actual"].to_numpy().astype(np.float64)
    score = {k: stats.quantile_score(actual=actual, quantiles=v) for k, v in quantiles.items()}
    return Scored(rows=rows, quantiles=quantiles, per_seed=per_seed, score=score)


def scopes(*, rows: pl.DataFrame) -> dict[str, np.ndarray]:
    """Return the row masks of every reported scope on the decision sites."""
    site = rows["site_index"].to_numpy()
    decision = np.isin(site, [common.SCORED_SITES.index(s) for s in common.DECISION_SITES])
    lead = rows["lead_hours"].to_numpy()
    out = {name: decision & (lead >= lo) & (lead < hi) for name, (lo, hi) in LEAD_BANDS.items()}
    winter = stats.is_winter(run_date=rows["run_date"])
    hour = rows["valid_time"].dt.hour().to_numpy()
    evening = (hour >= EVENING_HOURS[0]) & (hour < EVENING_HOURS[1])
    holidays = set(bank_holiday_names())
    day = rows["valid_time"].dt.date().to_list()
    off_day = stats.weekend_or_holiday(valid_time=rows["valid_time"]) | np.array(
        [d in holidays for d in day]
    )
    out["winter (Nov-Mar)"] = decision & winter
    out["summer (Apr-Oct)"] = decision & ~winter
    out["winter evening peak (16-20 UTC)"] = decision & winter & evening
    out["working days"] = decision & ~off_day
    out["weekends and bank holidays"] = decision & off_day
    return out


def contrast(
    *, scored: Scored, treatment: str, reference: str, mask: np.ndarray, metric: str = "score"
) -> dict:
    """Return a month-resampled paired difference of two arms over some rows."""
    rows = scored.rows
    actual = rows["actual"].to_numpy().astype(np.float64)
    if metric == "score":
        a, b = scored.score[treatment], scored.score[reference]
    else:
        a = np.abs(actual - scored.quantiles[treatment][:, 5])
        b = np.abs(actual - scored.quantiles[reference][:, 5])
    return stats.month_interval(
        treatment=a[mask],
        reference=b[mask],
        site=rows["site_index"].to_numpy()[mask],
        run=rows["run"].to_numpy()[mask],
        month=stats.month_of_run(run_date=rows["run_date"])[mask],
    )


def per_site(*, scored: Scored, treatment: str, reference: str, mask: np.ndarray) -> dict:
    """Return each decision site's mean paired difference, and how many improve."""
    site = scored.rows["site_index"].to_numpy()
    out = {}
    for name in (*common.DECISION_SITES, "D9"):
        at = mask & (site == common.SCORED_SITES.index(name)) if name != "D9" else None
        if name == "D9":
            at = (site == common.SCORED_SITES.index("D9")) & np.isfinite(scored.score["K0"])
            lead = scored.rows["lead_hours"].to_numpy()
            at &= lead < LEAD_BANDS["0.5-168 h"][1]
        out[name] = float((scored.score[treatment][at] - scored.score[reference][at]).mean())
    return out


def hard_days(*, scored: Scored, treatment: str, reference: str) -> dict:
    """Return the contrast on each hard-day subset with a 7-day block bootstrap interval."""
    masks = stats.hard_day_masks(rows=scored.rows)
    site = scored.rows["site_index"].to_numpy()
    decision = np.isin(site, [common.SCORED_SITES.index(s) for s in common.DECISION_SITES])
    out = {}
    for name, subset in masks.items():
        mask = subset & decision
        if mask.sum() == 0:
            out[name] = {"n_rows": 0}
            continue
        out[name] = stats.block_interval(
            treatment=scored.score[treatment][mask],
            reference=scored.score[reference][mask],
            run=scored.rows["run"].to_numpy()[mask],
        )
    return out


def tails(*, scored: Scored, arm: str, mask: np.ndarray) -> list[dict]:
    """Return coverage and pinball loss at q01, q05, q95, q99 over some rows."""
    frame = (
        scored.rows.filter(pl.Series(mask))
        .select("power_fcst_init_time", "actual")
        .with_columns(
            **{
                c: pl.Series(scored.quantiles[arm][mask][:, i])
                for i, c in enumerate(common.QUANTILE_COLUMNS)
            }
        )
    )
    return tail_calibration_table(frame=frame).to_dicts()


def score_direction(*, direction: cv.Direction) -> dict:
    """Compute every reported number for a direction."""
    scored = load(direction=direction)
    masks = scopes(rows=scored.rows)
    full = masks["0.5-168 h"]
    arms = ("K0", *ARMS)
    result: dict = {"direction": direction.name, "contrasts": {}, "absolute": {}, "tails": {}}
    for name, (a, b) in CONTRASTS.items():
        result["contrasts"][name] = {
            "pair": [a, b],
            "by_scope": {
                s: contrast(scored=scored, treatment=a, reference=b, mask=m)
                for s, m in masks.items()
            },
            "mae_median": contrast(
                scored=scored, treatment=a, reference=b, mask=full, metric="mae"
            ),
            "per_site": per_site(scored=scored, treatment=a, reference=b, mask=full),
            "hard_days": hard_days(scored=scored, treatment=a, reference=b),
        }
    result["contrasts"]["C5b"] = {
        "pair": ["ENC-c", "ENC-cpt"],
        "hard_days": hard_days(scored=scored, treatment="ENC-c", reference="ENC-cpt"),
    }
    site = scored.rows["site_index"].to_numpy()
    leave_out = {}
    for name in common.DECISION_SITES:
        keep = full & (site != common.SCORED_SITES.index(name))
        leave_out[name] = {
            c: contrast(scored=scored, treatment=a, reference=b, mask=keep)
            for c, (a, b) in CONTRASTS.items()
        }
    result["leave_one_site_out"] = leave_out
    for arm in arms:
        result["absolute"][arm] = {s: float(scored.score[arm][m].mean()) for s, m in masks.items()}
        result["tails"][arm] = {
            band: tails(scored=scored, arm=arm, mask=masks[band])
            for band in ("0.5-168 h", "D+1 (15.5-39 h)", "winter (Nov-Mar)")
        }
    mean_shift = scored.rows["mean"].to_numpy() * 0.0
    result["median_minus_mean"] = {
        arm: {
            name: float(
                (
                    scored.quantiles[arm][:, 5]
                    - (
                        scored.rows["mean"].to_numpy()
                        + (scored.quantiles[arm][:, 5] - scored.quantiles["K0"][:, 5])
                        + mean_shift
                    )
                )[full & (site == common.SCORED_SITES.index(name))].mean()
            )
            for name in common.DECISION_SITES
        }
        for arm in arms
    }
    seed_scores = {
        arm: [
            float(
                stats.quantile_score(
                    actual=scored.rows["actual"].to_numpy().astype(np.float64),
                    quantiles=np.sort(q, axis=1),
                )[full].mean()
            )
            for q in scored.per_seed[arm]
        ]
        for arm in ("ENC-c", "ENC-r")
    }
    result["per_seed_absolute"] = seed_scores
    result["n_rows"] = int(full.sum())
    result["n_runs"] = int(scored.rows["run"].n_unique())
    store_predictions(direction=direction, scored=scored)
    return result


def store_predictions(*, direction: cv.Direction, scored: Scored) -> None:
    """Write each arm's seed-mean forecasts as a named prediction set."""
    row_set = f"enc_{direction.name.lower()}_scored_v1"
    seed_sites = scored.rows["site_index"].to_numpy()
    series_id = scored.rows["time_series_id"].to_numpy()
    for arm in ("K0", *ARMS):
        frame = pl.DataFrame(
            {
                "time_series_id": series_id,
                "power_fcst_init_time": scored.rows["power_fcst_init_time"],
                "valid_time": scored.rows["valid_time"],
                "seed": np.zeros(len(series_id), dtype=np.int8),
                "lead_hours": scored.rows["lead_hours"],
                **{
                    c: scored.quantiles[arm][:, i].astype(np.float32)
                    for i, c in enumerate(common.QUANTILE_COLUMNS)
                },
                "actual": scored.rows["actual"],
                "arm": np.full(len(series_id), arm),
                "switching_mask": np.zeros(len(series_id), dtype=bool),
            }
        )
        name = f"{arm.lower().replace('-', '_')}_{direction.name.lower()}"
        try:
            write_predictions(
                experiment_id=common.EXPERIMENT_ID,
                name=name,
                frame=frame,
                kind="quantiles",
                row_set_id=row_set,
            )
        except FileExistsError:
            continue
    del seed_sites


def main() -> None:
    """Score the requested directions and write their results."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--directions", nargs="+", default=["F1", "F2"])
    args = parser.parse_args()
    for name in args.directions:
        result = score_direction(direction=cv.DIRECTIONS[name])
        common.write_json_atomic(path=common.WORK_DIR / f"results_{name}.json", content=result)
        c1 = result["contrasts"]["C1"]["by_scope"]["0.5-168 h"]
        print(
            f"{name} C1 {c1['difference']:+.4f} [{c1['lower']:+.4f}, {c1['upper']:+.4f}]",
            flush=True,
        )


if __name__ == "__main__":
    main()
