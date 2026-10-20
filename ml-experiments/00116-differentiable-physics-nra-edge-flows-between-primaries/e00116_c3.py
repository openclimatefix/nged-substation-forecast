"""C3 of experiment 116: do causal flows help the metered forecast during and after a transfer?

The rows are the forecasts of the arms in `e00116_run.py` for every injected transfer of set `C3`:
the rows whose valid time is inside the transfer or in the 48 hours after its end, at leads of 0.5
to 48 hours, at both ends of the transfer. The contrast is A5 (A4's fit with causal flows) minus
A4 (the plain filter), in quantile score, with a 98.33% interval from resampling calendar months
within each seed. Beside it: A5 minus A0, the oracle (the ceiling), the share of the A4-to-oracle
gap that A5 closes, the sites that improve, the effect on real rows outside the events, and the
tails.

    uv run python ml-experiments/00116-*/e00116_c3.py
"""

from typing import Any, Final

import e00116_common as common
import e00116_inject as inject
import e00116_run as run
import e00116_score as score
import numpy as np
import polars as pl
from studies.dp_features import FeatureCache

RESULT_PATH: Final = common.WORK_DIR / "c3.json"
TAIL_COLUMNS: Final[tuple[str, ...]] = ("q01", "q05", "q95", "q99")
TAIL_LEVELS: Final[dict[str, float]] = {"q01": 0.01, "q05": 0.05, "q95": 0.95, "q99": 0.99}


def read_rows(*, year_name: str) -> pl.DataFrame:
    """Read every per-transfer file of set C3 in a year."""
    folder = run.transfers_dir(set_name="C3", year_name=year_name)
    return pl.concat([pl.read_parquet(path) for path in sorted(folder.glob("*.parquet"))])


def tails(*, rows: pl.DataFrame) -> list[dict[str, Any]]:
    """Return coverage and pinball loss at the four tail quantiles for each arm."""
    out = []
    for arm in ("a0", "a4", "a5", "or"):
        part = rows.filter(pl.col("arm") == arm)
        entry: dict[str, Any] = {"arm": arm}
        for column, level in TAIL_LEVELS.items():
            error = part["actual"] - part[column]
            entry[f"coverage_{column}"] = float(
                np.mean(part["actual"].to_numpy() <= part[column].to_numpy())
            )
            entry[f"pinball_{column}"] = float(
                np.mean(np.maximum(level * error.to_numpy(), (level - 1.0) * error.to_numpy()))
            )
        out.append(entry)
    return out


def receiver_exceedance(*, rows: pl.DataFrame) -> dict[str, float]:
    """Return how often the receiver's power exceeds its q99, inside and after the transfer."""
    out = {}
    for arm in ("a4", "a5"):
        part = rows.filter(pl.col("arm") == arm, pl.col("role") == "receiver")
        for label, scope in (("inside", part["in_event"]), ("after", ~part["in_event"])):
            chosen = part.filter(scope)
            out[f"{arm}_{label}"] = float(
                np.mean(chosen["actual"].to_numpy() > chosen["q99"].to_numpy())
            )
    return out


def sites_improving(*, rows: pl.DataFrame) -> dict[str, float]:
    """Return the mean A5-minus-A4 score at each gate site."""
    keys = ["transfer_id", "node", "cutoff", "target"]
    paired = (
        rows.filter(pl.col("arm") == "a4")
        .select(*keys, a4=pl.col("score"))
        .join(rows.filter(pl.col("arm") == "a5").select(*keys, a5=pl.col("score")), on=keys)
    )
    by_site = paired.group_by("node").agg((pl.col("a5") - pl.col("a4")).mean().alias("difference"))
    return dict(zip(by_site["node"].to_list(), by_site["difference"].to_list(), strict=True))


def in_event_share(*, cache: FeatureCache, year_name: str) -> float:
    """Return the share of the gate sites' rows that lie inside the hand-filtered events."""
    year = common.year_slice(cache=cache, cut=common.CUTS[year_name])
    inside = total = 0
    for node in common.GATE_SITES:
        features = cache.sites[node]
        present = features.present[year] & np.isfinite(features.y[year])
        inside += int(np.sum(present & features.switching_mask[year]))
        total += int(np.sum(present))
    return inside / total


def real_row_effect(*, cache: FeatureCache, year_name: str) -> dict[str, float] | None:
    """Return A5 minus A4 on real rows outside the events, from the own-year runs, if they exist."""
    try:
        a4 = score.read_replica(year_name=year_name, arm="a4", own_year=True)
        a5 = score.read_replica(year_name=year_name, arm="a5", own_year=True)
    except ValueError:
        return None
    spreads = score.fit_spreads(cache=cache, year_name=year_name)
    losses = {}
    for arm, run_frame in (("a4", a4), ("a5", a5)):
        frame = run_frame.filter(~pl.col("switching"))
        target = frame["target"].to_numpy()
        mean = (frame["gam"] + frame["correction"]).to_numpy()
        quantiles = np.empty((frame.height, 11))
        for node in frame["node"].unique().to_list():
            chosen = (frame["node"] == node).to_numpy()
            quantiles[chosen] = spreads[(arm, node)].quantiles(
                mean=mean[chosen],
                band=cache.calendar.band[target[chosen]],
                quarter=cache.calendar.quarter[target[chosen]],
                lead_hours=frame["lead_hours"].to_numpy()[chosen],
            )
        losses[arm] = frame.with_columns(
            pl.Series(
                "score",
                score.quantile_score(
                    actual=frame["actual"].to_numpy(), quantiles=np.sort(quantiles, axis=1)
                ),
            )
        ).select("node", "cutoff", "target", "score")
    paired = losses["a4"].join(losses["a5"], on=["node", "cutoff", "target"], suffix="_a5")
    difference = (paired["score_a5"] - paired["score"]).to_numpy()
    return {"difference": float(difference.mean()), "n_rows": len(difference)}


def analyse(*, cache: FeatureCache, year_name: str) -> dict[str, Any]:
    """Compute C3's contrasts and its companions for one year."""
    transfers = inject.draw_set(set_name="C3", year_name=year_name, cache=cache)
    spreads = score.fit_spreads(cache=cache, year_name=year_name)
    rows = score.attach_seeds(
        rows=score.with_quantiles(
            cache=cache, rows=read_rows(year_name=year_name), spreads=spreads
        ),
        transfers=transfers,
    )
    rows.write_parquet(common.WORK_DIR / f"c3_rows_{year_name}.parquet")
    contrasts = {
        "a5_minus_a4": score.paired_contrast(
            rows=rows, treatment="a5", reference="a4", metric="score"
        ),
        "a5_minus_a0": score.paired_contrast(
            rows=rows, treatment="a5", reference="a0", metric="score"
        ),
        "or_minus_a4": score.paired_contrast(
            rows=rows, treatment="or", reference="a4", metric="score"
        ),
        "a5_minus_a4_mean_error": score.paired_contrast(
            rows=rows, treatment="a5", reference="a4", metric="abs_error"
        ),
    }
    gain = -contrasts["or_minus_a4"]["estimate"]
    gap_closed = -contrasts["a5_minus_a4"]["estimate"] / gain if gain > 0 else float("nan")
    first_day = rows.filter(pl.col("in_event"))
    winter = {}
    for scope in ("winter", "cold_snap", "evening_peak"):
        chosen = rows.filter(pl.col(scope))
        n_transfers = chosen["transfer_id"].n_unique() if chosen.height else 0
        winter[scope] = {
            "too_few": n_transfers < common.MIN_WINTER_TRANSFERS,
            "a5_minus_a4": score.paired_contrast(
                rows=chosen, treatment="a5", reference="a4", metric="score"
            )
            if n_transfers
            else None,
        }
    return {
        "year": year_name,
        "contrasts": contrasts,
        "gap_closed_share": gap_closed,
        "oracle_gain": gain,
        "oracle_gain_scaled_by_in_event_share": gain
        * in_event_share(cache=cache, year_name=year_name),
        "sites": sites_improving(rows=rows),
        "absolute": {
            arm: score.column_mean(frame=rows.filter(pl.col("arm") == arm), column="score")
            for arm in ("a0", "a4", "a5", "or")
        },
        "tails": tails(rows=rows),
        "receiver_q99_exceedance": receiver_exceedance(rows=rows),
        "winter_scopes": winter,
        "in_event_only": {
            "a5_minus_a4": score.paired_contrast(
                rows=first_day, treatment="a5", reference="a4", metric="score"
            )
        },
        "real_rows_outside_events": real_row_effect(cache=cache, year_name=year_name),
    }


def main() -> None:
    """Analyse the scored and the replicate year and write `c3.json`."""
    cache = common.load_cache()
    results = {year: analyse(cache=cache, year_name=year) for year in ("scored", "replicate")}
    score.write_json(path=RESULT_PATH, payload=results)
    print(results["scored"]["contrasts"]["a5_minus_a4"])


if __name__ == "__main__":
    main()
