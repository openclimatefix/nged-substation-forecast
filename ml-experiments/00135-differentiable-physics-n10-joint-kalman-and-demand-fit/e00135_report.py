"""Score experiment 135's arms and write `work/results.json` and `work/report.md`.

The scored year decides the verdict; the replicate months are the check. Contrasts are read at the
two-sided 97.5% month-resampled interval (the Bonferroni level for the two planned contrasts, C1 =
J1 minus TS and C2 = J2 minus TS). J2 minus J1 is descriptive. Breakouts and tails are descriptive.

    uv run --no-sync python ml-experiments/00135-*/e00135_report.py
"""

import json
from typing import Any, Final

import e00135_common as common
import e00135_score as score
import numpy as np
import polars as pl

ARMS: Final[tuple[str, ...]] = ("ts", "j1", "j2")
CONTRASTS: Final[tuple[tuple[str, str, str], ...]] = (
    ("C1", "j1", "ts"),
    ("C2", "j2", "ts"),
    ("J2 minus J1 (descriptive)", "j2", "j1"),
)
WINTER_MONTHS: Final[frozenset[int]] = frozenset({11, 12, 1, 2, 3})
COLD_NIGHT_C: Final[float] = 3.0
COLD_NIGHT_BANDS: Final[frozenset[int]] = frozenset({0, 3})
TAIL_LEVELS: Final[tuple[str, ...]] = ("q01", "q05", "q95", "q99")
MIN_SITE_IMPROVEMENT: Final[float] = 0.005


def load_common(*, periods: set[str], sites: tuple[str, ...]) -> dict[str, pl.DataFrame]:
    """Return every arm's rows on the rows all arms hold."""
    return score.common_rows(
        arms_by_name={name: score.load_arm(arm=name, periods=periods, sites=sites) for name in ARMS}
    )


def annotate(*, frame: pl.DataFrame) -> pl.DataFrame:
    """Add the month, hour band, ERA5 temperature, and holiday flag of each row's valid time."""
    cache = common.load_cache()
    pieces = []
    for site in frame["site"].unique().to_list():
        part = frame.filter(pl.col("site") == site)
        position = np.searchsorted(
            cache.end_times, part["valid_time"].dt.replace_time_zone(None).to_numpy()
        )
        features = cache.sites[site]
        pieces.append(
            part.with_columns(
                valid_month=pl.col("valid_time").dt.month(),
                hour_band=pl.Series(cache.calendar.band[position].astype(np.int64)),
                temperature_c=pl.Series(features.temperature_c[position]),
                holiday=pl.Series(cache.calendar.holiday_period[position] >= 0),
            )
        )
    return pl.concat(pieces)


def breakouts(*, frame: pl.DataFrame) -> dict[str, pl.Expr]:
    """Return the descriptive subsets as filter expressions on an annotated frame."""
    winter = pl.col("valid_month").is_in(sorted(WINTER_MONTHS))
    return {
        "winter": winter,
        "winter nights": winter & (pl.col("hour_band") == 0),
        "cold nights": pl.col("hour_band").is_in(sorted(COLD_NIGHT_BANDS))
        & (pl.col("temperature_c") <= COLD_NIGHT_C),
        "Christmas and bank holidays": pl.col("holiday"),
    }


def contrasts_for(*, frames: dict[str, pl.DataFrame], metric: str = "qs") -> dict:
    """Return the three contrasts for each lead band on already-common frames."""
    out = {}
    for band in score.LEAD_BANDS:
        banded = {name: score.lead_filter(frame=f, band=band) for name, f in frames.items()}
        losses = score.stack_losses(arms_by_name=banded)
        out[band[0]] = {
            label: score.contrast(losses=losses, treatment=t, reference=r, metric=metric)
            for label, t, r in CONTRASTS
        }
    return out


def absolute_scores(*, frames: dict[str, pl.DataFrame]) -> dict:
    """Return each arm's mean quantile score and mean-forecast absolute error per lead band."""
    out = {}
    for band in score.LEAD_BANDS:
        out[band[0]] = {}
        for name, frame in frames.items():
            losses = score.with_losses(frame=score.lead_filter(frame=frame, band=band))
            out[band[0]][name] = {
                "qs": float(losses["qs"].to_numpy().mean()),
                "abs_median": float(losses["abs_median"].to_numpy().mean()),
                "abs_mean": float(losses["abs_mean"].to_numpy().mean()),
                "n": losses.height,
            }
    return out


def tail_table(*, frames: dict[str, pl.DataFrame]) -> list[dict]:
    """Return coverage and pinball loss at the four tail levels, 12-36 h, and cold-night q95."""
    rows = []
    for name, frame in frames.items():
        banded = score.lead_filter(frame=frame, band=score.LEAD_BANDS[0])
        annotated = annotate(frame=banded)
        cold = annotated.filter(breakouts(frame=annotated)["cold nights"])
        for level in TAIL_LEVELS:
            tau = float(level[1:]) / 100.0
            error = banded["actual"] - banded[level]
            pinball = np.maximum(tau * error.to_numpy(), (tau - 1.0) * error.to_numpy())
            rows.append(
                {
                    "arm": name,
                    "level": level,
                    "coverage": float((banded["actual"] <= banded[level]).to_numpy().mean()),
                    "pinball": float(pinball.mean()),
                    "cold_night_coverage": float((cold["actual"] <= cold[level]).to_numpy().mean()),
                    "n_cold": cold.height,
                }
            )
    return rows


def breakout_contrasts(*, frames: dict[str, pl.DataFrame]) -> dict:
    """Return the 12-36 h contrasts C1 and C2 inside each descriptive subset."""
    annotated = {
        name: annotate(frame=score.lead_filter(frame=f, band=score.LEAD_BANDS[0]))
        for name, f in frames.items()
    }
    out: dict[str, Any] = {}
    for label, expression in breakouts(frame=annotated["ts"]).items():
        subset = {name: f.filter(expression) for name, f in annotated.items()}
        losses = score.stack_losses(arms_by_name=subset)
        out[label] = {
            name: score.contrast(losses=losses, treatment=t, reference=r)
            for name, t, r in CONTRASTS[:2]
        }
        out[label]["n"] = subset["ts"].height
    return out


def median_minus_mean(*, frames: dict[str, pl.DataFrame]) -> dict:
    """Return the mean of the median forecast minus the mean forecast, per arm and site."""
    return {
        name: {
            site: float((part["q50"] - part["mean"]).to_numpy().mean())
            for site, part in frame.group_by("site")
        }
        for name, frame in frames.items()
    }


def main() -> None:
    """Score the scored year and the replicate months and write the results."""
    decision = common.DECISION_SITES
    results: dict = {"ridge_arms": {s: score.chosen_ridge_arm(site=s) for s in common.SCORED_SITES}}
    for period in ("scored", "replicate"):
        frames = load_common(periods={period}, sites=decision)
        results[period] = {
            "contrasts": contrasts_for(frames=frames),
            "absolute": absolute_scores(frames=frames),
            "mean_forecast_error_contrasts": contrasts_for(frames=frames, metric="abs_mean"),
            "median_forecast_error_contrasts": contrasts_for(frames=frames, metric="abs_median"),
        }
        if period == "scored":
            results[period]["breakouts"] = breakout_contrasts(frames=frames)
            results[period]["tails"] = tail_table(frames=frames)
            results[period]["median_minus_mean"] = median_minus_mean(frames=frames)
    d9 = load_common(periods={"scored"}, sites=("D9",))
    results["d9_scored"] = absolute_scores(frames=d9)
    common.write_json_atomic(path=common.WORK_DIR / "results.json", content=results)
    headline = results["scored"]["contrasts"]["12-36 h"]
    for label, value in headline.items():
        print(
            f"{label}: {value['estimate']:+.4f} [{value['lower']:+.4f}, {value['upper']:+.4f}], "
            f"sites better {value['sites_better']}/{value['n_sites']}"
        )
    print(json.dumps(results["scored"]["absolute"]["12-36 h"], indent=1))


if __name__ == "__main__":
    main()
