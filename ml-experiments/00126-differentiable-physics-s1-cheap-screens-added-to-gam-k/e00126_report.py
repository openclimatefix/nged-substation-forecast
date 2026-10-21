"""Print experiment 126's numbers: scores, the 13 planned contrasts, labels, tails, and checks.

Every number the write-up quotes comes from this script. It reads the stored prediction sets
(`<year>_<arm>`), scores each row with the quantile score (twice the mean pinball loss over the 11
levels, in points of effective capacity) on rows outside the hand-filtered switching masks, and
pairs arms row by row at day D+1.

- **Contrasts X1 to X13**: arm-K minus T-all-K on the arm's primary rows, resampling whole calendar
  months. X1 to X9 use 99.44% intervals (Bonferroni over 9) and X10 to X13 use 98.75% (over 4).
  Each has the replicate year, the count of decision sites that improve in the scored year, the
  same arm with T-all's filter hyperparameters (`-b0`), and non-inferiority on all other rows.
- **Labels** from the decision rule, then downgraded by the gates in `work/gate3.json` and
  `work/skf_gate_*.parquet`.
- **Tails, absolute scores, median minus mean, and mean-forecast error** beside each contrast.

    uv run --no-sync python ml-experiments/00126-*/e00126_report.py > report.txt
"""

import json
from dataclasses import dataclass
from functools import cache
from typing import Any, Final

import e00126_common as common
import e00126_pilot as pilot
import numpy as np
import polars as pl
from studies.bootstrap import bootstrap_difference_at_level
from studies.dp_gam import QUANTILE_COLUMNS
from studies.experiment_store import losses_for_bootstrap

SITE_OF_ID: Final[dict[int, str]] = {7: "D9", 8: "D11", 10: "D10", 16: "D14", 17: "D5", 18: "D16"}
DECISION: Final[tuple[str, ...]] = common.DECISION_SITES
M_WINTER: Final[float] = 0.03
M_EVENT: Final[float] = 0.05
NI_FLOOR: Final[float] = 0.02
MIN_SITES_IMPROVING: Final[int] = 3
TAIL_LEVELS: Final[dict[str, float]] = {"q01": 0.01, "q05": 0.05, "q95": 0.95, "q99": 0.99}
KEYS: Final[list[str]] = ["time_series_id", "power_fcst_init_time", "valid_time", "seed"]
STN_END: Final[str] = "2025-12-31"
"""W-stn is compared on scored-year rows to this date only."""


@dataclass(frozen=True)
class Contrast:
    """One planned contrast: an arm's filtered set minus T-all-K on its primary rows.

    Attributes:
        name: The contrast's code, X1 to X13.
        arm: The arm's name.
        treatment: The treatment set (`<arm>_k` or the filter arm's set).
        subset: The primary row subset's name in `e00126_pilot.subset_masks`.
        level: The interval level in percent.
        margin: The gain margin M in points.
    """

    name: str
    arm: str
    treatment: str
    subset: str
    level: float
    margin: float


WINTER = "Nov-Mar"
CONTRASTS: Final[tuple[Contrast, ...]] = (
    Contrast("X1", "a_mono", "a_mono_k", WINTER, 99.44, M_WINTER),
    Contrast("X2", "a_free", "a_free_k", WINTER, 99.44, M_WINTER),
    Contrast("X3", "s_ewma", "s_ewma_k", WINTER, 99.44, M_WINTER),
    Contrast("X4", "s_mm", "s_mm_k", WINTER, 99.44, M_WINTER),
    Contrast("X5", "c_wind", "c_wind_k", WINTER, 99.44, M_WINTER),
    Contrast("X6", "c_hw", "c_hw_k", WINTER, 99.44, M_WINTER),
    Contrast("X7", "d_sch", "d_sch_k", "school-holiday weekdays", 99.44, M_EVENT),
    Contrast(
        "X8", "e_sol", "e_sol_k", "dark winter evenings (Nov-Feb, 15:30-19:00)", 99.44, M_EVENT
    ),
    Contrast("X9", "f_mix", "t_all_fmix_k", "cold-snap days (Nov-Mar)", 99.44, M_EVENT),
    Contrast("X10", "g_skf", "t_all_gskf_k", WINTER, 98.75, M_WINTER),
    Contrast("X11", "w_basis", "w_basis_k", WINTER, 98.75, M_WINTER),
    Contrast("X12", "w_mix", "w_mix_k", WINTER, 98.75, M_WINTER),
    Contrast("X13", "w_stn", "w_stn_k", "Nov-Dec (W-stn rows)", 98.75, M_WINTER),
)


@cache
def load_arm(*, year: str, arm: str) -> pl.DataFrame:
    """Read a stored set with its per-row score, mean absolute error, and subset masks."""
    frame = pl.read_parquet(
        common.DATA_DIR / "predictions" / f"{year}_{arm}" / "predictions.parquet"
    ).filter(~pl.col("switching_mask"))
    quantiles = frame.select(QUANTILE_COLUMNS).to_numpy().astype(np.float64)
    actual = frame["actual"].to_numpy().astype(np.float64)
    masks = pilot.subset_masks(frame=frame, cache=_cache())
    return frame.with_columns(
        qs=pl.Series(common.quantile_score(actual=actual, quantiles=quantiles)),
        ae=(pl.col("actual") - pl.col("mean")).abs(),
        site_name=pl.col("time_series_id").replace_strict(SITE_OF_ID),
        **{name: pl.Series(mask) for name, mask in masks.items()},
    )


@cache
def _cache() -> Any:
    """Return the feature cache."""
    return common.load_cache()


def paired(
    *,
    year: str,
    treatment: str,
    reference: str,
    sites: tuple[str, ...],
    subset: str | None,
    invert: bool = False,
    stn_only: bool = False,
) -> pl.DataFrame:
    """Return both arms' rows on the sites and subset, paired, for `studies.bootstrap`."""
    arms = {}
    for name in (treatment, reference):
        frame = load_arm(year=year, arm=name).filter(pl.col("site_name").is_in(sites))
        if subset is not None:
            frame = frame.filter(~pl.col(subset) if invert else pl.col(subset))
        if stn_only:
            frame = frame.filter(pl.col("valid_time").dt.date() <= pl.lit(STN_END).str.to_date())
        arms[name] = frame.select(*KEYS, "qs", "ae")
    first, second = list(arms.values())
    shared = first.select(KEYS).join(second.select(KEYS), on=KEYS, how="inner")
    arms = {name: frame.join(shared, on=KEYS, how="inner") for name, frame in arms.items()}
    return losses_for_bootstrap(arm_losses=arms)


def estimate(
    *,
    contrast: Contrast,
    reference: str = "t_all_k",
    treatment: str | None = None,
    year: str = "scored",
    sites: tuple[str, ...] = DECISION,
    subset: str | bool | None = True,
    invert: bool = False,
) -> tuple[float, float, int]:
    """Return the mean paired difference in score, in mean absolute error, and the row count."""
    chosen = contrast.subset if subset is True else (subset or None)
    losses = paired(
        year=year,
        treatment=treatment or contrast.treatment,
        reference=reference,
        sites=sites,
        subset=chosen,  # type: ignore[arg-type]
        invert=invert,
        stn_only=contrast.arm == "w_stn" and year == "scored",
    )
    means = losses.group_by("arm").agg(
        pl.col("qs").mean(), pl.col("ae").mean(), pl.len().alias("n")
    )
    first = means.filter(pl.col("arm") == (treatment or contrast.treatment)).row(0, named=True)
    second = means.filter(pl.col("arm") == reference).row(0, named=True)
    return first["qs"] - second["qs"], first["ae"] - second["ae"], int(first["n"])


def interval(
    *,
    contrast: Contrast,
    invert: bool = False,
    level: float | None = None,
    treatment: str | None = None,
) -> tuple[float, float]:
    """Return the month-resampled interval of the contrast on its primary rows (or the others)."""
    losses = paired(
        year="scored",
        treatment=treatment or contrast.treatment,
        reference="t_all_k",
        sites=DECISION,
        subset=contrast.subset,
        invert=invert,
        stn_only=contrast.arm == "w_stn",
    )
    return bootstrap_difference_at_level(
        losses=losses,
        treatment=treatment or contrast.treatment,
        reference="t_all_k",
        metric="qs",
        level=level or contrast.level,
        n_resamples=common.N_RESAMPLES,
    )


def label_of(
    *,
    contrast: Contrast,
    low: float,
    high: float,
    scored: float,
    replicate: float,
    improving: int,
    ni_ok: bool,
    half_width: float,
) -> str:
    """Return the decision rule's label (before the gates)."""
    if low > 0.0:
        return "worse"
    if high < 0.0:
        if not ni_ok:
            return "worse on other rows"
        if scored > -contrast.margin:
            return "small gain" if replicate < 0.0 else "not replicated"
        return "gain" if replicate < 0.0 and improving >= MIN_SITES_IMPROVING else "not replicated"
    return "undecidable" if half_width > contrast.margin else "tie"


def pilot_half_width(*, contrast: Contrast) -> float:
    """Return the pilot's half-width for the contrast's subset at its level."""
    record = json.loads((common.WORK_DIR / "pilot.json").read_text())
    key = "half_width_99.44" if contrast.level > 99 else "half_width_98.75"
    return float(record[contrast.subset][key])


def contrast_row(*, contrast: Contrast) -> dict[str, Any]:
    """Return one contrast's row of numbers and its label."""
    scored, scored_ae, rows = estimate(contrast=contrast)
    replicate, _, _ = estimate(contrast=contrast, year="replicate")
    low, high = interval(contrast=contrast)
    improving = sum(int(estimate(contrast=contrast, sites=(s,))[0] < 0.0) for s in DECISION)
    other, _, _ = estimate(contrast=contrast, invert=True)
    other_low, other_high = interval(contrast=contrast, invert=True)
    ni = max(NI_FLOOR, pilot_half_width(contrast=contrast))
    ni_ok = other_high <= ni
    kb0 = None
    if contrast.arm not in ("f_mix", "g_skf"):
        kb0_name = f"{contrast.arm}_kb0"
        diff, _, _ = estimate(contrast=contrast, treatment=kb0_name)
        kb0_low, kb0_high = interval(contrast=contrast, treatment=kb0_name)
        kb0 = {"estimate": diff, "low": kb0_low, "high": kb0_high}
    label = label_of(
        contrast=contrast,
        low=low,
        high=high,
        scored=scored,
        replicate=replicate,
        improving=improving,
        ni_ok=ni_ok,
        half_width=0.5 * (high - low),
    )
    return {
        "contrast": contrast.name,
        "arm": contrast.arm,
        "primary rows": contrast.subset,
        "rows": rows,
        "scored": scored,
        "low": low,
        "high": high,
        "half_width": 0.5 * (high - low),
        "replicate": replicate,
        "sites improving (of 5)": improving,
        "other rows": other,
        "other rows low": other_low,
        "other rows high": other_high,
        "NI margin": ni,
        "MAE of the mean": scored_ae,
        "with T-all's filter (-b0)": kb0,
        "label before gates": label,
    }


def tail_table() -> pl.DataFrame:
    """Return coverage and pinball loss at the four tail levels for the scored year, all rows."""
    rows = []
    for name in ("t_all_k", "t_all_fmix_k", "t_all_gskf_k"):
        frame = load_arm(year="scored", arm=name).filter(pl.col("site_name").is_in(DECISION))
        actual = frame["actual"].cast(pl.Float64)
        for column, level in TAIL_LEVELS.items():
            quantile = frame[column].cast(pl.Float64)
            error = actual - quantile
            pinball = pl.select(
                pl.when(error >= 0).then(level * error).otherwise((level - 1.0) * error).mean()
            ).item()
            rows.append(
                {
                    "set": name,
                    "level": column,
                    "coverage": float((actual <= quantile).mean()),
                    "pinball": pinball,
                }
            )
    return pl.DataFrame(rows)


def absolute_table() -> pl.DataFrame:
    """Return each set's mean score on the decision sites for the three years."""
    rows = []
    for year in ("tuning", "replicate", "scored"):
        for arm in ("t_all", "t_all_k", *(c.treatment for c in CONTRASTS)):
            try:
                frame = load_arm(year=year, arm=arm).filter(pl.col("site_name").is_in(DECISION))
            except FileNotFoundError:
                continue
            rows.append({"year": year, "set": arm, "score": float(frame["qs"].mean())})
    return pl.DataFrame(rows).pivot(on="year", index="set", values="score")


def median_minus_mean() -> pl.DataFrame:
    """Return the mean over rows of (q50 minus mean forecast) per decision site, T-all-K."""
    frame = load_arm(year="scored", arm="t_all_k").filter(pl.col("site_name").is_in(DECISION))
    return (
        frame.group_by("site_name")
        .agg((pl.col("q50") - pl.col("mean")).mean().alias("q50 - mean"))
        .sort("site_name")
    )


def main() -> None:
    """Print every table and write `work/numbers.json`."""
    rows = [contrast_row(contrast=contrast) for contrast in CONTRASTS]
    gate3 = (
        json.loads((common.WORK_DIR / "gate3.json").read_text())
        if (common.WORK_DIR / "gate3.json").exists()
        else {}
    )
    for row in rows:
        entry = gate3.get(row["arm"])
        row["gate 3 powered"] = None if entry is None else entry.get("powered")
    print(json.dumps(rows, indent=2))
    (common.WORK_DIR / "numbers.json").write_text(json.dumps(rows, indent=2) + "\n")
    print("\n== absolute scores (decision sites, all rows outside switching masks) ==")
    print(absolute_table())
    print("\n== tails, scored year ==")
    print(tail_table())
    print("\n== median minus mean of T-all-K per site ==")
    print(median_minus_mean())


if __name__ == "__main__":
    main()
