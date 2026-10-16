"""Score the known answers: contrasts C1 and C2, the gate, and the chosen arm. Writes the report.

uv run python ml-experiments/00073-pvdis-6-orientation-mix-and-dc-ac-ratio/e00073_score_known.py
"""

import json

import e00073_bootstrap as boot
import e00073_common as common
import numpy as np
import polars as pl

CHOSEN_PATH = common.WORK_DIR / "chosen_arm.json"


def unit_errors(*, table: pl.DataFrame) -> pl.DataFrame:
    """Return one error per (site, farm, tilt, mix, arm): the mean over years for O1 and O2."""
    return table.group_by("site", "farm", "tilt", "mix_id", "arm").agg(
        pl.col("tv").mean(), pl.col("bound").mean().alias("bound")
    )


def guess_error(*, mix_id: int, guess: tuple[float, ...]) -> float:
    """Return the total-variation error of guessing `guess` when the truth is mix `mix_id`."""
    return boot.as_float(0.5 * np.abs(np.array(guess) - np.array(common.MIXES[mix_id])).sum())


def orientation(*, lines: list[str]) -> str:
    """Score C1; return the chosen arm."""
    units = unit_errors(table=pl.read_parquet(common.KNOWN_ORIENTATION_PATH))
    main = units.filter(pl.col("tilt") == common.MAIN_TILT)
    tuning = main.filter(pl.col("farm").is_in(common.TUNING_FARMS))
    medians = tuning.group_by("arm").agg(pl.col("tv").median()).sort("tv")
    lines.append("Orientation, tuning units (farm locations A-C, tilt 25): median error by arm")
    lines.append(str(medians))
    chosen = str(medians["arm"][0])
    scoring = main.filter(pl.col("farm").is_in(common.SCORING_FARMS)).with_columns(
        guess=pl.col("mix_id").map_elements(
            lambda m: guess_error(mix_id=m, guess=common.FLEET_MIX), return_dtype=pl.Float64
        ),
        south=pl.col("mix_id").map_elements(
            lambda m: guess_error(mix_id=m, guess=(0.0, 1.0, 0.0)), return_dtype=pl.Float64
        ),
    )
    lines.append("\nScoring units (farm locations D-F, tilt 25): median error by arm and by guess")
    lines.append(str(scoring.group_by("arm").agg(pl.col("tv").median()).sort("arm")))
    mine = scoring.filter(pl.col("arm") == chosen).sort("site", "farm", "mix_id")
    diff = (mine["tv"] - mine["guess"]).to_numpy()
    clusters = (mine["site"] + mine["farm"]).to_numpy()
    mean, low, high = boot.cluster_interval(values=diff, clusters=clusters)
    median_error = boot.as_float(mine["tv"].median())
    lines.append(f"\nChosen arm: {chosen}")
    lines.append(f"Median error of the chosen arm on scoring units: {median_error:.3f}")
    lines.append(
        f"Median error of guessing the fleet mix: {boot.as_float(mine['guess'].median()):.3f}"
    )
    lines.append(
        f"C1 (chosen arm minus guess), mean {mean:+.3f}, interval {low:+.3f} to {high:+.3f}"
    )
    per_mix = mine.group_by("mix_id").agg(
        pl.col("tv").median().alias("arm_error"), pl.col("guess").first().alias("guess_error"),
        pl.col("south").first().alias("all_south_error"),
    ).sort("mix_id")  # fmt: skip
    lines.append("By true mix:\n" + str(per_mix))
    lines.append("Share of scoring units where a bound applied (negative share or <0.3 MW):")
    lines.append(f"  chosen arm {boot.as_float(mine['bound'].mean()):.3f}")
    lines.append("Robustness tilts (all farm locations), median error by arm:")
    lines.append(str(units.group_by("tilt", "arm").agg(pl.col("tv").median()).sort("tilt", "arm")))
    identified = median_error <= 0.15 and high < -0.10
    lines.append(f"\nO identified: {identified}  (median <= 0.15 and C1 upper < -0.10)")
    return chosen


def dcac(*, lines: list[str]) -> None:
    """Score the DC:AC known answers and the knee gate."""
    table = pl.read_parquet(common.KNOWN_DCAC_PATH)
    truths = common.FARM_DC_AC
    lines.append("\nDC:AC known answers (grid 1.00 to 1.60 step 0.05)")
    for case in ("alone", "injected"):
        rows = table.filter(pl.col("case") == case)
        wins = 0
        per_farm = []
        diffs = []
        for farm in common.FARMS:
            others = [truths[f] for f in common.FARMS if f != farm]
            guess = boot.as_float(np.median(others))
            fitted_error = boot.as_float(rows.filter(pl.col("farm") == farm)["error"].median())
            guess_err = abs(guess - truths[farm])
            wins += fitted_error < guess_err
            per_farm.append((farm, truths[farm], fitted_error, guess_err))
            diffs.append(fitted_error - guess_err)
        lines.append(f"\nCase {case}: farm, truth, median |fitted - truth|, |guess - truth|")
        lines.extend(f"  {f}  {t:.2f}  {a:.3f}  {g:.3f}" for f, t, a, g in per_farm)
        median_error = boot.as_float(np.median([a for _, _, a, _ in per_farm]))
        lines.append(
            f"  median over farms {median_error:.3f}; fit beats guess on {wins} of 6 farms; "
            f"mean difference {np.mean(diffs):+.3f} (descriptive: six farms)"
        )
    alone = table.filter(pl.col("case") == "alone")
    gate = alone.group_by("farm").agg(pl.col("knee_rows").median())
    lines.append("\nKnee gate: median clear-day half-hours at or above the knee (ratio 1.15)")
    lines.append(str(gate.sort("farm")))
    median_knee = boot.as_float(gate["knee_rows"].median())
    lines.append(
        f"Gate threshold {common.KNEE_GATE_HALF_HOURS}: "
        f"{'passed' if median_knee >= common.KNEE_GATE_HALF_HOURS else 'missed'}"
    )
    base = table.filter(pl.col("case") == "base")["fitted"]
    lines.append(
        f"Reference only: ratio searched on the 8 Primaries' own series without injection, "
        f"{base.len()} Primary-years, quartiles {base.quantile(0.25)} / {base.median()} / "
        f"{base.quantile(0.75)}, range {base.min()} to {base.max()}"
    )


def main() -> None:
    """Write `report.txt` for the known answers and the chosen arm."""
    lines: list[str] = []
    chosen = orientation(lines=lines)
    dcac(lines=lines)
    CHOSEN_PATH.write_text(json.dumps({"chosen": chosen}))
    common.REPORT_PATH.with_name("report_known.txt").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
