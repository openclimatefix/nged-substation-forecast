"""Run the sweep's stages in order, choosing every feature set on the selection folds only.

Stages (each writes predictions under `fits/<stage>/<fold>/`):

- `add_one_in`: each column group added to `W`, and to `W+clock`, plus the three treatments.
- `greedy`: forward selection from `W+clock`, adding the group that lowers the mean absolute error
  most, until the best gain is under `MIN_GAIN_POINTS` or `MAX_GREEDY_STEPS` is reached.
- `drop_one`: the chosen set with each group removed.
- `confirm`: the frozen set, run only on the confirmation fold, after `selection.json` exists.

    uv run python ml-experiments/00006-stage1a-demand-input-feature-sweep/e00006_sweep.py add_one_in select_2024
"""

import json
import sys
from pathlib import Path
from typing import Final

import e00006_fit as fit
import e00006_inputs as inputs
import polars as pl

MIN_GAIN_POINTS: Final[float] = 0.1
"""Greedy selection stops when the best addition lowers the error by less than this."""
MAX_GREEDY_STEPS: Final[int] = 6
SELECTION_PATH: Final[Path] = inputs.EXPERIMENT_DIR / "selection.json"
FEATURE_GROUPS: Final[tuple[str, ...]] = tuple(fit.GROUP_COLUMNS)


def read_stage(*, stage: str, fold_name: str, config: str = "series", seed: int = 0) -> pl.DataFrame:
    """Concatenate a stage's unit files for one fold."""
    folder = fit.FOLDS_DIR / stage / fold_name / config / f"seed{seed}"
    files = sorted(folder.glob("*.parquet"))
    if not files:
        msg = f"no files in {folder}"
        raise FileNotFoundError(msg)
    return pl.concat([pl.read_parquet(f) for f in files], how="diagonal")


def mae_points(*, frame: pl.DataFrame) -> dict[str, float]:
    """Return each arm's mean over series of the per-series mean absolute error, in points.

    A point is one hundredth of the series' effective capacity.
    """
    arms = [c for c in frame.columns if c not in ("time_series_id", "valid_time", "target")]
    per_series = frame.group_by("time_series_id").agg(
        [(pl.col(a) - pl.col("target")).abs().mean().alias(a) for a in arms]
    )
    return {a: float(per_series[a].mean()) * 100.0 for a in arms}  # ty: ignore[unsupported-operator]


def add_one_in(*, fold_name: str) -> None:
    """Fit `W`, `W+clock`, and every one-group and treatment addition to each."""
    names = [fit.Arm(groups=()).name]
    names += fit.arm_names_with(base=(), extra=FEATURE_GROUPS)
    names += fit.arm_names_with(base=("clock",), extra=FEATURE_GROUPS)
    for base in ((), ("clock",)):
        names += [fit.Arm(groups=base, treatments=(t,)).name for t in fit.TREATMENTS]
    fit.run_stage(
        stage="add_one_in", fold_name=fold_name, config="series", arm_names=list(dict.fromkeys(names)), seeds=[0]
    )


def greedy(*, fold_name: str) -> None:
    """Forward-select groups from `W+clock` on a selection fold and write `selection.json`."""
    chosen: list[str] = ["clock"]
    scores = mae_points(frame=read_stage(stage="add_one_in", fold_name=fold_name))
    current = scores[fit.Arm(groups=tuple(chosen)).name]
    path_taken = [{"set": list(chosen), "mae": current}]
    for step in range(1, MAX_GREEDY_STEPS + 1):
        candidates = fit.arm_names_with(base=tuple(chosen), extra=FEATURE_GROUPS)
        missing = [c for c in candidates if c not in scores]
        if missing:
            fit.run_stage(
                stage=f"greedy_step{step}",
                fold_name=fold_name,
                config="series",
                arm_names=candidates,
                seeds=[0],
            )
            scores |= mae_points(frame=read_stage(stage=f"greedy_step{step}", fold_name=fold_name))
        best = min(candidates, key=lambda c: scores[c])
        gain = current - scores[best]
        print(f"step {step}: best {best} gain {gain:.3f} points", flush=True)
        if gain < MIN_GAIN_POINTS:
            break
        chosen = list(fit.Arm.from_name(best).groups)
        current = scores[best]
        path_taken.append({"set": list(chosen), "mae": current})
    SELECTION_PATH.write_text(json.dumps({"fold": fold_name, "path": path_taken}, indent=2) + "\n")


def drop_one(*, fold_name: str) -> None:
    """Fit the selected set with each of its groups removed."""
    selected = tuple(json.loads(SELECTION_PATH.read_text())["path"][-1]["set"])
    names = [fit.Arm(groups=selected).name]
    names += [fit.Arm(groups=tuple(g for g in selected if g != drop)).name for drop in selected]
    fit.run_stage(stage="drop_one", fold_name=fold_name, config="series", arm_names=names, seeds=[0])


def selected_groups() -> tuple[str, ...]:
    """Return the groups of the set greedy selection ended on."""
    return tuple(json.loads(SELECTION_PATH.read_text())["path"][-1]["set"])


def candidates(*, fold_name: str) -> None:
    """Fit the selected set with each treatment and training start (one seed)."""
    chosen = selected_groups()
    without_flag = tuple(g for g in chosen if g != "covid_flag")
    names = [
        fit.Arm(groups=chosen).name,
        fit.Arm(groups=chosen, treatments=("recency",)).name,
        fit.Arm(groups=chosen, treatments=("block",)).name,
        fit.Arm(groups=without_flag, treatments=("nocovid",)).name,
        fit.Arm(groups=without_flag, treatments=("recency", "nocovid")).name,
        fit.Arm(groups=chosen, train_start="2023-09").name,
        fit.Arm(groups=chosen, train_start="2024-04").name,
        fit.Arm(groups=chosen, treatments=("recency",), train_start="2023-09").name,
    ]
    fit.run_stage(stage="candidates", fold_name=fold_name, config="series", arm_names=names, seeds=[0])


def planned(*, fold_name: str) -> None:
    """Fit the arms of the planned contrasts P1 to P7 with three seeds."""
    names = [
        fit.Arm(groups=()).name,
        fit.Arm(groups=("windchill",)).name,
        fit.Arm(groups=("clock",)).name,
        fit.Arm(groups=("clock", "doy1")).name,
        fit.Arm(groups=("clock", "hol_flags")).name,
        fit.Arm(groups=("clock", "t24")).name,
        fit.Arm(groups=("clock", "covid_flag")).name,
        fit.Arm(groups=("clock",), treatments=("recency",)).name,
    ]
    fit.run_stage(stage="planned", fold_name=fold_name, config="series", arm_names=names, seeds=[0, 1, 2])


def main() -> None:
    """Dispatch a stage named on the command line."""
    stage, fold_name = sys.argv[1], sys.argv[2]
    {"add_one_in": add_one_in, "greedy": greedy, "drop_one": drop_one, "candidates": candidates, "planned": planned}[stage](fold_name=fold_name)


if __name__ == "__main__":
    main()
