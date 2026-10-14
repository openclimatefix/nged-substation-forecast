"""Run the sweep's stages in order, choosing every feature set on the selection folds only.

Stages (each writes predictions under `fits/<stage>/<fold>/`):

- `add_one_in`: each column group added to `W`, and to `W+clock`, plus the three treatments.
- `greedy`: forward selection from `W+clock`, adding the group that lowers the mean absolute error
  most, until the best gain is under `MIN_GAIN_POINTS` or `MAX_GREEDY_STEPS` is reached.
- `drop_one`: the chosen set with each group removed.
- `confirm`: the frozen set, run only on the confirmation fold, after `selection.json` exists.

    cd ml-experiments/00006-stage1a-demand-input-feature-sweep
    uv run python e00006_sweep.py add_one_in select_2024
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
SELECTION_PATH: Final[Path] = inputs.EXPERIMENT_DIR / "selection_Primary.json"
FEATURE_GROUPS: Final[tuple[str, ...]] = tuple(
    g
    for g in fit.GROUP_COLUMNS
    if g not in ("month_cat", "covid_graded", "windchill_unguarded", "xmas_window", "season_half")
)
"""The groups the add-one-in and greedy stages consider. The month categorical and the graded
lockdown indicator are tested in their own stages."""


def read_stage(
    *, stage: str, fold_name: str, config: str = "series", seed: int = 0
) -> pl.DataFrame:
    """Concatenate a stage's unit files for one fold."""
    folder = fit.FOLDS_DIR / stage / fold_name / config / f"seed{seed}"
    files = sorted(f for f in folder.glob("*.parquet") if not f.stem.endswith("_importance"))
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
    return {a: float(per_series[a].to_numpy().mean()) * 100.0 for a in arms}


def add_one_in(*, fold_name: str) -> None:
    """Fit `W`, `W+clock`, and every one-group and treatment addition to each."""
    names = [fit.Arm(groups=()).name]
    names += fit.arm_names_with(base=(), extra=FEATURE_GROUPS)
    names += fit.arm_names_with(base=("clock",), extra=FEATURE_GROUPS)
    for base in ((), ("clock",)):
        names += [fit.Arm(groups=base, treatments=(t,)).name for t in fit.TREATMENTS]
    fit.run_stage(
        stage="add_one_in",
        fold_name=fold_name,
        config="series",
        arm_names=list(dict.fromkeys(names)),
        seeds=[0],
    )


def _only_type(*, frame: pl.DataFrame, substation_type: str | None) -> pl.DataFrame:
    """Keep the rows of one substation type, or every row if the type is None."""
    if substation_type is None:
        return frame
    series = pl.read_parquet(inputs.INPUTS_DIR / "series.parquet")
    keep = series.filter(pl.col("substation_type") == substation_type)["time_series_id"]
    return frame.filter(pl.col("time_series_id").is_in(keep.to_list()))


def greedy(*, fold_name: str, substation_type: str | None = None) -> None:
    """Forward-select groups from `W+clock` on a selection fold and write the selection file.

    Args:
        fold_name: The selection fold.
        substation_type: Score only this type's series (the file is `selection_<type>.json`), or
            every series (`selection.json`) if None.
    """
    chosen: list[str] = ["clock"]
    scores = mae_points(
        frame=_only_type(
            frame=read_stage(stage="add_one_in", fold_name=fold_name),
            substation_type=substation_type,
        )
    )
    current = scores[fit.Arm(groups=tuple(chosen)).name]
    path_taken = [{"set": list(chosen), "mae": current}]
    for step in range(1, MAX_GREEDY_STEPS + 1):
        candidates = fit.arm_names_with(base=tuple(chosen), extra=FEATURE_GROUPS)
        stage = "greedy_" + "+".join(chosen)
        if step > 1:
            fit.run_stage(
                stage=stage, fold_name=fold_name, config="series", arm_names=candidates, seeds=[0]
            )
            scores |= mae_points(
                frame=_only_type(
                    frame=read_stage(stage=stage, fold_name=fold_name),
                    substation_type=substation_type,
                )
            )
        best = min(candidates, key=lambda c: scores[c])
        gain = current - scores[best]
        print(f"step {step}: best {best} gain {gain:.3f} points", flush=True)
        if gain < MIN_GAIN_POINTS:
            break
        chosen = list(fit.Arm.from_name(best).groups)
        current = scores[best]
        path_taken.append({"set": list(chosen), "mae": current})
    name = "selection.json" if substation_type is None else f"selection_{substation_type}.json"
    out = inputs.EXPERIMENT_DIR / name
    out.write_text(json.dumps({"fold": fold_name, "path": path_taken}, indent=2) + "\n")


def drop_one(*, fold_name: str) -> None:
    """Fit the selected set with each of its groups removed."""
    selected = tuple(json.loads(SELECTION_PATH.read_text())["path"][-1]["set"])
    names = [fit.Arm(groups=selected).name]
    names += [fit.Arm(groups=tuple(g for g in selected if g != drop)).name for drop in selected]
    fit.run_stage(
        stage="drop_one", fold_name=fold_name, config="series", arm_names=names, seeds=[0]
    )


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
    fit.run_stage(
        stage="candidates", fold_name=fold_name, config="series", arm_names=names, seeds=[0]
    )


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
    fit.run_stage(
        stage="planned", fold_name=fold_name, config="series", arm_names=names, seeds=[0, 1, 2]
    )


def masking(*, fold_name: str) -> None:
    """Fit time-of-year encodings on sets whose weather block cannot carry the season."""
    removals = {
        "full": (),
        "no_sun": ("solar_zenith", "solar_azimuth"),
        "no_temperature": ("temperature_2m", "dew_point_temperature_2m"),
        "no_sun_no_temperature": (
            "solar_zenith",
            "solar_azimuth",
            "temperature_2m",
            "dew_point_temperature_2m",
        ),
    }
    names = [
        fit.Arm(groups=("clock", *extra), dropped=dropped).name
        for dropped in removals.values()
        for extra in ((), ("doy1",), ("doy2",), ("month_cat",))
    ]
    fit.run_stage(
        stage="masking", fold_name=fold_name, config="series", arm_names=names, seeds=[0, 1, 2]
    )


def graded_covid(*, fold_name: str) -> None:
    """Fit the Primary set with the graded lockdown indicator in place of the 0/1 one."""
    base = tuple(g for g in selected_groups() if g != "covid_flag")
    names = [
        fit.Arm(groups=base).name,
        fit.Arm(groups=(*base, "covid_flag")).name,
        fit.Arm(groups=(*base, "covid_graded")).name,
        fit.Arm(groups=(*base, "covid_flag", "covid_graded")).name,
    ]
    fit.run_stage(
        stage="graded_covid", fold_name=fold_name, config="series", arm_names=names, seeds=[0, 1, 2]
    )


def production_windchill(*, fold_name: str) -> None:
    """Fit production's unguarded windchill expression, which the guarded arms do not test."""
    names = [
        fit.Arm(groups=()).name,
        fit.Arm(groups=("windchill_unguarded",)).name,
        fit.Arm(groups=("clock",)).name,
        fit.Arm(groups=("clock", "windchill_unguarded")).name,
    ]
    fit.run_stage(
        stage="production_windchill",
        fold_name=fold_name,
        config="series",
        arm_names=names,
        seeds=[0, 1, 2],
    )


def christmas_check(*, fold_name: str) -> None:
    """Test whether `days_to_christmas` helps through Christmas or through its half-year step."""
    base = ("clock", "covid_flag")
    names = [
        fit.Arm(groups=base).name,
        fit.Arm(groups=(*base, "xmas")).name,
        fit.Arm(groups=(*base, "xmas_window")).name,
        fit.Arm(groups=(*base, "season_half")).name,
        fit.Arm(groups=(*base, "doy1")).name,
        fit.Arm(groups=(*base, "hol_flags")).name,
    ]
    fit.run_stage(
        stage="christmas_check",
        fold_name=fold_name,
        config="series",
        arm_names=names,
        seeds=[0, 1, 2],
    )


def importance_only(*, fold_name: str) -> None:
    """Refit the finalists for gain and permutation importance only: per series, and pooled."""
    chosen = selected_groups()
    arms = [
        fit.Arm(groups=("clock",)),
        fit.Arm(groups=chosen, treatments=("recency",)),
    ]
    fit.run_stage(
        stage="importance_series",
        fold_name=fold_name,
        config="series",
        arm_names=[a.name for a in arms],
        seeds=[0],
        importance=True,
    )
    pooled = [f"{a.name}|idcat" for a in arms]
    fit.run_stage(
        stage="importance_pooled",
        fold_name=fold_name,
        config="type",
        arm_names=pooled,
        seeds=[0],
        importance=True,
    )


def finals(*, fold_name: str) -> None:
    """Fit the frozen set and its reference arms: three seeds, quantiles, pooled, importance."""
    chosen = selected_groups()
    reference = [fit.Arm(groups=()), fit.Arm(groups=("clock",))]
    frozen = [
        fit.Arm(groups=chosen),
        fit.Arm(groups=chosen, treatments=("recency",)),
    ]
    recent = [
        fit.Arm(groups=chosen, train_start="2023-09"),
        fit.Arm(groups=chosen, treatments=("recency",), train_start="2023-09"),
    ]
    names = [a.name for a in (*reference, *frozen, *recent)]
    fit.run_stage(
        stage="finals", fold_name=fold_name, config="series", arm_names=names, seeds=[0, 1, 2]
    )
    median_arms = [a.name for a in (*reference, *frozen)]
    fit.run_stage(
        stage="finals_importance",
        fold_name=fold_name,
        config="series",
        arm_names=median_arms,
        seeds=[0],
        importance=True,
    )
    fit.run_stage(
        stage="finals_quantiles",
        fold_name=fold_name,
        config="series",
        arm_names=[a.name for a in (*reference, frozen[1])],
        seeds=[0, 1, 2],
        quantiles=True,
    )
    pooled = []
    for arm in (*reference, *frozen):
        pooled += [
            arm.name,
            f"{arm.name}|idint",
            f"{arm.name}|idcat",
        ]
    fit.run_stage(
        stage="pooled",
        fold_name=fold_name,
        config="type",
        arm_names=pooled,
        seeds=[0],
        importance=True,
    )


def type_sets(*, fold_name: str) -> None:
    """Fit each substation type's own selected set, and the all-series set, with three seeds."""
    sets = {"BSP": ("clock", "hol_flags"), "GSP": ("clock", "covid_flag", "hol_nearest", "hdh")}
    all_series = ("clock", "covid_flag", "hol_nearest")
    names = []
    for groups in (*sets.values(), all_series):
        names += [
            fit.Arm(groups=groups).name,
            fit.Arm(groups=groups, treatments=("recency",)).name,
        ]
    fit.run_stage(
        stage="type_sets", fold_name=fold_name, config="series", arm_names=names, seeds=[0, 1, 2]
    )


def main() -> None:
    """Dispatch a stage named on the command line."""
    stage, fold_name = sys.argv[1], sys.argv[2]
    if stage.startswith("greedy_"):
        greedy(fold_name=fold_name, substation_type=stage.removeprefix("greedy_"))
        return
    {
        "add_one_in": add_one_in,
        "greedy": greedy,
        "drop_one": drop_one,
        "candidates": candidates,
        "finals": finals,
        "importance_only": importance_only,
        "christmas_check": christmas_check,
        "production_windchill": production_windchill,
        "type_sets": type_sets,
        "graded_covid": graded_covid,
        "masking": masking,
        "planned": planned,
    }[stage](fold_name=fold_name)


if __name__ == "__main__":
    main()
