"""Print recipe feature inventories using the fitter's preparation on synthetic rows.

No observations, forecasts or fitted models are read. Counts distinguish recipe inputs,
equal-width duplicate padding, and the numeric site identifier added by pooled encoders.
"""

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import arms
import fixed_pipeline_fit as fitter
import polars as pl


def _sha256(*, path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _inventory(*, recipes: list[str], registry: dict[str, fitter.Recipe]) -> pl.DataFrame:
    kinds = ("PV", "Wind")
    features = sorted(
        {
            column
            for recipe in registry.values()
            for kind in kinds
            for column in recipe.arm.columns(kind)
        }
    )
    synthetic = pl.DataFrame({"time_series_type": kinds, **{name: [0.0, 0.0] for name in features}})
    rows: list[dict[str, Any]] = []
    for name in recipes:
        recipe = registry[name]
        _, columns = fitter._prepare(
            frame=synthetic, recipe=recipe, padding="equal", registry=registry
        )
        for kind in kinds:
            real_count = len(recipe.arm.columns(kind))
            ordered = columns[kind][:real_count]
            production = set(arms.PRODUCTION_COLUMNS)
            trained = [*columns[kind], *(["site_id"] if recipe.arm.pooled_site_id else [])]
            rows.append(
                {
                    "recipe": name,
                    "series_type": kind,
                    "recipe_input_count": real_count,
                    "removed_from_production": ", ".join(sorted(production - set(ordered))),
                    "added_to_production": ", ".join(sorted(set(ordered) - production)),
                    "ordered_recipe_inputs": ", ".join(ordered),
                    "equal_padding_width": len(columns[kind]),
                    "duplicate_padding_count": len(columns[kind]) - real_count,
                    "pooled_numeric_site_id": recipe.arm.pooled_site_id,
                    "trained_column_count": len(trained),
                    "ordered_trained_columns": ", ".join(trained),
                    "exact_reference": name in fitter.EXACT_RECIPES,
                    "unpadded": len(columns[kind]) == real_count,
                }
            )
    return pl.DataFrame(rows)


def main() -> None:
    """Save an immutable public feature inventory without fitting or reading study data."""
    registry = fitter.recipe_registry()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--recipes", nargs="+", choices=sorted(registry), default=sorted(registry))
    args = parser.parse_args()
    if len(set(args.recipes)) != len(args.recipes):
        raise ValueError("requested recipes must be distinct")
    frame = _inventory(recipes=args.recipes, registry=registry)
    lines = [
        "# Fixed-pipeline recipe feature inventory",
        "",
        "Recipe inputs precede encoding. Padding width excludes the numeric site identifier;",
        "trained counts and order include it when requested by a pooled recipe.",
        "The pooled site identifier is Float32 numeric.",
        "Lists come from the fitter's equal-padding preparation on two synthetic rows.",
        "No data were read and no model was fitted. Empty added/removed cells mean none.",
        "",
        "| " + " | ".join(frame.columns) + " |",
        "|" + "---|" * frame.width,
    ]
    lines.extend("| " + " | ".join(str(value) for value in row) + " |" for row in frame.iter_rows())
    report = "\n".join(lines) + "\n"
    args.output.mkdir(parents=True, exist_ok=False)
    frame.write_parquet(args.output / "recipe_inventory.parquet")
    (args.output / "report.md").write_text(report)
    provenance = {
        "recipes": args.recipes,
        "padding": "equal",
        "inputs": {
            str(path): _sha256(path=path)
            for path in (
                Path(__file__),
                Path(fitter.__file__),
                Path(arms.__file__),
                Path(fitter.fit_arms.__file__),
                Path(arms.inputs.PRODUCTION_MODEL_CONFIG),
            )
        },
        "outputs": {
            name: _sha256(path=args.output / name)
            for name in ("recipe_inventory.parquet", "report.md")
        },
    }
    (args.output / "manifest.json").write_text(json.dumps(provenance, indent=2) + "\n")
    print(report)


if __name__ == "__main__":
    main()
