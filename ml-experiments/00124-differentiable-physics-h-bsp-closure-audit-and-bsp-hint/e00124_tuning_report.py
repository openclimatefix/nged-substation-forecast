"""Print the tuning results of experiment 124 into `tuning_report.md`, by kind of injection.

    uv run python ml-experiments/00124-*/e00124_tuning_report.py

Every number the page quotes about the tuning year and the clean-year cost comes from this report.
Damage is an injected fit's mean absolute error minus the clean fit's at the same weight, in points
of effective capacity, summed over the injections of a kind.
"""

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from e00124_common import WORK_DIR


def section(*, suffix: str, title: str) -> list[str]:
    """Return the report lines of the tuning run with the files' `suffix`."""
    tuning = json.loads((WORK_DIR / f"tuning{suffix}.json").read_text())
    weights = tuning["weights"]
    clean = json.loads((WORK_DIR / f"clean_grid_tuning{suffix}.json").read_text())["clean_mae"]
    rows = [
        json.loads(line)
        for line in (WORK_DIR / f"tuning_injections{suffix}.jsonl").read_text().splitlines()
    ]
    by_kind: dict[str, np.ndarray] = defaultdict(lambda: np.zeros(len(weights)))
    counts: dict[str, int] = defaultdict(int)
    for row in rows:
        by_kind[row["kind"]] += np.array(row["mae"]) - np.array(clean[row["site"]])
        counts[row["kind"]] += 1
    lines = [f"## {title}", ""]
    lines.append(f"- weights tried: {', '.join(f'{w:g}' for w in weights)}")
    lines.append(f"- chosen weight W*: {tuning['w_star']:g} ({tuning['n_injections']} injections)")
    total = np.sum(list(by_kind.values()), axis=0)
    lines.append(
        "- summed damage over all injections by weight: "
        + ", ".join(f"{v:.1f}" for v in total)
        + f"; reduction at the largest weight {100 * (1 - total[-1] / total[0]):.0f}%"
    )
    for kind, values in by_kind.items():
        lines.append(
            f"- {kind} ({counts[kind]} injections): summed damage by weight "
            + ", ".join(f"{v:.2f}" for v in values)
            + (
                f"; reduction at the largest weight {100 * (1 - values[-1] / values[0]):.0f}%"
                if values[0] > 1.0
                else "; no damage to reduce at weight 0"
            )
        )
    lines.append(
        "- clean cost by weight, tuning year (mean over the MW Primaries of the clean error minus "
        "weight 0's): " + ", ".join(f"{v:.3f}" for v in tuning["clean_cost"])
    )
    for year in ("replicate", "scored"):
        grid = json.loads((WORK_DIR / f"clean_grid_{year}{suffix}.json").read_text())["clean_mae"]
        mean = np.mean(list(grid.values()), axis=0)
        lines.append(
            f"- clean cost by weight, {year} year: " + ", ".join(f"{v:.3f}" for v in mean - mean[0])
        )
    return [*lines, ""]


def main() -> None:
    """Write and print the tuning report: the first run, then the corrected-scale re-run."""
    lines = ["# Experiment 124, tuning year", ""]
    lines += section(
        suffix="", title="First run (BSP-row scale 1.2 MW, from the observed readings)"
    )
    lines += section(
        suffix="_corrected",
        title="Re-run after the plumbing check (BSP-row scale 2.9 MW, from the fitted Primaries)",
    )
    text = "\n".join(lines) + "\n"
    (WORK_DIR / "tuning_report.md").write_text(text)
    (
        Path(__file__).parents[2]
        / "docs"
        / "ml-experiments"
        / Path(__file__).parent.name
        / "tuning_report.md"
    ).write_text(text)
    print(text)


if __name__ == "__main__":
    main()
