"""Print every number that the experiment 116 write-up quotes, from the stored result files.

The write-up's numbers come from this script and nowhere else. It reads `decisions.json`,
`tune/*.json`, `gate.json`, `c2.json`, `c3.json`, the seeds of C1, `verdict.json`, and
`checks.json`, and prints them in the order the page uses them.

    uv run python ml-experiments/00116-*/e00116_report.py
"""

import json
from typing import Any

import e00116_common as common
import e00116_decide as decide
import e00116_tune as tune


def show(*, label: str, value: Any) -> None:
    """Print one labelled number or table row."""
    print(f"{label}: {value}")


def interval(*, entry: dict[str, Any]) -> str:
    """Format a contrast as estimate (lower to upper) at the 98.33% level."""
    estimate = entry.get("estimate", entry.get("difference"))
    return f"{estimate:+.3f} ({entry['lower']:+.3f} to {entry['upper']:+.3f})"


def main() -> None:  # noqa: PLR0912 - a flat sequence of prints, one block per result file
    """Print the tuning choice, the gate, C1, C2, C3, the verdict, and the controls."""
    work = common.WORK_DIR
    decisions = json.loads((work / "decisions.json").read_text())
    print("== Tuning (seeds T)")
    for name in ("A2", "A2-b", "chosen"):
        entry = decisions[name]
        show(
            label=name,
            value=(
                f"config {entry['config']}, recall {entry['recall']:.3f}, placebo "
                f"{ {k: round(v, 3) for k, v in entry['placebo_mw_days'].items()} }, "
                f"met limit {entry['met_limit']}"
            ),
        )
    for result in tune.read_results():
        config = result["config"]
        show(
            label=f"grid block {config['block']} c {config['step_c']:g} l {config['level_c']:g}",
            value=(
                f"recall {result['recall']:.3f} "
                f"placebo {max(result['placebo_mw_days'].values()):.3f}"
            ),
        )
    print("== Gate (seeds G)")
    gate = json.loads((work / "gate.json").read_text())
    for name, reading in gate.items():
        show(label=name, value=f"{reading['value']} pass={reading['pass']}")
    first = json.loads((work / "gate_tuning_only_first_run.json").read_text())
    show(label="first run, reading 5 (outside-path bug)", value=first["5_nra_gap_closed"]["value"])
    print("== C1")
    c1 = decide.c1_summary()
    show(
        label="reconstruction share",
        value=f"{c1['estimate']:.3f} ({c1['lower']:.3f} to {c1['upper']:.3f})",
    )
    show(label="transfers", value=c1["n_transfers"])
    show(
        label="clean-year MAE (points) a0/a4/oracle trained on injected history",
        value=c1["clean_year_mae"],
    )
    print("== C2")
    c2 = json.loads((work / "c2.json").read_text())
    for scope, entry in c2.items():
        if scope == "checks":
            continue
        for name in ("a4_minus_a0", "a4_minus_a1", "a1_minus_a0", "a4_minus_a0_mean_error"):
            show(
                label=f"{scope} {name}",
                value=(
                    f"{interval(entry=entry[name])}, sites better "
                    f"{entry[name]['sites_better']} of {entry[name]['sites']}"
                ),
            )
        show(label=f"{scope} absolute", value=entry["absolute"])
        show(label=f"{scope} days", value=entry["days"])
    for cut_name, entry in c2["checks"].items():
        show(label=f"{cut_name} median minus mean forecast", value=entry["median_minus_mean"])
        for row in entry["tails"]:
            show(label=f"{cut_name} tails", value=row)
    print("== Post hoc variants")
    for variant in json.loads((work / "variants.json").read_text())["post_hoc_variants"]:
        show(label=variant["variant"], value=variant)
    print("== Real events on the gate edges")
    show(label="events", value=(work / "real_events.json").read_text())
    print("== C3")
    c3 = json.loads((work / "c3.json").read_text())
    for year, entry in c3.items():
        for name, contrast in entry["contrasts"].items():
            show(
                label=f"{year} {name}",
                value=f"{interval(entry=contrast)} on {contrast['n_transfers']} transfers",
            )
        for key in (
            "gap_closed_share",
            "oracle_gain",
            "oracle_gain_scaled_by_in_event_share",
            "sites",
            "absolute",
            "receiver_q99_exceedance",
            "real_rows_outside_events",
        ):
            show(label=f"{year} {key}", value=entry[key])
        for scope, scoped in entry["winter_scopes"].items():
            show(label=f"{year} {scope}", value=scoped)
        for row in entry["tails"]:
            show(label=f"{year} tails", value=row)
    print("== Verdict")
    verdict = json.loads((work / "verdict.json").read_text())
    show(label="verdict", value=f"{verdict['verdict']}: {verdict['reason']}")
    show(label="C3 conditions", value=verdict["c3"]["conditions"])
    print("== Controls")
    show(label="checks", value=json.loads((work / "checks.json").read_text()))


if __name__ == "__main__":
    main()
