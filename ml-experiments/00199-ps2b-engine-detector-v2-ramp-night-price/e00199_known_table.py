# ruff: noqa: E501
"""Print the known-answer table: recall, precision, and energy error by arm, with intervals."""

import json

from e00199_common import DATA_DIR


def main() -> None:
    """One line per injection, arm, and row set."""
    d = json.loads((DATA_DIR / "known_answers.json").read_text())
    print(f"price gate: spark spread at or below {d['gate_spread']:.1f} pounds per MWh\n")
    for key, v in d.items():
        if "|" not in key:
            continue

        def cell(name: str, v: dict = v) -> str:
            m = v[name]
            return f"{m['value']:.2f} ({m['lo']:.2f}-{m['hi']:.2f})"

        print(
            f"{key:46s} r1 {cell('recall_1_unit'):17s} r23 {cell('recall_2_or_3_units'):17s} "
            f"prec {cell('precision'):17s} energy {v['energy_error']['value']:+.2f} n={v['n_scored']}"
        )


if __name__ == "__main__":
    main()
