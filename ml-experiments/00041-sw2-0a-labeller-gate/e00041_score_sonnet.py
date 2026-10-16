"""Merge a labeller's JSONL batches and score them on a known-answer set.

uv run python e00041_score_sonnet.py <set> <labels folder name>
"""

from __future__ import annotations

import json
import sys

from e00041_common import WORK_DIR
from e00041_gate import score


def load(*, which: str, folder: str) -> dict[str, dict]:
    """Load the labels in every JSONL file under `which`/`folder`, keyed by card id."""
    labels: dict[str, dict] = {}
    for path in sorted((WORK_DIR / which / folder).glob("*.jsonl")):
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                labels[row["card_id"]] = row
    return labels


if __name__ == "__main__":
    which, folder = sys.argv[1], sys.argv[2]
    truths = json.loads((WORK_DIR / which / "truths.json").read_text())
    labels = load(which=which, folder=folder)
    truths = {c: t for c, t in truths.items() if c in labels}
    print(len(labels), "labels")
    result = score(labels=labels, truths=truths)
    for k, v in result.items():
        print(k, v)
    (WORK_DIR / which / f"{folder}_merged.json").write_text(json.dumps(labels, indent=1))
