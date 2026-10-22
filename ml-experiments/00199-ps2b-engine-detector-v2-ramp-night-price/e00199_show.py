# ruff: noqa: E501
"""Print the false-alarm table: flagged share and false active days a month, by arm."""

import json

from e00199_common import DATA_DIR


def main() -> None:
    """Print one line per site and arm, night rows then all rows."""
    d = json.loads((DATA_DIR / "false_alarms.json").read_text())
    for site, r in d.items():
        print(
            f"== {site}: night share of rows {r['night_share_of_rows']:.2f}, "
            f"left tail (night) {r['left_tail_night']:.3f}, rows with a price {r['price_rows']}"
        )
        for k, v in r.items():
            if isinstance(v, dict) and k.endswith("| night"):
                a = r[k.replace("night", "all")]
                print(
                    f"  {k:34s} night {v['flagged_share'] * 100:5.1f}% {v['false_active_days_per_month']:5.2f} d/mo"
                    f" | all {a['flagged_share'] * 100:5.1f}% {a['false_active_days_per_month']:5.2f} d/mo"
                )


if __name__ == "__main__":
    main()
