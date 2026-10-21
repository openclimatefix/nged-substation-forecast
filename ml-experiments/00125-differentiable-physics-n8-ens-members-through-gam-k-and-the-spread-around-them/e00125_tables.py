"""Print the tables of experiment 125's write-up from `work/results.json` and `work/checks.json`.

uv run python ml-experiments/00125-*/e00125_tables.py
"""

import json

import e00125_common as common

ARMS = ("K0", "K0-w", "MP", "MP-s", "MM", "MO", "MP-1")


def main() -> None:
    """Print each table as markdown."""
    data = json.loads((common.WORK_DIR / "results.json").read_text())
    checks = json.loads((common.WORK_DIR / "checks.json").read_text())
    print(
        "| Arm | Period | Quantile score | Tail pinball | Cells in band | Median absolute error |"
    )
    print("|---|---|---|---|---|---|")
    for period in ("scored", "replicate"):
        for arm in ARMS:
            a = data[period]["arms"][arm]
            print(
                f"| {arm} | {period} | {a['quantile_score']:.3f} | {a['tail_pinball']:.4f} | "
                f"{a['calibration']['in_band']} of 16 | {a['mae_median']:.3f} |"
            )
    print()
    print("| Contrast | Period | Quantile score | Tail pinball |")
    print("|---|---|---|---|")
    for period in ("scored", "replicate"):
        for name, item in data[period]["contrasts"].items():
            s, t = item["score"], item["tail"]
            score = f"{s['difference']:+.4f} [{s['lower']:+.4f}, {s['upper']:+.4f}]"
            tail = f"{t['difference']:+.4f} [{t['lower']:+.4f}, {t['upper']:+.4f}]"
            print(f"| {name} | {period} | {score} | {tail} |")
    print()
    print("| Arm | Winter (Nov-Mar), scored year: score | tail | cells |")
    for arm in ARMS:
        a = data["scored"]["winter"][arm]
        cells = a["calibration"]["in_band"]
        print(f"| {arm} | {a['quantile_score']:.3f} | {a['tail_pinball']:.4f} | {cells} |")
    print()
    for arm, sites in checks["median_minus_mean"].items():
        print(arm, {site: round(value, 3) for site, value in sites.items()})
    for period in ("scored", "replicate"):
        print(
            period, "sites improving tail", data[period]["sites_improving_tail_pinball_MP_vs_K0w"]
        )
        print(
            period,
            "D9",
            {arm: round(data[period]["d9"][arm]["tail_pinball"], 4) for arm in ("K0-w", "MP")},
        )
        for band in range(4):
            print(
                period,
                band,
                {
                    arm: [
                        round(data[period]["arms"][arm]["by_band"][str(band)][m], 3)
                        for m in ("quantile_score", "tail_pinball")
                    ]
                    for arm in ("K0-w", "MP")
                },
            )
    print(json.dumps(json.loads((common.WORK_DIR / "tuned_n8.json").read_text())["mp_s"]))
    t = json.loads((common.WORK_DIR / "tuned_n8.json").read_text())
    print(
        {
            k: t[k]
            for k in (
                "stored_k0_factors",
                "k0w_factors",
                "mp_factors",
                "mm_factors",
                "mp_s",
                "mp_s_factors",
            )
        }
    )
    print(t["s_surface_tail_pinball"])


if __name__ == "__main__":
    main()
