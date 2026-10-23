"""Post hoc checks for experiment 218: why arm 3 is worse than arm 2, and the per-farm contrasts.

Prints the penalty that cross-validation chose for arm 3, arm 3 refitted with a large fixed
penalty (which should approach arm 2), and the arm 2 minus arm 1 contrast for each farm.
"""

import e00218_common as common
import e00218_fit as fit
import numpy as np
import polars as pl
from studies.pv_plant_fit import fit_response_table, fraction_of_output_fitted

LARGE_PENALTY: float = 100.0


def main() -> None:
    """Print the three checks for fold 1."""
    predictions = pl.read_parquet(fit.PREDICTIONS_PATH).filter(pl.col("fold") == 1)
    parameters = pl.read_parquet(fit.PARAMETERS_PATH).filter(pl.col("fold") == 1)
    print(
        parameters.filter(pl.col("arm") == "arm3_learned_response")
        .select("site", "penalty")
        .sort("site")
    )
    rows = pl.read_parquet(common.ROWS_PATH)
    results = []
    for site in common.FARMS:
        data = fit.farm_data(rows=rows, site=site, fold=1)
        if data is None:
            continue
        frame = predictions.filter(pl.col("site") == site).sort("time")
        prior = frame["arm2_per_farm"].to_numpy()
        fixed = frame["arm3_learned_response"].to_numpy()
        table = fit_response_table(
            grid=fit.RESPONSE_GRID,
            irradiance_w_m2=data.frame["ghi_w_m2"].to_numpy()[data.tuning],
            elevation_deg=data.frame["elevation_deg"].to_numpy()[data.tuning],
            output=data.frame["output"].to_numpy().astype(np.float64)[data.tuning],
            prior_output=prior[data.tuning],
            penalty=LARGE_PENALTY,
        )
        heavy = table.predict(
            irradiance_w_m2=data.frame["ghi_w_m2"].to_numpy(),
            elevation_deg=data.frame["elevation_deg"].to_numpy(),
        )
        scored = data.scored
        output = frame["output"].to_numpy().astype(np.float64)
        results.append(
            {
                "site": site,
                "arm2": fraction_of_output_fitted(output=output[scored], predicted=prior[scored]),
                "arm3_cv_penalty": fraction_of_output_fitted(
                    output=output[scored], predicted=fixed[scored]
                ),
                "arm3_penalty_100": fraction_of_output_fitted(
                    output=output[scored], predicted=heavy[scored]
                ),
                "arm1": fraction_of_output_fitted(
                    output=output[scored], predicted=frame["arm1_fleet_shape"].to_numpy()[scored]
                ),
            }
        )
    table = pl.DataFrame(results).with_columns(c1=pl.col("arm2") - pl.col("arm1"))
    print(table)
    print(table.select(pl.all().exclude("site").median()))
    table.write_csv(common.DATA_DIR / "posthoc_arm3_and_c1.csv")


if __name__ == "__main__":
    main()
