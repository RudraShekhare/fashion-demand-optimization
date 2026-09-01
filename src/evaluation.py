from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)


def calculate_metrics(
    actual,
    predictions
):

    actual = np.asarray(
        actual,
        dtype=float
    )

    predictions = np.asarray(
        predictions,
        dtype=float
    )

    mae = mean_absolute_error(
        actual,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            actual,
            predictions
        )
    )

    non_zero = actual != 0

    if np.any(non_zero):

        mape = np.mean(
            np.abs(
                (
                    actual[non_zero]
                    -
                    predictions[non_zero]
                )
                /
                actual[non_zero]
            )
        ) * 100

    else:

        mape = np.nan

    return {
        "MAE": float(mae),
        "RMSE": float(rmse),
        "MAPE": float(mape)
    }


def save_metrics(
    model_name,
    metrics,
    output_path
):

    output_path = Path(output_path)

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    row = {
        "Model": model_name,
        "MAE": metrics["MAE"],
        "RMSE": metrics["RMSE"],
        "MAPE": metrics["MAPE"]
    }

    pd.DataFrame(
        [row]
    ).to_csv(
        output_path,
        index=False
    )