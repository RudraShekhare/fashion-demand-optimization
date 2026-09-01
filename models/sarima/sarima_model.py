from pathlib import Path
import sys
import time
import warnings

import numpy as np
import pandas as pd

from statsmodels.tsa.statespace.sarimax import SARIMAX


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "visuelle2"
)

RESULTS_DIR = (
    PROJECT_ROOT
    / "results"
)


# ============================================================
# SETTINGS
# ============================================================

TRAIN_PATH = DATA_DIR / "stfore_train.csv"
TEST_PATH = DATA_DIR / "stfore_test.csv"

OUTPUT_PATH = (
    RESULTS_DIR
    / "sarima_predictions.csv"
)


# ============================================================
# DATA COLUMNS
# ============================================================

HISTORY_COLUMNS = [
    str(i)
    for i in range(11)
]


# ============================================================
# SARIMA FORECAST FUNCTION
# ============================================================

def forecast_sarima(history):
    """
    Forecast the next demand period using a small SARIMA model.

    The Visuelle 2.0 records used here contain 11 historical
    demand periods, so a lightweight model is used.

    Primary model:

        SARIMA(1,0,0) x (1,0,0,2)

    If the model cannot be fitted because of a numerical or
    convergence problem, a simpler ARIMA-style fallback is used.
    """

    history = np.asarray(
        history,
        dtype=float
    )

    history = np.nan_to_num(
        history,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    history = np.maximum(
        history,
        0.0
    )

    # --------------------------------------------------------
    # Constant / all-zero series
    # --------------------------------------------------------

    if len(history) == 0:
        return 0.0

    if np.allclose(history, 0.0):
        return 0.0

    if np.std(history) < 1e-8:
        return float(
            max(history[-1], 0.0)
        )

    # --------------------------------------------------------
    # Primary SARIMA model
    # --------------------------------------------------------

    try:

        with warnings.catch_warnings():

            warnings.simplefilter(
                "ignore"
            )

            model = SARIMAX(
                history,

                order=(
                    1,
                    0,
                    0
                ),

                seasonal_order=(
                    1,
                    0,
                    0,
                    2
                ),

                trend="c",

                enforce_stationarity=False,

                enforce_invertibility=False
            )

            fitted_model = model.fit(
                disp=False,
                maxiter=50
            )

            forecast = fitted_model.forecast(
                steps=1
            )

            prediction = float(
                forecast[0]
            )

            if np.isfinite(prediction):

                return max(
                    prediction,
                    0.0
                )

    except Exception:
        pass


    # --------------------------------------------------------
    # Fallback: simple ARIMA(1,0,0)
    # --------------------------------------------------------

    try:

        with warnings.catch_warnings():

            warnings.simplefilter(
                "ignore"
            )

            model = SARIMAX(
                history,

                order=(
                    1,
                    0,
                    0
                ),

                seasonal_order=(
                    0,
                    0,
                    0,
                    0
                ),

                trend="c",

                enforce_stationarity=False,

                enforce_invertibility=False
            )

            fitted_model = model.fit(
                disp=False,
                maxiter=50
            )

            forecast = fitted_model.forecast(
                steps=1
            )

            prediction = float(
                forecast[0]
            )

            if np.isfinite(prediction):

                return max(
                    prediction,
                    0.0
                )

    except Exception:
        pass


    # --------------------------------------------------------
    # Final fallback: last observed demand
    # --------------------------------------------------------

    return max(
        float(history[-1]),
        0.0
    )


# ============================================================
# MAIN SARIMA PIPELINE
# ============================================================

def run_sarima():

    print("=" * 70)
    print("SARIMA DEMAND FORECASTING")
    print("=" * 70)

    start_time = time.time()

    # --------------------------------------------------------
    # Load datasets
    # --------------------------------------------------------

    print("\nLoading datasets...")

    train_df = pd.read_csv(
        TRAIN_PATH
    )

    test_df = pd.read_csv(
        TEST_PATH
    )

    print(
        f"Training shape: {train_df.shape}"
    )

    print(
        f"Testing shape : {test_df.shape}"
    )

    # --------------------------------------------------------
    # Validate history columns
    # --------------------------------------------------------

    missing_columns = [
        column
        for column in HISTORY_COLUMNS
        if column not in test_df.columns
    ]

    if missing_columns:

        raise ValueError(
            "Missing historical demand columns: "
            f"{missing_columns}"
        )

    # --------------------------------------------------------
    # Extract test history
    # --------------------------------------------------------

    histories = (
        test_df[
            HISTORY_COLUMNS
        ]
        .astype(float)
        .to_numpy()
    )

    print(
        f"\nForecasting {len(histories):,} "
        "product-store records..."
    )

    # --------------------------------------------------------
    # Generate forecasts
    # --------------------------------------------------------

    predictions = []

    total_rows = len(histories)

    for index, history in enumerate(
        histories
    ):

        prediction = forecast_sarima(
            history
        )

        predictions.append(
            prediction
        )

        # Progress display
        if (
            (index + 1) % 500 == 0
            or index == 0
            or index == total_rows - 1
        ):

            elapsed = (
                time.time()
                - start_time
            )

            print(
                f"Processed "
                f"{index + 1:,}/{total_rows:,} "
                f"records | "
                f"Elapsed: {elapsed:.1f}s"
            )

    predictions = np.asarray(
        predictions,
        dtype=float
    )

    predictions = np.nan_to_num(
        predictions,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    predictions = np.maximum(
        predictions,
        0.0
    )

    # --------------------------------------------------------
    # Actual values
    # --------------------------------------------------------

    actual = (
        test_df["11"]
        .astype(float)
        .to_numpy()
    )

    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    prediction_output = pd.DataFrame(
        {
            "prediction": predictions,
            "actual": actual
        }
    )

    prediction_output.to_csv(
        OUTPUT_PATH,
        index=False
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    errors = (
        predictions
        - actual
    )

    mae = float(
        np.mean(
            np.abs(errors)
        )
    )

    rmse = float(
        np.sqrt(
            np.mean(
                errors ** 2
            )
        )
    )

    non_zero_actual = (
        np.abs(actual) > 1e-8
    )

    if np.any(non_zero_actual):

        mape = float(
            np.mean(
                np.abs(
                    (
                        actual[non_zero_actual]
                        -
                        predictions[non_zero_actual]
                    )
                    /
                    actual[non_zero_actual]
                )
            )
            * 100
        )

    else:

        mape = 0.0

    elapsed = (
        time.time()
        - start_time
    )

    # --------------------------------------------------------
    # Results
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("SARIMA RESULTS")
    print("=" * 70)

    print(
        f"MAE : {mae:.6f}"
    )

    print(
        f"RMSE: {rmse:.6f}"
    )

    print(
        f"MAPE: {mape:.2f}%"
    )

    print(
        f"Training/forecasting time: "
        f"{elapsed:.2f} seconds"
    )

    print(
        f"\nPredictions saved to:"
    )

    print(
        OUTPUT_PATH
    )

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    run_sarima()