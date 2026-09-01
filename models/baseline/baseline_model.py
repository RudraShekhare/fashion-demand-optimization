import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(
    0,
    str(PROJECT_ROOT / "src")
)

import numpy as np
import pandas as pd

from statsmodels.tsa.holtwinters import (
    SimpleExpSmoothing
)

from data_loader import load_test
from evaluation import calculate_metrics


RESULTS_DIR = (
    PROJECT_ROOT /
    "results"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


print("=" * 60)
print("VISUELLE 2.0 BASELINE MODELS")
print("=" * 60)


test_df = load_test()

history_columns = [
    str(i)
    for i in range(11)
]

actual = (
    test_df["11"]
    .astype(float)
    .to_numpy()
)

history = (
    test_df[history_columns]
    .astype(float)
    .to_numpy()
)


# ============================================================
# NAIVE
# ============================================================

print("\nRunning Naive baseline...")

naive_predictions = (
    history[:, -1]
)

naive_predictions = np.maximum(
    naive_predictions,
    0
)

naive_metrics = calculate_metrics(
    actual,
    naive_predictions
)

print("\nNAIVE RESULTS")

print(
    f"MAE  : {naive_metrics['MAE']:.6f}"
)

print(
    f"RMSE : {naive_metrics['RMSE']:.6f}"
)

print(
    f"MAPE : {naive_metrics['MAPE']:.2f}%"
)


pd.DataFrame({
    "external_code":
        test_df["external_code"],
    "retail":
        test_df["retail"],
    "season":
        test_df["season"],
    "category":
        test_df["category"],
    "actual":
        actual,
    "prediction":
        naive_predictions
}).to_csv(
    RESULTS_DIR /
    "naive_predictions.csv",
    index=False
)


pd.DataFrame([
    {
        "Model": "Naive",
        **naive_metrics
    }
]).to_csv(
    RESULTS_DIR /
    "naive_metrics.csv",
    index=False
)


# ============================================================
# SES
# ============================================================

print("\nRunning SES baseline...")

ses_predictions = []

for row in history:

    try:

        model = SimpleExpSmoothing(
            row,
            initialization_method="estimated"
        )

        fitted = model.fit(
            optimized=True
        )

        forecast = fitted.forecast(
            1
        )[0]

    except Exception:

        forecast = row[-1]

    ses_predictions.append(
        max(float(forecast), 0)
    )


ses_predictions = np.asarray(
    ses_predictions
)


ses_metrics = calculate_metrics(
    actual,
    ses_predictions
)


print("\nSES RESULTS")

print(
    f"MAE  : {ses_metrics['MAE']:.6f}"
)

print(
    f"RMSE : {ses_metrics['RMSE']:.6f}"
)

print(
    f"MAPE : {ses_metrics['MAPE']:.2f}%"
)


pd.DataFrame({
    "external_code":
        test_df["external_code"],
    "retail":
        test_df["retail"],
    "season":
        test_df["season"],
    "category":
        test_df["category"],
    "actual":
        actual,
    "prediction":
        ses_predictions
}).to_csv(
    RESULTS_DIR /
    "ses_predictions.csv",
    index=False
)


pd.DataFrame([
    {
        "Model": "SES",
        **ses_metrics
    }
]).to_csv(
    RESULTS_DIR /
    "ses_metrics.csv",
    index=False
)


print("\nBaseline models completed.")