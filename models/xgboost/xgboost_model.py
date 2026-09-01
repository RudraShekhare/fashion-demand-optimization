import sys
from pathlib import Path
import time

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(
    0,
    str(PROJECT_ROOT / "src")
)

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from xgboost import XGBRegressor

from data_loader import load_train, load_test
from evaluation import calculate_metrics


# ============================================================
# PATHS
# ============================================================

RESULTS_DIR = (
    PROJECT_ROOT /
    "results"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# ============================================================
# SETTINGS
# ============================================================

HISTORY_COLUMNS = [
    str(i)
    for i in range(11)
]

TARGET_COLUMN = "11"


# ============================================================
# FEATURE ENGINEERING
# ============================================================

def create_features(df):

    result = df.copy()

    # --------------------------------------------------------
    # Remove target if present
    # --------------------------------------------------------

    result = result.drop(
        columns=[TARGET_COLUMN],
        errors="ignore"
    )


    # --------------------------------------------------------
    # Release date
    # --------------------------------------------------------

    if "release_date" in result.columns:

        release_date = pd.to_datetime(
            result["release_date"],
            errors="coerce"
        )

        result["release_year"] = (
            release_date.dt.year
        )

        result["release_month"] = (
            release_date.dt.month
        )

        result["release_day"] = (
            release_date.dt.day
        )

        result = result.drop(
            columns=["release_date"]
        )


    # --------------------------------------------------------
    # Image path is not used as a feature
    # --------------------------------------------------------

    if "image_path" in result.columns:

        result = result.drop(
            columns=["image_path"]
        )


    # --------------------------------------------------------
    # Historical demand features
    # --------------------------------------------------------

    result["mean_demand"] = (
        result[HISTORY_COLUMNS]
        .mean(axis=1)
    )

    result["std_demand"] = (
        result[HISTORY_COLUMNS]
        .std(axis=1)
        .fillna(0)
    )

    result["min_demand"] = (
        result[HISTORY_COLUMNS]
        .min(axis=1)
    )

    result["max_demand"] = (
        result[HISTORY_COLUMNS]
        .max(axis=1)
    )

    result["total_demand"] = (
        result[HISTORY_COLUMNS]
        .sum(axis=1)
    )

    # Recent demand

    result["recent_mean_3"] = (
        result[
            ["8", "9", "10"]
        ]
        .mean(axis=1)
    )

    result["recent_mean_5"] = (
        result[
            [
                "6",
                "7",
                "8",
                "9",
                "10"
            ]
        ]
        .mean(axis=1)
    )

    # Demand trend

    result["trend"] = (
        result["10"]
        -
        result["0"]
    )


    return result


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 65)
print("VISUELLE 2.0 - XGBOOST FORECASTING")
print("=" * 65)


print("\nLoading datasets...")

train_df = load_train()
test_df = load_test()


print(
    f"Training shape: {train_df.shape}"
)

print(
    f"Testing shape : {test_df.shape}"
)


# ============================================================
# CREATE FEATURES
# ============================================================

print("\nCreating features...")

X_train = create_features(
    train_df
)

X_test = create_features(
    test_df
)


y_train = (
    train_df[
        TARGET_COLUMN
    ]
    .astype(float)
    .to_numpy()
)

y_test = (
    test_df[
        TARGET_COLUMN
    ]
    .astype(float)
    .to_numpy()
)


print(
    f"X_train shape before encoding: "
    f"{X_train.shape}"
)

print(
    f"X_test shape before encoding : "
    f"{X_test.shape}"
)


# ============================================================
# EXPLICIT COLUMN TYPES
# ============================================================

# Do NOT rely on pandas dtype detection here.
# These columns are explicitly categorical.

CATEGORICAL_COLUMNS = [
    "season",
    "category",
    "color",
    "fabric"
]


CATEGORICAL_COLUMNS = [
    column
    for column in CATEGORICAL_COLUMNS
    if column in X_train.columns
]


NUMERICAL_COLUMNS = [
    column
    for column in X_train.columns
    if column not in CATEGORICAL_COLUMNS
]


print("\nCategorical features:")

print(
    CATEGORICAL_COLUMNS
)

print(
    f"Number of categorical features: "
    f"{len(CATEGORICAL_COLUMNS)}"
)

print(
    f"Number of numerical features: "
    f"{len(NUMERICAL_COLUMNS)}"
)


# ============================================================
# FORCE CORRECT DATA TYPES
# ============================================================

# Numerical columns must actually be numeric.

for column in NUMERICAL_COLUMNS:

    X_train[column] = pd.to_numeric(
        X_train[column],
        errors="coerce"
    )

    X_test[column] = pd.to_numeric(
        X_test[column],
        errors="coerce"
    )


# Categorical columns are explicitly converted to strings.

for column in CATEGORICAL_COLUMNS:

    X_train[column] = (
        X_train[column]
        .astype("string")
    )

    X_test[column] = (
        X_test[column]
        .astype("string")
    )


# ============================================================
# PREPROCESSING
# ============================================================

numeric_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(
            strategy="median"
        )
    )
])


categorical_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(
            strategy="most_frequent"
        )
    ),

    (
        "onehot",
        OneHotEncoder(
            handle_unknown="ignore",
            sparse_output=True
        )
    )
])


preprocessor = ColumnTransformer(

    transformers=[

        (
            "numeric",
            numeric_pipeline,
            NUMERICAL_COLUMNS
        ),

        (
            "categorical",
            categorical_pipeline,
            CATEGORICAL_COLUMNS
        )
    ],

    remainder="drop"
)


# ============================================================
# XGBOOST MODEL
# ============================================================

model = XGBRegressor(

    n_estimators=300,

    max_depth=6,

    learning_rate=0.05,

    subsample=0.8,

    colsample_bytree=0.8,

    objective="reg:squarederror",

    random_state=42,

    n_jobs=-1
)


# ============================================================
# PIPELINE
# ============================================================

pipeline = Pipeline([

    (
        "preprocessor",
        preprocessor
    ),

    (
        "model",
        model
    )
])


# ============================================================
# TRAIN
# ============================================================

print("\nTraining XGBoost...")

start_time = time.time()


pipeline.fit(
    X_train,
    y_train
)


training_time = (
    time.time()
    -
    start_time
)


# ============================================================
# PREDICTION
# ============================================================

print("\nGenerating predictions...")

predictions = pipeline.predict(
    X_test
)


# Demand cannot be negative.

predictions = np.maximum(
    predictions,
    0
)


# ============================================================
# EVALUATION
# ============================================================

metrics = calculate_metrics(
    y_test,
    predictions
)


print("\n" + "=" * 65)
print("XGBOOST RESULTS")
print("=" * 65)


print(
    f"MAE  : {metrics['MAE']:.6f}"
)

print(
    f"RMSE : {metrics['RMSE']:.6f}"
)

print(
    f"MAPE : {metrics['MAPE']:.2f}%"
)

print(
    f"Training time: "
    f"{training_time:.2f} seconds"
)


# ============================================================
# SAVE PREDICTIONS
# ============================================================

prediction_df = test_df[
    [
        "external_code",
        "retail",
        "season",
        "category"
    ]
].copy()


prediction_df["actual"] = (
    y_test
)

prediction_df["prediction"] = (
    predictions
)


prediction_path = (
    RESULTS_DIR /
    "xgboost_predictions.csv"
)


prediction_df.to_csv(
    prediction_path,
    index=False
)


# ============================================================
# SAVE METRICS
# ============================================================

metrics_df = pd.DataFrame([

    {
        "Model": "XGBoost",

        "MAE": metrics["MAE"],

        "RMSE": metrics["RMSE"],

        "MAPE": metrics["MAPE"],

        "Training_Time_Seconds":
            training_time
    }

])


metrics_path = (
    RESULTS_DIR /
    "xgboost_metrics.csv"
)


metrics_df.to_csv(
    metrics_path,
    index=False
)


# ============================================================
# COMPLETE
# ============================================================

print(
    "\nPredictions saved to:"
)

print(
    prediction_path
)

print(
    "\nMetrics saved to:"
)

print(
    metrics_path
)

print(
    "\nXGBoost completed successfully."
)