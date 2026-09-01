from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(
    __file__
).resolve().parents[1]


RESULTS_DIR = (
    PROJECT_ROOT /
    "results"
)


metric_files = [

    "naive_metrics.csv",

    "ses_metrics.csv",

    "xgboost_metrics.csv",

    "lstm_metrics.csv"
]


results = []


for filename in metric_files:

    path = (
        RESULTS_DIR /
        filename
    )

    if path.exists():

        df = pd.read_csv(
            path
        )

        results.append(df)


if not results:

    raise FileNotFoundError(
        "No model metric files found."
    )


comparison = pd.concat(
    results,
    ignore_index=True
)


comparison = comparison.sort_values(
    "MAE"
).reset_index(
    drop=True
)


output_path = (
    RESULTS_DIR /
    "model_comparison.csv"
)


comparison.to_csv(
    output_path,
    index=False
)


print("=" * 60)
print("MODEL COMPARISON")
print("=" * 60)

print(
    comparison.to_string(
        index=False
    )
)


print(
    "\nComparison saved to:"
)

print(
    output_path
)