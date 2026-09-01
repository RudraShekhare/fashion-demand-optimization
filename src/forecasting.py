import numpy as np


SALES_COLUMNS = [
    str(i)
    for i in range(12)
]


def get_sales_matrix(df):

    missing = [
        column
        for column in SALES_COLUMNS
        if column not in df.columns
    ]

    if missing:

        raise ValueError(
            f"Missing sales columns: {missing}"
        )

    return df[
        SALES_COLUMNS
    ].astype(float).to_numpy()


def get_sales_history(
    df,
    row_index
):

    if (
        row_index < 0
        or row_index >= len(df)
    ):

        raise IndexError(
            "Invalid row index."
        )

    return (
        df.loc[
            row_index,
            SALES_COLUMNS
        ]
        .astype(float)
        .to_numpy()
    )


def naive_forecast(history):

    history = np.asarray(
        history,
        dtype=float
    )

    if len(history) == 0:
        raise ValueError(
            "History cannot be empty."
        )

    return float(
        history[-1]
    )


def mean_forecast(history):

    history = np.asarray(
        history,
        dtype=float
    )

    if len(history) == 0:
        raise ValueError(
            "History cannot be empty."
        )

    return float(
        np.mean(history)
    )


def create_lag_features(df):

    result = df.copy()

    result["lag_1"] = result["10"]
    result["lag_2"] = result["9"]
    result["lag_3"] = result["8"]

    result["mean_3"] = result[
        ["8", "9", "10"]
    ].mean(axis=1)

    result["mean_6"] = result[
        [
            "5",
            "6",
            "7",
            "8",
            "9",
            "10"
        ]
    ].mean(axis=1)

    result["max_demand"] = result[
        [str(i) for i in range(11)]
    ].max(axis=1)

    result["total_demand"] = result[
        [str(i) for i in range(11)]
    ].sum(axis=1)

    result["std_demand"] = result[
        [str(i) for i in range(11)]
    ].std(axis=1)

    return result