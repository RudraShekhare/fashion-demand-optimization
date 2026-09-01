from pathlib import Path
import pandas as pd

from data_loader import load_sales_data, get_sales_columns


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = (
    PROJECT_ROOT /
    "data" /
    "processed"
)


def prepare_sales_data(df):

    sales_columns = get_sales_columns(df)

    id_columns = [
        "external_code",
        "retail",
        "season",
        "category",
        "color",
        "image_path",
        "fabric",
        "release_date",
        "restock"
    ]

    id_columns = [
        column
        for column in id_columns
        if column in df.columns
    ]

    long_df = df.melt(
        id_vars=id_columns,
        value_vars=sales_columns,
        var_name="time",
        value_name="sales"
    )

    long_df["time"] = pd.to_numeric(
        long_df["time"],
        errors="coerce"
    )

    long_df["sales"] = pd.to_numeric(
        long_df["sales"],
        errors="coerce"
    )

    if "release_date" in long_df.columns:

        long_df["release_date"] = pd.to_datetime(
            long_df["release_date"],
            errors="coerce"
        )

    long_df = long_df.dropna(
        subset=["time", "sales"]
    )

    sort_columns = []

    if "external_code" in long_df.columns:
        sort_columns.append("external_code")

    if "retail" in long_df.columns:
        sort_columns.append("retail")

    sort_columns.append("time")

    long_df = long_df.sort_values(
        sort_columns
    ).reset_index(drop=True)

    return long_df


def save_processed_data(df):

    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = (
        PROCESSED_DIR /
        "sales_processed.csv"
    )

    df.to_csv(
        output_path,
        index=False
    )

    print(
        f"Processed data saved to:\n{output_path}"
    )

    return output_path


def preprocess_sales():

    print("Loading Visuelle 2.0 sales data...")

    df = load_sales_data()

    print(
        f"Original shape: {df.shape}"
    )

    print(
        "\nConverting sales data "
        "to long format..."
    )

    processed_df = prepare_sales_data(df)

    print(
        f"Processed shape: {processed_df.shape}"
    )

    save_processed_data(
        processed_df
    )

    return processed_df


if __name__ == "__main__":

    preprocess_sales()