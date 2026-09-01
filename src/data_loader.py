from pathlib import Path
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data" / "raw" / "visuelle2"


def load_csv(filename):
    path = DATA_DIR / filename

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset file not found:\n{path}"
        )

    df = pd.read_csv(path)

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    return df


def load_sales_data():
    return load_csv("sales.csv")


def load_train():
    return load_csv("stfore_train.csv")


def load_test():
    return load_csv("stfore_test.csv")


def get_sales_columns(df):
    return [
        str(i)
        for i in range(12)
        if str(i) in df.columns
    ]


if __name__ == "__main__":

    train = load_train()
    test = load_test()

    print("Visuelle 2.0 Dataset")
    print("=" * 50)

    print(f"Training shape: {train.shape}")
    print(f"Testing shape : {test.shape}")

    print("\nSales columns:")
    print(get_sales_columns(train))