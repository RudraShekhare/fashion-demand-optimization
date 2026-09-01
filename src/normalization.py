from pathlib import Path
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]

SCALAR_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "visuelle2"
    / "stfore_sales_norm_scalar.npy"
)


def load_normalization_scalar():
    """
    Load the Visuelle 2.0 sales normalization scalar.

    The dataset stores normalized sales values.
    For this dataset:

        actual_sales = normalized_sales * 53
    """

    if not SCALAR_PATH.exists():
        raise FileNotFoundError(
            f"Normalization scalar not found:\n{SCALAR_PATH}"
        )

    scalar = np.load(SCALAR_PATH, allow_pickle=True)

    if np.asarray(scalar).size != 1:
        raise ValueError(
            f"Expected a single normalization scalar, "
            f"but found shape {np.asarray(scalar).shape}"
        )

    return float(np.asarray(scalar).reshape(-1)[0])


def inverse_transform(values):
    """
    Convert normalized sales values into original sales units.

    Example:
        0.05 -> 2.65 units

    Negative predictions are clipped to zero because
    negative physical demand is not meaningful.
    """

    scalar = load_normalization_scalar()

    values = np.asarray(values, dtype=float)

    values = np.nan_to_num(
        values,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    original_values = values * scalar

    return np.maximum(original_values, 0.0)


def inverse_transform_and_round(values):
    """
    Convert normalized forecasts to practical whole-unit quantities.

    Example:
        2.65 -> 3 units
    """

    converted = inverse_transform(values)

    return np.rint(converted).astype(int)


def inspect_scalar():

    scalar = load_normalization_scalar()

    print("=" * 60)
    print("VISUELLE 2.0 SALES NORMALIZATION")
    print("=" * 60)

    print(f"File: {SCALAR_PATH}")
    print(f"Normalization scalar: {scalar}")

    print("\nConversion:")
    print(f"0.01 normalized -> {0.01 * scalar:.2f} sales units")
    print(f"0.02 normalized -> {0.02 * scalar:.2f} sales units")
    print(f"0.05 normalized -> {0.05 * scalar:.2f} sales units")

    print("\nFormula:")
    print("actual_sales = normalized_sales × normalization_scalar")


if __name__ == "__main__":
    inspect_scalar()