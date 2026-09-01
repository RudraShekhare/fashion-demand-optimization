import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = (
    Path(__file__)
    .resolve()
    .parents[1]
)


def run_script(
    script_path
):

    print("\n")
    print("=" * 70)
    print(
        f"RUNNING: {script_path}"
    )
    print("=" * 70)

    result = subprocess.run(
        [
            sys.executable,
            str(
                PROJECT_ROOT /
                script_path
            )
        ]
    )

    if result.returncode != 0:

        print(
            f"\nERROR while running:"
            f" {script_path}"
        )

        sys.exit(
            result.returncode
        )


def main():

    print("=" * 70)
    print(
        "FASHION DEMAND FORECASTING "
        "FULL PIPELINE"
    )
    print("=" * 70)


    # --------------------------------------------------------
    # Baseline
    # --------------------------------------------------------

    run_script(
        "models/baseline/"
        "baseline_model.py"
    )


    # --------------------------------------------------------
    # XGBoost
    # --------------------------------------------------------

    run_script(
        "models/xgboost/"
        "xgboost_model.py"
    )


    # --------------------------------------------------------
    # LSTM
    # --------------------------------------------------------

    run_script(
        "models/lstm/"
        "lstm_model.py"
    )


    # --------------------------------------------------------
    # SARIMA
    # --------------------------------------------------------

    run_script(
        "models/sarima/"
        "sarima_model.py"
    )


    # --------------------------------------------------------
    # Comparison
    # --------------------------------------------------------

    run_script(
        "src/run_comparison.py"
    )


    print("\n")
    print("=" * 70)
    print(
        "FULL PIPELINE COMPLETED"
    )
    print("=" * 70)

    print(
        "\nResults are available in:"
    )

    print(
        PROJECT_ROOT /
        "results"
    )


if __name__ == "__main__":

    main()