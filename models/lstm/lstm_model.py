import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(
    0,
    str(PROJECT_ROOT / "src")
)

import time
import random

import numpy as np
import pandas as pd

import torch
import torch.nn as nn

from torch.utils.data import (
    TensorDataset,
    DataLoader
)

from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error
)

from data_loader import (
    load_train,
    load_test
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

SEED = 42

random.seed(SEED)

np.random.seed(SEED)

torch.manual_seed(SEED)


# ============================================================
# PATHS
# ============================================================

RESULTS_DIR = (
    PROJECT_ROOT /
    "results"
)

MODEL_DIR = (
    PROJECT_ROOT /
    "models" /
    "lstm"
)

RESULTS_DIR.mkdir(
    parents=True,
    exist_ok=True
)

MODEL_DIR.mkdir(
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

BATCH_SIZE = 4096

MAX_EPOCHS = 20

PATIENCE = 3

LEARNING_RATE = 0.001

HIDDEN_SIZE = 16

VALIDATION_RATIO = 0.10


# ============================================================
# DEVICE
# ============================================================

if torch.backends.mps.is_available():

    DEVICE = torch.device("mps")

elif torch.cuda.is_available():

    DEVICE = torch.device("cuda")

else:

    DEVICE = torch.device("cpu")


print("=" * 65)
print("VISUELLE 2.0 - EFFICIENT LSTM")
print("=" * 65)

print(
    f"\nDevice: {DEVICE}"
)

print(
    f"PyTorch: {torch.__version__}"
)


# ============================================================
# LOAD DATA
# ============================================================

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
# PREPARE DATA
# ============================================================

X_all = (
    train_df[
        HISTORY_COLUMNS
    ]
    .to_numpy(
        dtype=np.float32
    )
)

y_all = (
    train_df[
        TARGET_COLUMN
    ]
    .to_numpy(
        dtype=np.float32
    )
)


X_all = X_all.reshape(
    X_all.shape[0],
    X_all.shape[1],
    1
)


# ============================================================
# TRAIN / VALIDATION SPLIT
# ============================================================

split_index = int(
    len(X_all)
    *
    (1 - VALIDATION_RATIO)
)


X_train = X_all[
    :split_index
]

y_train = y_all[
    :split_index
]

X_val = X_all[
    split_index:
]

y_val = y_all[
    split_index:
]


print(
    f"\nTraining samples: "
    f"{len(X_train):,}"
)

print(
    f"Validation samples: "
    f"{len(X_val):,}"
)


# ============================================================
# TORCH DATA
# ============================================================

X_train_tensor = torch.from_numpy(
    X_train
)

y_train_tensor = torch.from_numpy(
    y_train
).unsqueeze(1)


X_val_tensor = torch.from_numpy(
    X_val
)

y_val_tensor = torch.from_numpy(
    y_val
).unsqueeze(1)


train_dataset = TensorDataset(
    X_train_tensor,
    y_train_tensor
)

val_dataset = TensorDataset(
    X_val_tensor,
    y_val_tensor
)


train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False
)


# ============================================================
# MODEL
# ============================================================

class LSTMModel(nn.Module):

    def __init__(self):

        super().__init__()

        self.lstm = nn.LSTM(
            input_size=1,
            hidden_size=HIDDEN_SIZE,
            num_layers=1,
            batch_first=True
        )

        self.fc = nn.Linear(
            HIDDEN_SIZE,
            1
        )

    def forward(self, x):

        output, _ = self.lstm(x)

        last_output = (
            output[:, -1, :]
        )

        return self.fc(
            last_output
        )


model = LSTMModel().to(
    DEVICE
)


parameter_count = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)


print("\nModel:")

print(model)

print(
    f"\nTrainable parameters: "
    f"{parameter_count:,}"
)


# ============================================================
# LOSS / OPTIMIZER
# ============================================================

criterion = nn.MSELoss()

optimizer = torch.optim.Adam(
    model.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# TRAINING
# ============================================================

print("\n" + "=" * 65)
print("TRAINING")
print("=" * 65)

best_val_loss = float("inf")

best_state = None

best_epoch = 0

patience_counter = 0

training_start = time.time()


for epoch in range(
    MAX_EPOCHS
):

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    model.train()

    train_loss = 0.0

    for X_batch, y_batch in train_loader:

        X_batch = X_batch.to(
            DEVICE
        )

        y_batch = y_batch.to(
            DEVICE
        )

        optimizer.zero_grad(
            set_to_none=True
        )

        prediction = model(
            X_batch
        )

        loss = criterion(
            prediction,
            y_batch
        )

        loss.backward()

        optimizer.step()

        train_loss += (
            loss.item()
            *
            X_batch.size(0)
        )


    train_loss /= len(
        train_dataset
    )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    model.eval()

    validation_loss = 0.0

    with torch.inference_mode():

        for X_batch, y_batch in val_loader:

            X_batch = X_batch.to(
                DEVICE
            )

            y_batch = y_batch.to(
                DEVICE
            )

            prediction = model(
                X_batch
            )

            loss = criterion(
                prediction,
                y_batch
            )

            validation_loss += (
                loss.item()
                *
                X_batch.size(0)
            )


    validation_loss /= len(
        val_dataset
    )


    elapsed = (
        time.time()
        -
        training_start
    )


    print(
        f"Epoch {epoch + 1:02d}/"
        f"{MAX_EPOCHS} | "
        f"Train Loss: "
        f"{train_loss:.6f} | "
        f"Val Loss: "
        f"{validation_loss:.6f} | "
        f"Time: "
        f"{elapsed:.2f}s"
    )


    # --------------------------------------------------------
    # BEST MODEL
    # --------------------------------------------------------

    if (
        validation_loss
        <
        best_val_loss
    ):

        best_val_loss = (
            validation_loss
        )

        best_epoch = (
            epoch + 1
        )

        patience_counter = 0

        best_state = {
            key: value.detach()
            .cpu()
            .clone()

            for key, value
            in model.state_dict().items()
        }

    else:

        patience_counter += 1

        if (
            patience_counter
            >= PATIENCE
        ):

            print(
                "\nEarly stopping."
            )

            break


training_time = (
    time.time()
    -
    training_start
)


# ============================================================
# RESTORE BEST MODEL
# ============================================================

model.load_state_dict(
    best_state
)


print("\n" + "=" * 65)
print("TRAINING COMPLETE")
print("=" * 65)

print(
    f"Best epoch: {best_epoch}"
)

print(
    f"Best validation loss: "
    f"{best_val_loss:.6f}"
)

print(
    f"Training time: "
    f"{training_time:.2f} seconds"
)


# ============================================================
# TEST DATA
# ============================================================

X_test = (
    test_df[
        HISTORY_COLUMNS
    ]
    .to_numpy(
        dtype=np.float32
    )
)

y_test = (
    test_df[
        TARGET_COLUMN
    ]
    .to_numpy(
        dtype=np.float32
    )
)


X_test = X_test.reshape(
    X_test.shape[0],
    X_test.shape[1],
    1
)


X_test_tensor = torch.from_numpy(
    X_test
)


# ============================================================
# PREDICTIONS
# ============================================================

print(
    "\nGenerating predictions..."
)

model.eval()

prediction_batches = []


with torch.inference_mode():

    for start in range(
        0,
        len(X_test_tensor),
        BATCH_SIZE
    ):

        end = (
            start +
            BATCH_SIZE
        )

        X_batch = (
            X_test_tensor[start:end]
            .to(DEVICE)
        )

        prediction = model(
            X_batch
        )

        prediction_batches.append(
            prediction
            .cpu()
            .numpy()
        )


predictions = np.concatenate(
    prediction_batches
).flatten()


predictions = np.maximum(
    predictions,
    0
)


# ============================================================
# METRICS
# ============================================================

actual = y_test


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


non_zero = (
    actual != 0
)


if np.any(non_zero):

    mape = (
        np.mean(
            np.abs(
                (
                    actual[non_zero]
                    -
                    predictions[non_zero]
                )
                /
                actual[non_zero]
            )
        )
        *
        100
    )

else:

    mape = np.nan


print("\n" + "=" * 65)
print("LSTM RESULTS")
print("=" * 65)

print(
    f"MAE  : {mae:.6f}"
)

print(
    f"RMSE : {rmse:.6f}"
)

print(
    f"MAPE : {mape:.2f}%"
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
    actual
)

prediction_df["prediction"] = (
    predictions
)


prediction_df.to_csv(
    RESULTS_DIR /
    "lstm_predictions.csv",
    index=False
)


# ============================================================
# SAVE MODEL
# ============================================================

torch.save(
    {
        "model_state_dict":
            model.state_dict(),

        "hidden_size":
            HIDDEN_SIZE,

        "best_epoch":
            best_epoch,

        "validation_loss":
            best_val_loss
    },

    MODEL_DIR /
    "lstm_model.pt"
)


# ============================================================
# SAVE METRICS
# ============================================================

pd.DataFrame([
    {
        "Model": "LSTM",

        "MAE": mae,

        "RMSE": rmse,

        "MAPE": mape,

        "Training_Time_Seconds":
            training_time,

        "Best_Epoch":
            best_epoch
    }
]).to_csv(
    RESULTS_DIR /
    "lstm_metrics.csv",
    index=False
)


print(
    "\nPredictions saved to:"
)

print(
    RESULTS_DIR /
    "lstm_predictions.csv"
)

print(
    "\nModel saved to:"
)

print(
    MODEL_DIR /
    "lstm_model.pt"
)

print(
    "\nLSTM completed successfully."
)