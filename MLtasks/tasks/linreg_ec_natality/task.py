"""
Linear Regression on BigQuery Natality Dataset  (v4)
=====================================================
Predicts infant birth weight (weight_pounds) from polynomial-expanded and
interaction features loaded from the BigQuery public natality dataset.

Artifact outputs
  ├── model.pth                                      PyTorch weights
  ├── scaler.pkl                                     Fitted StandardScaler
  ├── metrics.json                                   All split metrics
  ├── training_history.json                          Per-epoch losses + LR
  ├── linreg_bq_natality_predictions_vs_actual.png   Scatter + training curve
  └── linreg_bq_natality_residuals.png               Residual distribution

Run:
    python task_linreg_natality.py

Requirements:
    pip install torch scikit-learn google-cloud-bigquery db-dtypes pandas numpy matplotlib

Authentication:
    gcloud auth application-default login
"""

import json
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from typing import Dict, Tuple, Any
import warnings

warnings.filterwarnings("ignore")

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Configuration
PROJECT_ID = "cmpe-188-hw1-ec-491005"
OUTPUT_DIR = "./tasks/linreg_ec_natality/output"

_RAW_COLS = [
    "mother_age",
    "gestation_weeks",
    "weight_gain_pounds",
    "plurality",
    "ever_born",
    "cigarette_use",
    "alcohol_use",
]
TARGET_COL = "weight_pounds"

FEATURE_COLS = _RAW_COLS + [
    "gestation_weeks_sq",
    "gestation_weeks_cu",
    "gestation_x_weight_gain",
    "gestation_x_plurality",
    "mother_age_x_cigarette_use",
]

BQ_QUERY = """
SELECT
    mother_age,
    gestation_weeks,
    weight_gain_pounds,
    plurality,
    ever_born,
    CAST(cigarette_use AS INT64) AS cigarette_use,
    CAST(alcohol_use   AS INT64) AS alcohol_use,
    weight_pounds
FROM
    `bigquery-public-data.samples.natality`
    TABLESAMPLE SYSTEM (2 PERCENT)
WHERE
    weight_pounds       IS NOT NULL
    AND gestation_weeks     IS NOT NULL
    AND mother_age          IS NOT NULL
    AND weight_gain_pounds  IS NOT NULL
    AND plurality           IS NOT NULL
    AND ever_born           IS NOT NULL
    AND cigarette_use       IS NOT NULL
    AND alcohol_use         IS NOT NULL
    AND gestation_weeks BETWEEN 20  AND 45
    AND weight_pounds   BETWEEN 1.0 AND 15.0
    AND mother_age      BETWEEN 14  AND 50
LIMIT 150000
"""


# Required interface functions
def get_task_metadata() -> Dict[str, Any]:
    return {
        "task_id": "linreg_lvl4_bq_natality",
        "task_name": "linear_regression_bq_natality",
        "series": "Linear Regression",
        "level": 4,
        "algorithm": "Polynomial Linear Regression (Adam + ReduceLROnPlateau + BigQuery Natality)",
        "description": (
            "Predict infant birth weight (weight_pounds) using polynomial-expanded and "
            "interaction features loaded from the BigQuery public natality dataset via "
            "TABLESAMPLE SYSTEM(2 PERCENT)."
        ),
        "data_source": "bigquery-public-data.samples.natality",
        "bq_features_used": [
            "TABLESAMPLE SYSTEM(2 PERCENT) for random block-level sampling",
            "SQL feature selection with CAST (boolean → INT64)",
            "NULL filtering and range constraints in WHERE clause",
            "LIMIT for final row budget control",
        ],
        "features": FEATURE_COLS,
        "target": TARGET_COL,
        "metrics": ["mse", "rmse", "r2", "mae"],
        "quality_thresholds": {"val_r2_min": 0.35, "val_mse_max": 1.8},
    }


def set_seed(seed: int = 42) -> None:
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _load_from_bigquery(project_id: str) -> "pd.DataFrame":
    """
    Submits BQ_QUERY via the google-cloud-bigquery Python client and returns
    a pandas DataFrame. Authentication uses Application Default Credentials.
    TABLESAMPLE SYSTEM(2 PERCENT) performs block-level random sampling in
    BigQuery before any WHERE filtering, avoiding a full 137 M-row scan.
    """
    try:
        from google.cloud import bigquery
    except ImportError as exc:
        raise ImportError("Run: pip install google-cloud-bigquery db-dtypes") from exc

    print(f"   Connecting to BigQuery project: {project_id}")
    client = bigquery.Client(project=project_id)
    print("   Running query (TABLESAMPLE SYSTEM 2%) …")
    df = client.query(BQ_QUERY).to_dataframe()
    print(f"   Fetched {len(df):,} rows from BigQuery.")
    return df


def _engineer_features(df: "pd.DataFrame") -> np.ndarray:
    df = df.copy()
    gw = df["gestation_weeks"]
    df["gestation_weeks_sq"] = gw**2
    df["gestation_weeks_cu"] = gw**3
    df["gestation_x_weight_gain"] = gw * df["weight_gain_pounds"]
    df["gestation_x_plurality"] = gw * df["plurality"]
    df["mother_age_x_cigarette_use"] = df["mother_age"] * df["cigarette_use"]
    return df[FEATURE_COLS].values.astype(np.float32)


def make_dataloaders(
    project_id: str = PROJECT_ID,
    val_size: float = 0.15,
    test_size: float = 0.15,
    batch_size: int = 512,
    random_state: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, StandardScaler, int]:
    df = _load_from_bigquery(project_id)

    X = _engineer_features(df)
    y = df[TARGET_COL].values.astype(np.float32).reshape(-1, 1)
    print(f"   Feature matrix : {X.shape}  ({len(FEATURE_COLS)} features)")

    temp_frac = val_size + test_size
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=temp_frac, random_state=random_state
    )
    relative_val = val_size / temp_frac
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=(1 - relative_val), random_state=random_state
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_val = scaler.transform(X_val)
    X_test = scaler.transform(X_test)

    def _make_loader(X_arr, y_arr, shuffle):
        ds = TensorDataset(
            torch.from_numpy(X_arr.astype(np.float32)),
            torch.from_numpy(y_arr.astype(np.float32)),
        )
        return DataLoader(
            ds, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=False
        )

    return (
        _make_loader(X_train, y_train, shuffle=True),
        _make_loader(X_val, y_val, shuffle=False),
        _make_loader(X_test, y_test, shuffle=False),
        scaler,
        X_train.shape[1],
    )


# Model
class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)
        nn.init.kaiming_uniform_(self.linear.weight, nonlinearity="linear")
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


def build_model(input_dim: int, device: torch.device) -> nn.Module:
    model = LinearRegressionModel(input_dim=input_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"   Model: LinearRegressionModel | input_dim={input_dim} | params={n_params}"
    )
    return model


# Training
def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 120,
    lr: float = 0.05,
    verbose: bool = True,
) -> Dict[str, list]:
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        patience=8,
        factor=0.5,
        min_lr=1e-6,
    )

    history: Dict[str, list] = {
        "train_losses": [],
        "val_losses": [],
        "val_r2": [],
        "lr": [],
    }

    best_val_loss = float("inf")
    best_state = None
    best_epoch = 0

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            loss = criterion(model(X_batch), y_batch)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        train_loss = running_loss / len(train_loader)

        val_metrics = evaluate(model, val_loader, device)
        scheduler.step(val_metrics["mse"])

        if val_metrics["mse"] < best_val_loss:
            best_val_loss = val_metrics["mse"]
            best_epoch = epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        current_lr = optimizer.param_groups[0]["lr"]
        history["train_losses"].append(round(train_loss, 6))
        history["val_losses"].append(round(val_metrics["mse"], 6))
        history["val_r2"].append(round(val_metrics["r2"], 6))
        history["lr"].append(current_lr)

        if verbose and epoch % 10 == 0:
            print(
                f"   Epoch [{epoch:3d}/{epochs}] "
                f"train_mse={train_loss:.4f}  "
                f"val_mse={val_metrics['mse']:.4f}  "
                f"val_r2={val_metrics['r2']:.4f}  "
                f"lr={current_lr:.5f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)
        print(
            f"   Restored best checkpoint (epoch={best_epoch}, val_mse={best_val_loss:.4f})"
        )

    history["best_epoch"] = best_epoch
    history["best_val_loss"] = round(best_val_loss, 6)
    history["early_stopped"] = False

    return history


# Evaluation
def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
) -> Dict[str, float]:
    """
    Evaluates the model on the given DataLoader.
    Returns mse, rmse, r2, mae, predictions, targets.
    """
    model.eval()
    all_preds: list = []
    all_targets: list = []

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            preds = model(X_batch.to(device)).cpu().numpy().flatten()
            targets = y_batch.numpy().flatten()
            all_preds.extend(preds)
            all_targets.extend(targets)

    preds_arr = np.array(all_preds, dtype=np.float32)
    targets_arr = np.array(all_targets, dtype=np.float32)

    mse = float(mean_squared_error(targets_arr, preds_arr))
    rmse = float(np.sqrt(mse))
    r2 = float(r2_score(targets_arr, preds_arr))
    mae = float(mean_absolute_error(targets_arr, preds_arr))

    return {
        "mse": mse,
        "rmse": rmse,
        "r2": r2,
        "mae": mae,
        "predictions": preds_arr,
        "targets": targets_arr,
    }


# Inference
def predict(
    model: nn.Module,
    X_raw: np.ndarray,
    scaler: StandardScaler,
    device: torch.device,
) -> np.ndarray:
    """
    Inference on raw (un-scaled) feature matrix of shape (n, 7).
    Columns: [mother_age, gestation_weeks, weight_gain_pounds, plurality,
               ever_born, cigarette_use, alcohol_use]
    """
    import pandas as pd

    model.eval()
    df_tmp = pd.DataFrame(X_raw, columns=_RAW_COLS)
    X_eng = _engineer_features(df_tmp)
    X_sc = scaler.transform(X_eng).astype(np.float32)
    with torch.no_grad():
        return model(torch.from_numpy(X_sc).to(device)).cpu().numpy().flatten()


# Save artifacts
def save_artifacts(
    model: nn.Module,
    scaler: StandardScaler,
    history: Dict[str, list],
    train_metrics: Dict[str, float],
    val_metrics: Dict[str, float],
    test_metrics: Dict[str, float],
) -> None:
    """
    Saves all artifacts to OUTPUT_DIR:
      model.pth                                  PyTorch state dict
      scaler.pkl                                 Fitted StandardScaler
      metrics.json                               All split metrics
      training_history.json                      Per-epoch losses (matches uploaded schema)
      linreg_bq_natality_predictions_vs_actual.png
      linreg_bq_natality_residuals.png
    """
    import pickle

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # model + scaler
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "model.pth"))
    with open(os.path.join(OUTPUT_DIR, "scaler.pkl"), "wb") as fh:
        pickle.dump(scaler, fh)

    # metrics.json
    _skip = {"predictions", "targets"}

    def _clean(m):
        return {k: round(float(v), 6) for k, v in m.items() if k not in _skip}

    metrics_out = {
        "metadata": {
            "task_name": "linear_regression_bq_natality",
            "task_type": "regression",
            "dataset": "bigquery-public-data.samples.natality",
            "frameworks": ["pytorch"],
            "metrics": ["mse", "rmse", "r2", "mae"],
            "bq_features_used": [
                "TABLESAMPLE SYSTEM(2 PERCENT)",
                "SQL feature selection with CAST",
                "NULL filtering and range constraints",
            ],
            "output_dir": OUTPUT_DIR,
        },
        "train_metrics": _clean(train_metrics),
        "val_metrics": _clean(val_metrics),
        "test_metrics": _clean(test_metrics),
        "quality_thresholds": {"val_r2_min": 0.35, "val_mse_max": 1.8},
        "quality_passed": {
            "r2": val_metrics["r2"] >= 0.35,
            "mse": val_metrics["mse"] <= 1.8,
        },
    }
    with open(os.path.join(OUTPUT_DIR, "metrics.json"), "w") as fh:
        json.dump(metrics_out, fh, indent=2)
    print("   metrics.json saved.")

    # training_history.json
    history_out = {
        "training": {
            "train_losses": history["train_losses"],
            "val_losses": history["val_losses"],
            "val_r2": history["val_r2"],
            "lr": history["lr"],
            "best_epoch": history["best_epoch"],
            "best_val_loss": history["best_val_loss"],
            "early_stopped": history["early_stopped"],
        }
    }
    with open(os.path.join(OUTPUT_DIR, "training_history.json"), "w") as fh:
        json.dump(history_out, fh, indent=2)
    print("   training_history.json saved.")

    # visualizations
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        preds = val_metrics["predictions"]
        targets = val_metrics["targets"]
        residuals = targets - preds

        # Plot 1: Predictions vs Actual + Training curve
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        lo = min(targets.min(), preds.min()) - 0.2
        hi = max(targets.max(), preds.max()) + 0.2
        axes[0].scatter(
            targets, preds, alpha=0.10, s=3, color="#2563eb", label="val predictions"
        )
        axes[0].plot([lo, hi], [lo, hi], "r--", linewidth=1.4, label="perfect fit")
        axes[0].set_xlabel("Actual birth weight (lbs)")
        axes[0].set_ylabel("Predicted birth weight (lbs)")
        axes[0].set_title(
            f"Predicted vs Actual — Validation\n"
            f"R²={val_metrics['r2']:.4f}  "
            f"MSE={val_metrics['mse']:.4f}  "
            f"RMSE={val_metrics['rmse']:.4f}"
        )
        axes[0].legend(markerscale=4)
        axes[0].grid(True, alpha=0.3)

        epochs_range = range(1, len(history["train_losses"]) + 1)
        axes[1].plot(
            epochs_range, history["train_losses"], label="Train MSE", color="#2563eb"
        )
        axes[1].plot(
            epochs_range, history["val_losses"], label="Val MSE", color="#dc2626"
        )
        best_ep = history["best_epoch"]
        axes[1].axvline(
            x=best_ep,
            color="green",
            linestyle="--",
            linewidth=1.2,
            label=f"Best epoch ({best_ep})",
        )
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("MSE Loss")
        axes[1].set_title("Training History")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        p1 = os.path.join(OUTPUT_DIR, "linreg_bq_natality_predictions_vs_actual.png")
        plt.savefig(p1, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"   Plot saved → {p1}")

        # Plot 2: Residuals
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].scatter(preds, residuals, alpha=0.10, s=3, color="#7c3aed")
        axes[0].axhline(0, color="red", linewidth=1.2, linestyle="--")
        axes[0].set_xlabel("Predicted birth weight (lbs)")
        axes[0].set_ylabel("Residual (actual − predicted)")
        axes[0].set_title("Residuals vs Predicted")
        axes[0].grid(True, alpha=0.3)

        axes[1].hist(
            residuals, bins=60, color="#7c3aed", edgecolor="white", linewidth=0.4
        )
        axes[1].axvline(0, color="red", linewidth=1.2, linestyle="--")
        axes[1].set_xlabel("Residual (lbs)")
        axes[1].set_ylabel("Frequency")
        axes[1].set_title(
            f"Residual Distribution\n"
            f"Mean={residuals.mean():.4f}  Std={residuals.std():.4f}"
        )
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        p2 = os.path.join(OUTPUT_DIR, "linreg_bq_natality_residuals.png")
        plt.savefig(p2, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"   Plot saved → {p2}")

    except ImportError:
        print("   (matplotlib not available — skipping plots)")

    print(f"   All artifacts saved → {OUTPUT_DIR}/")


# Main
def main() -> int:
    print("=" * 65)
    print(" Linear Regression  ·  BigQuery Natality Dataset  (v4)")
    print("=" * 65)

    meta = get_task_metadata()
    print(f"\n Task     : {meta['task_id']}")
    print(f" Target   : {meta['target']}")
    print(f" Features : {meta['features']}")

    set_seed(42)
    device = get_device()
    print(f"\n Device   : {device}")

    print("\n[1] Loading data from BigQuery …")
    train_loader, val_loader, test_loader, scaler, n_features = make_dataloaders(
        project_id=PROJECT_ID
    )
    print(f"   Train : {len(train_loader.dataset):,} samples")
    print(f"   Val   : {len(val_loader.dataset):,} samples")
    print(f"   Test  : {len(test_loader.dataset):,} samples")

    print("\n[2] Building model …")
    model = build_model(input_dim=n_features, device=device)

    print("\n[3] Training …")
    history = train(
        model,
        train_loader,
        val_loader,
        device,
        epochs=120,
        lr=0.05,
        verbose=True,
    )

    print("\n[4] Evaluating …")
    train_metrics = evaluate(model, train_loader, device)
    val_metrics = evaluate(model, val_loader, device)
    test_metrics = evaluate(model, test_loader, device)

    for split_name, m in [
        ("TRAIN", train_metrics),
        ("VALIDATION", val_metrics),
        ("TEST", test_metrics),
    ]:
        print(f"\n   ── {split_name} {'─' * (38 - len(split_name))}")
        print(f"   MSE  : {m['mse']:.4f}")
        print(f"   RMSE : {m['rmse']:.4f}")
        print(f"   R²   : {m['r2']:.4f}")
        print(f"   MAE  : {m['mae']:.4f}")

    print("\n[5] Saving artifacts …")
    save_artifacts(model, scaler, history, train_metrics, val_metrics, test_metrics)

    print("\n[6] Quality check …")
    thresholds = meta["quality_thresholds"]
    passed = True

    val_r2 = val_metrics["r2"]
    val_mse = val_metrics["mse"]

    if val_r2 >= thresholds["val_r2_min"]:
        print(f"   ✓  Val R²  {val_r2:.4f} ≥ {thresholds['val_r2_min']} (threshold)")
    else:
        print(
            f"   ✗  Val R²  {val_r2:.4f} < {thresholds['val_r2_min']} (threshold) — FAILED"
        )
        passed = False

    if val_mse <= thresholds["val_mse_max"]:
        print(f"   ✓  Val MSE {val_mse:.4f} ≤ {thresholds['val_mse_max']} (threshold)")
    else:
        print(
            f"   ✗  Val MSE {val_mse:.4f} > {thresholds['val_mse_max']} (threshold) — FAILED"
        )
        passed = False

    print("\n" + "=" * 65)
    if passed:
        print(" ✓  Task completed successfully!")
        print("=" * 65)
        return 0
    else:
        print(" ✗  Task FAILED quality thresholds.")
        print("=" * 65)
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
