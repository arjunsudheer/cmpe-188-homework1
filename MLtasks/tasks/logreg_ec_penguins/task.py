"""
Logistic Regression on BigQuery Penguins Dataset  (v2)
=======================================================
Binary classification: predicts whether a penguin is Gentoo (1) or not (0)
using physical measurements loaded from bigquery-public-data.ml_datasets.penguins.
The binary label is derived via CASE WHEN entirely in SQL.

Artifact outputs
  ├── model.pth                          PyTorch weights
  ├── scaler.pkl                         Fitted StandardScaler
  ├── metrics.json                       All split metrics
  ├── training_history.json              Per-epoch losses (matches uploaded schema)
  ├── logreg_bq_penguins_roc_curve.png   ROC curve + training history
  └── logreg_bq_penguins_confusion.png   Confusion matrix heatmap

Run:
    python task_logreg_penguins.py

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
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    roc_auc_score,
    mean_squared_error,
    confusion_matrix,
    roc_curve,
)
from typing import Dict, Tuple, Any
import warnings

warnings.filterwarnings("ignore")

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Configuration
PROJECT_ID = "cmpe-188-hw1-ec-491005"
OUTPUT_DIR = "./tasks/logreg_ec_penguins/output"

FEATURE_COLS = [
    "culmen_length_mm",
    "culmen_depth_mm",
    "flipper_length_mm",
    "body_mass_g",
]
TARGET_COL = "is_gentoo"

# Binary label derived in SQL via CASE WHEN — no Python label encoding needed
BQ_QUERY = """
SELECT
    CAST(culmen_length_mm  AS FLOAT64) AS culmen_length_mm,
    CAST(culmen_depth_mm   AS FLOAT64) AS culmen_depth_mm,
    CAST(flipper_length_mm AS FLOAT64) AS flipper_length_mm,
    CAST(body_mass_g       AS FLOAT64) AS body_mass_g,
    CASE
        WHEN species LIKE '%Gentoo%' THEN 1
        ELSE 0
    END AS is_gentoo
FROM
    `bigquery-public-data.ml_datasets.penguins`
WHERE
    culmen_length_mm  IS NOT NULL
    AND culmen_depth_mm   IS NOT NULL
    AND flipper_length_mm IS NOT NULL
    AND body_mass_g       IS NOT NULL
    AND species           IS NOT NULL
"""


# Required interface functions
def get_task_metadata() -> Dict[str, Any]:
    return {
        "task_id": "logreg_lvl4_bq_penguins",
        "task_name": "logistic_regression_bq_penguins",
        "series": "Logistic Regression",
        "level": 4,
        "algorithm": "Logistic Regression (SGD + CosineAnnealingLR + BigQuery Penguins)",
        "description": (
            "Binary classification: predict whether a penguin is Gentoo (1) or "
            "another species (0) using physical measurements from the BigQuery "
            "public ml_datasets.penguins table. Binary label derived in SQL."
        ),
        "data_source": "bigquery-public-data.ml_datasets.penguins",
        "bq_features_used": [
            "SQL column selection with CAST for type safety",
            "Binary label derived via CASE WHEN directly in SQL",
            "NULL filtering in WHERE clause",
        ],
        "features": FEATURE_COLS,
        "target": TARGET_COL,
        "metrics": ["accuracy", "f1", "roc_auc", "mse", "brier"],
        "quality_thresholds": {"val_accuracy_min": 0.88, "val_roc_auc_min": 0.92},
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
    Submits BQ_QUERY via the google-cloud-bigquery Python client.
    The CASE WHEN expression computes the binary label entirely in BigQuery
    so no Python-side label encoding is required.
    """
    try:
        from google.cloud import bigquery
    except ImportError as exc:
        raise ImportError("Run: pip install google-cloud-bigquery db-dtypes") from exc

    print(f"   Connecting to BigQuery project: {project_id}")
    client = bigquery.Client(project=project_id)
    print("   Running query …")
    df = client.query(BQ_QUERY).to_dataframe()
    print(f"   Fetched {len(df):,} rows from BigQuery.")
    counts = df[TARGET_COL].value_counts().to_dict()
    print(
        f"   Class distribution → {{0 (non-Gentoo): {counts.get(0,0)}, "
        f"1 (Gentoo): {counts.get(1,0)}}}"
    )
    return df


def make_dataloaders(
    project_id: str = PROJECT_ID,
    val_size: float = 0.15,
    test_size: float = 0.15,
    batch_size: int = 32,
    random_state: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader, StandardScaler, int]:
    df = _load_from_bigquery(project_id)

    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df[TARGET_COL].values.astype(np.float32).reshape(-1, 1)

    temp_frac = val_size + test_size
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=temp_frac, random_state=random_state, stratify=y.flatten()
    )
    relative_val = val_size / temp_frac
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp,
        y_temp,
        test_size=(1 - relative_val),
        random_state=random_state,
        stratify=y_temp.flatten(),
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
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.linear(x))


def build_model(input_dim: int, device: torch.device) -> nn.Module:
    model = LogisticRegressionModel(input_dim=input_dim).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(
        f"   Model: LogisticRegressionModel | input_dim={input_dim} | params={n_params}"
    )
    return model


# Training
def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    device: torch.device,
    epochs: int = 150,
    lr: float = 0.01,
    verbose: bool = True,
) -> Dict[str, list]:
    criterion = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    history: Dict[str, list] = {
        "train_losses": [],
        "val_losses": [],
        "val_accuracy": [],
        "val_f1": [],
        "val_roc_auc": [],
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
        scheduler.step()

        if val_metrics["bce_loss"] < best_val_loss:
            best_val_loss = val_metrics["bce_loss"]
            best_epoch = epoch
            best_state = {k: v.clone() for k, v in model.state_dict().items()}

        history["train_losses"].append(round(train_loss, 6))
        history["val_losses"].append(round(val_metrics["bce_loss"], 6))
        history["val_accuracy"].append(round(val_metrics["accuracy"], 6))
        history["val_f1"].append(round(val_metrics["f1"], 6))
        history["val_roc_auc"].append(round(val_metrics["roc_auc"], 6))

        if verbose and epoch % 15 == 0:
            print(
                f"   Epoch [{epoch:3d}/{epochs}] "
                f"train_loss={train_loss:.4f}  "
                f"val_acc={val_metrics['accuracy']:.4f}  "
                f"val_f1={val_metrics['f1']:.4f}  "
                f"val_auc={val_metrics['roc_auc']:.4f}"
            )

    if best_state is not None:
        model.load_state_dict(best_state)
        print(
            f"   Restored best checkpoint (epoch={best_epoch}, val_bce={best_val_loss:.4f})"
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
    threshold: float = 0.5,
) -> Dict[str, float]:
    """
    Evaluates the model on the given DataLoader.
    Returns bce_loss, accuracy, f1, roc_auc, mse (Brier), brier,
            probabilities, predictions, targets.
    """
    model.eval()
    criterion = nn.BCELoss()
    all_probs: list = []
    all_targets: list = []
    total_loss = 0.0

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            probs = model(X_batch)
            total_loss += criterion(probs, y_batch).item()
            all_probs.extend(probs.cpu().numpy().flatten())
            all_targets.extend(y_batch.cpu().numpy().flatten())

    probs_arr = np.array(all_probs, dtype=np.float32)
    targets_arr = np.array(all_targets, dtype=np.float32)
    preds_arr = (probs_arr >= threshold).astype(np.float32)

    return {
        "bce_loss": total_loss / len(data_loader),
        "accuracy": float(accuracy_score(targets_arr, preds_arr)),
        "f1": float(f1_score(targets_arr, preds_arr, zero_division=0)),
        "roc_auc": float(roc_auc_score(targets_arr, probs_arr)),
        "mse": float(mean_squared_error(targets_arr, probs_arr)),
        "brier": float(mean_squared_error(targets_arr, probs_arr)),
        "probabilities": probs_arr,
        "predictions": preds_arr,
        "targets": targets_arr,
    }


# Inference
def predict(
    model: nn.Module,
    X: np.ndarray,
    scaler: StandardScaler,
    device: torch.device,
    threshold: float = 0.5,
) -> Tuple[np.ndarray, np.ndarray]:
    model.eval()
    X_sc = scaler.transform(X).astype(np.float32)
    with torch.no_grad():
        probs = model(torch.from_numpy(X_sc).to(device)).cpu().numpy().flatten()
    return probs, (probs >= threshold).astype(np.int32)


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
      model.pth
      scaler.pkl
      metrics.json                       (matches uploaded schema)
      training_history.json              (matches uploaded schema)
      logreg_bq_penguins_roc_curve.png
      logreg_bq_penguins_confusion.png
    """
    import pickle

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # model + scaler
    torch.save(model.state_dict(), os.path.join(OUTPUT_DIR, "model.pth"))
    with open(os.path.join(OUTPUT_DIR, "scaler.pkl"), "wb") as fh:
        pickle.dump(scaler, fh)

    # metrics.json
    _skip = {"probabilities", "predictions", "targets"}

    def _clean(m):
        return {k: round(float(v), 6) for k, v in m.items() if k not in _skip}

    metrics_out = {
        "metadata": {
            "task_name": "logistic_regression_bq_penguins",
            "task_type": "binary_classification",
            "dataset": "bigquery-public-data.ml_datasets.penguins",
            "frameworks": ["pytorch"],
            "metrics": ["accuracy", "f1", "roc_auc", "mse", "brier", "bce_loss"],
            "bq_features_used": [
                "SQL column selection with CAST",
                "Binary label via CASE WHEN in SQL",
                "NULL filtering in WHERE clause",
            ],
            "output_dir": OUTPUT_DIR,
        },
        "train_metrics": _clean(train_metrics),
        "val_metrics": _clean(val_metrics),
        "test_metrics": _clean(test_metrics),
        "quality_thresholds": {"val_accuracy_min": 0.88, "val_roc_auc_min": 0.92},
        "quality_passed": {
            "accuracy": val_metrics["accuracy"] >= 0.88,
            "roc_auc": val_metrics["roc_auc"] >= 0.92,
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
            "val_accuracy": history["val_accuracy"],
            "val_f1": history["val_f1"],
            "val_roc_auc": history["val_roc_auc"],
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

        probs = val_metrics["probabilities"]
        targets = val_metrics["targets"]
        preds = val_metrics["predictions"]

        # Plot 1: ROC curve + training history
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        fpr, tpr, _ = roc_curve(targets, probs)
        auc_val = val_metrics["roc_auc"]
        axes[0].plot(
            fpr,
            tpr,
            color="#2563eb",
            linewidth=2,
            label=f"ROC curve (AUC = {auc_val:.4f})",
        )
        axes[0].plot([0, 1], [0, 1], "r--", linewidth=1.2, label="Random classifier")
        axes[0].set_xlabel("False Positive Rate")
        axes[0].set_ylabel("True Positive Rate")
        axes[0].set_title(
            f"ROC Curve — Validation\n"
            f"Accuracy={val_metrics['accuracy']:.4f}  "
            f"F1={val_metrics['f1']:.4f}  "
            f"AUC={auc_val:.4f}"
        )
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        epochs_range = range(1, len(history["train_losses"]) + 1)
        axes[1].plot(
            epochs_range, history["train_losses"], label="Train BCE", color="#2563eb"
        )
        axes[1].plot(
            epochs_range, history["val_losses"], label="Val BCE", color="#dc2626"
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
        axes[1].set_ylabel("BCE Loss")
        axes[1].set_title("Training History")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        p1 = os.path.join(OUTPUT_DIR, "logreg_bq_penguins_roc_curve.png")
        plt.savefig(p1, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"   Plot saved → {p1}")

        # Plot 2: Confusion matrix
        cm = confusion_matrix(targets.astype(int), preds.astype(int))
        fig, ax = plt.subplots(figsize=(6, 5))
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        plt.colorbar(im, ax=ax)
        classes = ["Non-Gentoo (0)", "Gentoo (1)"]
        tick_marks = range(len(classes))
        ax.set_xticks(tick_marks)
        ax.set_xticklabels(classes, rotation=15)
        ax.set_yticks(tick_marks)
        ax.set_yticklabels(classes)
        thresh = cm.max() / 2.0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(
                    j,
                    i,
                    format(cm[i, j], "d"),
                    ha="center",
                    va="center",
                    color="white" if cm[i, j] > thresh else "black",
                    fontsize=14,
                    fontweight="bold",
                )
        ax.set_ylabel("True label")
        ax.set_xlabel("Predicted label")
        ax.set_title(
            f"Confusion Matrix — Validation\n"
            f"Accuracy={val_metrics['accuracy']:.4f}  F1={val_metrics['f1']:.4f}"
        )
        plt.tight_layout()
        p2 = os.path.join(OUTPUT_DIR, "logreg_bq_penguins_confusion.png")
        plt.savefig(p2, dpi=150, bbox_inches="tight")
        plt.close()
        print(f"   Plot saved → {p2}")

    except ImportError:
        print("   (matplotlib not available — skipping plots)")

    print(f"   All artifacts saved → {OUTPUT_DIR}/")


# Main
def main() -> int:
    print("=" * 65)
    print(" Logistic Regression  ·  BigQuery Penguins Dataset  (v2)")
    print("=" * 65)

    meta = get_task_metadata()
    print(f"\n Task     : {meta['task_id']}")
    print(f" Target   : {meta['target']}  (1 = Gentoo, 0 = other species)")
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
        epochs=150,
        lr=0.01,
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
        print(f"   Accuracy  : {m['accuracy']:.4f}")
        print(f"   F1 Score  : {m['f1']:.4f}")
        print(f"   ROC-AUC   : {m['roc_auc']:.4f}")
        print(f"   MSE/Brier : {m['mse']:.4f}")
        print(f"   BCE Loss  : {m['bce_loss']:.4f}")

    print("\n[5] Saving artifacts …")
    save_artifacts(model, scaler, history, train_metrics, val_metrics, test_metrics)

    print("\n[6] Quality check …")
    thresholds = meta["quality_thresholds"]
    passed = True

    val_acc = val_metrics["accuracy"]
    val_auc = val_metrics["roc_auc"]

    if val_acc >= thresholds["val_accuracy_min"]:
        print(
            f"   ✓  Val Accuracy {val_acc:.4f} ≥ {thresholds['val_accuracy_min']} (threshold)"
        )
    else:
        print(
            f"   ✗  Val Accuracy {val_acc:.4f} < {thresholds['val_accuracy_min']} (threshold) — FAILED"
        )
        passed = False

    if val_auc >= thresholds["val_roc_auc_min"]:
        print(
            f"   ✓  Val ROC-AUC  {val_auc:.4f} ≥ {thresholds['val_roc_auc_min']} (threshold)"
        )
    else:
        print(
            f"   ✗  Val ROC-AUC  {val_auc:.4f} < {thresholds['val_roc_auc_min']} (threshold) — FAILED"
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
