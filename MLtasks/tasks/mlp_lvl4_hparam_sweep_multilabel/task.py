"""
MLP hyperparameter search on sklearn RCV1 multilabel text classification.

Uses fetch_rcv1 (TF-IDF bag-of-words, sparse multilabel targets). High-dimensional
sparse features are reduced with TruncatedSVD (fit on train only), then standardized.
Training uses BCEWithLogitsLoss (multilabel) and AdamW.
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from typing import Dict, List, Tuple, Any, Optional
import matplotlib.pyplot as plt

from sklearn.datasets import fetch_rcv1
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, r2_score

_TASK_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(_TASK_DIR, 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Filled after make_dataloaders() so get_task_metadata() reflects real shapes.
_DATA_INFO: Dict[str, Any] = {
    'input_dim': None,
    'output_dim': None,
    'svd_components': 256,
}


def get_task_metadata() -> Dict[str, Any]:
    """Return metadata about the ML task."""
    return {
        'task_type': 'multilabel_classification',
        'input_dim': _DATA_INFO['input_dim'] or _DATA_INFO['svd_components'],
        'output_dim': _DATA_INFO['output_dim'] or 103,
        'description': (
            'MLP hyperparameter search on sklearn RCV1 multilabel topics; '
            'TruncatedSVD + StandardScaler; BCEWithLogitsLoss; AdamW.'
        ),
        'metrics': ['mse', 'r2', 'mae', 'micro_f1', 'macro_f1'],
        'hyperparameters': ['depth', 'width', 'learning_rate', 'weight_decay'],
        'dataset': 'sklearn.datasets.fetch_rcv1',
    }


def set_seed(seed: int = 42) -> None:
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device() -> torch.device:
    """Get the appropriate device (GPU or CPU)."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def make_dataloaders(
    max_samples: int = 6000,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    batch_size: int = 64,
    svd_components: int = 256,
    random_state: int = 42,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Load RCV1 (train subset), subsample, split train/val/test, SVD + scale (train-only fit).
    Targets are dense float multilabel indicators in {0,1}^{L}.
    """
    set_seed(random_state)
    rcv1 = fetch_rcv1(subset='train', shuffle=True, random_state=random_state, download_if_missing=True)
    X = rcv1.data
    y = rcv1.target
    n_total = X.shape[0]
    n_take = min(max_samples, n_total)
    rng = np.random.RandomState(random_state)
    idx = rng.choice(n_total, size=n_take, replace=False)
    X = X[idx]
    y = y[idx].toarray().astype(np.float32)

    n_labels = y.shape[1]
    test_ratio = 1.0 - train_ratio - val_ratio
    if test_ratio <= 0:
        raise ValueError('train_ratio + val_ratio must be < 1')

    X_trainval, X_test, y_trainval, y_test = train_test_split(
        X, y, test_size=test_ratio, random_state=random_state
    )
    rel_val = val_ratio / (train_ratio + val_ratio)
    X_train, X_val, y_train, y_val = train_test_split(
        X_trainval, y_trainval, test_size=rel_val, random_state=random_state
    )

    n_comp = min(svd_components, X_train.shape[0] - 1, X_train.shape[1] - 1)
    n_comp = max(1, n_comp)
    svd = TruncatedSVD(n_components=n_comp, random_state=random_state)
    X_train_svd = svd.fit_transform(X_train)
    X_val_svd = svd.transform(X_val)
    X_test_svd = svd.transform(X_test)

    scaler = StandardScaler()
    X_train_d = scaler.fit_transform(X_train_svd).astype(np.float32)
    X_val_d = scaler.transform(X_val_svd).astype(np.float32)
    X_test_d = scaler.transform(X_test_svd).astype(np.float32)

    input_dim = X_train_d.shape[1]
    _DATA_INFO['input_dim'] = input_dim
    _DATA_INFO['output_dim'] = n_labels
    _DATA_INFO['svd_components'] = input_dim

    X_train_t = torch.from_numpy(X_train_d)
    y_train_t = torch.from_numpy(y_train)
    X_val_t = torch.from_numpy(X_val_d)
    y_val_t = torch.from_numpy(y_val)
    X_test_t = torch.from_numpy(X_test_d)
    y_test_t = torch.from_numpy(y_test)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t), batch_size=batch_size, shuffle=True
    )
    val_loader = DataLoader(
        TensorDataset(X_val_t, y_val_t), batch_size=batch_size, shuffle=False
    )
    test_loader = DataLoader(
        TensorDataset(X_test_t, y_test_t), batch_size=batch_size, shuffle=False
    )

    return train_loader, val_loader, test_loader


class MLP(nn.Module):
    """MLP producing per-label logits for multilabel classification."""

    def __init__(self, input_dim: int, output_dim: int, hidden_dims: List[int], dropout: float = 0.1):
        super().__init__()
        layers: List[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


def build_model(
    input_dim: int,
    output_dim: int,
    depth: int = 3,
    width: int = 64,
    dropout: float = 0.1,
) -> MLP:
    """Build an MLP with specified depth/width."""
    hidden_dims = [width] * depth
    return MLP(input_dim, output_dim, hidden_dims, dropout)


def _probs_from_logits(logits: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(logits)


def sigmoid_np(logits: np.ndarray) -> np.ndarray:
    """Sigmoid in NumPy without exp overflow (large |logits|)."""
    z = np.asarray(logits, dtype=np.float64)
    out = np.empty_like(z, dtype=np.float64)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    neg = ~pos
    exp_z = np.exp(z[neg])
    out[neg] = exp_z / (1.0 + exp_z)
    return out.astype(np.float32, copy=False)


def train(
    model: nn.Module,
    train_loader: DataLoader,
    val_loader: DataLoader,
    learning_rate: float = 0.001,
    weight_decay: float = 0.0,
    epochs: int = 100,
    patience: int = 10,
    device: Optional[torch.device] = None,
) -> Dict[str, List[float]]:
    """Train with BCEWithLogitsLoss, AdamW, ReduceLROnPlateau, early stopping on val loss."""
    if device is None:
        device = get_device()

    model.to(device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=4
    )

    history: Dict[str, List[float]] = {'train_loss': [], 'val_loss': [], 'val_r2': []}
    best_val_loss = float('inf')
    patience_counter = 0
    best_state_dict = None

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= max(len(train_loader), 1)

        model.eval()
        val_loss = 0.0
        all_logits: List[np.ndarray] = []
        all_y: List[np.ndarray] = []
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                logits = model(X_batch)
                loss = criterion(logits, y_batch)
                val_loss += loss.item()
                all_logits.append(logits.cpu().numpy())
                all_y.append(y_batch.cpu().numpy())
        val_loss /= max(len(val_loader), 1)
        logits_np = np.concatenate(all_logits, axis=0)
        y_np = np.concatenate(all_y, axis=0)
        probs_np = sigmoid_np(logits_np)
        val_r2 = float(r2_score(y_np.reshape(-1), probs_np.reshape(-1)))

        scheduler.step(val_loss)
        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['val_r2'].append(val_r2)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1}")
                break

        if (epoch + 1) % 15 == 0:
            print(
                f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.6f}, "
                f"Val Loss: {val_loss:.6f}, Val R2 (prob vs y): {val_r2:.4f}"
            )

    if best_state_dict is not None:
        model.load_state_dict(best_state_dict)
    return history


def evaluate(
    model: nn.Module,
    data_loader: DataLoader,
    device: Optional[torch.device] = None,
) -> Dict[str, float]:
    """
    Multilabel metrics on the given loader: MSE/MAE/R2 on probabilities vs labels,
    plus micro and macro F1 at 0.5 threshold.
    """
    if device is None:
        device = get_device()

    model.eval()
    criterion = nn.BCEWithLogitsLoss(reduction='sum')
    all_logits: List[np.ndarray] = []
    all_y: List[np.ndarray] = []
    sum_loss = 0.0
    n_elems = 0

    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            logits = model(X_batch)
            sum_loss += criterion(logits, y_batch).item()
            ne = y_batch.numel()
            n_elems += ne
            all_logits.append(logits.cpu().numpy())
            all_y.append(y_batch.cpu().numpy())

    logits_np = np.concatenate(all_logits, axis=0)
    y_np = np.concatenate(all_y, axis=0)
    probs_np = sigmoid_np(logits_np)
    mse = float(np.mean((probs_np - y_np) ** 2))
    mae = float(np.mean(np.abs(probs_np - y_np)))
    r2 = float(r2_score(y_np.reshape(-1), probs_np.reshape(-1)))
    y_hat = (probs_np >= 0.5).astype(np.int32)
    micro_f1 = float(f1_score(y_np, y_hat, average='micro', zero_division=0))
    macro_f1 = float(f1_score(y_np, y_hat, average='macro', zero_division=0))
    bce_mean = sum_loss / max(n_elems, 1)

    return {
        'mse': mse,
        'r2': r2,
        'mae': mae,
        'micro_f1': micro_f1,
        'macro_f1': macro_f1,
        'bce': bce_mean,
    }


def predict(
    model: nn.Module,
    X: np.ndarray,
    device: Optional[torch.device] = None,
) -> np.ndarray:
    """Return predicted multilabel probabilities (sigmoid of logits)."""
    if device is None:
        device = get_device()
    model.eval()
    X_tensor = torch.as_tensor(X, dtype=torch.float32, device=device)
    with torch.no_grad():
        probs = _probs_from_logits(model(X_tensor))
    return probs.cpu().numpy()


def save_artifacts(
    model: nn.Module,
    history: Dict[str, List[float]],
    best_config: Dict[str, Any],
    metrics: Dict[str, Any],
    output_dir: str = OUTPUT_DIR,
) -> None:
    """Save model, history, metrics, config, and training plot."""
    os.makedirs(output_dir, exist_ok=True)
    torch.save(model.state_dict(), os.path.join(output_dir, 'model.pt'))
    with open(os.path.join(output_dir, 'history.json'), 'w') as f:
        json.dump(history, f, indent=2)
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2, default=str)
    with open(os.path.join(output_dir, 'config.json'), 'w') as f:
        json.dump(best_config, f, indent=2)

    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.plot(history['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('BCE (logits)')
    plt.legend()
    plt.title('Training History')

    plt.subplot(1, 2, 2)
    plt.plot(history['val_r2'], label='Val R2 (prob vs y)')
    plt.xlabel('Epoch')
    plt.ylabel('R2')
    plt.legend()
    plt.title('Validation R2')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_plot.png'))
    plt.close()
    print(f"Artifacts saved to {output_dir}")


def _metrics_match(
    a: Dict[str, float],
    b: Dict[str, float],
    keys: Tuple[str, ...],
    rtol: float = 1e-4,
    atol: float = 1e-5,
) -> bool:
    for k in keys:
        if k not in a or k not in b:
            return False
        if not np.isfinite(a[k]) or not np.isfinite(b[k]):
            return False
        if not np.isclose(a[k], b[k], rtol=rtol, atol=atol):
            return False
    return True


def run_self_checks_from_disk(
    model_path: str,
    metrics_path: str,
    config_path: str,
    input_dim: int,
    output_dim: int,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
    expected_val_metrics: Dict[str, float],
    expected_test_metrics: Dict[str, float],
    baseline_val: Dict[str, float],
) -> None:
    """
    Load best weights and metrics.json from disk; re-evaluate; assert quality and parity.
    """
    if not os.path.isfile(model_path):
        print(f"ERROR: Missing checkpoint: {model_path}", file=sys.stderr)
        sys.exit(1)
    with open(config_path, 'r') as f:
        saved_cfg = json.load(f)

    loaded = build_model(
        input_dim=input_dim,
        output_dim=output_dim,
        depth=int(saved_cfg['depth']),
        width=int(saved_cfg['width']),
    ).to(device)
    try:
        state = torch.load(model_path, map_location=device, weights_only=True)
    except TypeError:
        state = torch.load(model_path, map_location=device)
    loaded.load_state_dict(state)
    loaded.eval()

    val_ck = evaluate(loaded, val_loader, device)
    test_ck = evaluate(loaded, test_loader, device)
    train_ck = evaluate(loaded, train_loader, device)

    metric_keys = ('mse', 'r2', 'mae', 'micro_f1', 'macro_f1', 'bce')
    print("\n" + "-" * 60)
    print("Self-check: metrics from reloaded model.pt vs in-memory (validation)")
    print("-" * 60)
    for k in metric_keys:
        print(f"  {k}: disk-reload={val_ck[k]:.8f}  pre-save={expected_val_metrics[k]:.8f}")

    if not _metrics_match(val_ck, expected_val_metrics, metric_keys):
        print(
            "ERROR: Reloaded model validation metrics do not match pre-save metrics.",
            file=sys.stderr,
        )
        sys.exit(1)

    if not _metrics_match(test_ck, expected_test_metrics, metric_keys):
        print(
            "ERROR: Reloaded model test metrics do not match pre-save metrics.",
            file=sys.stderr,
        )
        sys.exit(1)

    with open(metrics_path, 'r') as f:
        saved_metrics = json.load(f)
    file_val = saved_metrics['validation']
    file_test = saved_metrics['test']
    for k in metric_keys:
        if not np.isclose(val_ck[k], float(file_val[k]), rtol=1e-4, atol=1e-5):
            print(
                f"ERROR: val {k} from reload ({val_ck[k]}) != metrics.json ({file_val[k]}).",
                file=sys.stderr,
            )
            sys.exit(1)
        if not np.isclose(test_ck[k], float(file_test[k]), rtol=1e-4, atol=1e-5):
            print(
                f"ERROR: test {k} from reload ({test_ck[k]}) != metrics.json ({file_test[k]}).",
                file=sys.stderr,
            )
            sys.exit(1)

    # Task-quality gates on the checkpoint (same thresholds as main)
    if val_ck['micro_f1'] < baseline_val['micro_f1'] - 1e-6:
        print(
            "ERROR (self-check): Reloaded model val micro-F1 below baseline.",
            file=sys.stderr,
        )
        sys.exit(1)
    if val_ck['micro_f1'] < 0.12:
        print("ERROR (self-check): Reloaded model val micro-F1 below minimum.", file=sys.stderr)
        sys.exit(1)
    if val_ck['mse'] > 0.45:
        print("ERROR (self-check): Reloaded model val MSE above maximum.", file=sys.stderr)
        sys.exit(1)
    if train_ck['bce'] <= 0 or not np.isfinite(train_ck['bce']):
        print("ERROR (self-check): Invalid train BCE on reloaded model.", file=sys.stderr)
        sys.exit(1)
    if not np.isfinite(val_ck['r2']) or not np.isfinite(val_ck['mae']):
        print("ERROR (self-check): Non-finite val R2 or MAE on reloaded model.", file=sys.stderr)
        sys.exit(1)

    print("Self-check passed: checkpoint matches metrics.json and meets thresholds.")


def hyperparameter_search(
    train_loader: DataLoader,
    val_loader: DataLoader,
    input_dim: int,
    output_dim: int,
    search_type: str = 'grid',
    n_iterations: int = 10,
    device: Optional[torch.device] = None,
) -> Tuple[Dict[str, Any], Dict[str, Any], nn.Module, Dict[str, List[float]]]:
    """
    Grid or random search over depth, width, lr, weight_decay.
    Best config is chosen by validation micro-F1 (higher is better).
    """
    if device is None:
        device = get_device()

    depth_options = [2, 3]
    width_options = [64, 128]
    lr_options = [0.001, 0.003]
    weight_decay_options = [0.0, 1e-4]

    best_config: Optional[Dict[str, Any]] = None
    best_micro_f1 = float('-inf')
    best_model: Optional[nn.Module] = None
    best_history: Optional[Dict[str, List[float]]] = None
    sweep_results: List[Dict[str, Any]] = []

    if search_type == 'grid':
        configs: List[Dict[str, Any]] = []
        for depth in depth_options:
            for width in width_options:
                for lr in lr_options:
                    for wd in weight_decay_options:
                        configs.append({
                            'depth': depth,
                            'width': width,
                            'learning_rate': lr,
                            'weight_decay': wd,
                        })
    else:
        rng = np.random.RandomState(42)
        configs = []
        for _ in range(n_iterations):
            configs.append({
                'depth': int(rng.choice(depth_options)),
                'width': int(rng.choice(width_options)),
                'learning_rate': float(rng.choice(lr_options)),
                'weight_decay': float(rng.choice(weight_decay_options)),
            })

    print(f"Searching over {len(configs)} configurations...")

    for i, config in enumerate(configs):
        print(f"\n[{i + 1}/{len(configs)}] Config: {config}")
        set_seed(42)
        model = build_model(
            input_dim=input_dim,
            output_dim=output_dim,
            depth=config['depth'],
            width=config['width'],
        )
        history = train(
            model,
            train_loader,
            val_loader,
            learning_rate=config['learning_rate'],
            weight_decay=config['weight_decay'],
            epochs=35,
            patience=6,
            device=device,
        )
        val_metrics = evaluate(model, val_loader, device)
        sweep_results.append({'config': config, 'val_metrics': val_metrics})
        print(
            f"  Val micro-F1: {val_metrics['micro_f1']:.4f}, "
            f"macro-F1: {val_metrics['macro_f1']:.4f}, MSE: {val_metrics['mse']:.6f}"
        )
        if val_metrics['micro_f1'] > best_micro_f1:
            best_micro_f1 = val_metrics['micro_f1']
            best_config = config
            best_model = model
            best_history = history

    assert best_config is not None and best_model is not None and best_history is not None

    leaderboard: List[Dict[str, Any]] = []
    for result in sweep_results:
        c = result['config']
        m = result['val_metrics']
        leaderboard.append({
            'depth': c['depth'],
            'width': c['width'],
            'learning_rate': c['learning_rate'],
            'weight_decay': c['weight_decay'],
            'val_mse': m['mse'],
            'val_r2': m['r2'],
            'val_mae': m['mae'],
            'val_micro_f1': m['micro_f1'],
            'val_macro_f1': m['macro_f1'],
        })
    leaderboard.sort(key=lambda x: x['val_micro_f1'], reverse=True)

    metrics_out: Dict[str, Any] = {
        'sweep': leaderboard,
        'best_config': best_config,
        'best_val_micro_f1': best_micro_f1,
    }
    return best_config, metrics_out, best_model, best_history


def main() -> None:
    """Run hyperparameter search, final eval on train/val/test, assert quality, save artifacts."""
    print("=" * 60)
    print("MLP Hyperparameter Search — RCV1 multilabel")
    print("=" * 60)

    device = get_device()
    print(f"Using device: {device}")

    set_seed(42)
    train_loader, val_loader, test_loader = make_dataloaders(
        max_samples=6000, batch_size=64, svd_components=256
    )
    input_dim = _DATA_INFO['input_dim']
    output_dim = _DATA_INFO['output_dim']
    assert input_dim is not None and output_dim is not None

    print("\n" + "-" * 60)
    print("Starting hyperparameter search...")
    print("-" * 60)

    best_config, metrics, best_model, _ = hyperparameter_search(
        train_loader,
        val_loader,
        input_dim,
        output_dim,
        search_type='grid',
        device=device,
    )

    print("\n" + "-" * 60)
    print("Hyperparameter search complete")
    print("-" * 60)
    print(f"\nBest configuration: {best_config}")
    print(f"Best validation micro-F1: {metrics['best_val_micro_f1']:.4f}")

    set_seed(42)
    best_model = build_model(
        input_dim=input_dim,
        output_dim=output_dim,
        depth=best_config['depth'],
        width=best_config['width'],
    ).to(device)
    best_history = train(
        best_model,
        train_loader,
        val_loader,
        learning_rate=best_config['learning_rate'],
        weight_decay=best_config['weight_decay'],
        epochs=80,
        patience=12,
        device=device,
    )

    print("\n" + "-" * 60)
    print("Evaluating on train / val / test")
    print("-" * 60)
    train_metrics = evaluate(best_model, train_loader, device)
    val_metrics = evaluate(best_model, val_loader, device)
    test_metrics = evaluate(best_model, test_loader, device)

    for name, tm in [('Train', train_metrics), ('Val', val_metrics), ('Test', test_metrics)]:
        print(
            f"{name} — MSE: {tm['mse']:.6f}, R2: {tm['r2']:.4f}, MAE: {tm['mae']:.6f}, "
            f"micro-F1: {tm['micro_f1']:.4f}, macro-F1: {tm['macro_f1']:.4f}"
        )

    set_seed(42)
    baseline = build_model(
        input_dim=input_dim,
        output_dim=output_dim,
        depth=2,
        width=32,
    ).to(device)
    train(
        baseline,
        train_loader,
        val_loader,
        learning_rate=0.001,
        weight_decay=0.0,
        epochs=40,
        patience=8,
        device=device,
    )
    baseline_val = evaluate(baseline, val_loader, device)

    print("\nBaseline (depth=2, width=32) val micro-F1:", f"{baseline_val['micro_f1']:.4f}")
    if val_metrics['micro_f1'] < baseline_val['micro_f1'] - 1e-6:
        print(
            "ERROR: Best config from sweep should beat or match baseline on val micro-F1.",
            file=sys.stderr,
        )
        sys.exit(1)

    # pytorch_task_v1-style quality checks (validation split)
    if val_metrics['micro_f1'] < 0.12:
        print("ERROR: Val micro-F1 below minimum threshold.", file=sys.stderr)
        sys.exit(1)
    if val_metrics['mse'] > 0.45:
        print("ERROR: Val MSE above maximum threshold.", file=sys.stderr)
        sys.exit(1)

    final_metrics = {
        **metrics,
        'train': train_metrics,
        'validation': val_metrics,
        'test': test_metrics,
        'baseline_val': baseline_val,
    }

    print("\n" + "-" * 60)
    print("Saving artifacts...")
    print("-" * 60)
    save_artifacts(best_model, best_history, best_config, final_metrics)

    run_self_checks_from_disk(
        model_path=os.path.join(OUTPUT_DIR, 'model.pt'),
        metrics_path=os.path.join(OUTPUT_DIR, 'metrics.json'),
        config_path=os.path.join(OUTPUT_DIR, 'config.json'),
        input_dim=input_dim,
        output_dim=output_dim,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        expected_val_metrics=val_metrics,
        expected_test_metrics=test_metrics,
        baseline_val=baseline_val,
    )

    print("\n" + "=" * 60)
    print("Task complete.")
    print("=" * 60)


if __name__ == '__main__':
    main()
