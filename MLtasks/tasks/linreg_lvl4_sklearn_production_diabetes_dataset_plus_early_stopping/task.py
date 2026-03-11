"""
Linear Regression Task: PyTorch vs Sklearn Comparison
Industrial Comparison with EDA, Preprocessing, and Standardized Output
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Output directory
OUTPUT_DIR = "./tasks/linreg_lvl4_sklearn_production_diabetes_dataset_plus_early_stopping/output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Device setup
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def get_task_metadata():
    """Return task metadata as dictionary."""
    return {
        "task_name": "linear_regression",
        "task_type": "regression",
        "dataset": "diabetes",
        "frameworks": ["pytorch", "sklearn"],
        "metrics": ["mse", "rmse", "r2", "mae"],
        "early_stopping": True,
        "output_dir": OUTPUT_DIR
    }


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device():
    """Get computation device."""
    return torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def make_dataloaders(test_size=0.2, batch_size=64):
    """
    Load and preprocess Diabetes data.
    Returns train/val dataloaders and scalers.
    """
    # Load dataset
    data = load_diabetes()
    X, y = data.data, data.target
    
    # Split data
    X_train_full, X_val, y_train_full, y_val = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    # Manual StandardScaler for PyTorch
    X_scaler = StandardScaler()
    y_scaler = StandardScaler()
    
    X_train_full = X_scaler.fit_transform(X_train_full)
    X_val = X_scaler.transform(X_val)
    
    # Reshape y for scaling
    y_train_full = y_train_full.reshape(-1, 1)
    y_val = y_val.reshape(-1, 1)
    
    y_train_full = y_scaler.fit_transform(y_train_full).flatten()
    y_val = y_scaler.transform(y_val).flatten()
    
    # Convert to PyTorch tensors
    X_train_tensor = torch.FloatTensor(X_train_full)
    y_train_tensor = torch.FloatTensor(y_train_full).unsqueeze(1)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.FloatTensor(y_val).unsqueeze(1)
    
    # Create DataLoaders
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    # Store original data for sklearn comparison
    sklearn_data = {
        'X_train': X_train_full,
        'X_val': X_val,
        'y_train': y_train_full,
        'y_val': y_val,
        'feature_names': data.feature_names,
        'target_name': data.target_names if hasattr(data, 'target_names') else ['disease_progression']
    }
    
    return {
        'train_loader': train_loader,
        'val_loader': val_loader,
        'X_train': X_train_full,
        'X_val': X_val,
        'y_train': y_train_full,
        'y_val': y_val,
        'X_scaler': X_scaler,
        'y_scaler': y_scaler,
        'sklearn_data': sklearn_data,
        'n_features': X_train_full.shape[1]
    }


class PyTorchLinearRegression(nn.Module):
    """PyTorch Linear Regression model with sklearn-style API."""
    
    def __init__(self, n_features):
        super(PyTorchLinearRegression, self).__init__()
        self.linear = nn.Linear(n_features, 1)
        
    def forward(self, x):
        return self.linear(x)
    
    def fit(self, train_loader, val_loader=None, epochs=100, lr=0.01, verbose=False, early_stopping=False, patience=20):
        """Fit the model using training data with optional early stopping.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Maximum number of epochs
            lr: Learning rate
            verbose: Whether to print training progress
            early_stopping: Whether to use early stopping
            patience: Number of epochs to wait for improvement before stopping
        """
        self.to(device)
        criterion = nn.MSELoss()
        optimizer = optim.SGD(self.parameters(), lr=lr)
        
        train_losses = []
        val_losses = []
        best_val_loss = float('inf')
        patience_counter = 0
        best_epoch = 0
        
        for epoch in range(epochs):
            self.train()
            epoch_loss = 0.0
            for X_batch, y_batch in train_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                
                optimizer.zero_grad()
                outputs = self(X_batch)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
            
            train_losses.append(epoch_loss / len(train_loader))
            
            if val_loader is not None:
                self.eval()
                with torch.no_grad():
                    val_loss = 0.0
                    for X_batch, y_batch in val_loader:
                        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                        outputs = self(X_batch)
                        loss = criterion(outputs, y_batch)
                        val_loss += loss.item()
                    val_loss = val_loss / len(val_loader)
                    val_losses.append(val_loss)
                    
                    # Early stopping logic
                    if early_stopping:
                        if val_loss < best_val_loss:
                            best_val_loss = val_loss
                            patience_counter = 0
                            best_epoch = epoch
                        else:
                            patience_counter += 1
                            if patience_counter >= patience:
                                if verbose:
                                    print(f"Early stopping at epoch {epoch+1} (best: {best_epoch+1})")
                                break
            
            if verbose and (epoch + 1) % 10 == 0:
                val_loss_str = f"{val_losses[-1]:.4f}" if val_losses else "N/A"
                print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {train_losses[-1]:.4f}, "
                      f"Val Loss: {val_loss_str}")
        
        return {
            'train_losses': train_losses,
            'val_losses': val_losses if val_loader else None,
            'best_epoch': best_epoch,
            'best_val_loss': best_val_loss if early_stopping else None,
            'early_stopped': early_stopping and patience_counter >= patience
        }
    
    def predict(self, X):
        """Make predictions."""
        self.eval()
        with torch.no_grad():
            if isinstance(X, np.ndarray):
                X = torch.FloatTensor(X).to(device)
            else:
                X = X.to(device)
            outputs = self(X)
            return outputs.cpu().numpy().flatten()
    
    def save(self, filepath):
        """Save model state dict."""
        torch.save(self.state_dict(), filepath)
    
    def load(self, filepath):
        """Load model state dict."""
        self.load_state_dict(torch.load(filepath))


def build_model(n_features):
    """Build and return PyTorch linear regression model."""
    model = PyTorchLinearRegression(n_features)
    return model


def train(model, dataloaders, epochs=100, lr=0.01, verbose=True, early_stopping=False, patience=20):
    """Train the model and return training history."""
    history = model.fit(
        dataloaders['train_loader'],
        val_loader=dataloaders['val_loader'],
        epochs=epochs,
        lr=lr,
        verbose=verbose,
        early_stopping=early_stopping,
        patience=patience
    )
    return history


def evaluate(model, dataloaders, y_scaler):
    """
    Evaluate model on validation set and return metrics.
    Returns dict with MSE, RMSE, R2, MAE.
    """
    model.eval()
    with torch.no_grad():
        # Get predictions for validation set
        X_val = dataloaders['X_val']
        y_val = dataloaders['y_val']
        
        if isinstance(X_val, torch.Tensor):
            X_val = X_val.cpu().numpy()
        if isinstance(y_val, torch.Tensor):
            y_val = y_val.cpu().numpy()
        
        y_pred = model.predict(X_val)
        
        # Inverse transform to get actual values
        y_val_actual = y_scaler.inverse_transform(y_val.reshape(-1, 1)).flatten()
        y_pred_actual = y_scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
        
        # Calculate metrics
        mse = mean_squared_error(y_val_actual, y_pred_actual)
        rmse = np.sqrt(mse)
        r2 = r2_score(y_val_actual, y_pred_actual)
        mae = mean_absolute_error(y_val_actual, y_pred_actual)
        
        return {
            'mse': float(mse),
            'rmse': float(rmse),
            'r2': float(r2),
            'mae': float(mae)
        }


def predict(model, X):
    """Make predictions on new data."""
    return model.predict(X)


def save_artifacts(model_no_es, model_with_es, dataloaders, metrics_no_es, metrics_with_es, sklearn_metrics, history_no_es, history_with_es, metadata):
    """Save all artifacts to output directory."""
    
    # Save models
    model_no_es_path = os.path.join(OUTPUT_DIR, 'pytorch_model_no_early_stopping.pth')
    model_no_es.save(model_no_es_path)
    
    model_with_es_path = os.path.join(OUTPUT_DIR, 'pytorch_model_with_early_stopping.pth')
    model_with_es.save(model_with_es_path)
    
    # Save metrics comparing both approaches
    metrics_path = os.path.join(OUTPUT_DIR, 'metrics.json')
    all_metrics = {
        'metadata': metadata,
        'pytorch_metrics_no_early_stopping': metrics_no_es,
        'pytorch_metrics_with_early_stopping': metrics_with_es,
        'sklearn_metrics': sklearn_metrics,
        'comparison_early_stopping': {
            'r2_improvement': metrics_with_es['r2'] - metrics_no_es['r2'],
            'rmse_reduction': metrics_no_es['rmse'] - metrics_with_es['rmse'],
            'mse_reduction': metrics_no_es['mse'] - metrics_with_es['mse']
        },
        'sklearn_comparison': {
            'r2_diff_no_es': abs(metrics_no_es['r2'] - sklearn_metrics['r2']),
            'rmse_diff_no_es': abs(metrics_no_es['rmse'] - sklearn_metrics['rmse']),
            'r2_diff_with_es': abs(metrics_with_es['r2'] - sklearn_metrics['r2']),
            'rmse_diff_with_es': abs(metrics_with_es['rmse'] - sklearn_metrics['rmse'])
        }
    }
    with open(metrics_path, 'w') as f:
        json.dump(all_metrics, f, indent=2)
    
    # Save training histories
    history_path = os.path.join(OUTPUT_DIR, 'training_history.json')
    history_data = {
        'no_early_stopping': history_no_es,
        'with_early_stopping': history_with_es
    }
    with open(history_path, 'w') as f:
        json.dump(history_data, f, indent=2)
    
    # Save EDA plots
    # Correlation matrix
    X_train = dataloaders['X_train']
    X_train_inv = dataloaders['X_scaler'].inverse_transform(X_train)
    y_train = dataloaders['y_train']
    y_train_inv = dataloaders['y_scaler'].inverse_transform(y_train.reshape(-1, 1)).flatten()
    
    # Create correlation matrix plot
    feature_names = dataloaders['sklearn_data']['feature_names']
    all_data = np.column_stack([X_train_inv, y_train_inv])
    all_features = list(feature_names) + ['disease_progression']
    
    plt.figure(figsize=(10, 8))
    corr_matrix = np.corrcoef(all_data.T)
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', 
                xticklabels=all_features, yticklabels=all_features,
                cmap='coolwarm', center=0)
    plt.title('Correlation Matrix - Diabetes Dataset')
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'correlation_matrix.png'), dpi=150)
    plt.close()
    
    # Target distribution
    plt.figure(figsize=(10, 6))
    plt.hist(y_train_inv, bins=30, edgecolor='black', alpha=0.7)
    plt.xlabel('Disease Progression')
    plt.ylabel('Frequency')
    plt.title('Target Distribution - Diabetes Dataset')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'target_distribution.png'), dpi=150)
    plt.close()
    
    # Training loss comparison: no early stopping vs with early stopping
    plt.figure(figsize=(12, 5))
    
    # Plot 1: No early stopping
    plt.subplot(1, 2, 1)
    if history_no_es and 'train_losses' in history_no_es:
        plt.plot(history_no_es['train_losses'], label='Train Loss', linewidth=2)
        if history_no_es.get('val_losses'):
            plt.plot(history_no_es['val_losses'], label='Val Loss', linewidth=2)
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training History - NO Early Stopping')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Plot 2: With early stopping
    plt.subplot(1, 2, 2)
    if history_with_es and 'train_losses' in history_with_es:
        plt.plot(history_with_es['train_losses'], label='Train Loss', linewidth=2)
        if history_with_es.get('val_losses'):
            plt.plot(history_with_es['val_losses'], label='Val Loss', linewidth=2)
        if history_with_es.get('best_epoch') is not None:
            best_epoch = history_with_es['best_epoch']
            plt.axvline(x=best_epoch, color='red', linestyle='--', label=f'Best Epoch ({best_epoch+1})')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.title('Training History - WITH Early Stopping')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_history_comparison.png'), dpi=150)
    plt.close()
    
    # Overlay comparison of the two approaches
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    if history_no_es and 'train_losses' in history_no_es and history_with_es and 'train_losses' in history_with_es:
        plt.plot(history_no_es['train_losses'], label='No ES - Train', linewidth=2, alpha=0.7)
        plt.plot(history_with_es['train_losses'], label='With ES - Train', linewidth=2, alpha=0.7)
    plt.xlabel('Epoch')
    plt.ylabel('Train Loss')
    plt.title('Train Loss Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(1, 2, 2)
    if history_no_es and history_with_es and history_no_es.get('val_losses') and history_with_es.get('val_losses'):
        plt.plot(history_no_es['val_losses'], label='No ES - Val', linewidth=2, alpha=0.7)
        plt.plot(history_with_es['val_losses'], label='With ES - Val', linewidth=2, alpha=0.7)
        if history_with_es.get('best_epoch') is not None:
            best_epoch = history_with_es['best_epoch']
            plt.axvline(x=best_epoch, color='red', linestyle='--', label=f'ES Stop ({best_epoch+1})')
    plt.xlabel('Epoch')
    plt.ylabel('Val Loss')
    plt.title('Validation Loss Comparison')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'training_history_overlay.png'), dpi=150)
    plt.close()


def main():
    """Main function to run the complete ML pipeline with early stopping comparison."""
    print("=" * 70)
    print("Linear Regression: Diabetes Dataset - Early Stopping Comparison")
    print("=" * 70)
    
    # Get metadata
    metadata = get_task_metadata()
    print(f"\nTask: {metadata['task_name']}")
    print(f"Dataset: {metadata['dataset']}")
    print(f"Device: {get_device()}")
    
    # Create dataloaders
    print("\n[1/6] Loading and preprocessing data...")
    dataloaders = make_dataloaders(test_size=0.2, batch_size=32)
    print(f"Training samples: {len(dataloaders['X_train'])}")
    print(f"Validation samples: {len(dataloaders['X_val'])}")
    print(f"Features: {dataloaders['n_features']}")
    
    # Build and train PyTorch model WITHOUT early stopping
    print("\n[2/6] Training PyTorch model WITHOUT early stopping...")
    pytorch_model_no_es = build_model(dataloaders['n_features'])
    print(f"Model architecture: {pytorch_model_no_es}")
    
    history_no_es = train(pytorch_model_no_es, dataloaders, epochs=200, lr=0.01, verbose=True, early_stopping=False)
    epochs_no_es = len(history_no_es['train_losses'])
    print(f"Completed {epochs_no_es} epochs")
    
    # Build and train PyTorch model WITH early stopping
    print("\n[3/6] Training PyTorch model WITH early stopping...")
    pytorch_model_with_es = build_model(dataloaders['n_features'])
    
    history_with_es = train(pytorch_model_with_es, dataloaders, epochs=200, lr=0.01, verbose=True, early_stopping=True, patience=20)
    epochs_with_es = len(history_with_es['train_losses'])
    early_stopped = history_with_es.get('early_stopped', False)
    print(f"Completed {epochs_with_es} epochs (Early Stopped: {early_stopped})")
    
    # Train sklearn model for comparison
    print("\n[4/6] Training Sklearn model...")
    sklearn_model = LinearRegression()
    sklearn_model.fit(dataloaders['X_train'], dataloaders['y_train'])
    
    # Evaluate models on validation set
    print("\n[5/6] Evaluating models...")
    pytorch_metrics_no_es = evaluate(pytorch_model_no_es, dataloaders, dataloaders['y_scaler'])
    pytorch_metrics_with_es = evaluate(pytorch_model_with_es, dataloaders, dataloaders['y_scaler'])
    
    # Sklearn evaluation
    y_val_pred_sklearn = sklearn_model.predict(dataloaders['X_val'])
    y_val_actual = dataloaders['y_scaler'].inverse_transform(
        dataloaders['y_val'].reshape(-1, 1)
    ).flatten()
    y_val_pred_sklearn_inv = dataloaders['y_scaler'].inverse_transform(
        y_val_pred_sklearn.reshape(-1, 1)
    ).flatten()
    
    sklearn_metrics = {
        'mse': float(mean_squared_error(y_val_actual, y_val_pred_sklearn_inv)),
        'rmse': float(np.sqrt(mean_squared_error(y_val_actual, y_val_pred_sklearn_inv))),
        'r2': float(r2_score(y_val_actual, y_val_pred_sklearn_inv)),
        'mae': float(mean_absolute_error(y_val_actual, y_val_pred_sklearn_inv))
    }
    
    # Print results
    print("\n" + "=" * 70)
    print("RESULTS - VALIDATION SET")
    print("=" * 70)
    
    print(f"\nPyTorch Model (NO Early Stopping - {epochs_no_es} epochs):")
    print(f"  MSE:  {pytorch_metrics_no_es['mse']:.6f}")
    print(f"  RMSE: {pytorch_metrics_no_es['rmse']:.6f}")
    print(f"  R2:   {pytorch_metrics_no_es['r2']:.6f}")
    print(f"  MAE:  {pytorch_metrics_no_es['mae']:.6f}")
    
    print(f"\nPyTorch Model (WITH Early Stopping - {epochs_with_es} epochs):")
    print(f"  MSE:  {pytorch_metrics_with_es['mse']:.6f}")
    print(f"  RMSE: {pytorch_metrics_with_es['rmse']:.6f}")
    print(f"  R2:   {pytorch_metrics_with_es['r2']:.6f}")
    print(f"  MAE:  {pytorch_metrics_with_es['mae']:.6f}")
    
    print(f"\nSklearn Model:")
    print(f"  MSE:  {sklearn_metrics['mse']:.6f}")
    print(f"  RMSE: {sklearn_metrics['rmse']:.6f}")
    print(f"  R2:   {sklearn_metrics['r2']:.6f}")
    print(f"  MAE:  {sklearn_metrics['mae']:.6f}")
    
    # Early Stopping Analysis
    print("\n" + "=" * 70)
    print("EARLY STOPPING COMPARISON")
    print("=" * 70)
    print(f"\nEpochs trained (No ES): {epochs_no_es}")
    print(f"Epochs trained (With ES): {epochs_with_es}")
    print(f"Epochs saved: {epochs_no_es - epochs_with_es}")
    
    r2_diff_es = pytorch_metrics_with_es['r2'] - pytorch_metrics_no_es['r2']
    rmse_diff_es = pytorch_metrics_no_es['rmse'] - pytorch_metrics_with_es['rmse']
    
    print(f"\nPerformance Difference (With ES - No ES):")
    print(f"  R2 improvement:   {r2_diff_es:.6f}")
    print(f"  RMSE reduction:   {rmse_diff_es:.6f}")
    print(f"  MSE reduction:    {pytorch_metrics_no_es['mse'] - pytorch_metrics_with_es['mse']:.6f}")
    print(f"  MAE reduction:    {pytorch_metrics_no_es['mae'] - pytorch_metrics_with_es['mae']:.6f}")
    
    if r2_diff_es > 0:
        print(f"\n✓ Early stopping achieved BETTER validation performance (R2 +{r2_diff_es:.6f})")
    elif abs(r2_diff_es) < 0.01:
        print(f"\n≈ Early stopping achieved SIMILAR performance with fewer epochs")
    else:
        print(f"\n✗ Early stopping achieved slightly lower performance (R2 {r2_diff_es:.6f})")
    
    # Comparison with Sklearn
    print("\n" + "=" * 70)
    print("SKLEARN COMPARISON")
    print("=" * 70)
    r2_diff_no_es = abs(pytorch_metrics_no_es['r2'] - sklearn_metrics['r2'])
    rmse_diff_no_es = abs(pytorch_metrics_no_es['rmse'] - sklearn_metrics['rmse'])
    r2_diff_with_es = abs(pytorch_metrics_with_es['r2'] - sklearn_metrics['r2'])
    rmse_diff_with_es = abs(pytorch_metrics_with_es['rmse'] - sklearn_metrics['rmse'])
    
    print(f"\nPyTorch (No ES) vs Sklearn:")
    print(f"  R2 Difference: {r2_diff_no_es:.6f}")
    print(f"  RMSE Difference: {rmse_diff_no_es:.6f}")
    
    print(f"\nPyTorch (With ES) vs Sklearn:")
    print(f"  R2 Difference: {r2_diff_with_es:.6f}")
    print(f"  RMSE Difference: {rmse_diff_with_es:.6f}")
    
    # Quality checks
    print("\n" + "=" * 70)
    print("QUALITY CHECKS")
    print("=" * 70)
    
    checks_passed = True
    
    # Check 1: No ES model R2 > 0.4
    if pytorch_metrics_no_es['r2'] > 0.4:
        print(f"✓ PyTorch (No ES) R2 > 0.4: {pytorch_metrics_no_es['r2']:.6f}")
    else:
        print(f"✗ PyTorch (No ES) R2 > 0.4: {pytorch_metrics_no_es['r2']:.6f}")
        checks_passed = False
    
    # Check 2: With ES model R2 > 0.3
    if pytorch_metrics_with_es['r2'] > 0.3:
        print(f"✓ PyTorch (With ES) R2 > 0.3: {pytorch_metrics_with_es['r2']:.6f}")
    else:
        print(f"✗ PyTorch (With ES) R2 > 0.3: {pytorch_metrics_with_es['r2']:.6f}")
        checks_passed = False
    
    # Check 3: No ES model RMSE < 100
    if pytorch_metrics_no_es['rmse'] < 100:
        print(f"✓ PyTorch (No ES) RMSE < 100: {pytorch_metrics_no_es['rmse']:.6f}")
    else:
        print(f"✗ PyTorch (No ES) RMSE < 100: {pytorch_metrics_no_es['rmse']:.6f}")
        checks_passed = False
    
    # Check 4: With ES model RMSE < 100
    if pytorch_metrics_with_es['rmse'] < 100:
        print(f"✓ PyTorch (With ES) RMSE < 100: {pytorch_metrics_with_es['rmse']:.6f}")
    else:
        print(f"✗ PyTorch (With ES) RMSE < 100: {pytorch_metrics_with_es['rmse']:.6f}")
        checks_passed = False
    
    # Check 5: No ES R2 vs Sklearn < 0.05 difference
    if r2_diff_no_es < 0.05:
        print(f"✓ PyTorch (No ES) vs Sklearn R2 diff < 0.05: {r2_diff_no_es:.6f}")
    else:
        print(f"✗ PyTorch (No ES) vs Sklearn R2 diff < 0.05: {r2_diff_no_es:.6f}")
        checks_passed = False
    
    # Check 6: Early stopping saved significant epochs
    epochs_saved = epochs_no_es - epochs_with_es
    if epochs_saved > 50:
        print(f"✓ Early stopping saved > 50 epochs: {epochs_saved} epochs")
    else:
        print(f"✗ Early stopping saved > 50 epochs: {epochs_saved} epochs")
        checks_passed = False
    
    # Check 7: Both models completed training
    if epochs_no_es > 0 and epochs_with_es > 0:
        print(f"✓ Both models completed training successfully")
    else:
        print(f"✗ One or both models failed to complete training")
        checks_passed = False
    
    # Save artifacts
    print("\n[6/6] Saving artifacts...")
    save_artifacts(pytorch_model_no_es, pytorch_model_with_es, dataloaders, 
                   pytorch_metrics_no_es, pytorch_metrics_with_es, sklearn_metrics, 
                   history_no_es, history_with_es, metadata)
    
    print(f"\nAll artifacts saved to: {OUTPUT_DIR}")
    
    # Final summary
    print("=" * 70)
    if checks_passed:
        print("PASS: All quality checks passed!")
    else:
        print("FAIL: Some quality checks failed!")
    print("=" * 70)

    # Exit with appropriate code
    return 0 if checks_passed else 1


if __name__ == "__main__":
    exit(main())
