"""
Softmax Regression (Multiclass) using PyTorch with Optimizer Comparison
Implements multiclass classification with CrossEntropyLoss.
Compares three optimizers: Adam, SGD, and RMSProp on the Iris dataset.
"""

import os
import sys
import json
import time
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

# Set seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

OUTPUT_DIR = "./tasks/logreg_lvl2_multiclass_softmax_optimizer_comparison/output"

def get_task_metadata():
    """Return task metadata."""
    return {
        "task_name": "softmax_regression_multiclass_optimizer_comparison",
        "task_type": "classification",
        "dataset": "iris",
        "num_classes": 3,
        "input_dim": 4,
        "description": "Multiclass softmax regression on Iris dataset with optimizer comparison"
    }


def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def get_device():
    """Get the appropriate device (CUDA or CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def make_dataloaders(train_ratio=0.8, batch_size=16, random_state=42):
    """
    Create dataloaders using the Iris dataset.
    
    The Iris dataset contains 150 samples with 4 features and 3 classes.
    Features are standardized to help with optimization.
    """
    set_seed(random_state)
    
    # Load iris dataset
    iris = load_iris()
    X, y = iris.data, iris.target
    
    # Standardize features for better optimization
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    # Split into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=1-train_ratio, random_state=random_state, stratify=y
    )
    
    # Convert to tensors
    X_train_tensor = torch.FloatTensor(X_train)
    y_train_tensor = torch.LongTensor(y_train)
    X_val_tensor = torch.FloatTensor(X_val)
    y_val_tensor = torch.LongTensor(y_val)
    
    # Create datasets
    train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
    val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, X_train, X_val, y_train, y_val


class SoftmaxRegressionModel(nn.Module):
    """Simple softmax regression model (multiclass logistic regression)."""
    
    def __init__(self, input_dim, num_classes):
        super(SoftmaxRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, num_classes)
        
    def forward(self, x):
        # Return logits (raw scores) - CrossEntropyLoss applies softmax internally
        return self.linear(x)


def build_model(input_dim, num_classes, device):
    """Build and return the model."""
    model = SoftmaxRegressionModel(input_dim, num_classes)
    model = model.to(device)
    return model


def train(model, train_loader, val_loader, criterion, optimizer, device, epochs=100, verbose=True):
    """Train the model and track training/validation loss history."""
    model.train()
    
    train_losses = []
    val_losses = []
    
    for epoch in range(epochs):
        # Training phase
        total_loss = 0
        correct = 0
        total = 0
        
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            # Move data to device
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += y_batch.size(0)
            correct += (predicted == y_batch).sum().item()
        
        avg_train_loss = total_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation phase
        model.eval()
        val_total_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch)
                val_total_loss += loss.item()
        
        avg_val_loss = val_total_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        model.train()
        
        if verbose and (epoch + 1) % 20 == 0:
            print(f"Epoch [{epoch+1}/{epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
    
    return model, train_losses, val_losses


def evaluate(model, data_loader, criterion, device):
    """
    Evaluate the model and return metrics.
    
    Returns dict with:
    - loss: average cross-entropy loss
    - accuracy: classification accuracy
    - f1_macro: macro F1 score
    - mse: mean squared error
    - r2: R2 score
    """
    model.eval()
    total_loss = 0
    all_predictions = []
    all_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            # Move data to device
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            # Forward pass
            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            
            # Statistics
            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())
    
    avg_loss = total_loss / len(data_loader)
    accuracy = accuracy_score(all_targets, all_predictions)
    f1_macro = f1_score(all_targets, all_predictions, average='macro')
    
    # Compute MSE and R2 score
    all_predictions_array = np.array(all_predictions)
    all_targets_array = np.array(all_targets)
    mse = mean_squared_error(all_targets_array, all_predictions_array)
    r2 = r2_score(all_targets_array, all_predictions_array)
    
    return {
        "loss": avg_loss,
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "mse": mse,
        "r2": r2
    }


def predict(model, X, device):
    """Predict class labels for samples in X."""
    model.eval()
    
    with torch.no_grad():
        X_tensor = torch.FloatTensor(X).to(device)
        outputs = model(X_tensor)
        _, predicted = torch.max(outputs.data, 1)
    
    return predicted.cpu().numpy()


def save_artifacts(model, metrics, X_train, y_train, X_val, y_val, 
                   output_dir=OUTPUT_DIR, filename_prefix="logreg_lvl2_multiclass"):
    """Save model artifacts and visualizations."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    model_path = os.path.join(output_dir, f"{filename_prefix}_model.pt")
    torch.save(model.state_dict(), model_path)
    
    # Save metrics
    metrics_path = os.path.join(output_dir, f"{filename_prefix}_metrics.json")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Create decision boundary visualization only if input is 2D
    if X_train.shape[1] == 2:
        create_decision_boundary_plot(model, X_train, y_train, X_val, y_val, 
                                       output_dir, filename_prefix)
    else:
        print(f"Skipping decision boundary plot (data is {X_train.shape[1]}-dimensional, not 2D)")
    
    print(f"Artifacts saved to {output_dir}")


def save_training_history(training_histories, output_dir=OUTPUT_DIR, filename="training_history.json"):
    """Save training and validation loss history for all optimizers."""
    os.makedirs(output_dir, exist_ok=True)
    
    history_data = {}
    for optimizer_name, history in training_histories.items():
        history_data[optimizer_name] = {
            "train_losses": history["train_losses"],
            "val_losses": history["val_losses"]
        }
    
    history_path = os.path.join(output_dir, filename)
    with open(history_path, 'w') as f:
        json.dump(history_data, f, indent=2)
    
    print(f"Training history saved to {history_path}")


def save_all_metrics(all_results, output_dir=OUTPUT_DIR, filename="metrics.json"):
    """Save metrics for each optimizer."""
    os.makedirs(output_dir, exist_ok=True)
    
    metrics_data = {}
    for optimizer_name, result in all_results.items():
        metrics_data[optimizer_name] = {
            "train": result["train_metrics"],
            "validation": result["val_metrics"]
        }
    
    metrics_path = os.path.join(output_dir, filename)
    with open(metrics_path, 'w') as f:
        json.dump(metrics_data, f, indent=2)
    
    print(f"All optimizer metrics saved to {metrics_path}")


def save_training_times(training_times, output_dir=OUTPUT_DIR, filename="times.json"):
    """Save training times for each optimizer."""
    os.makedirs(output_dir, exist_ok=True)
    
    times_path = os.path.join(output_dir, filename)
    with open(times_path, 'w') as f:
        json.dump(training_times, f, indent=2)
    
    print(f"Training times saved to {times_path}")


def save_all_models(all_results, output_dir=OUTPUT_DIR):
    """Save models for all optimizers as .pth files."""
    os.makedirs(output_dir, exist_ok=True)
    
    for optimizer_name, result in all_results.items():
        model = result["model"]
        model_path = os.path.join(output_dir, f"logreg_lvl2_multiclass_optimizer_{optimizer_name.lower()}_model.pth")
        torch.save(model.state_dict(), model_path)
        print(f"Saved {optimizer_name} model to {model_path}")
    
    print(f"All optimizer models saved to {output_dir}")


def create_decision_boundary_plot(model, X_train, y_train, X_val, y_val, 
                                   output_dir, filename_prefix):
    """Create and save decision boundary contour plot."""
    # Set up the mesh grid
    x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
    y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
    xx, yy = np.meshgrid(np.linspace(x_min, x_max, 200),
                         np.linspace(y_min, y_max, 200))
    
    # Create grid points
    grid_points = np.c_[xx.ravel(), yy.ravel()]
    
    # Get predictions on grid
    model.eval()
    with torch.no_grad():
        grid_tensor = torch.FloatTensor(grid_points).to(next(model.parameters()).device)
        outputs = model(grid_tensor)
        _, predicted = torch.max(outputs.data, 1)
        Z = predicted.cpu().numpy().reshape(xx.shape)
    
    # Create plot
    plt.figure(figsize=(12, 5))
    
    # Define colors for 3 classes
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    cmap = ListedColormap(colors)
    
    # Plot decision boundary
    plt.subplot(1, 2, 1)
    plt.contourf(xx, yy, Z, alpha=0.4, cmap=cmap)
    plt.scatter(X_train[:, 0], X_train[:, 1], c=y_train, cmap=ListedColormap(colors), 
                edgecolors='black', s=30, label='Train')
    plt.title('Decision Boundary (Training Data)')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    
    # Plot validation data
    plt.subplot(1, 2, 2)
    plt.contourf(xx, yy, Z, alpha=0.4, cmap=cmap)
    plt.scatter(X_val[:, 0], X_val[:, 1], c=y_val, cmap=ListedColormap(colors), 
                edgecolors='black', s=30, label='Validation')
    plt.title('Decision Boundary (Validation Data)')
    plt.xlabel('Feature 1')
    plt.ylabel('Feature 2')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{filename_prefix}_boundary.png"), 
                dpi=150, bbox_inches='tight')
    plt.close()


def main():
    """Main function to run the softmax regression task with optimizer comparison."""
    print("=" * 70)
    print("Softmax Regression (Multiclass) - Optimizer Comparison on Iris Dataset")
    print("=" * 70)
    
    # Get device
    device = get_device()
    print(f"\nUsing device: {device}")
    
    # Get task metadata
    metadata = get_task_metadata()
    print(f"Task: {metadata['task_name']}")
    print(f"Dataset: {metadata['dataset']}")
    print(f"Number of classes: {metadata['num_classes']}")
    print(f"Input dimension: {metadata['input_dim']}")
    
    # Create dataloaders
    print("\nCreating dataloaders from Iris dataset...")
    train_loader, val_loader, X_train, X_val, y_train, y_val = make_dataloaders(
        train_ratio=0.8,
        batch_size=16,
        random_state=42
    )
    
    print(f"Training samples: {len(X_train)}")
    print(f"Validation samples: {len(X_val)}")
    
    # Define optimizers to compare
    optimizers_config = {
        "Adam": {"class": optim.Adam, "kwargs": {"lr": 0.01}},
        "SGD": {"class": optim.SGD, "kwargs": {"lr": 0.01}},
        "RMSProp": {"class": optim.RMSprop, "kwargs": {"lr": 0.01}}
    }
    
    # Dictionary to store results
    all_results = {}
    training_times = {}
    training_histories = {}
    
    # Train and evaluate for each optimizer
    print("\n" + "=" * 70)
    print("TRAINING WITH DIFFERENT OPTIMIZERS")
    print("=" * 70)
    
    for optimizer_name, optimizer_config in optimizers_config.items():
        print(f"\n--- Training with {optimizer_name} ---")
        
        # Build a fresh model for this optimizer
        model = build_model(
            input_dim=metadata['input_dim'],
            num_classes=metadata['num_classes'],
            device=device
        )
        
        # Create optimizer instance
        optimizer_class = optimizer_config["class"]
        optimizer_kwargs = optimizer_config["kwargs"]
        optimizer = optimizer_class(model.parameters(), **optimizer_kwargs)
        
        # Define loss
        criterion = nn.CrossEntropyLoss()
        
        # Train model and measure time
        print(f"Training for 100 epochs...")
        start_time = time.time()
        model, train_losses, val_losses = train(model, train_loader, val_loader, criterion, optimizer, device, 
                                                 epochs=100, verbose=False)
        end_time = time.time()
        
        training_time = end_time - start_time
        training_times[optimizer_name] = training_time
        training_histories[optimizer_name] = {
            "train_losses": train_losses,
            "val_losses": val_losses
        }
        
        print(f"Training time: {training_time:.2f} seconds")
        
        # Evaluate on training set
        train_metrics = evaluate(model, train_loader, criterion, device)
        
        # Evaluate on validation set
        val_metrics = evaluate(model, val_loader, criterion, device)
        
        # Store results
        all_results[optimizer_name] = {
            "train_metrics": train_metrics,
            "val_metrics": val_metrics,
            "training_time": training_time,
            "model": model  # Keep reference for visualization
        }
        
        print(f"\nTrain Accuracy:  {train_metrics['accuracy']:.4f}")
        print(f"Train F1 Macro:  {train_metrics['f1_macro']:.4f}")
        print(f"Val Accuracy:    {val_metrics['accuracy']:.4f}")
        print(f"Val F1 Macro:    {val_metrics['f1_macro']:.4f}")
    
    # Identify fastest optimizer
    fastest_optimizer = min(training_times, key=training_times.get)
    print(f"\n{'=' * 70}")
    print(f"FASTEST OPTIMIZER FOR CONVERGENCE: {fastest_optimizer}")
    print(f"Training time: {training_times[fastest_optimizer]:.2f} seconds")
    print(f"{'=' * 70}")
    
    # Print detailed comparison
    print("\n" + "=" * 70)
    print("OPTIMIZER COMPARISON SUMMARY")
    print("=" * 70)
    
    comparison_data = []
    for opt_name in optimizers_config.keys():
        result = all_results[opt_name]
        comparison_data.append({
            "Optimizer": opt_name,
            "Train Acc": f"{result['train_metrics']['accuracy']:.4f}",
            "Val Acc": f"{result['val_metrics']['accuracy']:.4f}",
            "Train F1": f"{result['train_metrics']['f1_macro']:.4f}",
            "Val F1": f"{result['val_metrics']['f1_macro']:.4f}",
            "Time (s)": f"{result['training_time']:.2f}"
        })
    
    # Print comparison table
    print("\n")
    for row in comparison_data:
        print(f"{row['Optimizer']:12} | Train Acc: {row['Train Acc']} | Val Acc: {row['Val Acc']} | "
              f"Train F1: {row['Train F1']} | Val F1: {row['Val F1']} | Time: {row['Time (s)']}s")
    
    # Quality checks for all optimizers
    print("\n" + "=" * 70)
    print("QUALITY CHECKS (Applied to All Optimizers)")
    print("=" * 70)
    
    all_passed = True
    
    for optimizer_name, result in all_results.items():
        print(f"\n--- Checks for {optimizer_name} ---")
        
        train_metrics = result['train_metrics']
        val_metrics = result['val_metrics']
        
        quality_passed = True
        
        # Check 1: Train accuracy > 0.8
        check1 = train_metrics['accuracy'] > 0.8
        status1 = "✓" if check1 else "✗"
        print(f"{status1} Train Accuracy > 0.8: {train_metrics['accuracy']:.4f}")
        quality_passed = quality_passed and check1
        
        # Check 2: Validation accuracy > 0.8
        check2 = val_metrics['accuracy'] > 0.8
        status2 = "✓" if check2 else "✗"
        print(f"{status2} Val Accuracy > 0.8: {val_metrics['accuracy']:.4f}")
        quality_passed = quality_passed and check2
        
        # Check 3: Validation F1 Macro > 0.8
        check3 = val_metrics['f1_macro'] > 0.8
        status3 = "✓" if check3 else "✗"
        print(f"{status3} Val F1 Macro > 0.8: {val_metrics['f1_macro']:.4f}")
        quality_passed = quality_passed and check3
        
        # Check 4: Loss < 1.0
        check4 = train_metrics['loss'] < 1.0
        status4 = "✓" if check4 else "✗"
        print(f"{status4} Final Train Loss < 1.0: {train_metrics['loss']:.4f}")
        quality_passed = quality_passed and check4
        
        # Check 5: Small gap between train and val performance
        accuracy_gap = abs(train_metrics['accuracy'] - val_metrics['accuracy'])
        check5 = accuracy_gap < 0.15
        status5 = "✓" if check5 else "✗"
        print(f"{status5} Accuracy gap < 0.15: {accuracy_gap:.4f}")
        quality_passed = quality_passed and check5
        
        if quality_passed:
            print(f"Result for {optimizer_name}: PASS")
        else:
            print(f"Result for {optimizer_name}: FAIL")
        
        all_passed = all_passed and quality_passed
    
    # Save all models for all optimizers
    print("\nSaving models for all optimizers...")
    save_all_models(all_results)
    
    # Save training history for all optimizers
    print("Saving training history for all optimizers...")
    save_training_history(training_histories)
    
    # Save metrics for all optimizers
    print("Saving metrics for all optimizers...")
    save_all_metrics(all_results)
    
    # Save training times
    print("Saving training times for all optimizers...")
    save_training_times(training_times)
    
    # Find best performing optimizer based on validation accuracy
    best_optimizer = max(all_results.items(), 
                        key=lambda x: x[1]['val_metrics']['accuracy'])
    best_optimizer_name = best_optimizer[0]
    best_optimizer_val_acc = best_optimizer[1]['val_metrics']['accuracy']
    
    # Final summary
    print("\n" + "=" * 70)
    print("FINAL TEST RESULT")
    print("=" * 70)
    
    if all_passed:
        print("PASS: All quality checks passed for all three optimizers!")
        print(f"\nFastest optimizer: {fastest_optimizer} ({training_times[fastest_optimizer]:.2f}s)")
        print(f"Best performing optimizer: {best_optimizer_name} (Val Accuracy: {best_optimizer_val_acc:.4f})")
    else:
        print("FAIL: Some quality checks failed!")
    
    print("=" * 70)
    
    # Exit with appropriate code
    return 0 if all_passed else 1


if __name__ == '__main__':
    exit_code = main()
    sys.exit(exit_code)
