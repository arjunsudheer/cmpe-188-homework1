"""
Transfer Learning (ResNet18 on Olivetti Faces + Optimizer Comparison)
Fine-tune pretrained ResNet18 on sklearn Olivetti Faces (400 images, 40 classes); 
compare Adam vs SGD with Nesterov; implement grayscale-to-RGB expansion.
"""

import os
import sys
import time
import json
import copy
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import models, transforms
from sklearn.datasets import fetch_olivetti_faces
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, mean_squared_error, r2_score
import matplotlib.pyplot as plt
from PIL import Image

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Output directory for artifacts
OUTPUT_DIR = './tasks/cnn_lvl3_resnet_transfer_olivetti_faces/output'
os.makedirs(OUTPUT_DIR, exist_ok=True)

class OlivettiDataset(Dataset):
    """Custom Dataset for sklearn Olivetti Faces."""
    def __init__(self, images, targets, transform=None):
        # Images should be (N, 64, 64) numpy array
        self.images = images
        self.targets = targets
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Transpose to (64, 64) and convert to PIL image for torchvision transforms
        image = Image.fromarray((self.images[idx] * 255).astype(np.uint8))
        target = self.targets[idx]
        
        if self.transform:
            image = self.transform(image)
        
        return image, target


def get_task_metadata():
    """Return metadata about the ML task."""
    return {
        'series': 'Convolutional Neural Networks',
        'level': 3,
        'id': 'cnn_lvl3_resnet_transfer_olivetti_faces',
        'algorithm': 'Transfer Learning (ResNet18 on Olivetti Faces + Optimizer Comparison)',
        'description': 'Fine-tune pretrained ResNet18 on sklearn Olivetti Faces dataset; compare Adam vs SGD with Nesterov; handle grayscale to RGB conversion.',
        'interface_protocol': 'pytorch_task_v1',
        'going_beyond': 'Uses the sklearn Olivetti faces dataset, which requires handling grayscale images for a model (ResNet) pretrained on RGB ImageNet. It compares two different optimization strategies — Adam (adaptive) vs SGD with Nesterov Momentum and Weight Decay (classic) — to evaluate which converges better on a very small, high-dimensional dataset (10 samples per class). It also implements a staged fine-tuning approach: first training the head, then unfreezing and fine-tuning the entire network.',
        'requirements': {
            'data': 'sklearn.datasets.fetch_olivetti_faces; 400 images of 64x64 pixel grayscale faces; stratified 80/20 train/val split (since dataset is very small, we skip a separate test set or use val as test).',
            'preprocessing': 'Convert grayscale [1, 64, 64] to RGB [3, 64, 64] by repeating channels; Normalize with ImageNet mean/std.',
            'implementation': 'Compare two optimizers: (1) Adam (lr=1e-3), (2) SGD with Nesterov Momentum (lr=1e-3, momentum=0.9, nesterov=True) and Weight Decay (1e-4). Use a two-stage training strategy: 10 epochs with a frozen base (head only), followed by 10 epochs unfrozen (fine-tuning).',
            'evaluation': 'evaluate() returns accuracy, loss, MSE, and R2 (between probabilities and one-hot labels). Report metrics for both optimization methods.',
            'visualization': "Save 'optimizer_comparison.png' showing validation accuracy curves; Save 'olivetti_predictions.png' showing a grid of test images with predictions.",
            'validation': 'Final validation accuracy for the best model must be >= 0.85; Both optimizers must reach >= 0.50 val accuracy. Compare and print which optimizer performed better.'
        }
    }

def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def get_device():
    """Get the computation device (GPU/CPU)."""
    return device

def make_dataloaders(batch_size=32, val_ratio=0.2):
    """
    Create data loaders for the Olivetti faces dataset from sklearn.
    
    Returns:
        train_loader, val_loader, class_mapping
    """
    # Fetch dataset
    faces = fetch_olivetti_faces(shuffle=True, random_state=42)
    X = faces.images
    y = faces.target
    
    # Split: 80% train / 20% validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=val_ratio, random_state=42, stratify=y
    )
    
    # Transformations: 
    # 1. Grayscale to RGB (Handled by transforms.Grayscale(3))
    # 2. ResNet standards (Resize to at least 64x64, though they are already 64x64)
    # 3. Normalization (ImageNet)
    
    transform_train = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),  # Expands grayscale to RGB
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    transform_val = transforms.Compose([
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Datasets
    train_dataset = OlivettiDataset(X_train, y_train, transform=transform_train)
    val_dataset = OlivettiDataset(X_val, y_val, transform=transform_val)
    
    # Loaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    
    return train_loader, val_loader, {}

def build_model(num_classes=40, pretrained=True, freeze_base=True):
    """Build ResNet18 model for transfer learning."""
    if pretrained:
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    else:
        model = models.resnet18(weights=None)
    
    # Modify the final FC layer
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    
    if freeze_base:
        for name, param in model.named_parameters():
            if "fc" not in name:
                param.requires_grad = False
                
    return model.to(device)

def convert_to_python_scalars(obj):
    """Recursively convert tensors and numpy arrays to Python scalars."""
    if isinstance(obj, torch.Tensor):
        return obj.item() if obj.numel() == 1 else obj.tolist()
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {k: convert_to_python_scalars(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_python_scalars(item) for item in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    else:
        return obj

def train(model, train_loader, val_loader, criterion, optimizer, scheduler=None, num_epochs=10):
    """Standard training loop."""
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device).long()
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
        train_loss = running_loss / total
        train_acc = correct / total
        
        # Evaluate
        val_results = evaluate(model, val_loader, criterion)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_results['loss'])
        history['val_acc'].append(val_results['accuracy'])
        
        if scheduler:
            scheduler.step(val_results['loss'])
            
    return model, history

def evaluate(model, data_loader, criterion):
    """Evaluate model and return metrics."""
    model.eval()
    running_loss = 0.0
    all_preds = []
    all_labels = []
    all_probs = []
    
    with torch.no_grad():
        for inputs, labels in data_loader:
            inputs, labels = inputs.to(device), labels.to(device).long()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.extend(probs.cpu().numpy())
            
    avg_loss = running_loss / len(data_loader.dataset)
    accuracy = accuracy_score(all_labels, all_preds)
    
    # Calculate MSE and R2 between one-hot labels and probabilities
    num_classes = 40
    one_hot_labels = np.zeros((len(all_labels), num_classes))
    for i, l in enumerate(all_labels):
        one_hot_labels[i, l] = 1.0
    
    mse = mean_squared_error(one_hot_labels, np.array(all_probs))
    r2 = r2_score(one_hot_labels.flatten(), np.array(all_probs).flatten())
    
    return {
        'loss': float(avg_loss),
        'accuracy': float(accuracy),
        'mse': float(mse),
        'r2': float(r2),
        'predictions': all_preds,
        'labels': all_labels
    }

def predict(model, data_loader):
    """Generate predictions."""
    model.eval()
    predictions = []
    with torch.no_grad():
        for inputs, _ in data_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            predictions.extend(predicted.cpu().numpy())
    return predictions

def save_artifacts(model, history, metadata, output_dir=OUTPUT_DIR):
    """Save model and plots."""
    os.makedirs(output_dir, exist_ok=True)
    
    # Save model
    torch.save(model.state_dict(), os.path.join(output_dir, 'model.pth'))
    
    # Save combined history if available, else best
    with open(os.path.join(output_dir, 'metrics.json'), 'w') as f:
        json.dump(convert_to_python_scalars(history), f, indent=2)
        
    # Plot accuracy curves
    plt.figure(figsize=(10, 6))
    for opt_name, hist in history.get('optimizer_comparison_histories', {}).items():
        plt.plot(hist['val_acc'], label=f'{opt_name} Val Acc')
    plt.title('Validation Accuracy Comparison')
    plt.xlabel('Epoch (Total)')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.savefig(os.path.join(output_dir, 'optimizer_comparison.png'))
    plt.close()

def main():
    set_seed(42)
    metadata = get_task_metadata()
    print(f"Starting Task: {metadata['id']}")
    
    # Data loaders
    train_loader, val_loader, _ = make_dataloaders(batch_size=32)
    print(f"Data ready. Classes: 40, Train: {len(train_loader.dataset)}, Val: {len(val_loader.dataset)}")
    
    criterion = nn.CrossEntropyLoss()
    
    optimizers_to_compare = {
        'Adam': {
            'type': optim.Adam,
            'params': {'lr': 1e-3}
        },
        'SGD_Nesterov': {
            'type': optim.SGD,
            'params': {'lr': 1e-3, 'momentum': 0.9, 'nesterov': True, 'weight_decay': 1e-4}
        }
    }
    
    comparison_results = {}
    best_val_acc = -1.0
    best_optimizer_name = ""
    best_model = None
    best_history = None
    
    for opt_name, opt_config in optimizers_to_compare.items():
        print(f"\nEvaluating Optimizer: {opt_name}")
        
        # Build model - reset per optimizer
        model = build_model(num_classes=40, pretrained=True, freeze_base=True)
        
        # Phase 1: Train Head (Frozen Base)
        optimizer = opt_config['type'](filter(lambda p: p.requires_grad, model.parameters()), **opt_config['params'])
        model, hist1 = train(model, train_loader, val_loader, criterion, optimizer, num_epochs=10)
        
        # Phase 2: Unfreeze and Fine-tune (Entire Model)
        for param in model.parameters():
            param.requires_grad = True
        
        # Use lower LR for fine-tuning
        ft_params = copy.deepcopy(opt_config['params'])
        ft_params['lr'] = ft_params['lr'] * 0.1
        optimizer = opt_config['type'](model.parameters(), **ft_params)
        model, hist2 = train(model, train_loader, val_loader, criterion, optimizer, num_epochs=10)
        
        # Merge histories
        combined_hist = {k: hist1[k] + hist2[k] for k in hist1}
        
        final_val_results = evaluate(model, val_loader, criterion)
        print(f"{opt_name} Final Accuracy: {final_val_results['accuracy']:.4f}")
        
        comparison_results[opt_name] = {
            'final_accuracy': final_val_results['accuracy'],
            'final_loss': final_val_results['loss'],
            'mse': final_val_results['mse'],
            'r2': final_val_results['r2'],
            'history': combined_hist
        }
        
        if final_val_results['accuracy'] > best_val_acc:
            best_val_acc = final_val_results['accuracy']
            best_optimizer_name = opt_name
            best_model = copy.deepcopy(model)
            best_history = combined_hist
            
    # Prepare artifacts data
    artifacts_history = {
        'best_optimizer': best_optimizer_name,
        'optimizer_comparison_metrics': {name: {k: v for k, v in res.items() if k != 'history'} for name, res in comparison_results.items()},
        'optimizer_comparison_histories': {name: res['history'] for name, res in comparison_results.items()},
        'final_test_metrics': evaluate(best_model, val_loader, criterion)
    }
    
    # Save artifacts
    save_artifacts(best_model, artifacts_history, metadata)
    print(f"\nBest Optimizer: {best_optimizer_name} with Accuracy {best_val_acc:.4f}")
    
    # Quality Assessment with Checkmarks
    print("\n" + "=" * 60)
    print("Quality Assessment")
    print("=" * 60)
    
    val_acc_pass = best_val_acc >= 0.85
    print(f"  {'✓ PASS' if val_acc_pass else '✗ FAIL'}: Best Validation Accuracy >= 0.85 (actual: {best_val_acc:.4f})")
    
    for opt_name, res in comparison_results.items():
        opt_pass = res['final_accuracy'] >= 0.50
        print(f"  {'✓ PASS' if opt_pass else '✗ FAIL'}: Optimizer {opt_name} reached threshold 0.50 (actual: {res['final_accuracy']:.4f})")
    
    print("\nQuality Thresholds Checked.")
    
    # Final assertions for exit code
    assert val_acc_pass, f"Validation accuracy too low: {best_val_acc:.4f}"
    for opt_name, res in comparison_results.items():
        assert res['final_accuracy'] >= 0.50, f"Optimizer {opt_name} failed threshold."
        
    print("\n" + "=" * 60)
    print("Task Complete")
    print("=" * 60)

if __name__ == '__main__':
    main()
