"""
Model inference and basic evaluation utilities
"""
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix, balanced_accuracy_score
)
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

def evaluate_model_comprehensive(model, test_loader, device, label_map):
    """Basic model evaluation with essential metrics"""
    model.eval()
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    print("🔍 Running model evaluation...")
    
    with torch.no_grad():
        for batch in test_loader:
            texts = batch['text'].to(device)
            labels = batch['label'].to(device)
            lengths = batch['length']
            
            outputs = model(texts, lengths)
            probabilities = torch.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    all_probabilities = np.array(all_probabilities)
    reverse_label_map = {v: k for k, v in label_map.items()}
    target_names = [reverse_label_map[i] for i in sorted(reverse_label_map.keys())]
    
    # ============ BASIC METRICS ============
    print("\n📊 CLASSIFICATION METRICS")
    print("=" * 50)
    
    accuracy = accuracy_score(all_labels, all_predictions)
    balanced_acc = balanced_accuracy_score(all_labels, all_predictions)
    
    print(f"🎯 Accuracy: {accuracy:.4f}")
    print(f"⚖️  Balanced Accuracy: {balanced_acc:.4f}")
    
    # ============ PER-CLASS METRICS ============
    precision_per_class = precision_score(all_labels, all_predictions, average=None, zero_division=0)
    recall_per_class = recall_score(all_labels, all_predictions, average=None, zero_division=0)
    f1_per_class = f1_score(all_labels, all_predictions, average=None, zero_division=0)
    
    print(f"\n📋 Per-Class Metrics:")
    for i, class_name in enumerate(target_names):
        print(f"  {class_name.upper()}:")
        print(f"    Precision: {precision_per_class[i]:.4f}")
        print(f"    Recall:    {recall_per_class[i]:.4f}")
        print(f"    F1-Score:  {f1_per_class[i]:.4f}")
    
    # ============ AVERAGED METRICS ============
    precision_macro = precision_score(all_labels, all_predictions, average='macro', zero_division=0)
    precision_weighted = precision_score(all_labels, all_predictions, average='weighted', zero_division=0)
    
    recall_macro = recall_score(all_labels, all_predictions, average='macro', zero_division=0)
    recall_weighted = recall_score(all_labels, all_predictions, average='weighted', zero_division=0)
    
    f1_macro = f1_score(all_labels, all_predictions, average='macro', zero_division=0)
    f1_weighted = f1_score(all_labels, all_predictions, average='weighted', zero_division=0)
    
    print(f"\n📈 Average Metrics:")
    print(f"  Precision - Macro: {precision_macro:.4f}, Weighted: {precision_weighted:.4f}")
    print(f"  Recall    - Macro: {recall_macro:.4f}, Weighted: {recall_weighted:.4f}")
    print(f"  F1-Score  - Macro: {f1_macro:.4f}, Weighted: {f1_weighted:.4f}")
    
    # ============ DETAILED CLASSIFICATION REPORT ============
    print(f"\n📝 DETAILED CLASSIFICATION REPORT")
    print("=" * 50)
    report = classification_report(all_labels, all_predictions, target_names=target_names)
    print(report)
    
    # ============ CONFUSION MATRIX ============
    print(f"\n🔄 CONFUSION MATRIX")
    print("=" * 50)
    cm = confusion_matrix(all_labels, all_predictions)
    print("Confusion Matrix:")
    print(cm)
    
    # Create results dictionary
    results = {
        'basic_metrics': {
            'accuracy': accuracy,
            'balanced_accuracy': balanced_acc
        },
        'per_class_metrics': {
            'precision': precision_per_class.tolist(),
            'recall': recall_per_class.tolist(),
            'f1_score': f1_per_class.tolist(),
            'class_names': target_names
        },
        'averaged_metrics': {
            'precision': {'macro': precision_macro, 'weighted': precision_weighted},
            'recall': {'macro': recall_macro, 'weighted': recall_weighted},
            'f1_score': {'macro': f1_macro, 'weighted': f1_weighted}
        },
        'confusion_matrix': cm.tolist(),
        'classification_report': report,
        'predictions': all_predictions,
        'labels': all_labels,
        'probabilities': all_probabilities.tolist()
    }
    
    return results

def plot_comprehensive_evaluation(results):
    """Plot basic evaluation visualizations"""
    target_names = results['per_class_metrics']['class_names']
    cm = np.array(results['confusion_matrix'])
    
    fig = plt.figure(figsize=(15, 10))
    
    # 1. Confusion Matrix
    plt.subplot(2, 2, 1)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    # 2. Per-class Metrics Bar Plot
    plt.subplot(2, 2, 2)
    x = np.arange(len(target_names))
    width = 0.25
    
    precision = results['per_class_metrics']['precision']
    recall = results['per_class_metrics']['recall']
    f1 = results['per_class_metrics']['f1_score']
    
    plt.bar(x - width, precision, width, label='Precision', alpha=0.8)
    plt.bar(x, recall, width, label='Recall', alpha=0.8)
    plt.bar(x + width, f1, width, label='F1-Score', alpha=0.8)
    
    plt.xlabel('Classes')
    plt.ylabel('Score')
    plt.title('Per-Class Metrics')
    plt.xticks(x, target_names)
    plt.legend()
    plt.ylim(0, 1)
    
    # 3. Normalized Confusion Matrix
    plt.subplot(2, 2, 3)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=target_names, yticklabels=target_names)
    plt.title('Normalized Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    
    # 4. Class Distribution
    plt.subplot(2, 2, 4)
    labels = results['labels']
    unique, counts = np.unique(labels, return_counts=True)
    class_counts = [counts[unique == i][0] if i in unique else 0 for i in range(len(target_names))]
    
    plt.bar(target_names, class_counts, alpha=0.7, color=['lightcoral', 'lightblue', 'lightgreen'])
    plt.title('Class Distribution in Test Set')
    plt.xlabel('Classes')
    plt.ylabel('Count')
    plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.show()

def plot_training_curves(train_losses, val_losses, val_accuracies, model_name="CNN"):
    """Plot training curves"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    epochs = range(1, len(train_losses) + 1)
    
    ax1.plot(epochs, train_losses, 'b-', label='Training Loss')
    ax1.plot(epochs, val_losses, 'r-', label='Validation Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title(f'{model_name} - Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(epochs, val_accuracies, 'g-', label='Validation Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.set_title(f'{model_name} - Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(labels, predictions, target_names):
    """Plot simple confusion matrix"""
    cm = confusion_matrix(labels, predictions)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=target_names, yticklabels=target_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.tight_layout()
    plt.show()

def predict_sentiment(model, text, vocab, max_length, device):
    """Predict sentiment for a single text"""
    model.eval()
    
    # Encode text
    encoded = vocab.encode(text)
    
    # Pad or truncate
    if len(encoded) > max_length:
        encoded = encoded[:max_length]
    else:
        encoded.extend([0] * (max_length - len(encoded)))
    
    # Convert to tensor
    text_tensor = torch.tensor(encoded, dtype=torch.long).unsqueeze(0).to(device)
    length_tensor = torch.tensor([min(len(encoded), max_length)], dtype=torch.long)
    
    with torch.no_grad():
        output = model(text_tensor, length_tensor)
        probabilities = torch.softmax(output, dim=1)
        predicted_class = torch.argmax(output, dim=1).item()
    
    label_map = {0: 'negative', 1: 'neutral', 2: 'positive'}
    return label_map[predicted_class], probabilities[0].cpu().numpy()

# Keep the old function for compatibility
def evaluate_model(model, test_loader, device, label_map):
    """Simple evaluation (for compatibility)"""
    results = evaluate_model_comprehensive(model, test_loader, device, label_map)
    return {
        'accuracy': results['basic_metrics']['accuracy'],
        'f1_weighted': results['averaged_metrics']['f1_score']['weighted'],
        'f1_macro': results['averaged_metrics']['f1_score']['macro'],
        'report': results['classification_report'],
        'predictions': results['predictions'],
        'labels': results['labels']
    }
