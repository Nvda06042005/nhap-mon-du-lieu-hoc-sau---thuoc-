"""
=====================================================================
Đồ án: Phân loại viên thuốc từ hình ảnh bằng CNN
=====================================================================
- Dataset: VAIPE Pill Classification (cropped images)
- Models: ResNet-50, EfficientNet-B0, Vision Transformer (ViT)
- Technique: Multi-class classification + Transfer Learning
- Output: Confusion Matrix, Accuracy/Loss curves, Model comparison
=====================================================================
"""

import os
import sys
import json
import time
import copy
import random
import warnings
import argparse
from pathlib import Path
from collections import Counter

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from torchvision import transforms
import timm

from sklearn.metrics import (
    confusion_matrix, classification_report,
    accuracy_score, precision_score, recall_score, f1_score
)
from sklearn.model_selection import train_test_split
from tqdm import tqdm

warnings.filterwarnings('ignore')

def safe_name(name):
    """Thay ký tự không hợp lệ trong tên file (VD: ViT-B/16 → ViT-B-16)."""
    return name.replace('/', '-').replace('\\', '-').replace(':', '-')

# =====================================================================
# CONFIG
# =====================================================================
class Config:
    # Paths - đọc từ 3 thư mục đã chia sẵn
    DATASET_DIR = r'd:\projectdulieuhocsau17\dataset'
    TRAIN_DIR = os.path.join(DATASET_DIR, 'train')
    VAL_DIR = os.path.join(DATASET_DIR, 'val')
    TEST_DIR = os.path.join(DATASET_DIR, 'test')
    OUTPUT_DIR = r'd:\projectdulieuhocsau17\results'
    
    # Dataset
    IMG_SIZE = 224                 # Resize cho ResNet/EfficientNet/ViT
    
    # Training
    BATCH_SIZE = 32               # RTX 3050 4GB: mặc định cho ResNet/EfficientNet
    NUM_EPOCHS = 15
    LEARNING_RATE = 1e-4
    WEIGHT_DECAY = 1e-4
    NUM_WORKERS = 2               # GPU mode: dùng 2 workers load data song song
    USE_AMP = True                 # Mixed Precision (FP16) → tiết kiệm VRAM, tăng tốc
    
    # Batch size riêng cho từng model (RTX 3050 4GB VRAM)
    MODEL_BATCH_SIZE = {
        'resnet50': 32,
        'efficientnet_b0': 32,
        'vit': 16,              # ViT lớn hơn, giảm batch để không tràn VRAM
    }
    
    # Device
    DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Random seed
    SEED = 42


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =====================================================================
# DATASET - Đọc từ thư mục train/ val/ test/ đã chia sẵn
# =====================================================================
def prepare_data(config):
    """Load dataset từ 3 thư mục train/val/test đã chia sẵn."""
    print("=" * 60)
    print("BƯỚC 1: Chuẩn bị dữ liệu")
    print("=" * 60)
    print(f"  Dataset: {config.DATASET_DIR}")
    
    # Đếm ảnh trong từng thư mục
    for split_name, split_dir in [('Train', config.TRAIN_DIR), 
                                   ('Val', config.VAL_DIR), 
                                   ('Test', config.TEST_DIR)]:
        classes = [d for d in os.listdir(split_dir) if os.path.isdir(os.path.join(split_dir, d))]
        total = sum(len(os.listdir(os.path.join(split_dir, c))) for c in classes)
        print(f"  {split_name:5s}: {total:6d} ảnh, {len(classes)} lớp  ({split_dir})")
    
    # Lấy danh sách class
    train_classes = sorted(os.listdir(config.TRAIN_DIR))
    num_classes = len(train_classes)
    class_to_idx = {c: i for i, c in enumerate(train_classes)}
    idx_to_class = {i: c for c, i in class_to_idx.items()}
    
    print(f"\n  Tổng: {num_classes} lớp")
    
    return num_classes, class_to_idx, idx_to_class


def get_transforms(config):
    """Data augmentation + normalize theo ImageNet."""
    train_transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE + 32, config.IMG_SIZE + 32)),
        transforms.RandomCrop(config.IMG_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((config.IMG_SIZE, config.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    
    return train_transform, val_test_transform


def get_weighted_sampler(dataset):
    """Tạo WeightedRandomSampler để xử lý class imbalance."""
    targets = [s[1] for s in dataset.samples]  # ImageFolder.samples = [(path, class_idx), ...]
    class_counts = Counter(targets)
    weights = [1.0 / class_counts[label] for label in targets]
    sampler = WeightedRandomSampler(
        weights=weights,
        num_samples=len(weights),
        replacement=True
    )
    return sampler


# =====================================================================
# MODELS (Transfer Learning)
# =====================================================================
def create_model(model_name, num_classes):
    """Tạo model với transfer learning (freeze backbone, thay head)."""
    
    if model_name == 'resnet50':
        model = timm.create_model('resnet50', pretrained=True, num_classes=num_classes)
        # Thêm Dropout trước FC head để giảm overfitting
        in_features = model.fc.in_features
        model.fc = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes)
        )
        # Freeze tất cả layers trừ layer4 + fc
        for name, param in model.named_parameters():
            if 'layer4' not in name and 'fc' not in name:
                param.requires_grad = False
                
    elif model_name == 'efficientnet_b0':
        model = timm.create_model('efficientnet_b0', pretrained=True, num_classes=num_classes)
        # Thêm Dropout trước classifier head để giảm overfitting
        in_features = model.classifier.in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes)
        )
        # Freeze tất cả trừ các block cuối + classifier
        for name, param in model.named_parameters():
            if 'blocks.6' not in name and 'classifier' not in name and 'conv_head' not in name and 'bn2' not in name:
                param.requires_grad = False
                
    elif model_name == 'vit':
        model = timm.create_model('vit_base_patch16_224', pretrained=True, num_classes=num_classes)
        # Thêm Dropout trước head để giảm overfitting
        in_features = model.head.in_features
        model.head = nn.Sequential(
            nn.Dropout(p=0.3),
            nn.Linear(in_features, num_classes)
        )
        # Freeze tất cả trừ 2 block cuối + head
        for name, param in model.named_parameters():
            if 'blocks.10' not in name and 'blocks.11' not in name and 'head' not in name and 'norm' not in name:
                param.requires_grad = False
    else:
        raise ValueError(f"Model {model_name} not supported")
    
    # Đếm params
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    
    print(f"\n  Model: {model_name}")
    print(f"    Total params:     {total_params:,}")
    print(f"    Trainable params: {trainable_params:,}")
    print(f"    Frozen params:    {frozen_params:,}")
    
    return model


# =====================================================================
# TRAINING
# =====================================================================
def train_one_epoch(model, dataloader, criterion, optimizer, device, scaler=None):
    """Train 1 epoch với Mixed Precision (AMP)."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for images, labels in tqdm(dataloader, desc="  Training", leave=False):
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        optimizer.zero_grad(set_to_none=True)
        
        # Mixed Precision Training
        if scaler is not None:
            with torch.amp.autocast('cuda'):
                outputs = model(images)
                loss = criterion(outputs, labels)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
        
        running_loss += loss.item() * images.size(0)
        _, predicted = torch.max(outputs, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def evaluate(model, dataloader, criterion, device):
    """Evaluate on val/test set với AMP."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for images, labels in tqdm(dataloader, desc="  Evaluating", leave=False):
            images = images.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            with torch.amp.autocast('cuda', enabled=(device.type == 'cuda')):
                outputs = model(images)
                loss = criterion(outputs, labels)
            
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc, np.array(all_preds), np.array(all_labels)


def train_model(model, model_name, train_loader, val_loader, config):
    """Train loop với early stopping."""
    print(f"\n{'='*60}")
    print(f"TRAINING: {model_name.upper()}")
    print(f"{'='*60}")
    
    model = model.to(config.DEVICE)
    
    # Mixed Precision scaler
    scaler = torch.amp.GradScaler('cuda') if config.USE_AMP and config.DEVICE.type == 'cuda' else None
    
    # Class weights cho loss function
    criterion = nn.CrossEntropyLoss()
    
    # Optimizer: chỉ update trainable params
    optimizer = optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=config.LEARNING_RATE,
        weight_decay=config.WEIGHT_DECAY
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config.NUM_EPOCHS)
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    best_val_acc = 0.0
    best_model_state = None
    patience = 7
    patience_counter = 0
    
    start_time = time.time()
    
    for epoch in range(config.NUM_EPOCHS):
        epoch_start = time.time()
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, config.DEVICE, scaler
        )
        
        # Validate
        val_loss, val_acc, _, _ = evaluate(
            model, val_loader, criterion, config.DEVICE
        )
        
        scheduler.step()
        
        # Save history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        epoch_time = time.time() - epoch_start
        
        print(f"  Epoch [{epoch+1}/{config.NUM_EPOCHS}] "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
              f"Time: {epoch_time:.0f}s")
        
        # Best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_model_state = copy.deepcopy(model.state_dict())
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"  Early stopping at epoch {epoch+1}")
                break
    
    total_time = time.time() - start_time
    print(f"  Training time: {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"  Best Val Accuracy: {best_val_acc:.4f}")
    
    # Load best model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    return model, history, total_time


# =====================================================================
# EVALUATION & VISUALIZATION
# =====================================================================
def plot_training_curves(history, model_name, output_dir):
    """Vẽ Training/Validation Accuracy & Loss curves."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    # Loss
    ax1.plot(epochs, history['train_loss'], 'b-o', label='Training Loss', markersize=4)
    ax1.plot(epochs, history['val_loss'], 'r-o', label='Validation Loss', markersize=4)
    ax1.set_title(f'{model_name} - Training vs Validation Loss', fontsize=13)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Accuracy
    ax2.plot(epochs, history['train_acc'], 'b-o', label='Training Accuracy', markersize=4)
    ax2.plot(epochs, history['val_acc'], 'r-o', label='Validation Accuracy', markersize=4)
    ax2.set_title(f'{model_name} - Training vs Validation Accuracy', fontsize=13)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    path = os.path.join(output_dir, f'{safe_name(model_name)}_training_curves.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_confusion_matrix(y_true, y_pred, model_name, num_classes, output_dir, idx_to_class=None):
    """Vẽ Confusion Matrix."""
    cm = confusion_matrix(y_true, y_pred)
    
    fig, ax = plt.subplots(figsize=(20, 18))
    
    if num_classes <= 30:
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                    xticklabels=range(num_classes), yticklabels=range(num_classes))
    else:
        # Quá nhiều class → không annotate
        sns.heatmap(cm, annot=False, cmap='Blues', ax=ax)
    
    ax.set_title(f'{model_name} - Confusion Matrix ({num_classes} classes)', fontsize=14)
    ax.set_xlabel('Predicted Label', fontsize=12)
    ax.set_ylabel('True Label', fontsize=12)
    
    plt.tight_layout()
    path = os.path.join(output_dir, f'{safe_name(model_name)}_confusion_matrix.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")
    
    return cm


def plot_top_classes_cm(y_true, y_pred, model_name, output_dir, top_n=20):
    """Vẽ confusion matrix cho top N classes có nhiều mẫu nhất (dễ đọc hơn)."""
    class_counts = Counter(y_true)
    top_classes = [c for c, _ in class_counts.most_common(top_n)]
    
    mask = np.isin(y_true, top_classes)
    y_true_top = y_true[mask]
    y_pred_top = y_pred[mask]
    
    cm = confusion_matrix(y_true_top, y_pred_top, labels=top_classes)
    
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=top_classes, yticklabels=top_classes)
    ax.set_title(f'{model_name} - Confusion Matrix (Top {top_n} classes)', fontsize=14)
    ax.set_xlabel('Predicted', fontsize=12)
    ax.set_ylabel('True', fontsize=12)
    plt.tight_layout()
    path = os.path.join(output_dir, f'{safe_name(model_name)}_confusion_matrix_top{top_n}.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def compute_metrics(y_true, y_pred, model_name):
    """Tính Accuracy, Precision, Recall, F1."""
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, average='macro', zero_division=0)
    rec = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1 = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    print(f"\n  {model_name} - Test Metrics:")
    print(f"    Accuracy:  {acc:.4f}")
    print(f"    Precision: {prec:.4f} (macro)")
    print(f"    Recall:    {rec:.4f} (macro)")
    print(f"    F1-Score:  {f1:.4f} (macro)")
    
    return {'accuracy': acc, 'precision': prec, 'recall': rec, 'f1': f1}


def save_classification_report(y_true, y_pred, model_name, output_dir, idx_to_class=None):
    """Lưu classification report chi tiết per-class (Precision, Recall, F1 từng lớp)."""
    # Tạo target_names nếu có mapping
    labels = sorted(set(y_true) | set(y_pred))
    if idx_to_class:
        target_names = [str(idx_to_class.get(i, i)) for i in labels]
    else:
        target_names = [str(i) for i in labels]
    
    # Classification report dạng text
    report_text = classification_report(
        y_true, y_pred, labels=labels, target_names=target_names, zero_division=0
    )
    
    # Lưu file text
    txt_path = os.path.join(output_dir, f'{safe_name(model_name)}_classification_report.txt')
    with open(txt_path, 'w', encoding='utf-8') as f:
        f.write(f"Classification Report: {model_name}\n")
        f.write("=" * 70 + "\n\n")
        f.write(report_text)
    print(f"  Saved: {txt_path}")
    
    # Classification report dạng dict → JSON
    report_dict = classification_report(
        y_true, y_pred, labels=labels, target_names=target_names,
        zero_division=0, output_dict=True
    )
    json_path = os.path.join(output_dir, f'{safe_name(model_name)}_classification_report.json')
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(report_dict, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {json_path}")
    
    return report_dict


def analyze_overfitting(history, model_name, output_dir):
    """Phân tích overfitting dựa trên training vs validation curves."""
    train_acc = history['train_acc']
    val_acc = history['val_acc']
    train_loss = history['train_loss']
    val_loss = history['val_loss']
    
    final_train_acc = train_acc[-1]
    final_val_acc = val_acc[-1]
    best_val_acc = max(val_acc)
    best_val_epoch = val_acc.index(best_val_acc) + 1
    
    final_train_loss = train_loss[-1]
    final_val_loss = val_loss[-1]
    
    acc_gap = final_train_acc - final_val_acc
    loss_gap = final_val_loss - final_train_loss
    
    # Phân tích mức độ overfitting
    if acc_gap > 0.15:
        overfit_level = "CAO (High Overfitting)"
        suggestion = "Cần tăng Dropout, thêm augmentation, hoặc giảm model complexity."
    elif acc_gap > 0.08:
        overfit_level = "TRUNG BÌNH (Moderate Overfitting)"
        suggestion = "Có thể chấp nhận. Thử tăng nhẹ Dropout hoặc thêm regularization."
    elif acc_gap > 0.03:
        overfit_level = "THẤP (Mild Overfitting)"
        suggestion = "Tốt. Model generalize khá tốt."
    else:
        overfit_level = "KHÔNG OVERFITTING (Good Fit)"
        suggestion = "Model generalize rất tốt trên validation set."
    
    # Kiểm tra underfitting
    if final_val_acc < 0.5:
        underfit_note = "⚠ Val accuracy thấp (<50%), model có thể underfitting. Cần train thêm epochs hoặc unfreeze thêm layers."
    elif final_val_acc < 0.65:
        underfit_note = "⚠ Val accuracy chưa cao (<65%), có thể cần thêm epochs hoặc tuning."
    else:
        underfit_note = "Val accuracy ở mức chấp nhận."
    
    analysis = {
        'model': model_name,
        'epochs_trained': len(train_acc),
        'best_val_acc': round(best_val_acc, 4),
        'best_val_epoch': best_val_epoch,
        'final_train_acc': round(final_train_acc, 4),
        'final_val_acc': round(final_val_acc, 4),
        'accuracy_gap': round(acc_gap, 4),
        'final_train_loss': round(final_train_loss, 4),
        'final_val_loss': round(final_val_loss, 4),
        'loss_gap': round(loss_gap, 4),
        'overfitting_level': overfit_level,
        'suggestion': suggestion,
        'underfitting_note': underfit_note
    }
    
    # In phân tích
    print(f"\n  {model_name} - Phân tích Overfitting:")
    print(f"    Epochs trained:    {analysis['epochs_trained']}")
    print(f"    Best Val Acc:      {analysis['best_val_acc']:.4f} (epoch {best_val_epoch})")
    print(f"    Final Train Acc:   {analysis['final_train_acc']:.4f}")
    print(f"    Final Val Acc:     {analysis['final_val_acc']:.4f}")
    print(f"    Accuracy Gap:      {analysis['accuracy_gap']:.4f}")
    print(f"    Mức overfitting:   {analysis['overfitting_level']}")
    print(f"    Nhận xét:          {analysis['suggestion']}")
    print(f"    {analysis['underfitting_note']}")
    
    return analysis


def plot_sample_predictions(test_dataset, y_true, y_pred, model_name, output_dir, idx_to_class=None, num_samples=16):
    """Vẽ grid ảnh thuốc với kết quả dự đoán (đúng + sai)."""
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    def denormalize(tensor):
        """Convert tensor về lại ảnh hiển thị được."""
        img = tensor.cpu().numpy().transpose(1, 2, 0)
        img = img * std + mean
        img = np.clip(img, 0, 1)
        return img
    
    # Tìm các mẫu dự đoán ĐÚNG và SAI
    correct_indices = [i for i in range(len(y_true)) if y_pred[i] == y_true[i]]
    wrong_indices = [i for i in range(len(y_true)) if y_pred[i] != y_true[i]]
    
    # --- 1. Grid ảnh dự đoán ĐÚNG ---
    n_correct = min(num_samples, len(correct_indices))
    if n_correct > 0:
        selected = random.sample(correct_indices, n_correct)
        cols = 4
        rows = (n_correct + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
        if rows == 1:
            axes = [axes] if cols == 1 else axes
        axes_flat = np.array(axes).flatten()
        
        for idx, ax in enumerate(axes_flat):
            if idx < n_correct:
                sample_idx = selected[idx]
                img_tensor, _ = test_dataset[sample_idx]
                img = denormalize(img_tensor)
                true_label = y_true[sample_idx]
                
                pill_id = str(idx_to_class.get(true_label, true_label)) if idx_to_class else str(true_label)
                
                ax.imshow(img)
                ax.set_title(f'Thuốc ID: {pill_id}\n✅ Dự đoán đúng', 
                           fontsize=11, color='green', fontweight='bold')
                ax.axis('off')
            else:
                ax.axis('off')
        
        plt.suptitle(f'{model_name} - Mẫu dự đoán ĐÚNG ({n_correct} ảnh)', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        path = os.path.join(output_dir, f'{safe_name(model_name)}_correct_predictions.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {path}")
    
    # --- 2. Grid ảnh dự đoán SAI ---
    n_wrong = min(num_samples, len(wrong_indices))
    if n_wrong > 0:
        selected = random.sample(wrong_indices, n_wrong)
        cols = 4
        rows = (n_wrong + cols - 1) // cols
        fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
        if rows == 1:
            axes = [axes] if cols == 1 else axes
        axes_flat = np.array(axes).flatten()
        
        for idx, ax in enumerate(axes_flat):
            if idx < n_wrong:
                sample_idx = selected[idx]
                img_tensor, _ = test_dataset[sample_idx]
                img = denormalize(img_tensor)
                true_label = y_true[sample_idx]
                pred_label = y_pred[sample_idx]
                
                true_id = str(idx_to_class.get(true_label, true_label)) if idx_to_class else str(true_label)
                pred_id = str(idx_to_class.get(pred_label, pred_label)) if idx_to_class else str(pred_label)
                
                ax.imshow(img)
                ax.set_title(f'Thực tế: Thuốc {true_id}\n❌ Dự đoán: Thuốc {pred_id}', 
                           fontsize=11, color='red', fontweight='bold')
                ax.axis('off')
            else:
                ax.axis('off')
        
        plt.suptitle(f'{model_name} - Mẫu dự đoán SAI ({n_wrong} ảnh)', 
                    fontsize=14, fontweight='bold')
        plt.tight_layout()
        path = os.path.join(output_dir, f'{safe_name(model_name)}_wrong_predictions.png')
        plt.savefig(path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"  Saved: {path}")
    
    print(f"  Tổng: {len(correct_indices)} đúng / {len(wrong_indices)} sai "
          f"({len(correct_indices)/len(y_true)*100:.1f}% đúng)")


def plot_model_comparison(all_results, output_dir):
    """So sánh 3 models."""
    models = list(all_results.keys())
    metrics = ['accuracy', 'precision', 'recall', 'f1']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(metrics))
    width = 0.25
    colors = ['#2196F3', '#4CAF50', '#FF9800']
    
    for i, model_name in enumerate(models):
        values = [all_results[model_name]['metrics'][m] for m in metrics]
        bars = ax.bar(x + i * width, values, width, label=model_name, color=colors[i], alpha=0.85)
        # Thêm giá trị trên cột
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width()/2., bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Metrics', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Model Comparison - Test Performance', fontsize=14)
    ax.set_xticks(x + width)
    ax.set_xticklabels(['Accuracy', 'Precision', 'Recall', 'F1-Score'])
    ax.legend(fontsize=11)
    ax.set_ylim(0, 1.1)
    ax.grid(True, axis='y', alpha=0.3)
    
    plt.tight_layout()
    path = os.path.join(output_dir, 'model_comparison.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_all_training_curves(all_results, output_dir):
    """Vẽ training curves của cả 3 models trên 1 hình."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    colors = ['#2196F3', '#4CAF50', '#FF9800']
    
    for i, (model_name, result) in enumerate(all_results.items()):
        h = result['history']
        epochs = range(1, len(h['train_loss']) + 1)
        
        ax1.plot(epochs, h['val_loss'], '-o', color=colors[i], label=model_name, markersize=3)
        ax2.plot(epochs, h['val_acc'], '-o', color=colors[i], label=model_name, markersize=3)
    
    ax1.set_title('Validation Loss - All Models', fontsize=13)
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_title('Validation Accuracy - All Models', fontsize=13)
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    path = os.path.join(output_dir, 'all_models_comparison_curves.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def plot_dataset_distribution(train_df, val_df, test_df, num_classes, output_dir):
    """Vẽ biểu đồ phân bố dataset."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    for ax, df, title in zip(axes,
                              [train_df, val_df, test_df],
                              ['Train', 'Validation', 'Test']):
        counts = df['label_encoded'].value_counts().sort_index()
        ax.bar(counts.index, counts.values, color='steelblue', alpha=0.7)
        ax.set_title(f'{title} Set ({len(df)} images)', fontsize=12)
        ax.set_xlabel('Class ID')
        ax.set_ylabel('Count')
        ax.grid(True, axis='y', alpha=0.3)
    
    plt.suptitle(f'Dataset Distribution ({num_classes} classes)', fontsize=14, y=1.02)
    plt.tight_layout()
    path = os.path.join(output_dir, 'dataset_distribution.png')
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {path}")


def save_summary_report(all_results, num_classes, train_size, val_size, test_size, output_dir):
    """Lưu báo cáo tổng hợp."""
    report = {
        'dataset': {
            'num_classes': num_classes,
            'train_size': train_size,
            'val_size': val_size,
            'test_size': test_size,
            'image_size': '224x224',
            'augmentation': 'RandomCrop, HFlip, VFlip, Rotation, ColorJitter'
        },
        'models': {}
    }
    
    for model_name, result in all_results.items():
        report['models'][model_name] = {
            'test_metrics': result['metrics'],
            'best_val_acc': max(result['history']['val_acc']),
            'training_time_seconds': result['training_time'],
            'epochs_trained': len(result['history']['train_loss']),
            'overfitting_analysis': result.get('overfitting_analysis', {})
        }
    
    path = os.path.join(output_dir, 'training_report.json')
    with open(path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"  Saved: {path}")
    
    # Print summary table
    print("\n" + "=" * 75)
    print("TỔNG KẾT")
    print("=" * 75)
    print(f"{'Model':<20} {'Accuracy':>10} {'Precision':>10} {'Recall':>10} {'F1':>10} {'Time':>10}")
    print("-" * 75)
    for model_name, result in all_results.items():
        m = result['metrics']
        t = result['training_time']
        print(f"{model_name:<20} {m['accuracy']:>10.4f} {m['precision']:>10.4f} "
              f"{m['recall']:>10.4f} {m['f1']:>10.4f} {t:>8.0f}s")
    print("=" * 75)


# =====================================================================
# MAIN
# =====================================================================
def main():
    parser = argparse.ArgumentParser(description='Pill Classification with CNN')
    parser.add_argument('--models', nargs='+', 
                        default=['resnet50', 'efficientnet_b0', 'vit'],
                        choices=['resnet50', 'efficientnet_b0', 'vit'],
                        help='Models to train')
    parser.add_argument('--epochs', type=int, default=None,
                        help='Override number of epochs')
    parser.add_argument('--batch_size', type=int, default=None, 
                        help='Override batch size')
    args = parser.parse_args()
    
    config = Config()
    if args.epochs:
        config.NUM_EPOCHS = args.epochs
    if args.batch_size:
        config.BATCH_SIZE = args.batch_size
    
    set_seed(config.SEED)
    
    print(f"Device: {config.DEVICE}")
    print(f"Epochs: {config.NUM_EPOCHS}")
    if args.batch_size:
        print(f"Batch Size (override): {args.batch_size}")
    else:
        print(f"Batch Size per model: {config.MODEL_BATCH_SIZE}")
    
    # Create output dir
    os.makedirs(config.OUTPUT_DIR, exist_ok=True)
    
    # ===== 1. Prepare Data =====
    num_classes, class_to_idx, idx_to_class = prepare_data(config)
    
    # ===== 2. Transforms & Datasets (ImageFolder đọc từ thư mục) =====
    train_transform, val_test_transform = get_transforms(config)
    
    from torchvision.datasets import ImageFolder
    train_dataset = ImageFolder(config.TRAIN_DIR, transform=train_transform)
    val_dataset = ImageFolder(config.VAL_DIR, transform=val_test_transform)
    test_dataset = ImageFolder(config.TEST_DIR, transform=val_test_transform)
    
    # Weighted sampler cho class imbalance
    sampler = get_weighted_sampler(train_dataset)
    
    # ===== 3. Plot dataset distribution =====
    train_labels = [s[1] for s in train_dataset.samples]
    val_labels = [s[1] for s in val_dataset.samples]
    test_labels = [s[1] for s in test_dataset.samples]
    train_df = pd.DataFrame({'label_encoded': train_labels})
    val_df = pd.DataFrame({'label_encoded': val_labels})
    test_df = pd.DataFrame({'label_encoded': test_labels})
    plot_dataset_distribution(train_df, val_df, test_df, num_classes, config.OUTPUT_DIR)
    
    # ===== 4. Train each model =====
    all_results = {}
    model_configs = {
        'resnet50': 'ResNet-50',
        'efficientnet_b0': 'EfficientNet-B0',
        'vit': 'ViT-B/16'
    }
    
    criterion = nn.CrossEntropyLoss()
    
    for model_key in args.models:
        display_name = model_configs[model_key]
        
        # Batch size riêng cho từng model (để không tràn VRAM)
        bs = config.MODEL_BATCH_SIZE.get(model_key, config.BATCH_SIZE)
        if args.batch_size:
            bs = args.batch_size
        print(f"\n  Batch size cho {display_name}: {bs}")
        
        # Tạo DataLoader với batch size phù hợp
        train_loader = DataLoader(train_dataset, batch_size=bs,
                                  sampler=sampler, num_workers=config.NUM_WORKERS,
                                  pin_memory=True, persistent_workers=config.NUM_WORKERS > 0)
        val_loader = DataLoader(val_dataset, batch_size=bs,
                                shuffle=False, num_workers=config.NUM_WORKERS,
                                pin_memory=True, persistent_workers=config.NUM_WORKERS > 0)
        test_loader = DataLoader(test_dataset, batch_size=bs,
                                 shuffle=False, num_workers=config.NUM_WORKERS,
                                 pin_memory=True, persistent_workers=config.NUM_WORKERS > 0)
        
        # Create model
        model = create_model(model_key, num_classes)
        
        # Train
        model, history, training_time = train_model(
            model, display_name, train_loader, val_loader, config
        )
        
        # Evaluate on test set
        test_loss, test_acc, y_pred, y_true = evaluate(
            model, test_loader, criterion, config.DEVICE
        )
        
        metrics = compute_metrics(y_true, y_pred, display_name)
        
        # Classification Report per-class
        cls_report = save_classification_report(
            y_true, y_pred, display_name, config.OUTPUT_DIR, idx_to_class
        )
        
        # Phân tích Overfitting
        overfit_analysis = analyze_overfitting(history, display_name, config.OUTPUT_DIR)
        
        # Plot training curves
        plot_training_curves(history, display_name, config.OUTPUT_DIR)
        
        # Confusion Matrix
        plot_confusion_matrix(y_true, y_pred, display_name, num_classes, config.OUTPUT_DIR)
        plot_top_classes_cm(y_true, y_pred, display_name, config.OUTPUT_DIR, top_n=20)
        
        # Kết quả dự đoán trên ảnh thuốc thật (grid đúng + sai)
        plot_sample_predictions(
            test_dataset, y_true, y_pred, display_name,
            config.OUTPUT_DIR, idx_to_class, num_samples=16
        )
        
        # Save model
        model_path = os.path.join(config.OUTPUT_DIR, f'{model_key}_best.pth')
        torch.save(model.state_dict(), model_path)
        print(f"  Model saved: {model_path}")
        
        all_results[display_name] = {
            'metrics': metrics,
            'history': history,
            'training_time': training_time,
            'overfitting_analysis': overfit_analysis
        }
        
        # Free memory
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # ===== 5. Comparison =====
    if len(all_results) > 1:
        print(f"\n{'='*60}")
        print("SO SÁNH MODELS")
        print(f"{'='*60}")
        plot_model_comparison(all_results, config.OUTPUT_DIR)
        plot_all_training_curves(all_results, config.OUTPUT_DIR)
    
    # ===== 6. Save report =====
    save_summary_report(all_results, num_classes,
                        len(train_dataset), len(val_dataset), len(test_dataset),
                        config.OUTPUT_DIR)
    
    # Save class mapping
    mapping_path = os.path.join(config.OUTPUT_DIR, 'class_mapping.json')
    with open(mapping_path, 'w') as f:
        json.dump({str(k): v for k, v in idx_to_class.items()}, f, indent=2)
    
    print(f"\nTất cả kết quả đã lưu tại: {config.OUTPUT_DIR}")
    print("HOÀN THÀNH!")


if __name__ == '__main__':
    main()
