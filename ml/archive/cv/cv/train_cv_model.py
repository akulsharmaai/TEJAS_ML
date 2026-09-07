"""
TEJAS Computer Vision Pipeline: Fast In-Memory Augmented Transfer Learning
Architecture: MobileNetV3-Small (Pretrained)
Dataset: 5,153 Railway Track Surface Fault Images
Classes: ['Cracks', 'Flakings', 'Grooves', 'Joints', 'Shellings', 'Spallings', 'Squats']
"""

import os
import sys
import json
import random
import time
from collections import Counter
import numpy as np
import pandas as pd
from PIL import Image

if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(line_buffering=True)

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as transforms
import torchvision.transforms.functional as TF
import torchvision.models as models
from typing import cast, Dict, Any
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix, precision_recall_fscore_support, accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

DATASET_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", "Railway Track Surface Faults Dataset"))
OUTPUT_DIR = os.path.abspath(os.path.dirname(__file__))
os.makedirs(OUTPUT_DIR, exist_ok=True)

CLASS_NAMES = ['Cracks', 'Flakings', 'Grooves', 'Joints', 'Shellings', 'Spallings', 'Squats']
NUM_CLASSES = len(CLASS_NAMES)
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASS_NAMES)}
IDX_TO_CLASS = {i: c for i, c in enumerate(CLASS_NAMES)}

CV_TO_TEJAS_DEFECT_MAP = {
    'Cracks': 'structural crack indication',
    'Flakings': 'surface shelling/squat',
    'Grooves': 'corrugation/wear',
    'Joints': 'switch rail gap',
    'Shellings': 'surface shelling/squat',
    'Spallings': 'surface shelling/squat',
    'Squats': 'surface shelling/squat'
}


class InMemoryRailwayDataset(Dataset):
    def __init__(self, images_array, labels, transform=None):
        self.images = images_array  # (N, 224, 224, 3) uint8
        self.labels = np.array(labels, dtype=np.int64)
        self.transform = transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, index):
        img_np = self.images[index]
        img_pil = Image.fromarray(img_np)
        if self.transform:
            img_tensor = self.transform(img_pil)
        else:
            img_tensor = TF.to_tensor(img_pil)
        return img_tensor, self.labels[index]


def load_and_cache_dataset():
    print(f"Loading and pre-caching images (224x224) from: {DATASET_DIR}", flush=True)
    t0 = time.time()
    
    images_list = []
    labels_list = []
    file_paths_list = []
    
    for cls_name in CLASS_NAMES:
        cls_dir = os.path.join(DATASET_DIR, cls_name)
        files = [f for f in os.listdir(cls_dir) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        cls_idx = CLASS_TO_IDX[cls_name]
        
        for f in files:
            p = os.path.join(cls_dir, f)
            with Image.open(p) as img:
                img_rgb = img.convert('RGB').resize((224, 224), Image.Resampling.BILINEAR)
                images_list.append(np.array(img_rgb, dtype=np.uint8))
                labels_list.append(cls_idx)
                file_paths_list.append(p)
                
    images_array = np.stack(images_list, axis=0)
    print(f"Pre-cached {len(images_array)} images in {time.time()-t0:.1f}s | Array shape: {images_array.shape} ({images_array.nbytes / (1024**2):.1f} MB)", flush=True)
    
    counts = Counter(labels_list)
    for idx, count in sorted(counts.items()):
        print(f"  [{idx}] {IDX_TO_CLASS[idx]}: {count} images", flush=True)
        
    return images_array, labels_list, file_paths_list


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Executing CV Training on device: {device}", flush=True)
    
    images_array, labels_list, file_paths = load_and_cache_dataset()
    indices = np.arange(len(labels_list))
    
    # Stratified Split: 70% Train, 15% Val, 15% Test
    train_idx, test_idx, train_labels, test_labels = train_test_split(
        indices, labels_list, test_size=0.30, random_state=SEED, stratify=labels_list
    )
    val_idx, test_idx, val_labels, test_labels = train_test_split(
        test_idx, test_labels, test_size=0.50, random_state=SEED, stratify=test_labels
    )
    
    print(f"\nPartitioning: Train={len(train_idx)}, Val={len(val_idx)}, Test={len(test_idx)}", flush=True)
    
    # Transforms
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    eval_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    train_dataset = InMemoryRailwayDataset(images_array[train_idx], train_labels, transform=train_transform)
    val_dataset = InMemoryRailwayDataset(images_array[val_idx], val_labels, transform=eval_transform)
    test_dataset = InMemoryRailwayDataset(images_array[test_idx], test_labels, transform=eval_transform)
    
    # Loss Weights
    class_counts = np.bincount(train_labels, minlength=NUM_CLASSES)
    class_weights = len(train_labels) / (NUM_CLASSES * np.maximum(class_counts, 1).astype(float))
    class_weights = np.clip(class_weights, 0.3, 6.0)
    loss_weights = torch.tensor(class_weights, dtype=torch.float32).to(device)
    
    criterion = nn.CrossEntropyLoss(weight=loss_weights)
    
    sample_weights = [class_weights[l] for l in train_labels]
    sampler = WeightedRandomSampler(weights=sample_weights, num_samples=len(sample_weights), replacement=True)
    
    BATCH_SIZE = 64
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, sampler=sampler)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    # Build Model
    print("Building MobileNetV3-Small classifier...", flush=True)
    model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.DEFAULT)
    in_features = int(getattr(model.classifier[3], "in_features", 1024))
    model.classifier[3] = nn.Linear(in_features, NUM_CLASSES)
    model = model.to(device)
    
    optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-3)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=8, eta_min=1e-5)
    
    EPOCHS = 8
    best_val_f1 = 0.0
    best_model_state = None
    
    print(f"\nStarting Fast In-Memory Training ({EPOCHS} Epochs)...", flush=True)
    start_train = time.time()
    
    for epoch in range(1, EPOCHS + 1):
        ep_t0 = time.time()
        model.train()
        running_loss = 0.0
        
        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
            
        train_loss = running_loss / len(train_dataset)
        
        # Eval
        model.eval()
        val_loss = 0.0
        val_preds = []
        val_targets = []
        
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                targets = targets.to(device)
                outputs = model(images)
                loss = criterion(outputs, targets)
                val_loss += loss.item() * images.size(0)
                
                _, preds = torch.max(outputs, 1)
                val_preds.extend(preds.cpu().numpy())
                val_targets.extend(targets.cpu().numpy())
                
        val_loss = val_loss / len(val_dataset)
        val_acc = accuracy_score(val_targets, val_preds)
        val_f1 = precision_recall_fscore_support(val_targets, val_preds, average='weighted', zero_division=0)[2]
        
        scheduler.step()
        
        print(f"Epoch [{epoch:02d}/{EPOCHS:02d}] ({time.time()-ep_t0:.1f}s) - Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f} | Val F1: {val_f1:.4f}", flush=True)
        
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_model_state = model.state_dict().copy()
            print(f"  --> Best Model Updated (Val F1: {best_val_f1:.4f})", flush=True)
            
    print(f"\nTraining completed in {time.time()-start_train:.1f}s.", flush=True)
    
    # Load Best Model
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    # Final Untouched Test Set Evaluation
    print("\n" + "=" * 70, flush=True)
    print("FINAL EVALUATION ON UNTOUCHED TEST SET (773 SAMPLES)", flush=True)
    print("=" * 70, flush=True)
    
    model.eval()
    test_preds = []
    test_probs = []
    test_targets = []
    
    with torch.no_grad():
        for images, targets in test_loader:
            images = images.to(device)
            outputs = model(images)
            probs = torch.softmax(outputs, dim=1)
            _, preds = torch.max(outputs, 1)
            
            test_probs.extend(probs.cpu().numpy())
            test_preds.extend(preds.cpu().numpy())
            test_targets.extend(targets.numpy())
            
    test_acc = accuracy_score(test_targets, test_preds)
    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(test_targets, test_preds, average='macro', zero_division=0)
    p_wt, r_wt, f1_wt, _ = precision_recall_fscore_support(test_targets, test_preds, average='weighted', zero_division=0)
    
    print(f"\nOverall Test Metrics:", flush=True)
    print(f"  Top-1 Accuracy:  {test_acc:.4f} ({test_acc*100:.2f}%)", flush=True)
    print(f"  Weighted F1:     {f1_wt:.4f}", flush=True)
    print(f"  Macro F1:        {f1_macro:.4f}", flush=True)
    print(f"  Weighted Recall: {r_wt:.4f}", flush=True)
    
    print("\nDetailed Per-Class Classification Report:", flush=True)
    report_dict = cast(Dict[str, Any], classification_report(
        test_targets, test_preds, target_names=CLASS_NAMES, zero_division=0, output_dict=True
    ))
    report_text = classification_report(
        test_targets, test_preds, target_names=CLASS_NAMES, zero_division=0
    )
    print(report_text, flush=True)
    
    # Save Model Weights
    model_save_path = os.path.join(OUTPUT_DIR, "railway_track_defect_classifier.pth")
    torch.save(best_model_state, model_save_path)
    print(f"\nSaved model weights to: {model_save_path}", flush=True)
    
    # Save Confusion Matrix Plot
    cm = confusion_matrix(test_targets, test_preds)
    plt.figure(figsize=(9, 7))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.title('TEJAS Computer Vision: Railway Track Fault Classification Matrix (Test Set)')
    plt.xlabel('Predicted Defect Class')
    plt.ylabel('Ground Truth Class')
    plt.tight_layout()
    cm_path = os.path.join(OUTPUT_DIR, "cv_confusion_matrix.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix plot to: {cm_path}", flush=True)
    
    # Save Metrics CSV
    metrics_rows = []
    for cls_name in CLASS_NAMES:
        cls_data = report_dict[cls_name]
        metrics_rows.append({
            "class_name": cls_name,
            "precision": round(float(cls_data["precision"]), 4),
            "recall": round(float(cls_data["recall"]), 4),
            "f1_score": round(float(cls_data["f1-score"]), 4),
            "support": int(cls_data["support"]),
            "tejas_pipeline_defect_mapping": CV_TO_TEJAS_DEFECT_MAP[cls_name]
        })
    metrics_df = pd.DataFrame(metrics_rows)
    metrics_csv_path = os.path.join(OUTPUT_DIR, "cv_test_metrics.csv")
    metrics_df.to_csv(metrics_csv_path, index=False)
    print(f"Saved metrics CSV to: {metrics_csv_path}", flush=True)
    
    # Save Metadata JSON
    metadata = {
        "model_architecture": "MobileNetV3-Small (Transfer Learning)",
        "num_classes": NUM_CLASSES,
        "class_names": CLASS_NAMES,
        "class_to_idx": CLASS_TO_IDX,
        "cv_to_tejas_defect_map": CV_TO_TEJAS_DEFECT_MAP,
        "training_epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "total_dataset_size": len(images_array),
        "train_samples": len(train_idx),
        "val_samples": len(val_idx),
        "test_samples": len(test_idx),
        "overall_test_accuracy": round(float(test_acc), 4),
        "overall_test_weighted_f1": round(float(f1_wt), 4),
        "overall_test_macro_f1": round(float(f1_macro), 4),
        "overall_test_weighted_recall": round(float(r_wt), 4),
        "input_size": [224, 224],
        "normalization": {
            "mean": [0.485, 0.456, 0.406],
            "std": [0.229, 0.224, 0.225]
        },
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S")
    }
    metadata_path = os.path.join(OUTPUT_DIR, "cv_model_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    print(f"Saved metadata JSON to: {metadata_path}", flush=True)
    
    # Generate Evaluation Report MD
    report_md = f"""# TEJAS Computer Vision: Railway Track Defect Classification Report

**Dataset**: `Railway Track Surface Faults Dataset` (5,153 real images, 720p HD)  
**Architecture**: MobileNetV3-Small Transfer Learning with In-Memory Augmentation  
**Splits**: Train: {len(train_idx)} | Val: {len(val_idx)} | Test: {len(test_idx)} (Stratified)  

---

## 1. Overall Test Set Performance (773 Untouched Samples)

| Metric | Score |
| :--- | :--- |
| **Top-1 Accuracy** | **{test_acc*100:.2f}%** ({test_acc:.4f}) |
| **Weighted F1-Score** | **{f1_wt:.4f}** |
| **Macro F1-Score** | **{f1_macro:.4f}** |
| **Weighted Recall** | **{r_wt:.4f}** |

---

## 2. Per-Class Performance Breakdown

| Defect Class | Precision | Recall | F1-Score | Support | Downstream TEJAS Mapping |
| :--- | :--- | :--- | :--- | :--- | :--- |
"""
    for _, row in metrics_df.iterrows():
        report_md += f"| **{row['class_name']}** | {row['precision']:.4f} | {row['recall']:.4f} | **{row['f1_score']:.4f}** | {int(row['support'])} | `{row['tejas_pipeline_defect_mapping']}` |\n"
        
    report_md += f"""
---

## 3. Downstream Pipeline Integration

When an inspection image is attached:
1. MobileNetV3-Small processes the image tensor and outputs 7 class softmax probabilities.
2. The top predicted defect class (e.g. `Cracks` with 98.2% confidence) is extracted.
3. TEJAS maps `Cracks` $\\to$ `structural crack indication`.
4. Downstream `preprocessing_pipeline.pkl` incorporates the predicted defect type into the 165-dim feature space.
5. 4 Champion ML Models compute failure risk, block requirements, priority score, and repair duration.
6. Provenance tracks `DATASET:CV_MOBILENET_MODEL` with predicted confidence percentage.
"""
    report_md_path = os.path.join(OUTPUT_DIR, "cv_evaluation_report.md")
    with open(report_md_path, "w", encoding="utf-8") as f:
        f.write(report_md)
    print(f"Saved evaluation report markdown to: {report_md_path}", flush=True)
    print("\n" + "=" * 70, flush=True)
    print("ALL CV TRAINING & EVALUATION STEPS COMPLETE (100% SUCCESS)!", flush=True)
    print("=" * 70, flush=True)


if __name__ == "__main__":
    main()
