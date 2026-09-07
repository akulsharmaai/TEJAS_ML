# TEJAS Computer Vision: Railway Track Defect Classification Report

**Dataset**: `Railway Track Surface Faults Dataset` (5,153 real images, 720p HD)  
**Architecture**: MobileNetV3-Small Transfer Learning with In-Memory Augmentation  
**Splits**: Train: 3607 | Val: 773 | Test: 773 (Stratified)  

---

## 1. Overall Test Set Performance (773 Untouched Samples)

| Metric | Score |
| :--- | :--- |
| **Top-1 Accuracy** | **90.17%** (0.9017) |
| **Weighted F1-Score** | **0.9035** |
| **Macro F1-Score** | **0.8704** |
| **Weighted Recall** | **0.9017** |

---

## 2. Per-Class Performance Breakdown

| Defect Class | Precision | Recall | F1-Score | Support | Downstream TEJAS Mapping |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Cracks** | 1.0000 | 1.0000 | **1.0000** | 6 | `structural crack indication` |
| **Flakings** | 0.9243 | 0.9222 | **0.9233** | 424 | `surface shelling/squat` |
| **Grooves** | 0.5000 | 1.0000 | **0.6667** | 1 | `corrugation/wear` |
| **Joints** | 1.0000 | 1.0000 | **1.0000** | 2 | `switch rail gap` |
| **Shellings** | 0.7500 | 0.9000 | **0.8182** | 20 | `surface shelling/squat` |
| **Spallings** | 0.6615 | 0.9773 | **0.7890** | 44 | `surface shelling/squat` |
| **Squats** | 0.9402 | 0.8551 | **0.8956** | 276 | `surface shelling/squat` |

---

## 3. Downstream Pipeline Integration

When an inspection image is attached:
1. MobileNetV3-Small processes the image tensor and outputs 7 class softmax probabilities.
2. The top predicted defect class (e.g. `Cracks` with 98.2% confidence) is extracted.
3. TEJAS maps `Cracks` $\to$ `structural crack indication`.
4. Downstream `preprocessing_pipeline.pkl` incorporates the predicted defect type into the 165-dim feature space.
5. 4 Champion ML Models compute failure risk, block requirements, priority score, and repair duration.
6. Provenance tracks `DATASET:CV_MOBILENET_MODEL` with predicted confidence percentage.
