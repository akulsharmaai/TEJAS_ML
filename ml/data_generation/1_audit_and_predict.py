import os
import io
import glob
import pandas as pd
import numpy as np
import cv2
import sys
from PIL import Image
import torch
import warnings

# Add project root to sys.path so we can import from ml
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

# Suppress torchvision warnings
warnings.filterwarnings('ignore')

from ml.backend.cv_service import TrackCVDefectClassifier

DATASET_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "Railway Track Surface Faults Dataset"))
OUTPUT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "cv_labels"))

def main():
    print(f"Scanning directory: {DATASET_DIR}")
    if not os.path.exists(DATASET_DIR):
        raise FileNotFoundError(f"Dataset directory not found: {DATASET_DIR}")
        
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load model
    print("Loading CV Model...")
    classifier = TrackCVDefectClassifier.get_instance()
    
    image_paths = glob.glob(os.path.join(DATASET_DIR, "**", "*.[jJ][pP]*[gG]"), recursive=True)
    print(f"Found {len(image_paths)} images.")
    
    if len(image_paths) != 5153:
        print(f"WARNING: Expected 5153 images, found {len(image_paths)}")
        
    results = []
    
    print("Starting inference and feature extraction...")
    for i, path in enumerate(image_paths):
        if i > 0 and i % 500 == 0:
            print(f"Processed {i}/{len(image_paths)} images...")
            
        filename = os.path.basename(path)
        defect_type = os.path.basename(os.path.dirname(path))
        
        with open(path, "rb") as f:
            image_bytes = f.read()
            
        try:
            pred = classifier.predict_image_bytes(image_bytes)
            confidence = pred["confidence"]
            probs = list(pred["class_probabilities"].values())
            margin = probs[0] - probs[1] if len(probs) > 1 else confidence
        except Exception as e:
            print(f"Inference failed for {filename}: {e}")
            confidence = 0.0
            margin = 0.0
            
        # The CV model classifies defect TYPE (e.g. Crack, Flaking).
        # It does NOT predict severity, and image texture/edge-density is not a valid proxy for track fault severity.
        # Therefore, we cannot safely assign AI_ONLY severity labels.
        ai_severity = None
        requires_review = True
        
        results.append({
            "image_path": path,
            "filename": filename,
            "folder_defect_type": defect_type,
            "ai_severity": ai_severity,
            "ai_confidence": round(confidence, 4),
            "margin": round(margin, 4),
            "edge_density": 0.0, # Removed invalid heuristic
            "requires_review": requires_review
        })
        
    df = pd.DataFrame(results)
    
    out_path = os.path.join(OUTPUT_DIR, "severity_predictions.csv")
    df.to_csv(out_path, index=False)
    print(f"Predictions saved to {out_path}")
    
    # Summary
    print("\n--- Audit Summary ---")
    print(f"Total Images Processed: {len(df)}")
    print(f"Requires Human Review: {df['requires_review'].sum()} ({(df['requires_review'].sum() / len(df)) * 100:.1f}%)")

if __name__ == "__main__":
    main()
