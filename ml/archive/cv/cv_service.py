"""
TEJAS Computer Vision Inference Service
Loads and serves the trained MobileNetV3 railway track surface fault classifier.
"""

import os
import io
import json
import logging
from typing import Dict, Any, Optional
from PIL import Image

import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.models as models

logger = logging.getLogger("tejas.cv_service")

# Defect Mapping from CV Categories to TEJAS Tabular Pipeline Taxonomy
DEFAULT_CV_TO_TEJAS_DEFECT_MAP = {
    'Cracks': 'structural crack indication',
    'Flakings': 'surface shelling/squat',
    'Grooves': 'corrugation/wear',
    'Joints': 'switch rail gap',
    'Shellings': 'surface shelling/squat',
    'Spallings': 'surface shelling/squat',
    'Squats': 'surface shelling/squat'
}

NO_IMAGE_DEFAULT_SEVERITY = "MEDIUM"

DEFAULT_CLASS_NAMES = ['Cracks', 'Flakings', 'Grooves', 'Joints', 'Shellings', 'Spallings', 'Squats']


class TrackCVDefectClassifier:
    """
    Inference service for railway track defect computer vision model.
    """
    _instance: Optional["TrackCVDefectClassifier"] = None

    def __init__(self, model_dir: Optional[str] = None):
        if model_dir is None:
            model_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "cv"))
        
        self.model_dir = model_dir
        self.weights_path = os.path.join(model_dir, "railway_track_defect_classifier.pth")
        self.metadata_path = os.path.join(model_dir, "cv_model_metadata.json")
        
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model: Optional[nn.Module] = None
        self.class_names = DEFAULT_CLASS_NAMES
        self.cv_to_tejas_map = DEFAULT_CV_TO_TEJAS_DEFECT_MAP
        self.metadata: Dict[str, Any] = {}
        self.is_loaded = False
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        self.load_model()

    @classmethod
    def get_instance(cls, model_dir: Optional[str] = None) -> "TrackCVDefectClassifier":
        if cls._instance is None:
            cls._instance = TrackCVDefectClassifier(model_dir=model_dir)
        return cls._instance

    def load_model(self) -> bool:
        """Loads metadata and weights into memory."""
        try:
            if os.path.exists(self.metadata_path):
                with open(self.metadata_path, "r", encoding="utf-8") as f:
                    self.metadata = json.load(f)
                self.class_names = self.metadata.get("class_names", DEFAULT_CLASS_NAMES)
                self.cv_to_tejas_map = self.metadata.get("cv_to_tejas_defect_map", DEFAULT_CV_TO_TEJAS_DEFECT_MAP)
                logger.info(f"Loaded CV metadata from {self.metadata_path}")
            
            num_classes = len(self.class_names)
            
            # Recreate model architecture
            model = models.mobilenet_v3_small(weights=None)
            in_features = int(getattr(model.classifier[3], "in_features", 1024))
            model.classifier[3] = nn.Linear(in_features, num_classes)
            
            if os.path.exists(self.weights_path):
                state_dict = torch.load(self.weights_path, map_location=self.device)
                model.load_state_dict(state_dict)
                model.to(self.device)
                model.eval()
                self.model = model
                self.is_loaded = True
                logger.info(f"Loaded CV model weights from {self.weights_path} onto {self.device}")
                return True
            else:
                logger.warning(f"CV Model weights not found at {self.weights_path}. Model will load once training finishes.")
                return False
        except Exception as e:
            logger.error(f"Error loading CV model: {e}")
            self.is_loaded = False
            return False

    def predict_image_bytes(self, image_bytes: bytes) -> Dict[str, Any]:
        """
        Runs inference on raw image bytes.
        """
        if not self.is_loaded or self.model is None:
            # Try reloading in case training just finished
            if not self.load_model() or self.model is None:
                raise RuntimeError("CV Model is not loaded or weights are unavailable.")
                
        try:
            image = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            tensor = self.transform(image).unsqueeze(0).to(self.device)
            
            model = self.model
            with torch.no_grad():
                outputs = model(tensor)
                probs = torch.softmax(outputs, dim=1).squeeze(0)
                conf, pred_idx = torch.max(probs, dim=0)
                
            pred_class = self.class_names[int(pred_idx.item())]
            confidence = float(conf.item())
            
            # All class probabilities
            probs_dict = {
                self.class_names[i]: round(float(probs[i].item()), 4)
                for i in range(len(self.class_names))
            }
            # Sorted by probability desc
            sorted_probs = dict(sorted(probs_dict.items(), key=lambda item: item[1], reverse=True))
            
            mapped_defect = self.cv_to_tejas_map.get(pred_class, "surface shelling/squat")
            
            return {
                "defect_class": pred_class,
                "confidence": round(confidence, 4),
                "confidence_percent": round(confidence * 100, 2),
                "class_probabilities": sorted_probs,
                "mapped_tejas_defect": mapped_defect,
                "model_architecture": self.metadata.get("model_architecture", "MobileNetV3-Small"),
                "is_cv_analyzed": True
            }
        except Exception as e:
            logger.error(f"Failed to run CV inference on image: {e}")
            raise ValueError(f"Image inference failed: {str(e)}")
