"""
End-to-end inference pipeline: takes a raw image, runs both models, blends
their predictions, and produces the final structured result matching the
assessment's expected response shape:

  {
    "quality_score": 82,
    "quality_label": "ACCEPTABLE",
    "issues": [{"type": "noise", "severity": "low", "confidence": 0.71}],
    "image_stats": {...},          # extra: raw feature values, for explainability
    "model_breakdown": {...}       # extra: per-model predictions, for transparency
  }

Blend weights (0.65 classical / 0.35 CNN) are set from each model's relative
test macro F1 (0.9431 vs 0.8628) so the stronger model dominates the vote
while the CNN still contributes, particularly useful on corruption/defect
edge cases and for Grad-CAM explainability downstream.

CONFIDENCE_THRESHOLD controls how many non-"none" issues get reported;
tune based on desired precision/recall tradeoff.
"""

import os
import sys

import cv2
import joblib
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import models, transforms

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.ml.features import FEATURE_ORDER, extract_features, feature_vector  # noqa: E402

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODELS_DIR = os.path.join(BASE_DIR, "models")

CLASSES = ["none", "blur", "underexposure", "overexposure", "noise", "corruption"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

CLASSICAL_WEIGHT = 0.65
CNN_WEIGHT = 0.35
CONFIDENCE_THRESHOLD = 0.15  # min blended prob for a non-"none" class to be reported as an issue

# Calibration: "none" was under-represented 5:1 in training (150 clean vs
# 750-per-degradation-class per split), which biases both models toward
# over-predicting degradation on clean/near-clean images. This boost
# corrects for that training-prior mismatch at inference time.
NONE_PRIOR_BOOST = 3.0

IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

CNN_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def severity_to_label(severity_value: float) -> str:
    """Map continuous severity (0-5) to a human-readable label."""
    if severity_value < 0.5:
        return "none"
    elif severity_value < 1.75:
        return "low"
    elif severity_value < 3.25:
        return "medium"
    else:
        return "high"


class QualityInferencePipeline:
    """
    Loads all three models once, exposes analyze(image_bgr) for repeated use.
    Instantiate a single instance and reuse it across requests (loading models
    per-request would be slow) — the FastAPI app should hold this as a
    module-level singleton.
    """

    def __init__(self):
        clf_path = os.path.join(MODELS_DIR, "feature_classifier.joblib")
        scaler_path = os.path.join(MODELS_DIR, "feature_scaler.joblib")
        severity_path = os.path.join(MODELS_DIR, "severity_regressor.joblib")
        cnn_path = os.path.join(MODELS_DIR, "cnn_classifier.pt")

        for path in (clf_path, scaler_path, severity_path, cnn_path):
            if not os.path.exists(path):
                raise FileNotFoundError(
                    f"Missing model file: {path}. Run the training scripts first "
                    "(train_feature_classifier.py, train_severity_regressor.py, train_cnn.py)."
                )

        self.feature_classifier = joblib.load(clf_path)
        self.feature_scaler = joblib.load(scaler_path)
        self.severity_regressor = joblib.load(severity_path)

        checkpoint = torch.load(cnn_path, map_location=DEVICE)
        self.cnn_model = self._build_cnn_architecture()
        self.cnn_model.load_state_dict(checkpoint["model_state_dict"])
        self.cnn_model.to(DEVICE)
        self.cnn_model.eval()

    @staticmethod
    def _build_cnn_architecture():
        model = models.mobilenet_v2(weights=None)
        model.classifier = torch.nn.Sequential(
            torch.nn.Dropout(0.3),
            torch.nn.Linear(model.last_channel, len(CLASSES)),
        )
        return model

    def _classical_probs(self, feats: dict) -> np.ndarray:
        vec = feature_vector(feats).reshape(1, -1)
        vec_scaled = self.feature_scaler.transform(vec)
        probs = self.feature_classifier.predict_proba(vec_scaled)[0]
        # Align to CLASSES order (sklearn's classes_ attribute gives actual order)
        class_order = list(self.feature_classifier.classes_)
        aligned = np.array([probs[class_order.index(c)] for c in CLASSES])
        return aligned

    def _cnn_probs(self, image_bgr: np.ndarray) -> np.ndarray:
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(image_rgb)
        tensor = CNN_TRANSFORM(pil_image).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits = self.cnn_model(tensor)
            probs = F.softmax(logits, dim=1)[0].cpu().numpy()
        return probs

    def analyze(self, image_bgr: np.ndarray) -> dict:
        feats = extract_features(image_bgr)

        classical_probs = self._classical_probs(feats)
        cnn_probs = self._cnn_probs(image_bgr)
        blended_probs = CLASSICAL_WEIGHT * classical_probs + CNN_WEIGHT * cnn_probs
        
        # Apply the "none" prior-correction boost, then renormalize so
        # probabilities still sum to 1.
        blended_probs = blended_probs.copy()
        blended_probs[CLASS_TO_IDX["none"]] *= NONE_PRIOR_BOOST
        blended_probs = blended_probs / blended_probs.sum()

        severity_value = float(self.severity_regressor.predict(
            feature_vector(feats).reshape(1, -1)
        )[0])
        severity_value = float(np.clip(severity_value, 0, 5))

        issues = []
        for idx, class_name in enumerate(CLASSES):
            if class_name == "none":
                continue
            confidence = float(blended_probs[idx])
            if confidence >= CONFIDENCE_THRESHOLD:
                issues.append({
                    "type": class_name,
                    "severity": severity_to_label(severity_value),
                    "confidence": round(confidence, 3),
                })
        issues.sort(key=lambda x: x["confidence"], reverse=True)

        none_confidence = float(blended_probs[CLASS_TO_IDX["none"]])

        # quality_score: 100 = perfect, degrades with severity and inversely
        # with confidence that the image is clean. Weighted by top issue's
        # severity and the blended "none" probability.
        if not issues:
            quality_score = int(round(85 + 15 * none_confidence))
        else:
            top_confidence = issues[0]["confidence"]
            quality_score = int(round(
                100 - (severity_value / 5.0) * 70 * top_confidence - (1 - none_confidence) * 10
            ))
        quality_score = int(np.clip(quality_score, 0, 100))

        if quality_score >= 75:
            quality_label = "ACCEPTABLE"
        elif quality_score >= 40:
            quality_label = "DEGRADED"
        else:
            quality_label = "POTENTIALLY_DEFECTIVE"

        return {
            "quality_score": quality_score,
            "quality_label": quality_label,
            "issues": issues,
            "image_stats": {k: round(float(v), 4) for k, v in feats.items()},
            "model_breakdown": {
                "classical_probs": {c: round(float(p), 3) for c, p in zip(CLASSES, classical_probs)},
                "cnn_probs": {c: round(float(p), 3) for c, p in zip(CLASSES, cnn_probs)},
                "blend_weights": {"classical": CLASSICAL_WEIGHT, "cnn": CNN_WEIGHT},
                "predicted_severity_raw": round(severity_value, 2),
            },
        }


if __name__ == "__main__":
    # Quick manual smoke test: python -m app.ml.inference <image_path>
    if len(sys.argv) < 2:
        print("Usage: python -m app.ml.inference <path_to_image>")
        sys.exit(1)

    img = cv2.imread(sys.argv[1])
    if img is None:
        print(f"Could not read image: {sys.argv[1]}")
        sys.exit(1)

    pipeline = QualityInferencePipeline()
    result = pipeline.analyze(img)

    import json
    print(json.dumps(result, indent=2))