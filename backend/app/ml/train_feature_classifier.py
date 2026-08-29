"""
Extract classical CV features for every image in labels.csv and train a
gradient-boosted classifier to predict issue_type from those features.

Outputs:
  data/features.csv           -- feature matrix + labels for train/val/test
  models/feature_classifier.joblib   -- trained sklearn model
  models/feature_scaler.joblib       -- fitted StandardScaler

Run from backend/:  python -m app.ml.train_feature_classifier
"""

import os
import sys
import time

import cv2
import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.ml.features import FEATURE_ORDER, extract_features, feature_vector  # noqa: E402

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
LABELS_CSV = os.path.join(DATA_DIR, "labels.csv")
FEATURES_CSV = os.path.join(DATA_DIR, "features.csv")


def extract_all_features():
    df = pd.read_csv(LABELS_CSV)
    print(f"Loaded {len(df)} rows from labels.csv")

    rows = []
    start = time.time()
    for i, row in df.iterrows():
        img_path = os.path.join(DATA_DIR, row["filepath"])
        image = cv2.imread(img_path)
        if image is None:
            print(f"  WARNING: could not read {img_path}, skipping")
            continue
        feats = extract_features(image)
        feats["filepath"] = row["filepath"]
        feats["split"] = row["split"]
        feats["issue_type"] = row["issue_type"]
        feats["severity"] = row["severity"]
        feats["quality_label"] = row["quality_label"]
        rows.append(feats)

        if (i + 1) % 2000 == 0:
            elapsed = time.time() - start
            print(f"  processed {i + 1}/{len(df)}  ({elapsed:.1f}s elapsed)")

    feat_df = pd.DataFrame(rows)
    feat_df.to_csv(FEATURES_CSV, index=False)
    print(f"Feature extraction done in {time.time() - start:.1f}s. Wrote {FEATURES_CSV}")
    return feat_df


def train_classifier(feat_df: pd.DataFrame):
    train_df = feat_df[feat_df["split"] == "train"]
    val_df = feat_df[feat_df["split"] == "val"]
    test_df = feat_df[feat_df["split"] == "test"]

    X_train = np.stack([feature_vector(row) for row in train_df[FEATURE_ORDER].to_dict("records")])
    X_val = np.stack([feature_vector(row) for row in val_df[FEATURE_ORDER].to_dict("records")])
    X_test = np.stack([feature_vector(row) for row in test_df[FEATURE_ORDER].to_dict("records")])

    y_train = train_df["issue_type"].values
    y_val = val_df["issue_type"].values
    y_test = test_df["issue_type"].values

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)

    print("\nTraining GradientBoostingClassifier...")
    clf = GradientBoostingClassifier(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
    )
    start = time.time()
    clf.fit(X_train_scaled, y_train)
    print(f"Trained in {time.time() - start:.1f}s")

    print("\n--- Validation set performance ---")
    val_preds = clf.predict(X_val_scaled)
    print(classification_report(y_val, val_preds))
    print(f"Macro F1 (val): {f1_score(y_val, val_preds, average='macro'):.4f}")

    print("\n--- Test set performance ---")
    test_preds = clf.predict(X_test_scaled)
    print(classification_report(y_test, test_preds))
    print(f"Macro F1 (test): {f1_score(y_test, test_preds, average='macro'):.4f}")

    print("\nConfusion matrix (test), labels order:", sorted(set(y_test)))
    print(confusion_matrix(y_test, test_preds, labels=sorted(set(y_test))))

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(clf, os.path.join(MODELS_DIR, "feature_classifier.joblib"))
    joblib.dump(scaler, os.path.join(MODELS_DIR, "feature_scaler.joblib"))
    print(f"\nSaved model + scaler to {MODELS_DIR}")


def main():
    if os.path.exists(FEATURES_CSV):
        print(f"{FEATURES_CSV} already exists, loading instead of re-extracting.")
        print("(delete this file if you want to force re-extraction)")
        feat_df = pd.read_csv(FEATURES_CSV)
    else:
        feat_df = extract_all_features()
    train_classifier(feat_df)


if __name__ == "__main__":
    main()