"""
Train a regressor to predict severity (0-5) from classical features.
This is needed because the classifiers only predict issue_type; severity
is a separate, continuous signal used for the final quality_score.

Trained on ALL rows (issue_type == "none" has severity 0), so the regressor
learns the general relationship between feature magnitude and degradation
intensity across all issue types at once.

Outputs:
  models/severity_regressor.joblib

Run from backend/:  python -m app.ml.train_severity_regressor
"""

import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, r2_score

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from app.ml.features import FEATURE_ORDER, feature_vector  # noqa: E402

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
FEATURES_CSV = os.path.join(DATA_DIR, "features.csv")


def main():
    feat_df = pd.read_csv(FEATURES_CSV)

    train_df = feat_df[feat_df["split"] == "train"]
    val_df = feat_df[feat_df["split"] == "val"]
    test_df = feat_df[feat_df["split"] == "test"]

    X_train = np.stack([feature_vector(row) for row in train_df[FEATURE_ORDER].to_dict("records")])
    X_val = np.stack([feature_vector(row) for row in val_df[FEATURE_ORDER].to_dict("records")])
    X_test = np.stack([feature_vector(row) for row in test_df[FEATURE_ORDER].to_dict("records")])

    y_train = train_df["severity"].values.astype(np.float32)
    y_val = val_df["severity"].values.astype(np.float32)
    y_test = test_df["severity"].values.astype(np.float32)

    print("Training severity regressor (GradientBoostingRegressor)...")
    reg = GradientBoostingRegressor(
        n_estimators=150,
        max_depth=4,
        learning_rate=0.1,
        random_state=42,
    )
    reg.fit(X_train, y_train)

    val_preds = reg.predict(X_val)
    print(f"Val   MAE: {mean_absolute_error(y_val, val_preds):.3f}  R2: {r2_score(y_val, val_preds):.3f}")

    test_preds = reg.predict(X_test)
    print(f"Test  MAE: {mean_absolute_error(y_test, test_preds):.3f}  R2: {r2_score(y_test, test_preds):.3f}")

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(reg, os.path.join(MODELS_DIR, "severity_regressor.joblib"))
    print(f"Saved severity regressor to {MODELS_DIR}")


if __name__ == "__main__":
    main()