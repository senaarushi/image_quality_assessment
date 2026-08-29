"""
Train a MobileNetV2 transfer-learning classifier on the degraded image
dataset. This is the deep-learning half of the hybrid model, primarily
responsible for the harder "corruption / potential defect" calls and for
providing Grad-CAM explainability.

Predicts issue_type (6 classes: none, blur, underexposure, overexposure,
noise, corruption) directly from pixels.

Outputs:
  models/cnn_classifier.pt       -- trained model weights + class mapping

Run from backend/:  python -m app.ml.train_cnn
"""

import os
import sys
import time

import pandas as pd
import torch
import torch.nn as nn
from PIL import Image
from sklearn.metrics import classification_report, f1_score
from torch.utils.data import DataLoader, Dataset
from torchvision import models, transforms

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
LABELS_CSV = os.path.join(DATA_DIR, "labels.csv")

IMG_SIZE = 224
BATCH_SIZE = 32
NUM_EPOCHS = 8
LEARNING_RATE = 1e-4

CLASSES = ["none", "blur", "underexposure", "overexposure", "noise", "corruption"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

TRAIN_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

EVAL_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


class ImageQualityDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(DATA_DIR, row["filepath"])
        image = Image.open(img_path).convert("RGB")
        image = self.transform(image)
        label = CLASS_TO_IDX[row["issue_type"]]
        return image, label


def build_model():
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    # Freeze the feature extractor, only train the classifier head + last few layers
    for param in model.features.parameters():
        param.requires_grad = False
    for param in model.features[-3:].parameters():
        param.requires_grad = True
    model.classifier = nn.Sequential(
        nn.Dropout(0.3),
        nn.Linear(model.last_channel, len(CLASSES)),
    )
    return model.to(DEVICE)


def evaluate(model, loader):
    model.eval()
    all_preds, all_labels = [], []
    with torch.no_grad():
        for images, labels in loader:
            images = images.to(DEVICE)
            outputs = model(images)
            preds = outputs.argmax(dim=1).cpu().numpy()
            all_preds.extend(preds)
            all_labels.extend(labels.numpy())
    return all_labels, all_preds


def main():
    print(f"Using device: {DEVICE}")
    df = pd.read_csv(LABELS_CSV)

    train_df = df[df["split"] == "train"]
    val_df = df[df["split"] == "val"]
    test_df = df[df["split"] == "test"]
    print(f"train={len(train_df)}  val={len(val_df)}  test={len(test_df)}")

    train_loader = DataLoader(
        ImageQualityDataset(train_df, TRAIN_TRANSFORM),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=2,
    )
    val_loader = DataLoader(
        ImageQualityDataset(val_df, EVAL_TRANSFORM),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=2,
    )
    test_loader = DataLoader(
        ImageQualityDataset(test_df, EVAL_TRANSFORM),
        batch_size=BATCH_SIZE, shuffle=False, num_workers=2,
    )

    model = build_model()
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()), lr=LEARNING_RATE,
    )

    best_val_f1 = 0.0
    os.makedirs(MODELS_DIR, exist_ok=True)
    ckpt_path = os.path.join(MODELS_DIR, "cnn_classifier.pt")

    for epoch in range(1, NUM_EPOCHS + 1):
        model.train()
        start = time.time()
        running_loss = 0.0
        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * images.size(0)

        train_loss = running_loss / len(train_df)
        val_labels, val_preds = evaluate(model, val_loader)
        val_f1 = f1_score(val_labels, val_preds, average="macro")
        elapsed = time.time() - start
        print(f"Epoch {epoch}/{NUM_EPOCHS}  loss={train_loss:.4f}  val_macro_f1={val_f1:.4f}  ({elapsed:.1f}s)")

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            torch.save({
                "model_state_dict": model.state_dict(),
                "classes": CLASSES,
                "img_size": IMG_SIZE,
            }, ckpt_path)
            print(f"  -> saved new best checkpoint (val_macro_f1={val_f1:.4f})")

    print(f"\nBest val macro F1: {best_val_f1:.4f}")
    print("Loading best checkpoint for final test evaluation...")
    checkpoint = torch.load(ckpt_path, map_location=DEVICE)
    model.load_state_dict(checkpoint["model_state_dict"])

    test_labels, test_preds = evaluate(model, test_loader)
    print("\n--- Test set performance (CNN) ---")
    print(classification_report(
        test_labels, test_preds, target_names=CLASSES,
    ))
    print(f"Macro F1 (test): {f1_score(test_labels, test_preds, average='macro'):.4f}")


if __name__ == "__main__":
    main()