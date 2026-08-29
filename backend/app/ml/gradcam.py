"""
Grad-CAM explainability for the CNN classifier. Produces a heatmap overlay
highlighting which image regions most influenced the predicted issue class,
saved as a base64-encoded PNG for direct embedding in the API response.
"""

import base64
import io
import os
import sys

import cv2
import numpy as np
import torch
from PIL import Image
from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.image import show_cam_on_image
from torchvision import transforms

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

IMG_SIZE = 224
GRADCAM_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


def generate_gradcam_overlay(cnn_model, image_bgr: np.ndarray, target_class_idx: int, device) -> str:
    """
    Returns a base64-encoded PNG data URI string of the Grad-CAM heatmap
    overlaid on the (resized) input image, for the given target class index.
    """
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(image_rgb).resize((IMG_SIZE, IMG_SIZE))
    rgb_float = np.array(pil_image).astype(np.float32) / 255.0

    input_tensor = GRADCAM_TRANSFORM(Image.fromarray(image_rgb)).unsqueeze(0).to(device)

    # Last conv layer of MobileNetV2's feature extractor is a good default
    # target layer for Grad-CAM on this architecture.
    target_layers = [cnn_model.features[-1]]

    cam = GradCAM(model=cnn_model, target_layers=target_layers)
    grayscale_cam = cam(input_tensor=input_tensor, targets=None)[0]
    # targets=None makes it explain the model's own top prediction; if we
    # want to force explaining `target_class_idx` specifically instead:
    from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
    grayscale_cam = cam(
        input_tensor=input_tensor,
        targets=[ClassifierOutputTarget(target_class_idx)],
    )[0]

    overlay = show_cam_on_image(rgb_float, grayscale_cam, use_rgb=True)
    overlay_bgr = cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR)

    ok, buffer = cv2.imencode(".png", overlay_bgr)
    if not ok:
        raise RuntimeError("Failed to encode Grad-CAM overlay")

    b64 = base64.b64encode(buffer).decode("utf-8")
    return f"data:image/png;base64,{b64}"