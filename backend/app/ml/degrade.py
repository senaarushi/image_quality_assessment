"""
Controlled synthetic degradation generation.

Given a clean image, produces labeled degraded variants across issue types
and severity levels. Used to build the training/validation/test sets for
both the classical-feature classifier and the CNN.

Severity levels: 1 (mild) .. 5 (severe). Level 0 / "clean" is the original.
"""

import cv2
import numpy as np

SEVERITY_LEVELS = [1, 2, 3, 4, 5]


def apply_blur(image_bgr: np.ndarray, severity: int) -> np.ndarray:
    """Gaussian blur, kernel size scales with severity."""
    ksize = {1: 3, 2: 7, 3: 13, 4: 21, 5: 31}[severity]
    return cv2.GaussianBlur(image_bgr, (ksize, ksize), 0)


def apply_underexposure(image_bgr: np.ndarray, severity: int) -> np.ndarray:
    """Multiply brightness down, scales with severity."""
    factor = {1: 0.8, 2: 0.6, 3: 0.45, 4: 0.3, 5: 0.15}[severity]
    return np.clip(image_bgr.astype(np.float32) * factor, 0, 255).astype(np.uint8)


def apply_overexposure(image_bgr: np.ndarray, severity: int) -> np.ndarray:
    """Additive brightness boost + clipping, scales with severity."""
    offset = {1: 30, 2: 60, 3: 95, 4: 135, 5: 180}[severity]
    return np.clip(image_bgr.astype(np.float32) + offset, 0, 255).astype(np.uint8)


def apply_noise(image_bgr: np.ndarray, severity: int) -> np.ndarray:
    """Additive Gaussian sensor noise, sigma scales with severity."""
    sigma = {1: 8, 2: 15, 3: 25, 4: 40, 5: 60}[severity]
    noise = np.random.normal(0, sigma, image_bgr.shape).astype(np.float32)
    return np.clip(image_bgr.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def apply_corruption(image_bgr: np.ndarray, severity: int) -> np.ndarray:
    """
    Simulates block corruption / sensor failure / severe compression damage:
    randomly blanks out or scrambles increasingly large block regions.
    """
    h, w = image_bgr.shape[:2]
    out = image_bgr.copy()
    n_blocks = {1: 2, 2: 5, 3: 10, 4: 18, 5: 30}[severity]
    block_size = max(8, min(h, w) // 10)
    for _ in range(n_blocks):
        y = np.random.randint(0, max(1, h - block_size))
        x = np.random.randint(0, max(1, w - block_size))
        mode = np.random.choice(["blank", "scramble", "solid"])
        if mode == "blank":
            out[y:y + block_size, x:x + block_size] = 0
        elif mode == "solid":
            out[y:y + block_size, x:x + block_size] = np.random.randint(0, 255)
        else:
            patch = out[y:y + block_size, x:x + block_size].copy()
            flat = patch.reshape(-1, patch.shape[-1])
            np.random.shuffle(flat)
            out[y:y + block_size, x:x + block_size] = flat.reshape(patch.shape)
    # Heavy JPEG re-encoding at low quality adds realistic compression damage
    quality = {1: 40, 2: 25, 3: 15, 4: 8, 5: 3}[severity]
    ok, enc = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, quality])
    if ok:
        out = cv2.imdecode(enc, cv2.IMREAD_COLOR)
    return out


DEGRADATION_FUNCS = {
    "blur": apply_blur,
    "underexposure": apply_underexposure,
    "overexposure": apply_overexposure,
    "noise": apply_noise,
    "corruption": apply_corruption,
}


def generate_degraded_variants(image_bgr: np.ndarray, rng_seed: int | None = None):
    """
    Yields (issue_type, severity, degraded_image) for every issue type and
    severity level, for a single clean input image.
    """
    if rng_seed is not None:
        np.random.seed(rng_seed)
    for issue_type, func in DEGRADATION_FUNCS.items():
        for severity in SEVERITY_LEVELS:
            yield issue_type, severity, func(image_bgr, severity)