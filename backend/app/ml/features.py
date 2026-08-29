"""
Classical image-quality feature extraction.

These features are the interpretable half of the hybrid model — they feed
a gradient-boosted classifier directly, and are also surfaced to the
frontend/API response as human-readable image statistics for explainability.
"""

import cv2
import numpy as np
from skimage.restoration import estimate_sigma


def _to_gray(image_bgr: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)


def blur_score(image_bgr: np.ndarray) -> float:
    """
    Variance of the Laplacian. Lower values indicate a blurrier image
    (less high-frequency edge content). This is the standard, well-validated
    no-reference blur metric (Pech-Canul et al.).
    """
    gray = _to_gray(image_bgr)
    lap = cv2.Laplacian(gray, cv2.CV_64F)
    return float(lap.var())


def exposure_stats(image_bgr: np.ndarray) -> dict:
    """
    Brightness histogram stats used to detect under/overexposure.
    Returns mean brightness, and the fraction of pixels clipped at the
    shadow and highlight ends of the histogram.
    """
    gray = _to_gray(image_bgr)
    mean_brightness = float(gray.mean())
    total_px = gray.size
    shadow_clip = float(np.sum(gray <= 5)) / total_px      # near-black pixels
    highlight_clip = float(np.sum(gray >= 250)) / total_px  # near-white pixels
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256]).flatten()
    hist_norm = hist / (hist.sum() + 1e-8)
    entropy = float(-np.sum(hist_norm[hist_norm > 0] * np.log2(hist_norm[hist_norm > 0])))
    return {
        "mean_brightness": mean_brightness,
        "shadow_clip_fraction": shadow_clip,
        "highlight_clip_fraction": highlight_clip,
        "histogram_entropy": entropy,
    }


def noise_score(image_bgr: np.ndarray) -> float:
    """
    Estimated Gaussian noise sigma via wavelet-based robust median estimator
    (skimage's estimate_sigma). Averaged across channels.
    """
    img = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    sigma = estimate_sigma(img, average_sigmas=True, channel_axis=-1)
    return float(sigma)


def contrast_score(image_bgr: np.ndarray) -> float:
    """RMS contrast: standard deviation of pixel intensities."""
    gray = _to_gray(image_bgr)
    return float(gray.std())


def saturation_score(image_bgr: np.ndarray) -> float:
    """Mean saturation channel value in HSV space."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    return float(hsv[:, :, 1].mean())


def corruption_score(image_bgr: np.ndarray) -> float:
    """
    Heuristic corruption/severe-degradation indicator: fraction of 8x8 blocks
    that are near-uniform (a hallmark of block corruption / heavy compression
    artifacts / sensor failure regions) combined with edge-density collapse.
    """
    gray = _to_gray(image_bgr)
    h, w = gray.shape
    block = 8
    h_trim, w_trim = h - h % block, w - w % block
    gray = gray[:h_trim, :w_trim]
    blocks = gray.reshape(h_trim // block, block, w_trim // block, block).swapaxes(1, 2)
    block_std = blocks.std(axis=(2, 3))
    flat_fraction = float(np.mean(block_std < 2.0))
    edges = cv2.Canny(gray, 50, 150)
    edge_density = float(np.mean(edges > 0))
    # Combine: many flat blocks + very low edge density => likely corrupted/degraded
    return float(flat_fraction * (1.0 - edge_density))


def extract_features(image_bgr: np.ndarray) -> dict:
    """
    Full feature vector for a single image. Returns a flat dict suitable for
    both model input (see feature_vector()) and direct API/explainability
    display.
    """
    exp = exposure_stats(image_bgr)
    feats = {
        "blur_score": blur_score(image_bgr),
        "noise_score": noise_score(image_bgr),
        "contrast_score": contrast_score(image_bgr),
        "saturation_score": saturation_score(image_bgr),
        "corruption_score": corruption_score(image_bgr),
        **exp,
    }
    return feats


FEATURE_ORDER = [
    "blur_score",
    "noise_score",
    "contrast_score",
    "saturation_score",
    "corruption_score",
    "mean_brightness",
    "shadow_clip_fraction",
    "highlight_clip_fraction",
    "histogram_entropy",
]


def feature_vector(feats: dict) -> np.ndarray:
    """Convert the feature dict to an ordered numpy vector for the classifier."""
    return np.array([feats[k] for k in FEATURE_ORDER], dtype=np.float32)