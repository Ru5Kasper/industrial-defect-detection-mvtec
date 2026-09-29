"""Image-level and localisation metrics used by both project models."""

from __future__ import annotations

import cv2
import numpy as np
from sklearn.metrics import roc_auc_score


def _safe_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    return float(roc_auc_score(labels, scores)) if np.unique(labels).size == 2 else float("nan")


def aupro(masks: np.ndarray, score_maps: np.ndarray, max_fpr: float = 0.30, steps: int = 200) -> float:
    """Approximate AUPRO up to ``max_fpr`` following the MVTec protocol."""
    masks = masks.astype(bool)
    if not masks.any():
        return float("nan")
    normal_pixels = ~masks
    thresholds = np.linspace(float(score_maps.max()), float(score_maps.min()), steps)
    points: list[tuple[float, float]] = []
    components = [cv2.connectedComponents(mask.astype(np.uint8))[1] for mask in masks]
    for threshold in thresholds:
        predicted = score_maps >= threshold
        fpr = float(predicted[normal_pixels].mean())
        if fpr > max_fpr:
            continue
        overlaps: list[float] = []
        for prediction, component_map in zip(predicted, components):
            for component in range(1, component_map.max() + 1):
                region = component_map == component
                overlaps.append(float(prediction[region].mean()))
        points.append((fpr, float(np.mean(overlaps))))
    if len(points) < 2:
        return float("nan")
    # Keep the best overlap at duplicate FPR values, then extend the curve to
    # the requested FPR cap.  The extension matters for a perfect score map:
    # all of its useful thresholds can have FPR=0.
    best_pro_by_fpr: dict[float, float] = {}
    for fpr, pro in points:
        best_pro_by_fpr[fpr] = max(pro, best_pro_by_fpr.get(fpr, 0.0))
    fpr_values = np.array(sorted(best_pro_by_fpr))
    pro_values = np.array([best_pro_by_fpr[fpr] for fpr in fpr_values])
    if fpr_values[0] > 0:
        fpr_values = np.insert(fpr_values, 0, 0.0)
        pro_values = np.insert(pro_values, 0, 0.0)
    if fpr_values[-1] < max_fpr:
        fpr_values = np.append(fpr_values, max_fpr)
        pro_values = np.append(pro_values, pro_values[-1])
    return float(np.trapezoid(pro_values, fpr_values) / max_fpr)


def evaluate_predictions(labels: np.ndarray, image_scores: np.ndarray, masks: np.ndarray, score_maps: np.ndarray) -> dict[str, float]:
    return {
        "image_auroc": _safe_auc(labels, image_scores),
        "pixel_auroc": _safe_auc(masks.reshape(-1), score_maps.reshape(-1)),
        "aupro": aupro(masks, score_maps),
    }
