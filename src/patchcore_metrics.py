"""PatchCore evaluation metrics; AUPRO uses an explicit finite threshold grid."""
import cv2
import numpy as np
from sklearn.metrics import roc_auc_score

def safe_auc(labels, scores):
    labels = np.asarray(labels).reshape(-1)
    scores = np.asarray(scores).reshape(-1)
    return float(roc_auc_score(labels, scores)) if np.unique(labels).size == 2 else float("nan")

def compute_aupro(masks, score_maps, max_fpr=0.30, thresholds=300):
    masks = np.asarray(masks).astype(bool)
    scores = np.asarray(score_maps, dtype=np.float32)
    if not masks.any() or masks.all():
        return float("nan")
    normal = ~masks
    component_maps = [cv2.connectedComponents(m.astype(np.uint8))[1] for m in masks]
    levels = np.linspace(float(scores.max()), float(scores.min()), thresholds)
    points = []
    for threshold in levels:
        pred = scores >= threshold
        fpr = float(pred[normal].mean())
        if fpr > max_fpr:
            continue
        region_overlaps = []
        for p, labels in zip(pred, component_maps):
            for component in range(1, int(labels.max()) + 1):
                region = labels == component
                region_overlaps.append(float(p[region].mean()))
        if region_overlaps:
            points.append((fpr, float(np.mean(region_overlaps))))
    if len(points) < 2:
        return float("nan")
    # Collapse duplicate FPR values; append endpoint using last measured PRO.
    by_fpr = {}
    for fpr, pro in points:
        by_fpr[fpr] = max(pro, by_fpr.get(fpr, 0.0))
    xs = np.array(sorted(by_fpr))
    ys = np.array([by_fpr[x] for x in xs])
    if xs[0] > 0:
        xs, ys = np.insert(xs, 0, 0.0), np.insert(ys, 0, 0.0)
    if xs[-1] < max_fpr:
        xs, ys = np.append(xs, max_fpr), np.append(ys, ys[-1])
    return float(np.trapezoid(ys, xs) / max_fpr)

def evaluate(labels, image_scores, masks, score_maps):
    return {"image_auroc": safe_auc(labels, image_scores),
            "pixel_auroc": safe_auc(masks, score_maps),
            "aupro": compute_aupro(masks, score_maps)}
