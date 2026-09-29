"""Visualisations for anomaly detection examples."""

from __future__ import annotations

from pathlib import Path

import matplotlib

# Results are written to files; a GUI backend would fail on a headless machine or CI.
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def save_prediction_figure(
    image: np.ndarray, mask: np.ndarray, score_map: np.ndarray, predicted_mask: np.ndarray, output: str | Path, title: str
) -> None:
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axes = plt.subplots(1, 4, figsize=(16, 4))
    panels = [(image, "Image", None), (mask, "Ground truth", "gray"), (score_map, "Anomaly map", "turbo"), (predicted_mask, "Prediction", "gray")]
    for axis, (panel, label, cmap) in zip(axes, panels):
        axis.imshow(panel, cmap=cmap)
        axis.set_title(label)
        axis.axis("off")
    figure.suptitle(title)
    figure.tight_layout()
    figure.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(figure)
