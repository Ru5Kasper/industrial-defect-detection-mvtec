"""Run PaDiM inference, save examples, and append MVTec AD metrics.

Usage: ``python -m src.evaluate_padim --config configs/padim.yaml`` after
``python -m src.train_padim``.
"""

from __future__ import annotations

import argparse
import csv
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.data import MVTecDataset, denormalize_image
from src.metrics import evaluate_predictions
from src.padim import PaDiM
from src.train_padim import make_loader, seed_everything
from src.visualize import save_prediction_figure


def load_config(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as source:
        return yaml.safe_load(source)


def append_metrics(path: Path, row: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    existing: list[dict[str, str]] = []
    if path.exists():
        with path.open(newline="", encoding="utf-8") as source:
            existing = list(csv.DictReader(source))
    fieldnames = ["category", "model", "image_auroc", "pixel_auroc", "aupro", "inference_ms", "config"]
    existing = [item for item in existing if not (item.get("category") == row["category"] and item.get("model") == row["model"])]
    existing.append({key: str(row[key]) for key in fieldnames})
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(existing)


def save_representative_examples(
    batches: list[dict[str, object]], score_maps: list[np.ndarray], image_scores: list[float], threshold: float, output_dir: Path, per_type: int
) -> None:
    saved: defaultdict[str, int] = defaultdict(int)
    for batch, maps, scores in zip(batches, score_maps, image_scores):
        images = batch["image"]
        masks = batch["mask"].numpy()
        labels = batch["label"].numpy()
        defect_types = batch["defect_type"]
        paths = batch["path"]
        for index, defect_type in enumerate(defect_types):
            if saved[defect_type] >= per_type:
                continue
            score_map = maps[index]
            predicted = score_map >= threshold
            title = f"{defect_type}; label={labels[index]}; image score={scores[index]:.4f}"
            save_prediction_figure(
                denormalize_image(images[index]), masks[index, 0], score_map, predicted, output_dir / defect_type / f"{Path(paths[index]).stem}.png", title
            )
            saved[defect_type] += 1


def evaluate_category(category: str, config: dict, artifact_dir: Path, results_dir: Path, device: str) -> dict[str, float | str]:
    dataset = MVTecDataset(config["dataset"]["root"], category, config["dataset"]["image_size"], train=False)
    loader = make_loader(dataset, config)
    model_config = config["model"]
    model = PaDiM(
        embedding_dim=model_config["embedding_dim"],
        covariance_regularization=model_config["covariance_regularization"],
        pretrained=model_config.get("pretrained", True),
        device=device,
    )
    model.load(artifact_dir / category / "padim_stats.npz")
    labels, masks, all_image_scores, all_score_maps, batches = [], [], [], [], []
    elapsed = 0.0
    sample_count = 0
    for batch in loader:
        started = time.perf_counter()
        image_scores, score_maps = model.score(batch["image"])
        if device.startswith("cuda"):
            torch.cuda.synchronize()
        elapsed += time.perf_counter() - started
        sample_count += len(batch["label"])
        labels.append(batch["label"].numpy())
        masks.append(batch["mask"].numpy()[:, 0])
        all_image_scores.append(image_scores.cpu().numpy())
        all_score_maps.append(score_maps.cpu().numpy())
        batches.append(batch)
    label_array = np.concatenate(labels)
    mask_array = np.concatenate(masks)
    image_score_array = np.concatenate(all_image_scores)
    score_map_array = np.concatenate(all_score_maps)
    metrics = evaluate_predictions(label_array, image_score_array, mask_array, score_map_array)
    if config["evaluation"].get("save_visualizations", True):
        save_representative_examples(
            batches,
            all_score_maps,
            all_image_scores,
            model.statistics.pixel_threshold,
            results_dir / "figures" / "padim" / category,
            config["evaluation"].get("visualization_examples_per_type", 1),
        )
    row: dict[str, float | str] = {
        "category": category,
        "model": "padim",
        **metrics,
        "inference_ms": elapsed / sample_count * 1000,
        "config": "configs/padim.yaml",
    }
    append_metrics(results_dir / "metrics.csv", row)
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate PaDiM on MVTec AD.")
    parser.add_argument("--config", default="configs/padim.yaml")
    parser.add_argument("--artifact-dir", default="artifacts/padim")
    parser.add_argument("--results-dir", default="results")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    seed_everything(config["seed"])
    for category in config["dataset"]["categories"]:
        row = evaluate_category(category, config, Path(arguments.artifact_dir), Path(arguments.results_dir), arguments.device)
        print(" | ".join(f"{name}={value}" for name, value in row.items()))


if __name__ == "__main__":
    main()
