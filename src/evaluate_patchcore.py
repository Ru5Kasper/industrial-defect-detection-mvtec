"""Evaluate trained PatchCore models on the official MVTec test split."""
import argparse
import csv
import json
import time
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.patchcore import PatchCore
from src.patchcore_data import MVTecPatchDataset, MEAN, STD
from src.patchcore_metrics import evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/patchcore.yaml")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    cfg = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    ds, mc, tc, ec = cfg["dataset"], cfg["model"], cfg["training"], cfg["evaluation"]
    root = args.data_root or ds.get("root", "data/mvtec_ad")
    device = args.device or tc.get("device", "auto")
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable.")
    size = int(ds.get("image_size", 256))
    results_dir = Path(ec.get("results_dir", "results"))
    viz_root = results_dir / "patchcore_examples"
    viz_root.mkdir(parents=True, exist_ok=True)
    metric_path = results_dir / "patchcore_metrics.csv"
    image_path = results_dir / "patchcore_per_image.csv"
    metric_rows, image_rows = [], []
    for category in ds["categories"]:
        model_path = Path(tc.get("artifacts_dir", "artifacts/patchcore")) / f"{category}.pt"
        model = PatchCore.load(model_path, device=device).eval()
        # Calibration uses only normal train images; test labels never select threshold.
        train_set = MVTecPatchDataset(root, category, size, train=True)
        train_loader = DataLoader(train_set, batch_size=ec.get("batch_size", 1),
                                  shuffle=False, num_workers=ec.get("num_workers", 0))
        calibration_scores = []
        for batch in train_loader:
            scores, _ = model.predict(batch["image"].to(device))
            calibration_scores.extend(scores.detach().cpu().numpy().tolist())
        threshold = float(np.quantile(calibration_scores, 0.995))
        test_set = MVTecPatchDataset(root, category, size, train=False)
        test_loader = DataLoader(test_set, batch_size=ec.get("batch_size", 1),
                                 shuffle=False, num_workers=ec.get("num_workers", 0))
        labels, image_scores, masks, maps = [], [], [], []
        records = []
        inference_seconds, inference_count = 0.0, 0
        for batch in test_loader:
            images = batch["image"].to(device)
            if device == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            scores, anomaly_maps = model.predict(images)
            if device == "cuda":
                torch.cuda.synchronize()
            inference_seconds += time.perf_counter() - start
            inference_count += len(images)
            score_values = scores.detach().cpu().numpy()
            map_values = anomaly_maps[:, 0].detach().cpu().numpy()
            mask_values = batch["mask"][:, 0].cpu().numpy().astype(np.uint8)
            label_values = batch["label"].cpu().numpy()
            for i in range(len(images)):
                labels.append(int(label_values[i]))
                image_scores.append(float(score_values[i]))
                masks.append(mask_values[i])
                maps.append(map_values[i])
                record = {"relative_path": str(Path(batch["path"][i]).relative_to(Path(root))),
                          "category": category, "defect_type": batch["defect_type"][i],
                          "image_label": int(label_values[i]), "image_score": float(score_values[i]),
                          "image_threshold": threshold, "predicted_label": int(score_values[i] >= threshold)}
                image_rows.append(record)
                records.append((i, record))
        labels_np = np.asarray(labels)
        masks_np, maps_np = np.asarray(masks), np.asarray(maps)
        metric = evaluate(labels_np, np.asarray(image_scores), masks_np, maps_np)
        metric_rows.append({"category": category, "model": "patchcore",
            **metric, "inference_ms": 1000.0 * inference_seconds / max(1, inference_count),
            "config": str(args.config)})
        # A small, deterministic panel: first normal and first anomalous test image.
        if ec.get("save_visualizations", True):
            selected = []
            for wanted in (0, 1):
                found = next((item for item in records if item[1]["image_label"] == wanted), None)
                if found is not None:
                    selected.append(found)
            for index, record in selected:
                tensor = test_set[index]["image"]
                mean = torch.tensor(MEAN).view(3, 1, 1)
                std = torch.tensor(STD).view(3, 1, 1)
                rgb = ((tensor * std + mean).clamp(0, 1).permute(1, 2, 0).numpy())
                pred_mask = maps_np[index] >= threshold
                fig, axes = plt.subplots(1, 4, figsize=(12, 3))
                axes[0].imshow(rgb); axes[0].set_title("Input")
                axes[1].imshow(masks_np[index], cmap="gray"); axes[1].set_title("Ground truth")
                axes[2].imshow(maps_np[index], cmap="inferno"); axes[2].set_title("Anomaly map")
                axes[3].imshow(pred_mask, cmap="gray"); axes[3].set_title("Thresholded map")
                for ax in axes: ax.axis("off")
                fig.suptitle(f'{category} / {record["defect_type"]} / score={record["image_score"]:.4f}')
                fig.tight_layout()
                fig.savefig(viz_root / f'{category}_{Path(record["relative_path"]).stem}.png', dpi=130)
                plt.close(fig)
        print(f"{category}: image AUROC={metric['image_auroc']:.4f}; pixel AUROC={metric['pixel_auroc']:.4f}; AUPRO={metric['aupro']:.4f}; {metric_rows[-1]['inference_ms']:.2f} ms/image")
    results_dir.mkdir(parents=True, exist_ok=True)
    # Replace only PatchCore rows; do not touch the team's PaDiM metrics file.
    existing = []
    if metric_path.exists():
        with metric_path.open(newline="", encoding="utf-8") as handle:
            existing = [row for row in csv.DictReader(handle) if row.get("model") != "patchcore"]
    fields = ["category", "model", "image_auroc", "pixel_auroc", "aupro", "inference_ms", "config"]
    with metric_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields); writer.writeheader()
        writer.writerows(existing); writer.writerows(metric_rows)
    image_fields = ["relative_path", "category", "defect_type", "image_label", "image_score", "image_threshold", "predicted_label"]
    with image_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=image_fields); writer.writeheader(); writer.writerows(image_rows)
    valid = [r for r in metric_rows if np.isfinite(r["image_auroc"])]
    if valid:
        print("Macro-average:", {k: float(np.nanmean([r[k] for r in metric_rows])) for k in ("image_auroc", "pixel_auroc", "aupro", "inference_ms")})
    print(f"Saved metrics: {metric_path}")
    print(f"Saved per-image results: {image_path}")

if __name__ == "__main__":
    main()
