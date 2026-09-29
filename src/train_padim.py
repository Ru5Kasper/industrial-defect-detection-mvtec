"""Fit one PaDiM model per MVTec AD category.

Usage: ``python -m src.train_padim --config configs/padim.yaml``.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

from src.data import MVTecDataset
from src.padim import PaDiM


def load_config(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as source:
        return yaml.safe_load(source)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def make_loader(dataset: MVTecDataset, config: dict, shuffle: bool = False) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=config["dataset"]["batch_size"],
        shuffle=shuffle,
        num_workers=config["dataset"].get("num_workers", 0),
        pin_memory=torch.cuda.is_available(),
    )


def train_category(category: str, config: dict, output_dir: Path, device: str) -> Path:
    dataset = MVTecDataset(config["dataset"]["root"], category, config["dataset"]["image_size"], train=True)
    loader = make_loader(dataset, config)
    model_config = config["model"]
    model = PaDiM(
        embedding_dim=model_config["embedding_dim"],
        covariance_regularization=model_config["covariance_regularization"],
        pretrained=model_config.get("pretrained", True),
        device=device,
    )
    model.fit(loader, seed=config["seed"])
    threshold = model.calibrate_threshold(loader)
    path = output_dir / category / "padim_stats.npz"
    model.save(path)
    print(f"{category}: fitted on {len(dataset)} normal images; visualisation threshold={threshold:.6f}; saved {path}")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Train PaDiM on MVTec AD normal images only.")
    parser.add_argument("--config", default="configs/padim.yaml")
    parser.add_argument("--output-dir", default="artifacts/padim")
    parser.add_argument("--data-root", help="Local MVTec AD root; overrides dataset.root in the config.")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    arguments = parser.parse_args()
    config = load_config(arguments.config)
    if arguments.data_root:
        config["dataset"]["root"] = arguments.data_root
    seed_everything(config["seed"])
    output_dir = Path(arguments.output_dir)
    for category in config["dataset"]["categories"]:
        train_category(category, config, output_dir, arguments.device)


if __name__ == "__main__":
    main()
