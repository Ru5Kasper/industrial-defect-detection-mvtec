"""MVTec AD loading utilities.

The training split deliberately exposes only ``train/good``.  Test samples may
include defects and their ground-truth masks, but those masks are never used by
the PaDiM fitting code.
"""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}


def image_transform(image_size: int) -> Callable:
    """Return the ImageNet-normalised transform used by the ResNet backbone."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size), antialias=True),
            transforms.ToTensor(),
            transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]
    )


def mask_transform(image_size: int) -> Callable:
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size), interpolation=transforms.InterpolationMode.NEAREST),
            transforms.PILToTensor(),
        ]
    )


class MVTecDataset(Dataset):
    """One MVTec AD category with original labels and masks retained for test."""

    def __init__(self, root: str | Path, category: str, image_size: int, train: bool) -> None:
        self.root = Path(root)
        self.category = category
        self.train = train
        self.image_size = image_size
        self.transform = image_transform(image_size)
        self.mask_transform = mask_transform(image_size)
        category_root = self.root / category
        split_root = category_root / ("train" if train else "test")
        if not split_root.exists():
            raise FileNotFoundError(f"MVTec category not found: {split_root}")

        self.samples: list[tuple[Path, str, Path | None]] = []
        defect_dirs = [split_root / "good"] if train else sorted(path for path in split_root.iterdir() if path.is_dir())
        for defect_dir in defect_dirs:
            defect_type = defect_dir.name
            for image_path in sorted(defect_dir.iterdir()):
                if image_path.suffix.lower() not in IMAGE_EXTENSIONS:
                    continue
                mask_path: Path | None = None
                if not train and defect_type != "good":
                    candidate = category_root / "ground_truth" / defect_type / f"{image_path.stem}_mask.png"
                    if not candidate.exists():
                        raise FileNotFoundError(f"Missing ground-truth mask for {image_path}: {candidate}")
                    mask_path = candidate
                self.samples.append((image_path, defect_type, mask_path))
        if not self.samples:
            raise ValueError(f"No images found in {split_root}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> dict[str, object]:
        image_path, defect_type, mask_path = self.samples[index]
        image = Image.open(image_path).convert("RGB")
        image_tensor = self.transform(image)
        if mask_path is None:
            mask = torch.zeros((1, self.image_size, self.image_size), dtype=torch.uint8)
        else:
            raw_mask = Image.open(mask_path).convert("L")
            mask = (self.mask_transform(raw_mask) > 0).to(torch.uint8)
        return {
            "image": image_tensor,
            "mask": mask,
            "label": int(defect_type != "good"),
            "defect_type": defect_type,
            "path": str(image_path),
        }


def denormalize_image(image: torch.Tensor) -> np.ndarray:
    """Convert an ImageNet-normalised CHW tensor to RGB uint8 for plotting."""
    mean = torch.tensor(IMAGENET_MEAN, device=image.device).view(3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=image.device).view(3, 1, 1)
    restored = (image.detach() * std + mean).clamp(0, 1)
    return (restored.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
