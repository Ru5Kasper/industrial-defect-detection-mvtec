"""PatchCore core for MVTec AD: frozen ResNet-18 + greedy k-center memory bank."""
from __future__ import annotations

import random
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torchvision.models import ResNet18_Weights, resnet18


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class ResNet18Features(nn.Module):
    """Frozen ImageNet ResNet-18 feature extractor (layer2 + layer3)."""

    def __init__(self, pretrained: bool = True):
        super().__init__()
        net = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        self.stem = nn.Sequential(net.conv1, net.bn1, net.relu, net.maxpool)
        self.layer1, self.layer2, self.layer3 = net.layer1, net.layer2, net.layer3
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.eval()

    @torch.inference_mode()
    def forward(self, images: torch.Tensor) -> torch.Tensor:
        x = self.layer1(self.stem(images))
        f2 = self.layer2(x)
        f3 = self.layer3(f2)
        f3 = F.interpolate(f3, size=f2.shape[-2:], mode="bilinear", align_corners=False)
        return torch.cat((f2, f3), dim=1)


def greedy_coreset(features: torch.Tensor, count: int, seed: int = 42) -> torch.Tensor:
    """Greedy k-center coreset on CPU float32 feature vectors."""
    features = features.detach().to(device="cpu", dtype=torch.float32)
    n = features.shape[0]
    if count <= 0:
        raise ValueError("coreset size must be positive")
    if count >= n:
        return features
    generator = torch.Generator(device="cpu").manual_seed(seed)
    first = int(torch.randint(n, (1,), generator=generator).item())
    chosen = torch.empty(count, dtype=torch.long)
    chosen[0] = first
    min_dist = torch.full((n,), float("inf"))
    for index in range(1, count):
        dist = torch.cdist(features, features[chosen[index - 1:index]]).squeeze(1)
        min_dist = torch.minimum(min_dist, dist)
        chosen[index] = torch.argmax(min_dist)
    return features[chosen]


class PatchCore(nn.Module):
    """PatchCore model. fit() accepts only a loader over train/good images."""

    def __init__(
        self,
        device: str = "cpu",
        image_size: int = 256,
        coreset_ratio: float = 0.01,
        max_bank_size: int = 10000,
        nn_chunk_size: int = 2048,
        seed: int = 42,
        pretrained: bool = True,
    ):
        super().__init__()
        if not 0 < coreset_ratio <= 1:
            raise ValueError("coreset_ratio must be in (0, 1]")
        self.device_name = str(device)
        self.image_size = int(image_size)
        self.coreset_ratio = float(coreset_ratio)
        self.max_bank_size = int(max_bank_size)
        self.nn_chunk_size = int(nn_chunk_size)
        self.seed = int(seed)
        self.backbone = ResNet18Features(pretrained=pretrained).to(self.device_name).eval()
        self.register_buffer("memory_bank", torch.empty((0, 0), dtype=torch.float32))
        self.fitted = False

    @torch.inference_mode()
    def embed(self, images: torch.Tensor) -> torch.Tensor:
        feature_map = self.backbone(images.to(self.device_name))
        feature_map = F.avg_pool2d(feature_map, kernel_size=3, stride=1, padding=1)
        return feature_map.flatten(2).transpose(1, 2).contiguous()

    @torch.inference_mode()
    def fit(self, train_loader: Iterable) -> "PatchCore":
        set_seed(self.seed)
        feature_chunks = []
        for batch in train_loader:
            images = batch[0] if isinstance(batch, (tuple, list)) else batch
            descriptors = self.embed(images)
            feature_chunks.append(descriptors.reshape(-1, descriptors.shape[-1]).cpu())
        if not feature_chunks:
            raise ValueError("Empty training loader; expected normal train/good images.")
        features = torch.cat(feature_chunks, dim=0).float()
        bank_size = min(self.max_bank_size, max(1, round(len(features) * self.coreset_ratio)))
        bank = greedy_coreset(features, bank_size, self.seed)
        self.memory_bank = bank.to(self.device_name)
        self.fitted = True
        return self

    @torch.inference_mode()
    def _nearest_distances(self, queries: torch.Tensor) -> torch.Tensor:
        if not self.fitted or self.memory_bank.numel() == 0:
            raise RuntimeError("Model is not fitted; call fit() or load() first.")
        results = []
        for query_chunk in queries.split(self.nn_chunk_size):
            nearest = torch.full((len(query_chunk),), float("inf"), device=query_chunk.device)
            for bank_chunk in self.memory_bank.split(self.nn_chunk_size):
                distances = torch.cdist(query_chunk, bank_chunk)
                nearest = torch.minimum(nearest, distances.min(dim=1).values)
            results.append(nearest)
        return torch.cat(results)

    @torch.inference_mode()
    def predict(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        descriptors = self.embed(images)
        batch_size, locations, channels = descriptors.shape
        distances = self._nearest_distances(descriptors.reshape(-1, channels))
        side = int(round(locations ** 0.5))
        if side * side != locations:
            raise RuntimeError(f"Expected square feature grid, got {locations} locations.")
        anomaly_map = distances.reshape(batch_size, 1, side, side)
        anomaly_map = F.interpolate(
            anomaly_map, size=(self.image_size, self.image_size),
            mode="bilinear", align_corners=False,
        )
        image_scores = anomaly_map.flatten(1).amax(dim=1)
        return image_scores, anomaly_map

    def save(self, path: str | Path, metadata: dict | None = None) -> None:
        if not self.fitted:
            raise RuntimeError("Cannot save an unfitted model.")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "state_dict": self.state_dict(),
            "config": {
                "image_size": self.image_size,
                "coreset_ratio": self.coreset_ratio,
                "max_bank_size": self.max_bank_size,
                "nn_chunk_size": self.nn_chunk_size,
                "seed": self.seed,
            },
            "metadata": metadata or {},
        }, path)

    @classmethod
    def load(cls, path: str | Path, device: str = "cpu") -> "PatchCore":
        try:
            payload = torch.load(path, map_location=device, weights_only=False)
        except TypeError:  # PyTorch versions before weights_only
            payload = torch.load(path, map_location=device)
        model = cls(device=device, pretrained=False, **payload["config"])
        model.load_state_dict(payload["state_dict"])
        model.fitted = True
        return model
