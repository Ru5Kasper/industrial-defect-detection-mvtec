"""A compact PaDiM implementation for MVTec AD.

It uses frozen ImageNet ResNet-18 features, models each spatial location as a
Gaussian distribution, and scores a test patch by Mahalanobis distance.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as functional
from torchvision.models import ResNet18_Weights, resnet18
from torchvision.models.feature_extraction import create_feature_extractor


@dataclass
class PaDiMStatistics:
    channels: np.ndarray
    mean: np.ndarray
    inverse_covariance: np.ndarray
    pixel_threshold: float


class PaDiM:
    def __init__(
        self,
        embedding_dim: int = 100,
        covariance_regularization: float = 0.01,
        pretrained: bool = True,
        device: str = "cpu",
    ) -> None:
        self.embedding_dim = embedding_dim
        self.covariance_regularization = covariance_regularization
        self.device = torch.device(device)
        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)
        self.extractor = create_feature_extractor(
            backbone, return_nodes={"layer1": "layer1", "layer2": "layer2", "layer3": "layer3"}
        ).to(self.device).eval()
        for parameter in self.extractor.parameters():
            parameter.requires_grad_(False)
        self.statistics: PaDiMStatistics | None = None

    @torch.no_grad()
    def embed(self, images: torch.Tensor) -> torch.Tensor:
        features = self.extractor(images.to(self.device))
        target_size = features["layer3"].shape[-2:]
        pooled = [functional.adaptive_avg_pool2d(features[name], target_size) for name in ("layer1", "layer2", "layer3")]
        return torch.cat(pooled, dim=1)

    @torch.no_grad()
    def fit(self, train_loader: torch.utils.data.DataLoader, seed: int) -> None:
        """Estimate one Gaussian per spatial feature location from normal images."""
        embeddings = [self.embed(batch["image"]).cpu() for batch in train_loader]
        if not embeddings:
            raise ValueError("PaDiM needs at least two normal training images")
        embeddings_tensor = torch.cat(embeddings)
        if embeddings_tensor.shape[0] < 2:
            raise ValueError("PaDiM needs at least two normal training images")
        _, channels, height, width = embeddings_tensor.shape
        generator = torch.Generator().manual_seed(seed)
        chosen_channels = torch.randperm(channels, generator=generator)[: min(self.embedding_dim, channels)].sort().values
        selected = embeddings_tensor[:, chosen_channels].permute(0, 2, 3, 1).contiguous()  # N,H,W,C
        mean = selected.mean(dim=0)
        centered = selected - mean
        covariance = torch.einsum("nhwc,nhwd->hwcd", centered, centered) / (selected.shape[0] - 1)
        eye = torch.eye(selected.shape[-1], dtype=covariance.dtype).view(1, 1, selected.shape[-1], selected.shape[-1])
        inverse_covariance = torch.linalg.pinv(covariance + self.covariance_regularization * eye)
        self.statistics = PaDiMStatistics(
            channels=chosen_channels.numpy(),
            mean=mean.numpy(),
            inverse_covariance=inverse_covariance.numpy(),
            pixel_threshold=float("nan"),
        )

    @torch.no_grad()
    def score(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return image scores and full-resolution anomaly maps."""
        if self.statistics is None:
            raise RuntimeError("Call fit() or load() before score()")
        embedding = self.embed(images)
        channels = torch.as_tensor(self.statistics.channels, device=self.device)
        selected = embedding[:, channels].permute(0, 2, 3, 1)
        mean = torch.as_tensor(self.statistics.mean, device=self.device, dtype=selected.dtype)
        inverse_covariance = torch.as_tensor(self.statistics.inverse_covariance, device=self.device, dtype=selected.dtype)
        delta = selected - mean
        distances = torch.einsum("nhwc,hwcd,nhwd->nhw", delta, inverse_covariance, delta).clamp_min(0).sqrt()
        maps = functional.interpolate(distances.unsqueeze(1), size=images.shape[-2:], mode="bilinear", align_corners=False).squeeze(1)
        return maps.amax(dim=(1, 2)), maps

    @torch.no_grad()
    def calibrate_threshold(self, normal_loader: torch.utils.data.DataLoader, quantile: float = 0.995) -> float:
        """Set a visualisation threshold from normal training data only."""
        maps = [self.score(batch["image"])[1].flatten().cpu() for batch in normal_loader]
        self.statistics.pixel_threshold = float(torch.quantile(torch.cat(maps), quantile))
        return self.statistics.pixel_threshold

    def save(self, path: str | Path) -> None:
        if self.statistics is None:
            raise RuntimeError("No PaDiM statistics to save")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, **self.statistics.__dict__)

    def load(self, path: str | Path) -> None:
        archive = np.load(path)
        self.statistics = PaDiMStatistics(
            channels=archive["channels"],
            mean=archive["mean"],
            inverse_covariance=archive["inverse_covariance"],
            pixel_threshold=float(archive["pixel_threshold"]),
        )
