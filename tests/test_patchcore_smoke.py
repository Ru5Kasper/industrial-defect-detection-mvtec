"""CPU smoke tests for PatchCore checkpoint and prediction contract."""
import tempfile
import unittest
from pathlib import Path

import torch
from torch.utils.data import DataLoader, Dataset

from src.patchcore import PatchCore


class TinyGoodDataset(Dataset):
    def __len__(self):
        return 2
    def __getitem__(self, index):
        # Deterministic, ImageNet-normalised-looking RGB tensor.
        generator = torch.Generator().manual_seed(index + 10)
        return {"image": torch.rand((3, 256, 256), generator=generator)}


class PatchCoreSmokeTest(unittest.TestCase):
    def test_fit_predict_save_load(self):
        model = PatchCore(device="cpu", image_size=256, coreset_ratio=0.001,
                          max_bank_size=8, nn_chunk_size=64, seed=42, pretrained=False)
        model.fit(DataLoader(TinyGoodDataset(), batch_size=1, shuffle=False))
        scores, maps = model.predict(torch.zeros((1, 3, 256, 256)))
        self.assertEqual(tuple(scores.shape), (1,))
        self.assertEqual(tuple(maps.shape), (1, 1, 256, 256))
        self.assertTrue(torch.isfinite(scores).all())
        self.assertTrue(torch.isfinite(maps).all())
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "smoke.pt"
            model.save(path, metadata={"test": True})
            restored = PatchCore.load(path, device="cpu")
            restored_scores, restored_maps = restored.predict(torch.zeros((1, 3, 256, 256)))
            self.assertTrue(torch.allclose(scores, restored_scores, atol=1e-5, rtol=1e-5))
            self.assertTrue(torch.allclose(maps, restored_maps, atol=1e-5, rtol=1e-5))


if __name__ == "__main__":
    unittest.main()
