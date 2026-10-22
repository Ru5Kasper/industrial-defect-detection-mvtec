"""Mini tests for scripts/analyze_dataset.py on a synthetic MVTec-like tree.

The fake dataset lives in a temporary directory and is removed after each test,
so no test data ends up in the repository.

Run: python -m unittest discover -s tests -v
"""

from __future__ import annotations

import contextlib
import io
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

import analyze_dataset  # noqa: E402


def save_image(path: Path, size: int = 32, mode: str = "RGB") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    channels = 3 if mode == "RGB" else 1
    data = np.full((size, size, channels), 128, dtype=np.uint8).squeeze()
    Image.fromarray(data).save(path)


def save_mask(path: Path, size: int = 32, defect: tuple[slice, slice] | None = (slice(0, 8), slice(0, 4))) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    data = np.zeros((size, size), dtype=np.uint8)
    if defect is not None:
        data[defect] = 255
    Image.fromarray(data).save(path)


def build_category(root: Path, name: str = "widget") -> Path:
    """3 train/good, 2 test/good, 2 test/scratch with masks, 1 test/dent with mask."""
    category = root / name
    for index in range(3):
        save_image(category / "train" / "good" / f"{index:03d}.png")
    for index in range(2):
        save_image(category / "test" / "good" / f"{index:03d}.png")
        save_image(category / "test" / "scratch" / f"{index:03d}.png")
        save_mask(category / "ground_truth" / "scratch" / f"{index:03d}_mask.png")
    save_image(category / "test" / "dent" / "000.png", mode="L")
    save_mask(category / "ground_truth" / "dent" / "000_mask.png")
    (category / "license.txt").write_text("test", encoding="utf-8")
    return category


class AnalyzeDatasetTest(unittest.TestCase):
    def setUp(self) -> None:
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.data = self.tmp / "data"
        self.out = self.tmp / "out"
        self.category = build_category(self.data)

    def tearDown(self) -> None:
        self._tmp.cleanup()

    def run_cli(self) -> int:
        argv = [
            "--data-root", str(self.data),
            "--output-dir", str(self.out / "eda"),
            "--tables-dir", str(self.out),
            "--categories", "widget",
            "--pilot", "widget",
        ]  # fmt: skip
        with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
            return analyze_dataset.main(argv)

    def quality(self) -> dict:
        return json.loads((self.out / "data_quality.json").read_text(encoding="utf-8"))

    def test_counts_and_pairs_on_valid_dataset(self) -> None:
        self.assertEqual(self.run_cli(), 0)
        summary = pd.read_csv(self.out / "dataset_summary.csv")
        self.assertEqual(list(summary.columns), analyze_dataset.SUMMARY_COLUMNS)
        row = summary.iloc[0].to_dict()
        self.assertEqual(
            {k: row[k] for k in ("train_good", "test_good", "test_defect", "test_total", "defect_types", "mask_count")},
            {"train_good": 3, "test_good": 2, "test_defect": 3, "test_total": 5, "defect_types": 2, "mask_count": 3},
        )

        defects = pd.read_csv(self.out / "defect_summary.csv")
        self.assertEqual(list(defects["defect_type"]), ["good", "dent", "scratch"])
        self.assertEqual(list(defects["image_count"]), [2, 1, 2])
        self.assertEqual(list(defects["mask_count"]), [0, 1, 2])
        self.assertEqual(defects["image_count"].sum(), row["test_total"])

        masks = pd.read_csv(self.out / "eda" / "mask_stats.csv")
        self.assertEqual(len(masks), 3)
        self.assertTrue((masks["defect_pixels"] == 32).all())
        self.assertAlmostEqual(masks["defect_fraction"].iloc[0], 32 / (32 * 32))
        self.assertIn("widget/test/scratch/000.png", set(masks["image"]))
        self.assertIn("widget/ground_truth/scratch/000_mask.png", set(masks["mask"]))

        quality = self.quality()
        self.assertEqual(quality["status"], "ok")
        self.assertEqual(quality["checked"]["image_mask_pairs_checked"], 3)
        for name in ("critical_issues", "missing_masks", "extra_masks", "unreadable_images", "size_mismatches",
                     "empty_defect_masks", "non_binary_masks"):  # fmt: skip
            self.assertEqual(quality[name], [], name)
        self.assertEqual(quality["per_category"]["widget"]["image_modes"], {"L": 1, "RGB": 7})
        for name in ("train_test_counts.png", "test_composition_pilot.png", "defect_area_pilot.png",
                     "gallery_widget.jpg"):  # fmt: skip
            self.assertTrue((self.out / "eda" / name).is_file(), name)

    def test_missing_mask_is_critical(self) -> None:
        (self.category / "ground_truth" / "scratch" / "001_mask.png").unlink()
        self.assertEqual(self.run_cli(), 1)
        quality = self.quality()
        self.assertEqual(quality["status"], "critical")
        self.assertEqual(quality["missing_masks"], ["widget/test/scratch/001.png"])
        self.assertFalse(quality["loader_compatibility"]["every_defect_image_has_mask"])

    def test_extra_mask_is_reported(self) -> None:
        save_mask(self.category / "ground_truth" / "scratch" / "099_mask.png")
        self.assertEqual(self.run_cli(), 0)
        self.assertEqual(self.quality()["extra_masks"], ["widget/ground_truth/scratch/099_mask.png"])

    def test_size_mismatch_and_empty_mask(self) -> None:
        save_mask(self.category / "ground_truth" / "scratch" / "000_mask.png", size=16)
        save_mask(self.category / "ground_truth" / "dent" / "000_mask.png", defect=None)
        self.assertEqual(self.run_cli(), 1)
        quality = self.quality()
        self.assertEqual(quality["size_mismatches"], ["widget/ground_truth/scratch/000_mask.png"])
        self.assertEqual(quality["empty_defect_masks"], ["widget/ground_truth/dent/000_mask.png"])

    def test_unreadable_image_and_missing_train(self) -> None:
        (self.category / "test" / "good" / "000.png").write_bytes(b"not a png")
        self.assertEqual(self.run_cli(), 1)
        self.assertEqual(self.quality()["unreadable_images"], ["widget/test/good/000.png"])

        for path in (self.category / "train" / "good").iterdir():
            path.unlink()
        self.assertEqual(self.run_cli(), 1)
        kinds = {issue["kind"] for issue in self.quality()["critical_issues"]}
        self.assertIn("empty_train_good", kinds)

    def test_dataset_is_not_modified_and_rerun_is_identical(self) -> None:
        before = sorted((p.relative_to(self.data).as_posix(), p.stat().st_mtime_ns) for p in self.data.rglob("*"))
        self.assertEqual(self.run_cli(), 0)
        first = {p.name: p.read_bytes() for p in self.out.rglob("*") if p.is_file()}
        self.assertEqual(self.run_cli(), 0)
        second = {p.name: p.read_bytes() for p in self.out.rglob("*") if p.is_file()}
        self.assertEqual(first.keys(), second.keys())
        for name in first:
            self.assertEqual(first[name], second[name], name)
        after = sorted((p.relative_to(self.data).as_posix(), p.stat().st_mtime_ns) for p in self.data.rglob("*"))
        self.assertEqual(before, after)


if __name__ == "__main__":
    unittest.main()
