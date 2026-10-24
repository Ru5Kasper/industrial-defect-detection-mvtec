"""Exploratory analysis and quality checks for the MVTec AD dataset.

The script reads the original files only (it never writes into the dataset),
processes images one by one and produces:

* ``<tables-dir>/dataset_summary.csv`` - one row per category;
* ``<tables-dir>/defect_summary.csv`` - one row per category and test type;
* ``<tables-dir>/data_quality.json`` - verified counts and every found problem;
* ``<output-dir>/mask_stats.csv`` - defect area of every ground-truth mask;
* ``<output-dir>/*.png`` - summary plots, ``gallery_<category>.jpg`` - galleries.

Sizes and defect areas are measured on the original files, before any resize.
The exit code is 1 when a critical problem is found, so the script can be used
as a gate before training.

Usage::

    python scripts/analyze_dataset.py --data-root "<path to MVTec AD>" --output-dir results/eda
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.ticker import NullFormatter  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from PIL import Image  # noqa: E402


EXPECTED_CATEGORIES = (
    "bottle",
    "cable",
    "capsule",
    "carpet",
    "grid",
    "hazelnut",
    "leather",
    "metal_nut",
    "pill",
    "screw",
    "tile",
    "toothbrush",
    "transistor",
    "wood",
    "zipper",
)
PILOT_CATEGORIES = ("bottle", "cable", "carpet")
# Same extensions as MVTecDataset in src/data.py (branch bashmakov).
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp"}
# Files shipped by MVTec next to the splits; they are not data and not issues.
KNOWN_EXTRA_FILES = {"license.txt", "readme.txt"}
MODEL_IMAGE_SIZE = 256
GALLERY_THUMB_SIZE = 256
# PNG "Software" metadata contains the matplotlib version; drop it so reruns
# produce byte-identical files.
PNG_METADATA = {"Software": None}
AREA_TICKS = [0.1, 0.3, 1, 3, 10, 30]

SUMMARY_COLUMNS = [
    "category",
    "train_good",
    "test_good",
    "test_defect",
    "test_total",
    "defect_types",
    "mask_count",
    "min_width",
    "max_width",
    "min_height",
    "max_height",
]
DEFECT_COLUMNS = ["category", "defect_type", "image_count", "mask_count"]
MASK_STATS_COLUMNS = [
    "category",
    "defect_type",
    "image",
    "mask",
    "width",
    "height",
    "defect_pixels",
    "defect_fraction",
    "defect_pixels_256",
]

# Problem kinds that make the dataset unusable for the team pipeline.
CRITICAL_KINDS = {
    "missing_category",
    "missing_train_good",
    "empty_train_good",
    "missing_test_dir",
    "missing_test_good",
    "unreadable_image",
    "missing_mask",
    "size_mismatch",
    "empty_defect_mask",
}
ISSUE_LISTS = {
    "missing_masks": "missing_mask",
    "extra_masks": "extra_mask",
    "unreadable_images": "unreadable_image",
    "size_mismatches": "size_mismatch",
    "empty_defect_masks": "empty_defect_mask",
    "non_binary_masks": "non_binary_mask",
    "masks_for_good_images": "mask_for_good",
    "non_square_images": "non_square_image",
    "unexpected_files": "unexpected_file",
    "masks_empty_after_resize": "mask_empty_after_resize",
}


@dataclass
class CategoryStats:
    name: str
    train_good: int = 0
    test_counts: dict[str, int] = field(default_factory=dict)
    mask_counts: dict[str, int] = field(default_factory=dict)
    widths: list[int] = field(default_factory=list)
    heights: list[int] = field(default_factory=list)
    modes: Counter = field(default_factory=Counter)
    sizes: Counter = field(default_factory=Counter)
    mask_values: set[int] = field(default_factory=set)
    train_subdirs: list[str] = field(default_factory=list)


@dataclass
class Analysis:
    data_root: Path
    categories: list[str]
    stats: dict[str, CategoryStats] = field(default_factory=dict)
    issues: list[dict[str, str]] = field(default_factory=list)
    mask_rows: list[dict[str, object]] = field(default_factory=list)
    unexpected_root_entries: list[str] = field(default_factory=list)
    images_read: int = 0
    masks_read: int = 0
    pairs_checked: int = 0

    def rel(self, path: Path) -> str:
        return path.relative_to(self.data_root).as_posix()

    def add_issue(self, kind: str, category: str, path: Path | str, detail: str) -> None:
        if isinstance(path, Path):
            path = self.rel(path)
        severity = "critical" if kind in CRITICAL_KINDS else "warning"
        self.issues.append(
            {"severity": severity, "kind": kind, "category": category, "path": path, "detail": detail}
        )

    @property
    def has_critical(self) -> bool:
        return any(issue["severity"] == "critical" for issue in self.issues)


def list_images(directory: Path) -> list[Path]:
    return sorted(
        path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_EXTENSIONS
    )


def subdirs(directory: Path) -> list[Path]:
    return sorted(path for path in directory.iterdir() if path.is_dir())


def report_non_images(analysis: Analysis, category: str, directory: Path) -> None:
    for path in sorted(directory.iterdir()):
        if path.is_file() and path.suffix.lower() not in IMAGE_EXTENSIONS:
            analysis.add_issue("unexpected_file", category, path, "file is not an image and is ignored")


def read_image_info(path: Path) -> tuple[int, int, str]:
    """Fully decode an image to make sure it is readable; return width, height, mode."""
    with Image.open(path) as image:
        image.load()
        return image.width, image.height, image.mode


def read_mask(path: Path) -> np.ndarray:
    with Image.open(path) as mask:
        mask.load()
        # MVTecDataset reads masks with convert("L") and treats every value > 0
        # as defect; reproduce that interpretation here.
        return np.asarray(mask.convert("L"))


def scan_image(analysis: Analysis, stats: CategoryStats, path: Path) -> tuple[int, int] | None:
    try:
        width, height, mode = read_image_info(path)
    except Exception as error:  # noqa: BLE001 - every decoding failure is reported
        analysis.add_issue("unreadable_image", stats.name, path, f"{type(error).__name__}: {error}")
        return None
    analysis.images_read += 1
    stats.widths.append(width)
    stats.heights.append(height)
    stats.modes[mode] += 1
    stats.sizes[f"{width}x{height}"] += 1
    if width != height:
        analysis.add_issue(
            "non_square_image",
            stats.name,
            path,
            f"{width}x{height}: resize to {MODEL_IMAGE_SIZE}x{MODEL_IMAGE_SIZE} changes aspect ratio",
        )
    return width, height


def scan_mask(
    analysis: Analysis,
    stats: CategoryStats,
    defect_type: str,
    image_path: Path,
    image_size: tuple[int, int] | None,
    mask_path: Path,
) -> None:
    try:
        mask = read_mask(mask_path)
    except Exception as error:  # noqa: BLE001
        analysis.add_issue("unreadable_image", stats.name, mask_path, f"{type(error).__name__}: {error}")
        return
    analysis.masks_read += 1
    height, width = mask.shape
    values = np.unique(mask)
    stats.mask_values.update(int(value) for value in values)
    if not set(values.tolist()) <= {0, 255}:
        analysis.add_issue(
            "non_binary_mask",
            stats.name,
            mask_path,
            f"values {values.tolist()[:10]}; interpreted as defect where value > 0",
        )
    if image_size is not None and (width, height) != image_size:
        analysis.add_issue(
            "size_mismatch",
            stats.name,
            mask_path,
            f"mask {width}x{height} vs image {image_size[0]}x{image_size[1]} ({analysis.rel(image_path)})",
        )
    defect_pixels = int((mask > 0).sum())
    if defect_pixels == 0:
        analysis.add_issue("empty_defect_mask", stats.name, mask_path, "mask of a defective image has no defect pixels")
    # Nearest-neighbour resize, as in the team loader (torchvision uses PIL here).
    resized = Image.fromarray(mask).resize((MODEL_IMAGE_SIZE, MODEL_IMAGE_SIZE), Image.Resampling.NEAREST)
    defect_pixels_256 = int((np.asarray(resized) > 0).sum())
    if defect_pixels > 0 and defect_pixels_256 == 0:
        analysis.add_issue(
            "mask_empty_after_resize",
            stats.name,
            mask_path,
            f"{defect_pixels} defect pixels vanish after nearest resize to {MODEL_IMAGE_SIZE}",
        )
    analysis.mask_rows.append(
        {
            "category": stats.name,
            "defect_type": defect_type,
            "image": analysis.rel(image_path),
            "mask": analysis.rel(mask_path),
            "width": width,
            "height": height,
            "defect_pixels": defect_pixels,
            "defect_fraction": round(defect_pixels / mask.size, 8),
            "defect_pixels_256": defect_pixels_256,
        }
    )


def analyze_category(analysis: Analysis, category: str) -> None:
    root = analysis.data_root / category
    stats = CategoryStats(category)
    analysis.stats[category] = stats
    if not root.is_dir():
        analysis.add_issue("missing_category", category, category, "category directory not found")
        return
    for path in sorted(root.iterdir()):
        if path.is_file() and path.name not in KNOWN_EXTRA_FILES:
            analysis.add_issue("unexpected_file", category, path, "unexpected file in category root")

    train_root = root / "train"
    train_good = train_root / "good"
    if train_root.is_dir():
        stats.train_subdirs = [path.name for path in subdirs(train_root)]
        for name in stats.train_subdirs:
            if name != "good":
                analysis.add_issue(
                    "unexpected_file", category, train_root / name, "train must contain only good images"
                )
    if not train_good.is_dir():
        analysis.add_issue("missing_train_good", category, f"{category}/train/good", "directory not found")
    else:
        report_non_images(analysis, category, train_good)
        images = list_images(train_good)
        if not images:
            analysis.add_issue("empty_train_good", category, train_good, "no training images")
        for image_path in images:
            if scan_image(analysis, stats, image_path) is not None:
                stats.train_good += 1

    test_root = root / "test"
    gt_root = root / "ground_truth"
    if not test_root.is_dir():
        analysis.add_issue("missing_test_dir", category, f"{category}/test", "directory not found")
        return
    test_types = [path.name for path in subdirs(test_root)]
    if "good" not in test_types:
        analysis.add_issue("missing_test_good", category, f"{category}/test/good", "directory not found")
    gt_types = [path.name for path in subdirs(gt_root)] if gt_root.is_dir() else []

    for defect_type in test_types:
        type_dir = test_root / defect_type
        report_non_images(analysis, category, type_dir)
        images = list_images(type_dir)
        gt_dir = gt_root / defect_type
        masks = list_images(gt_dir) if gt_dir.is_dir() else []
        stats.test_counts[defect_type] = 0
        stats.mask_counts[defect_type] = len(masks)
        expected_masks = set()
        for image_path in images:
            image_size = scan_image(analysis, stats, image_path)
            if image_size is not None:
                stats.test_counts[defect_type] += 1
            if defect_type == "good":
                continue
            mask_path = gt_dir / f"{image_path.stem}_mask.png"
            expected_masks.add(mask_path.name)
            analysis.pairs_checked += 1
            if not mask_path.is_file():
                analysis.add_issue(
                    "missing_mask", category, image_path, f"expected {category}/ground_truth/{defect_type}/{mask_path.name}"
                )
                continue
            scan_mask(analysis, stats, defect_type, image_path, image_size, mask_path)
        for mask_path in masks:
            if defect_type == "good":
                analysis.add_issue("mask_for_good", category, mask_path, "good images must not have masks")
            elif mask_path.name not in expected_masks:
                analysis.add_issue("extra_mask", category, mask_path, "mask without a matching test image")

    for defect_type in gt_types:
        if defect_type not in test_types:
            for mask_path in list_images(gt_root / defect_type):
                analysis.add_issue("extra_mask", category, mask_path, f"no test/{defect_type} directory")


def analyze(data_root: Path, categories: list[str]) -> Analysis:
    analysis = Analysis(data_root=data_root, categories=categories)
    for path in sorted(data_root.iterdir()):
        if path.name in categories:
            continue
        analysis.unexpected_root_entries.append(path.name)
        if path.is_dir():
            analysis.add_issue("unexpected_file", path.name, path, "directory is not an expected category")
    for category in categories:
        print(f"[analyze] {category}", flush=True)
        analyze_category(analysis, category)
    return analysis


# --------------------------------------------------------------------------- tables


def summary_table(analysis: Analysis) -> pd.DataFrame:
    rows = []
    for category in analysis.categories:
        stats = analysis.stats[category]
        test_good = stats.test_counts.get("good", 0)
        test_total = sum(stats.test_counts.values())
        rows.append(
            {
                "category": category,
                "train_good": stats.train_good,
                "test_good": test_good,
                "test_defect": test_total - test_good,
                "test_total": test_total,
                "defect_types": sum(1 for name in stats.test_counts if name != "good"),
                "mask_count": sum(stats.mask_counts.values()),
                "min_width": min(stats.widths, default=0),
                "max_width": max(stats.widths, default=0),
                "min_height": min(stats.heights, default=0),
                "max_height": max(stats.heights, default=0),
            }
        )
    return pd.DataFrame(rows, columns=SUMMARY_COLUMNS)


def defect_table(analysis: Analysis) -> pd.DataFrame:
    rows = []
    for category in analysis.categories:
        stats = analysis.stats[category]
        # "good" first, then defect types alphabetically.
        for defect_type in sorted(stats.test_counts, key=lambda name: (name != "good", name)):
            rows.append(
                {
                    "category": category,
                    "defect_type": defect_type,
                    "image_count": stats.test_counts[defect_type],
                    "mask_count": stats.mask_counts.get(defect_type, 0),
                }
            )
    return pd.DataFrame(rows, columns=DEFECT_COLUMNS)


def quality_report(analysis: Analysis, summary: pd.DataFrame, masks: pd.DataFrame) -> dict[str, object]:
    def issues_of(kind: str) -> list[dict[str, str]]:
        return [issue for issue in analysis.issues if issue["kind"] == kind]

    per_category = {}
    for category in analysis.categories:
        stats = analysis.stats[category]
        per_category[category] = {
            "image_modes": dict(sorted(stats.modes.items())),
            "image_sizes": dict(sorted(stats.sizes.items())),
            "mask_values": sorted(stats.mask_values),
            "train_subdirs": stats.train_subdirs,
        }

    gray_categories = [
        category
        for category in analysis.categories
        if analysis.stats[category].modes and set(analysis.stats[category].modes) <= {"L", "1", "I;16"}
    ]
    mixed_mode_categories = [
        category for category in analysis.categories if len(analysis.stats[category].modes) > 1
    ]
    smallest = []
    if not masks.empty:
        for row in masks.sort_values(["defect_pixels_256", "mask"]).head(5).itertuples():
            smallest.append(
                {
                    "mask": row.mask,
                    "defect_pixels": int(row.defect_pixels),
                    "defect_pixels_256": int(row.defect_pixels_256),
                }
            )
    loader = {
        "image_size": MODEL_IMAGE_SIZE,
        "train_contains_only_good": all(
            analysis.stats[category].train_subdirs in ([], ["good"]) for category in analysis.categories
        ),
        "every_defect_image_has_mask": not issues_of("missing_mask"),
        "good_test_images_without_masks": not issues_of("mask_for_good"),
        "all_images_square": not issues_of("non_square_image"),
        "grayscale_categories_converted_to_rgb": gray_categories,
        "mixed_mode_categories": mixed_mode_categories,
        "masks_binary_0_255": not issues_of("non_binary_mask"),
        "masks_empty_after_nearest_resize": len(issues_of("mask_empty_after_resize")),
        "smallest_masks_after_resize": smallest,
    }

    totals = {
        column: int(summary[column].sum())
        for column in ("train_good", "test_good", "test_defect", "test_total", "mask_count")
    }
    report: dict[str, object] = {
        "status": "critical" if analysis.has_critical else "ok",
        "expected_categories": list(analysis.categories),
        "found_categories": [c for c in analysis.categories if (analysis.data_root / c).is_dir()],
        "unexpected_root_entries": analysis.unexpected_root_entries,
        "totals": totals,
        "checked": {
            "images_read": analysis.images_read,
            "masks_read": analysis.masks_read,
            "image_mask_pairs_checked": analysis.pairs_checked,
        },
        "critical_issues": [issue for issue in analysis.issues if issue["severity"] == "critical"],
        "warnings": [issue for issue in analysis.issues if issue["severity"] == "warning"],
    }
    for name, kind in ISSUE_LISTS.items():
        report[name] = [issue["path"] for issue in issues_of(kind)]
    report["per_category"] = per_category
    report["loader_compatibility"] = loader
    return report


# --------------------------------------------------------------------------- plots


def save_figure(fig: plt.Figure, path: Path) -> None:
    if path.suffix == ".jpg":
        # Photo galleries are several times smaller as JPEG.
        fig.savefig(path, dpi=110, bbox_inches="tight", pil_kwargs={"quality": 85})
    else:
        fig.savefig(path, dpi=110, bbox_inches="tight", metadata=PNG_METADATA)
    plt.close(fig)


def plot_train_test_counts(summary: pd.DataFrame, path: Path) -> None:
    data = summary.set_index("category")[["train_good", "test_good", "test_defect"]]
    fig, ax = plt.subplots(figsize=(11, 6))
    positions = np.arange(len(data))
    width = 0.27
    labels = {"train_good": "train/good", "test_good": "test/good", "test_defect": "test, дефекты"}
    colors = {"train_good": "#4C72B0", "test_good": "#55A868", "test_defect": "#C44E52"}
    for offset, column in zip((-width, 0, width), data.columns):
        bars = ax.bar(positions + offset, data[column], width, label=labels[column], color=colors[column])
        ax.bar_label(bars, fontsize=6, padding=1)
    ax.set_xticks(positions, data.index, rotation=35, ha="right")
    ax.set_ylabel("Число изображений")
    ax.set_title("MVTec AD: обучающая и тестовая выборки по категориям")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    save_figure(fig, path)


def plot_test_composition(defects: pd.DataFrame, pilots: list[str], path: Path) -> None:
    fig, axes = plt.subplots(1, len(pilots), figsize=(5 * len(pilots), 4.5), squeeze=False)
    for ax, category in zip(axes[0], pilots):
        rows = defects[defects["category"] == category]
        colors = ["#55A868" if name == "good" else "#C44E52" for name in rows["defect_type"]]
        bars = ax.barh(rows["defect_type"], rows["image_count"], color=colors)
        ax.bar_label(bars, padding=2, fontsize=8)
        ax.invert_yaxis()
        ax.set_title(f"{category}: тест, всего {int(rows['image_count'].sum())}")
        ax.set_xlabel("Число изображений")
        ax.grid(axis="x", alpha=0.3)
    fig.suptitle("Состав тестовой выборки (зелёный — исправные)")
    fig.tight_layout()
    save_figure(fig, path)


def plot_defect_area(masks: pd.DataFrame, pilots: list[str], path: Path) -> None:
    fig, axes = plt.subplots(1, len(pilots), figsize=(5 * len(pilots), 4.5), squeeze=False)
    rng = np.random.default_rng(0)
    for ax, category in zip(axes[0], pilots):
        rows = masks[(masks["category"] == category) & (masks["defect_pixels"] > 0)]
        types = sorted(rows["defect_type"].unique())
        values = [rows.loc[rows["defect_type"] == name, "defect_fraction"].to_numpy() * 100 for name in types]
        if types:
            ax.boxplot(values, orientation="horizontal", showfliers=False)
            for index, value in enumerate(values, start=1):
                ax.scatter(value, index + rng.uniform(-0.15, 0.15, len(value)), s=8, alpha=0.6, color="#C44E52")
            ax.set_yticks(range(1, len(types) + 1), types)
        ax.set_xscale("log")
        ax.set_xticks(AREA_TICKS, [f"{tick:g}" for tick in AREA_TICKS])
        ax.xaxis.set_minor_formatter(NullFormatter())
        ax.invert_yaxis()
        ax.set_xlabel("Доля дефектных пикселей, % (лог. шкала)")
        ax.set_title(category)
        ax.grid(axis="x", alpha=0.3, which="both")
    fig.suptitle("Размер дефектных областей по исходным маскам")
    fig.tight_layout()
    save_figure(fig, path)


def load_thumbnail(path: Path, resample: Image.Resampling, mode: str) -> np.ndarray:
    with Image.open(path) as image:
        image = image.convert(mode)
        return np.asarray(image.resize((GALLERY_THUMB_SIZE, GALLERY_THUMB_SIZE), resample))


def plot_gallery(analysis: Analysis, category: str, path: Path) -> list[dict[str, str]]:
    """Plot the first readable image (sorted by name) of every test type with its mask."""
    root = analysis.data_root / category
    unreadable = {issue["path"] for issue in analysis.issues if issue["kind"] == "unreadable_image"}
    types = sorted(analysis.stats[category].test_counts, key=lambda name: (name != "good", name))
    examples = []
    for defect_type in types:
        images = [p for p in list_images(root / "test" / defect_type) if analysis.rel(p) not in unreadable]
        if not images:
            continue
        mask_path = root / "ground_truth" / defect_type / f"{images[0].stem}_mask.png"
        if defect_type != "good" and analysis.rel(mask_path) in unreadable:
            mask_path = root / "ground_truth" / defect_type / "unreadable"
        examples.append((defect_type, images[0], mask_path if defect_type != "good" else None))
    if not examples:
        return []

    fig, axes = plt.subplots(len(examples), 3, figsize=(7.5, 2.6 * len(examples)), squeeze=False)
    selected = []
    for row, (defect_type, image_path, mask_path) in zip(axes, examples):
        image = load_thumbnail(image_path, Image.Resampling.LANCZOS, "RGB")
        if mask_path is not None and mask_path.is_file():
            mask = load_thumbnail(mask_path, Image.Resampling.NEAREST, "L") > 0
            mask_label = analysis.rel(mask_path).split("/", 1)[1]
        else:
            mask = np.zeros(image.shape[:2], dtype=bool)
            mask_label = "маска нулевая" if mask_path is None else "маска отсутствует!"
        row[0].imshow(image)
        row[0].set_title(f"{defect_type}\n{analysis.rel(image_path).split('/', 1)[1]}", fontsize=8)
        row[1].imshow(mask, cmap="gray", vmin=0, vmax=1)
        row[1].set_title(mask_label, fontsize=8)
        row[2].imshow(image)
        if mask.any():
            row[2].contour(mask, levels=[0.5], colors="red", linewidths=1.2)
        row[2].set_title("контур маски", fontsize=8)
        for ax in row:
            ax.axis("off")
        selected.append(
            {
                "defect_type": defect_type,
                "image": analysis.rel(image_path),
                "mask": analysis.rel(mask_path) if mask_path is not None else "",
            }
        )
    fig.suptitle(f"{category}: исправный объект и примеры каждого типа дефекта", y=1.0)
    fig.tight_layout()
    save_figure(fig, path)
    return selected


# --------------------------------------------------------------------------- CLI


def parse_args(argv: list[str] | None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="EDA and data-quality checks for MVTec AD (CPU only, read-only for the dataset)."
    )
    parser.add_argument("--data-root", type=Path, required=True, help="directory that contains MVTec AD categories")
    parser.add_argument("--output-dir", type=Path, default=Path("results/eda"), help="plots, galleries, mask_stats.csv")
    parser.add_argument(
        "--tables-dir",
        type=Path,
        default=Path("results"),
        help="dataset_summary.csv, defect_summary.csv, data_quality.json (default: results)",
    )
    parser.add_argument(
        "--categories", nargs="+", default=list(EXPECTED_CATEGORIES), help="expected categories (default: all 15)"
    )
    parser.add_argument(
        "--pilot", nargs="+", default=list(PILOT_CATEGORIES), help="categories for detailed plots and galleries"
    )
    args = parser.parse_args(argv)
    if not args.data_root.is_dir():
        parser.error(f"--data-root is not a directory: {args.data_root}")
    return args


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    data_root = args.data_root.resolve()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    args.tables_dir.mkdir(parents=True, exist_ok=True)

    analysis = analyze(data_root, list(args.categories))
    summary = summary_table(analysis)
    defects = defect_table(analysis)
    masks = pd.DataFrame(analysis.mask_rows, columns=MASK_STATS_COLUMNS).sort_values("mask", ignore_index=True)

    summary.to_csv(args.tables_dir / "dataset_summary.csv", index=False, lineterminator="\n")
    defects.to_csv(args.tables_dir / "defect_summary.csv", index=False, lineterminator="\n")
    masks.to_csv(args.output_dir / "mask_stats.csv", index=False, lineterminator="\n")

    plot_train_test_counts(summary, args.output_dir / "train_test_counts.png")
    pilots = [c for c in args.pilot if c in analysis.stats and analysis.stats[c].test_counts]
    for category in args.pilot:
        if category not in pilots:
            print(f"[warn] pilot category {category} has no test data, skipped in plots", file=sys.stderr)
    galleries = {}
    if pilots:
        plot_test_composition(defects, pilots, args.output_dir / "test_composition_pilot.png")
        plot_defect_area(masks, pilots, args.output_dir / "defect_area_pilot.png")
        for category in pilots:
            galleries[category] = plot_gallery(analysis, category, args.output_dir / f"gallery_{category}.jpg")

    report = quality_report(analysis, summary, masks)
    report["gallery_examples"] = galleries
    with open(args.tables_dir / "data_quality.json", "w", encoding="utf-8", newline="\n") as file:
        json.dump(report, file, ensure_ascii=False, indent=2)
        file.write("\n")

    totals = report["totals"]
    print(
        f"[done] categories={len(summary)} train_good={totals['train_good']} test_total={totals['test_total']} "
        f"masks={totals['mask_count']} critical={len(report['critical_issues'])} warnings={len(report['warnings'])}"
    )
    if analysis.has_critical:
        for issue in report["critical_issues"][:20]:
            print(f"[critical] {issue['kind']}: {issue['path']} - {issue['detail']}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
