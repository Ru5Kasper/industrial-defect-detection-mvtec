"""Train PatchCore memory banks from MVTec train/good only."""
import argparse
import json
import platform
from pathlib import Path
import torch
import torchvision
import yaml
from torch.utils.data import DataLoader
from src.patchcore import PatchCore, set_seed
from src.patchcore_data import MVTecPatchDataset

def resolve_device(value):
    if value == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if value == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false.")
    return value

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="configs/patchcore.yaml")
    parser.add_argument("--data-root", default=None)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    ds, mc, tc = config["dataset"], config["model"], config["training"]
    root = args.data_root or ds.get("root", "data/mvtec_ad")
    device, seed = resolve_device(args.device or tc.get("device", "auto")), int(config.get("seed", 42))
    set_seed(seed)
    out = Path(tc.get("artifacts_dir", "artifacts/patchcore"))
    out.mkdir(parents=True, exist_ok=True)
    manifest = {"experiment_name": config.get("experiment_name", "patchcore"),
                "data_root": str(root), "device": device, "seed": seed,
                "python": platform.python_version(), "torch": torch.__version__,
                "torchvision": torchvision.__version__, "categories": {}}
    for category in ds["categories"]:
        dataset = MVTecPatchDataset(root, category, ds.get("image_size", 256), train=True)
        loader = DataLoader(dataset, batch_size=tc.get("batch_size", 8), shuffle=False,
                            num_workers=tc.get("num_workers", 0))
        model = PatchCore(device=device, image_size=ds.get("image_size", 256),
                          coreset_ratio=mc.get("coreset_ratio", 0.01),
                          max_bank_size=mc.get("max_bank_size", 10000),
                          nn_chunk_size=mc.get("nn_chunk_size", 2048),
                          seed=seed, pretrained=mc.get("pretrained", True))
        model.fit(loader)
        path = out / f"{category}.pt"
        model.save(path, metadata={"category": category, "train_images": len(dataset),
            "training_split": "train/good only", "backbone": "ResNet-18 ImageNet1K_V1",
            "feature_layers": ["layer2", "layer3"], "local_aggregation": "3x3 average pooling",
            "coreset_algorithm": "greedy k-center"})
        manifest["categories"][category] = {"train_good_images": len(dataset),
            "memory_bank_vectors": len(model.memory_bank), "model_path": str(path)}
        print(f"{category}: {len(dataset)} train/good images; memory bank={len(model.memory_bank)}")
    (out / "training_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"Saved: {out / 'training_manifest.json'}")

if __name__ == "__main__":
    main()
