"""MVTec AD dataset helpers dedicated to PatchCore."""
from pathlib import Path
from PIL import Image
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode

MEAN = (0.485, 0.456, 0.406)
STD = (0.229, 0.224, 0.225)
EXTS = {".png", ".jpg", ".jpeg", ".bmp"}

def image_transform(size: int):
    return transforms.Compose([
        transforms.Resize((size, size), antialias=True),
        transforms.ToTensor(),
        transforms.Normalize(MEAN, STD),
    ])

class MVTecPatchDataset(Dataset):
    """Train mode indexes only train/good; test mode validates every defect mask."""
    def __init__(self, root, category, image_size=256, train=False):
        self.root = Path(root) / category
        self.category = category
        self.train = train
        split = self.root / ("train" if train else "test")
        if not split.is_dir():
            raise FileNotFoundError(f"Missing split directory: {split}")
        self.transform = image_transform(image_size)
        self.mask_transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.NEAREST),
            transforms.PILToTensor(),
        ])
        self.samples = []
        folders = [split / "good"] if train else sorted(p for p in split.iterdir() if p.is_dir())
        for folder in folders:
            if not folder.is_dir():
                raise FileNotFoundError(f"Missing expected folder: {folder}")
            defect = folder.name
            for path in sorted(p for p in folder.iterdir() if p.suffix.lower() in EXTS):
                mask = None
                if not train and defect != "good":
                    mask = self.root / "ground_truth" / defect / f"{path.stem}_mask.png"
                    if not mask.is_file():
                        raise FileNotFoundError(f"Missing mask for {path}: {mask}")
                self.samples.append((path, defect, mask))
        if not self.samples:
            raise ValueError(f"No images found under {split}")
    def __len__(self):
        return len(self.samples)
    def __getitem__(self, index):
        path, defect, mask_path = self.samples[index]
        image = self.transform(Image.open(path).convert("RGB"))
        if mask_path is None:
            mask = torch.zeros((1, self.transform.transforms[0].size[0], self.transform.transforms[0].size[1]), dtype=torch.uint8)
        else:
            mask = (self.mask_transform(Image.open(mask_path).convert("L")) > 0).to(torch.uint8)
        return {"image": image, "mask": mask, "label": int(defect != "good"),
                "defect_type": defect, "path": str(path)}
