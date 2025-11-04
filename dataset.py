import os
import torch
from torch.utils.data import Dataset
import albumentations as A
from albumentations.pytorch import ToTensorV2
from PIL import Image
import numpy as np
from typing import List, Tuple
from config import Config

class SegmentationDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        split: str = "train",
        image_size: Tuple[int, int] = Config.IMAGE_SIZE,
        val_split: float = Config.VAL_SPLIT,
        limit_samples: int = 0
    ):
        self.images: List[str] = []
        self.masks: List[str] = []
        self.image_size = image_size

        # 1. Collect all image and label files
        img_paths = []
        lbl_paths = []
        for root, dirs, _ in os.walk(data_dir):
            if "images" in dirs and "labels" in dirs:
                img_dir = os.path.join(root, "images")
                lbl_dir = os.path.join(root, "labels")

                imgs = [os.path.join(img_dir, f) for f in os.listdir(img_dir)
                        if f.lower().endswith('.tif')]
                lbls = [os.path.join(lbl_dir, f) for f in os.listdir(lbl_dir)
                        if f.lower().endswith('.tif')]

                img_paths.extend(imgs)
                lbl_paths.extend(lbls)

        if not img_paths or not lbl_paths:
            raise ValueError(f"No .tif files found in {data_dir}")

        # 2. Pair by basename
        img_dict = {os.path.splitext(os.path.basename(p))[0]: p for p in img_paths}
        lbl_dict = {os.path.splitext(os.path.basename(p))[0]: p for p in lbl_paths}

        common_keys = sorted(set(img_dict.keys()) & set(lbl_dict.keys()))
        if len(common_keys) == 0:
            raise ValueError(f"No paired images/labels found in {data_dir}")

        self.images = [img_dict[k] for k in common_keys]
        self.masks = [lbl_dict[k] for k in common_keys]

        print(f"Found {len(self.images)} paired samples across all regions")

        # 3. Train/val split
        total = len(self.images)
        split_idx = int(total * (1 - val_split))
        if split == "train":
            self.images = self.images[:split_idx]
            self.masks = self.masks[:split_idx]
        else:
            self.images = self.images[split_idx:]
            self.masks = self.masks[split_idx:]

        # 4. Limit samples
        if limit_samples > 0 and len(self.images) > limit_samples:
            self.images = self.images[:limit_samples]
            self.masks = self.masks[:limit_samples]
        print(f"Dataset ({split}): {len(self.images)} samples")

        # 5. Augmentation
        height, width = image_size if isinstance(image_size, tuple) else (image_size, image_size)
        self.transform = A.Compose([
            A.Resize(height=height, width=width),
            A.HorizontalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
            ToTensorV2()
        ])

    def __len__(self) -> int:
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        img = np.array(Image.open(self.images[idx]).convert("RGB"))
        mask = np.array(Image.open(self.masks[idx]), dtype=np.int64)

        transformed = self.transform(image=img, mask=mask)
        return transformed["image"].float(), transformed["mask"].long()