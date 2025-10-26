import os
import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
import numpy as np
from config import Config

class SegmentationDataset(Dataset):
    def __init__(self, data_dir, split="train", image_size=256, val_split=0.2, limit_samples=0):
        self.images = []
        self.masks = []
        self.image_size = image_size

        for root, dirs, _ in os.walk(data_dir):
            if "images" in dirs and "labels" in dirs:
                img_dir = os.path.join(root, "images")
                lbl_dir = os.path.join(root, "labels")

                imgs = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.lower().endswith(".tif")])
                lbls = sorted([os.path.join(lbl_dir, f) for f in os.listdir(lbl_dir) if f.lower().endswith(".tif")])

                img_dict = {os.path.splitext(os.path.basename(p))[0].lower(): p for p in imgs}
                lbl_dict = {os.path.splitext(os.path.basename(p))[0].lower(): p for p in lbls}
                common = sorted(set(img_dict.keys()) & set(lbl_dict.keys()))

                if len(common) < len(imgs) or len(common) < len(lbls):
                    skipped = (len(imgs) + len(lbls)) - 2*len(common)
                    print(f"⚠️ Skipping {skipped} unmatched files in {os.path.basename(root)}")

                self.images.extend([img_dict[k] for k in common])
                self.masks.extend([lbl_dict[k] for k in common])

        if len(self.images) == 0:
            raise ValueError(f"No paired images found in {data_dir}")

        if limit_samples > 0:
            self.images = self.images[:limit_samples]
            self.masks = self.masks[:limit_samples]

        total = len(self.images)
        split_idx = int(total * (1 - val_split))
        if split == "train":
            self.images = self.images[:split_idx]
            self.masks = self.masks[:split_idx]
        else:
            self.images = self.images[split_idx:]
            self.masks = self.masks[split_idx:]

        print(f"Dataset ({split}): {len(self.images)} samples")

        self.transform_img = transforms.Compose([
            transforms.Resize(self.image_size),
            transforms.ToTensor(),
        ])
        self.transform_mask = transforms.Compose([
            transforms.Resize(self.image_size, interpolation=Image.NEAREST),
        ])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img = Image.open(self.images[idx]).convert("RGB")
        mask = Image.open(self.masks[idx])
        mask = np.array(self.transform_mask(mask), dtype=np.int64)
        mask = torch.from_numpy(mask).long()
        img = self.transform_img(img)
        return img, mask
