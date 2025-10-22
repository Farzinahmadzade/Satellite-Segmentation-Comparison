import os
from torch.utils.data import Dataset
from PIL import Image
import torch
import torchvision.transforms as T
import albumentations as A
from albumentations.pytorch import ToTensorV2

class SegmentationDataset(Dataset):
    def __init__(self, image_paths, mask_paths, augment=True, image_size=256):
        self.image_paths = image_paths
        self.mask_paths = mask_paths
        self.augment = augment
        self.image_size = image_size

        self.transform = A.Compose([
            A.Resize(image_size, image_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            ToTensorV2()
        ]) if augment else A.Compose([A.Resize(image_size, image_size), ToTensorV2()])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = np.array(Image.open(self.image_paths[idx]).convert("RGB"))
        mask = np.array(Image.open(self.mask_paths[idx]))
        # Ensure mask is LongTensor for cross-entropy
        augmented = self.transform(image=image, mask=mask)
        image = augmented['image']
        mask = torch.as_tensor(augmented['mask'], dtype=torch.long)
        return image, mask
