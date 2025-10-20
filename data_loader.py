import os
import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2

class SatelliteDataset(Dataset):
    def __init__(self, csv_file, transform=None, img_size=256):
        self.data = pd.read_csv(csv_file)
        self.transform = transform
        self.img_size = img_size
        print(f"Loaded {len(self.data)} samples from {csv_file}")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        try:
            img_path = self.data.iloc[idx]['image_path']
            label_path = self.data.iloc[idx]['label_path']
            
            if not os.path.exists(img_path):
                print(f"Warning: Image file not found: {img_path}")
            if not os.path.exists(label_path):
                print(f"Warning: Label file not found: {label_path}")
            
            image = np.array(Image.open(img_path).convert('RGB'), dtype=np.float32) / 255.0
            mask = np.array(Image.open(label_path).convert('L'), dtype=np.int64)
            
            mask = np.clip(mask, 0, 4)
            
            if self.transform:
                augmented = self.transform(image=image, mask=mask)
                image, mask = augmented['image'], augmented['mask'].long()
            else:
                image = torch.from_numpy(image).float().permute(2, 0, 1)
                mask = torch.from_numpy(mask).long()
            
            return image, mask, os.path.basename(img_path)
            
        except Exception as e:
            print(f"Error loading {self.data.iloc[idx]['image_path']}: {e}")
            dummy_img = torch.rand(3, self.img_size, self.img_size)
            dummy_mask = torch.zeros(self.img_size, self.img_size, dtype=torch.long)
            return dummy_img, dummy_mask, 'error'

def get_transforms(phase='train', img_size=256):
    if phase == 'train':
        return A.Compose([
            A.Resize(img_size, img_size),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomRotate90(p=0.5),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])
    else:
        return A.Compose([
            A.Resize(img_size, img_size),
            A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ToTensorV2(),
        ])

def get_dataloaders(batch_size=8, img_size=256, num_workers=2):
    """ایجاد دیتالودرهای train, val, test"""
    
    for csv_file in ['train.csv', 'val.csv', 'test.csv']:
        if not os.path.exists(csv_file):
            print(f"Warning: {csv_file} not found!")
    
    train_dataset = SatelliteDataset(
        'train.csv', 
        transform=get_transforms('train', img_size),
        img_size=img_size
    )
    
    val_dataset = SatelliteDataset(
        'val.csv',
        transform=get_transforms('val', img_size), 
        img_size=img_size
    )
    
    test_dataset = SatelliteDataset(
        'test.csv',
        transform=get_transforms('val', img_size),
        img_size=img_size
    )
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    
    return train_loader, val_loader, test_loader