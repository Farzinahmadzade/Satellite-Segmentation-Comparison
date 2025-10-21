import os
import random
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

class Config:
    IMG_SIZE = 256
    NUM_CLASSES = 5
    BATCH_SIZE = 8
    MAX_SAMPLES = 50

def get_csv_paths(output_dir):
    return {
        'train': os.path.join(output_dir, 'train.csv'),
        'val': os.path.join(output_dir, 'val.csv'),
        'test': os.path.join(output_dir, 'test.csv')
    }

def sample_csv(csv_path, max_samples=Config.MAX_SAMPLES):
    df = pd.read_csv(csv_path)
    if len(df) > max_samples:
        df = df.sample(n=max_samples, random_state=42)
    return df

class DamageDataset(Dataset):
    def __init__(self, csv_path, data_dir, transform=None, max_samples=Config.MAX_SAMPLES):
        self.df = sample_csv(csv_path, max_samples)
        self.data_dir = data_dir
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = os.path.join(self.data_dir, row['image_path'])
        mask_path = os.path.join(self.data_dir, row['label_path'])
        
        try:
            image = np.array(Image.open(img_path).convert('RGB'), dtype=np.float32) / 255.0
            mask = np.array(Image.open(mask_path).convert('L'), dtype=np.int64)
            mask = np.clip(mask, 0, Config.NUM_CLASSES - 1)
            
            if self.transform:
                augmented = self.transform(image=image, mask=mask)
                image, mask = augmented['image'], augmented['mask'].long()
            else:
                image = torch.from_numpy(image).permute(2, 0, 1)
                mask = torch.from_numpy(mask).long()
                
            return image, mask, os.path.basename(img_path)
        except Exception:
            return (torch.rand(3, Config.IMG_SIZE, Config.IMG_SIZE), 
                   torch.zeros(Config.IMG_SIZE, Config.IMG_SIZE, dtype=torch.long), 
                   'error')

def get_transforms():
    return A.Compose([
        A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

def get_loaders(data_dir, output_dir):
    csv_paths = get_csv_paths(output_dir)
    transform = get_transforms()
    
    train_ds = DamageDataset(csv_paths['train'], data_dir, transform)
    val_ds = DamageDataset(csv_paths['val'], data_dir, transform)
    test_ds = DamageDataset(csv_paths['test'], data_dir, transform)
    
    train_loader = DataLoader(train_ds, Config.BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=True)
    val_loader = DataLoader(val_ds, Config.BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    test_loader = DataLoader(test_ds, Config.BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=True)
    
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--output_dir', default='csvs')
    args = parser.parse_args()
    
    train_loader, val_loader, test_loader = get_loaders(args.data_dir, args.output_dir)
    images, masks, names = next(iter(train_loader))
    print(f"Batch shape: {images.shape}, Masks: {masks.shape}")