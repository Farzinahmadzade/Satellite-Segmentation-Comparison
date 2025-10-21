import os
import random
import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.dataloader import default_collate
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

class Config:
    IMG_SIZE = 256
    NUM_CLASSES = 9
    BATCH_SIZE = 32
    NUM_WORKERS = 8

def get_csv_paths(output_dir):
    return {
        'train': os.path.join(output_dir, 'train.csv'),
        'val': os.path.join(output_dir, 'val.csv'),
        'test': os.path.join(output_dir, 'test.csv')
    }

def sample_csv(csv_path):
    df = pd.read_csv(csv_path)
    return df

class SatelliteDataset(Dataset):
    def __init__(self, csv_path, data_dir, transform=None):
        self.df = sample_csv(csv_path)
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
                return augmented['image'], augmented['mask'].long(), os.path.basename(img_path)
            else:
                return (torch.from_numpy(image).permute(2, 0, 1).float(),
                        torch.from_numpy(mask).long(),
                        os.path.basename(img_path))
        
        except Exception as e:
            print(f"\nSkipping file due to error: {img_path}. Error: {e}")
            return None 

def get_transforms():
    return A.Compose([
        A.Resize(Config.IMG_SIZE, Config.IMG_SIZE),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.3),
        A.RandomRotate90(p=0.3),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ])

def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    
    if not batch:
        print("\nWarning: Entire batch failed to load. Returning dummy batch.")
        dummy_img = torch.rand(1, 3, Config.IMG_SIZE, Config.IMG_SIZE) * 0.1  # Low noise
        dummy_mask = torch.zeros(1, Config.IMG_SIZE, Config.IMG_SIZE, dtype=torch.long)
        dummy_name = ['dummy']
        return dummy_img, dummy_mask, dummy_name
        
    return default_collate(batch)

def get_loaders(data_dir, output_dir, batch_size=Config.BATCH_SIZE):
    csv_paths = get_csv_paths(output_dir)
    transform = get_transforms()
    
    train_ds = SatelliteDataset(csv_paths['train'], data_dir, transform)
    val_ds = SatelliteDataset(csv_paths['val'], data_dir, transform)
    test_ds = SatelliteDataset(csv_paths['test'], data_dir, transform)
    
    train_loader = DataLoader(train_ds, batch_size, shuffle=True, num_workers=Config.NUM_WORKERS, pin_memory=True, collate_fn=collate_fn)
    val_loader = DataLoader(val_ds, batch_size, shuffle=False, num_workers=Config.NUM_WORKERS, pin_memory=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_ds, batch_size, shuffle=False, num_workers=Config.NUM_WORKERS, pin_memory=True, collate_fn=collate_fn)
    
    return train_loader, val_loader, test_loader

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--output_dir', default='csvs')
    args = parser.parse_args()
    
    train_loader, val_loader, _ = get_loaders(args.data_dir, args.output_dir)
    batch = next(iter(train_loader))
    print(f"Batch shapes: images={batch[0].shape}, masks={batch[1].shape}")