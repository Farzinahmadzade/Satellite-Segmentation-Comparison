import os
import glob
import shutil
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from datetime import datetime

from config import Config
from dataset import SegmentationDataset
from utils import iou_score, dice_score
from model import get_model

import warnings
warnings.filterwarnings("ignore")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def prepare_dataset(data_dir):
    image_paths = sorted(glob.glob(os.path.join(data_dir, "*", "images", "*.png")))
    mask_paths = sorted(glob.glob(os.path.join(data_dir, "*", "labels", "*.png")))

    assert len(image_paths) == len(mask_paths), "Mismatch image/mask count!"
    dataset = SegmentationDataset(image_paths, mask_paths, augment=True, image_size=Config.IMAGE_SIZE)
    val_size = int(len(dataset) * Config.VAL_SPLIT)
    train_size = len(dataset) - val_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    return train_dataset, val_dataset

def visualize_batch(writer, images, masks, preds, step):
    import torchvision
    masks_rgb = masks.unsqueeze(1).repeat(1,3,1,1)
    preds_rgb = preds.unsqueeze(1).repeat(1,3,1,1)
    grid_list = []
    for i in range(images.size(0)):
        row = torch.cat([images[i], masks_rgb[i], preds_rgb[i]], dim=2)
        grid_list.append(row)
    grid = torch.stack(grid_list)
    writer.add_images("Batch Predictions", grid, global_step=step)

def train_model(data_dir, model_name="unet", encoder_name="resnet34", output_dir="./outputs"):
    # پاک کردن logهای قبلی
    if os.path.exists(Config.LOG_DIR):
        shutil.rmtree(Config.LOG_DIR)
    os.makedirs(Config.LOG_DIR, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    writer = SummaryWriter(log_dir=os.path.join(Config.LOG_DIR, f"{model_name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}"))

    train_dataset, val_dataset = prepare_dataset(data_dir)
    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=Config.NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS)

    model = get_model(model_name, encoder_name, num_classes=Config.NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=Config.LR)

    best_iou = 0

    for epoch in range(Config.EPOCHS):
        model.train()
        running_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{Config.EPOCHS}]")
        for images, masks in pbar:
            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        train_loss = running_loss / len(train_loader)
        writer.add_scalar("Train/Loss", train_loss, epoch)
        print(f"📉 Epoch {epoch+1}: Train loss = {train_loss:.4f}")

        # Validation
        model.eval()
        val_loss = 0
        all_preds, all_masks = [], []
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()
                all_preds.append(outputs.cpu())
                all_masks.append(masks.cpu())

            val_loss /= len(val_loader)
            all_preds = torch.cat(all_preds, dim=0)
            all_masks = torch.cat(all_masks, dim=0)

            iou = iou_score(all_preds, all_masks, num_classes=Config.NUM_CLASSES)
            dice = dice_score(all_preds, all_masks, num_classes=Config.NUM_CLASSES)

        writer.add_scalar("Val/Loss", val_loss, epoch)
        writer.add_scalar("Val/IoU", iou, epoch)
        writer.add_scalar("Val/Dice", dice, epoch)
        print(f"📊 Validation: loss={val_loss:.4f}, IoU={iou:.4f}, Dice={dice:.4f}")

        # Visualize first batch of validation
        visualize_batch(writer, images.cpu(), masks.cpu(), torch.argmax(outputs.cpu(), dim=1), step=epoch)

        # Save best model
        if iou > best_iou:
            best_iou = iou
            torch.save(model.state_dict(), Config.BEST_MODEL_PATH)
            print(f"💾 Checkpoint saved: {Config.BEST_MODEL_PATH}")

    writer.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="unet")
    parser.add_argument("--encoder_name", type=str, default="resnet34")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    args = parser.parse_args()

    train_model(args.data_dir, args.model_name, args.encoder_name, args.output_dir)
