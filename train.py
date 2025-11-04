import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datetime import datetime
from typing import Dict
from dataset import SegmentationDataset
from model import get_model
from utils import Metrics
from config import Config
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


def denormalize(tensor, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    mean = torch.tensor(mean).view(3, 1, 1).to(tensor.device)
    std = torch.tensor(std).view(3, 1, 1).to(tensor.device)
    return tensor * std + mean


def tensor_to_pil(tensor):
    img = denormalize(tensor).cpu().clamp(0, 1).numpy()
    img = np.transpose(img, (1, 2, 0))
    return Image.fromarray((img * 255).astype(np.uint8))


def mask_to_rgb(mask, num_classes=9):
    cmap = plt.get_cmap('tab20', num_classes)
    mask_np = mask.cpu().numpy().astype(np.float32) / (num_classes - 1)
    colored = cmap(mask_np)[..., :3]
    return Image.fromarray((colored * 255).astype(np.uint8))


def train_model(
    data_dir: str,
    model_name: str,
    encoder_name: str,
    output_dir: str,
    limit_samples: int = 0,
    epochs: int = 5,
    writer: SummaryWriter | None = None
) -> Dict:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training {model_name} with {encoder_name} on {device}")

    # --- Datasets ---
    train_dataset = SegmentationDataset(data_dir, split="train", limit_samples=limit_samples)
    val_dataset = SegmentationDataset(data_dir, split="val", limit_samples=limit_samples)

    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True, num_workers=Config.NUM_WORKERS)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False, num_workers=Config.NUM_WORKERS)

    # --- Model ---
    model = get_model(model_name, encoder_name, classes=Config.NUM_CLASSES).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)

    # --- TensorBoard ---
    tb_writer = writer or SummaryWriter(log_dir=os.path.join(output_dir, "logs"))

    # --- Collect up to 10 fixed validation samples ---
    print("Collecting up to 10 validation samples for visualization...")
    val_iter = iter(val_loader)
    fixed_images_list, fixed_masks_list = [], []
    collected = 0
    max_samples = 10

    while collected < max_samples:
        try:
            images, masks = next(val_iter)
            images, masks = images.to(device), masks.to(device)
            num_to_take = min(len(images), max_samples - collected)
            fixed_images_list.append(images[:num_to_take])
            fixed_masks_list.append(masks[:num_to_take])
            collected += num_to_take
        except StopIteration:
            break

    if collected == 0:
        print("Warning: No validation samples. Skipping image logging.")
        fixed_images = fixed_masks = None
    else:
        fixed_images = torch.cat(fixed_images_list, dim=0)
        fixed_masks = torch.cat(fixed_masks_list, dim=0)
        fixed_images_pil = [tensor_to_pil(img) for img in fixed_images]
        fixed_masks_rgb = [mask_to_rgb(mask) for mask in fixed_masks]
        print(f"Using {len(fixed_images)} validation samples for logging.")

    best_val_iou = 0.0
    metric_names = ["loss", "accuracy", "precision", "recall", "f1", "iou"]

    for epoch in range(1, epochs + 1):
        # --- Train ---
        model.train()
        train_metrics = {k: 0.0 for k in metric_names}
        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]"):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            batch_metrics = Metrics.compute_batch_metrics(outputs, masks)
            for k in metric_names:
                train_metrics[k] += batch_metrics[k]
        for k in train_metrics:
            train_metrics[k] /= len(train_loader)

        # --- Validation ---
        model.eval()
        val_metrics = {k: 0.0 for k in metric_names}
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                batch_metrics = Metrics.compute_batch_metrics(outputs, masks)
                for k in metric_names:
                    val_metrics[k] += batch_metrics[k]
            for k in val_metrics:
                val_metrics[k] /= len(val_loader)

        # --- Log Scalars ---
        for k in metric_names:
            tb_writer.add_scalar(f"Train/{k.capitalize()}", train_metrics[k], epoch)
            tb_writer.add_scalar(f"Val/{k.capitalize()}", val_metrics[k], epoch)

        # --- Log Images ---
        if fixed_images is not None:
            with torch.no_grad():
                outputs = model(fixed_images)
                preds = torch.argmax(outputs, dim=1)
                pred_masks_rgb = [mask_to_rgb(pred) for pred in preds]

            num_samples = len(fixed_images)
            fig, axes = plt.subplots(num_samples, 3, figsize=(12, 4 * num_samples))
            if num_samples == 1:
                axes = [axes]

            for i in range(num_samples):
                row = axes[i] if num_samples > 1 else axes
                row[0].imshow(fixed_images_pil[i])
                row[0].set_title("Input")
                row[1].imshow(fixed_masks_rgb[i])
                row[1].set_title("GT")
                row[2].imshow(pred_masks_rgb[i])
                row[2].set_title("Pred")
                for ax in row: ax.axis('off')

            plt.tight_layout()
            tb_writer.add_figure(f"Predictions/{model_name}", fig, global_step=epoch)
            plt.close(fig)

        # --- Checkpoint ---
        if val_metrics["iou"] > best_val_iou:
            best_val_iou = val_metrics["iou"]
            torch.save({
                "state_dict": model.state_dict(),
                "best_iou": best_val_iou,
                "epoch": epoch
            }, os.path.join(output_dir, "best_model.pth"))

        print(f"Epoch {epoch}: Val IoU = {val_metrics['iou']:.4f}")

    if not writer:
        tb_writer.close()
    return {"best_iou": best_val_iou}