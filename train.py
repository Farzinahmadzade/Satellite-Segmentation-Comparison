import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datetime import datetime
from typing import Dict, List
from dataset import SegmentationDataset
from model import get_model
from utils import Metrics


def train_model(
    data_dir: str,
    model_name: str,
    encoder_name: str,
    output_dir: str,
    limit_samples: int = 0,
    epochs: int = 5,
    batch_size: int = 4,
    writer: SummaryWriter | None = None
) -> Dict[str, List[float]]:
    """
    Train a single segmentation model with TensorBoard logging and model checkpointing.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training {model_name} with {encoder_name} on {device}")

    # Datasets
    train_dataset = SegmentationDataset(data_dir, split="train", limit_samples=limit_samples)
    val_dataset = SegmentationDataset(data_dir, split="val", limit_samples=limit_samples)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    # Model
    model = get_model(model_name, encoder_name, classes=9).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    # Logging
    log_dir = os.path.join(output_dir, "logs")
    tb_writer = writer or SummaryWriter(
        log_dir=log_dir,
        name=f"{model_name}_{encoder_name}",
        comment=f"{model_name.upper()} segmentation"
    )

    # Model info
    tb_writer.add_text(
        "Model Configuration",
        f"""
**Model**: `{model_name}`  
**Encoder**: `{encoder_name}`  
**Classes**: `9` (OpenEarthMap)  
**Input Size**: `256×256`  
**Epochs**: `{epochs}`  
**Batch Size**: `{batch_size}`  
**Learning Rate**: `1e-3`  
**Optimizer**: `Adam`  
**Loss**: `CrossEntropy`  
**Device**: `{device}`  
**Started**: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`
        """.strip(),
        global_step=0
    )

    best_val_iou = 0.0
    history = {"train_loss": [], "val_loss": [], "val_iou": []}

    for epoch in range(1, epochs + 1):
        model.train()
        running_loss = 0.0

        for images, masks in tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [Train]"):
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        history["train_loss"].append(train_loss)

        # Validation
        model.eval()
        val_loss = 0.0
        total_iou = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

                metrics = Metrics.compute_batch_metrics(outputs, masks, num_classes=9)
                total_iou += metrics["iou"]

        val_loss /= len(val_loader)
        val_iou = total_iou / len(val_loader)
        history["val_loss"].append(val_loss)
        history["val_iou"].append(val_iou)

        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, IoU={val_iou:.3f}")

        # Checkpoint
        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pth"))

        # TensorBoard
        if tb_writer:
            tb_writer.add_scalar("Loss/train", train_loss, epoch)
            tb_writer.add_scalar("Loss/val", val_loss, epoch)
            tb_writer.add_scalar("IoU/val", val_iou, epoch)

    if not writer:
        tb_writer.close()

    print(f"Training completed! Best IoU: {best_val_iou:.3f}")
    return history