import os
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from dataset import SegmentationDataset
from model import get_model

def train_model(data_dir, model_name, encoder_name, output_dir,
                limit_samples=0, epochs=5, writer=None, batch_size=4):
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = SegmentationDataset(data_dir, split="train", limit_samples=limit_samples)
    val_dataset = SegmentationDataset(data_dir, split="val", limit_samples=limit_samples)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    model = get_model(model_name, encoder_name, classes=9).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    best_val_iou = 0.0

    train_history = {"train_loss": [], "val_loss": [], "val_iou": []}

    for epoch in range(1, epochs+1):
        model.train()
        running_loss = 0.0

        loop = tqdm(train_loader, desc=f"Epoch {epoch}/{epochs} [{model_name}]")
        for images, masks in loop:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

        train_loss = running_loss / len(train_loader)
        train_history["train_loss"].append(train_loss)

        model.eval()
        val_loss = 0.0
        total_iou = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item()

                preds = torch.argmax(outputs, dim=1)

                intersection = ((preds == masks) & (masks >= 0)).float().sum()
                union = ((preds >= 0) | (masks >= 0)).float().sum()
                iou = (intersection / (union + 1e-6)).item()
                total_iou += iou

        val_loss /= len(val_loader)
        val_iou = total_iou / len(val_loader)
        train_history["val_loss"].append(val_loss)
        train_history["val_iou"].append(val_iou)

        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Loss={val_loss:.4f}, IoU={val_iou:.3f}")

        if val_iou > best_val_iou:
            best_val_iou = val_iou
            torch.save(model.state_dict(), os.path.join(output_dir, "best_model.pth"))

        if writer:
            writer.add_scalar("Loss/train", train_loss, epoch)
            writer.add_scalar("Loss/val", val_loss, epoch)
            writer.add_scalar("IoU/val", val_iou, epoch)

    print(f"Training completed! Best IoU: {best_val_iou:.3f}")
    return train_history
