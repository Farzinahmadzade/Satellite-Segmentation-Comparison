import os
import shutil
import torch
from torch.utils.data import DataLoader, random_split
from torch import nn, optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from dataset import SegmentationDataset
from config import Config
from utils import Metrics
import segmentation_models_pytorch as smp
import argparse

def get_model(model_name, encoder_name='resnet34', in_channels=3, classes=Config.NUM_CLASSES):
    model_name = model_name.lower()
    if model_name == 'unet':
        return smp.Unet(encoder_name=encoder_name, in_channels=in_channels, classes=classes)
    elif model_name == 'unet++':
        return smp.UnetPlusPlus(encoder_name=encoder_name, in_channels=in_channels, classes=classes)
    elif model_name == 'fpn':
        return smp.FPN(encoder_name=encoder_name, in_channels=in_channels, classes=classes)
    elif model_name == 'pspnet':
        return smp.PSPNet(encoder_name=encoder_name, in_channels=in_channels, classes=classes)
    elif model_name == 'deeplabv3':
        return smp.DeepLabV3(encoder_name=encoder_name, in_channels=in_channels, classes=classes)
    elif model_name == 'deeplabv3+':
        return smp.DeepLabV3Plus(encoder_name=encoder_name, in_channels=in_channels, classes=classes)
    elif model_name == 'linknet':
        return smp.Linknet(encoder_name=encoder_name, in_channels=in_channels, classes=classes)
    elif model_name == 'manet':
        return smp.MAnet(encoder_name=encoder_name, in_channels=in_channels, classes=classes)
    elif model_name == 'pan':
        return smp.PAN(encoder_name=encoder_name, in_channels=in_channels, classes=classes)
    elif model_name == 'upernet':
        return smp.UPerNet(encoder_name=encoder_name, in_channels=in_channels, classes=classes)
    elif model_name == 'segformer':
        return smp.Segformer(encoder_name=encoder_name, in_channels=in_channels, classes=classes)
    elif model_name == 'dpt':
        return smp.DPT(encoder_name=encoder_name, in_channels=in_channels, classes=classes)
    else:
        raise NotImplementedError(f"Model {model_name} not implemented")

def train_model(data_dir, model_name, encoder_name, output_dir, limit_samples=0):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    os.makedirs(output_dir, exist_ok=True)
    log_dir = os.path.join(output_dir, "logs")
    shutil.rmtree(log_dir, ignore_errors=True)
    writer = SummaryWriter(log_dir)

    full_dataset = SegmentationDataset(
        data_dir, split="train", image_size=Config.IMAGE_SIZE, limit_samples=limit_samples)
    total_size = len(full_dataset)
    val_size = int(total_size * Config.VAL_SPLIT)
    train_size = total_size - val_size

    train_dataset, val_dataset = random_split(
        full_dataset, [train_size, val_size],
        generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train_dataset, batch_size=Config.BATCH_SIZE, shuffle=True,
                              num_workers=Config.NUM_WORKERS, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.BATCH_SIZE, shuffle=False,
                            num_workers=Config.NUM_WORKERS, pin_memory=True)

    model = get_model(model_name, encoder_name).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=2)

    best_val_loss = float("inf")
    patience_counter = 0
    early_stop_patience = 5

    print(f"Training {model_name} ({encoder_name}) for {Config.EPOCHS} epochs | Batch: {Config.BATCH_SIZE} | Classes: {Config.NUM_CLASSES}")

    for epoch in range(Config.EPOCHS):
        model.train()
        train_loss = 0.0
        train_metrics = {"iou":0, "dice":0, "acc":0}
        num_batches = 0

        loop = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{Config.EPOCHS}] Training", leave=False)
        for images, masks in loop:
            images, masks = images.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()

            batch_metrics = Metrics.compute_batch_metrics(outputs, masks, Config.NUM_CLASSES)
            train_loss += loss.item()
            for k in train_metrics:
                train_metrics[k] += batch_metrics[k]
            num_batches += 1

        train_loss /= num_batches
        for k in train_metrics:
            train_metrics[k] /= num_batches

        model.eval()
        val_loss = 0.0
        val_metrics = {"iou":0, "dice":0, "acc":0}
        val_batches = 0
        with torch.no_grad():
            for batch_idx, (images, masks) in enumerate(val_loader):
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                batch_metrics = Metrics.compute_batch_metrics(outputs, masks, Config.NUM_CLASSES)

                val_loss += loss.item()
                for k in val_metrics:
                    val_metrics[k] += batch_metrics[k]
                val_batches += 1

                if batch_idx==0:
                    for j in range(min(2, images.size(0))):
                        Metrics.visualize_sample(images[j], masks[j], outputs[j], j, writer, epoch)

        val_loss /= val_batches
        for k in val_metrics:
            val_metrics[k] /= val_batches

        scheduler.step(val_loss)

        writer.add_scalars("Loss", {"Train":train_loss,"Val":val_loss}, epoch)
        writer.add_scalars("IoU", {"Train":train_metrics["iou"],"Val":val_metrics["iou"]}, epoch)
        writer.add_scalars("Dice", {"Train":train_metrics["dice"],"Val":val_metrics["dice"]}, epoch)
        writer.add_scalars("Accuracy", {"Train":train_metrics["acc"],"Val":val_metrics["acc"]}, epoch)

        print(f"\nEpoch {epoch+1}/{Config.EPOCHS} | Train: Loss={train_loss:.4f}, IoU={train_metrics['iou']:.3f}, Dice={train_metrics['dice']:.3f}, Acc={train_metrics['acc']:.3f} | Val: Loss={val_loss:.4f}, IoU={val_metrics['iou']:.3f}, Dice={val_metrics['dice']:.3f}, Acc={val_metrics['acc']:.3f}")

        model_path = os.path.join(output_dir, f"epoch_{epoch+1}_val_{val_loss:.4f}.pth")
        torch.save(model.state_dict(), model_path)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_path = os.path.join(output_dir, "best_model.pth")
            torch.save(model.state_dict(), best_path)
            print(f"New best model saved: {best_path} (val_loss={best_val_loss:.4f})")
        else:
            patience_counter +=1
            if patience_counter >= early_stop_patience:
                print("Early stopping triggered.")
                break

    writer.close()
    print(f"\n✅ Training completed! Best model: {best_path}")
    print(f"Classes: {Config.CLASS_NAMES}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="unet", help="Model to train or 'all' for all models")
    parser.add_argument("--encoder_name", type=str, default="resnet34")
    parser.add_argument("--output_dir", type=str, default="./outputs")
    parser.add_argument("--limit_samples", type=int, default=0, help="Limit number of samples for quick test")
    args = parser.parse_args()

    model_list = [
        "unet","unet++","fpn","pspnet","deeplabv3","deeplabv3+","linknet",
        "manet","pan","upernet","segformer","dpt"]

    if args.model_name.lower() == "all":
        for m in model_list:
            out_dir = os.path.join(args.output_dir, m)
            os.makedirs(out_dir, exist_ok=True)
            train_model(args.data_dir, m, args.encoder_name, out_dir, args.limit_samples)
    else:
        train_model(args.data_dir, args.model_name, args.encoder_name, args.output_dir, args.limit_samples)
