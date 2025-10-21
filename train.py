import time
import argparse
from pathlib import Path
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import segmentation_models_pytorch as smp
from data_loader import get_loaders, Config
from model import get_model
from utils import Metrics, save_model, plot_training_history, print_final_results, get_writer, log_to_tensorboard

def train_epoch(model, loader, criterion, optimizer, device, scaler=None):
    model.train()
    total_loss = 0
    ious, accs = [], []

    pbar = tqdm(loader, desc="Train")
    for images, masks, _ in pbar:
        images, masks = images.to(device), masks.to(device)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast(enabled=(scaler is not None)):
            outputs = model(images)
            loss = criterion(outputs, masks)

        if scaler:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        total_loss += loss.item()
        iou = Metrics.iou(outputs, masks)
        acc = Metrics.accuracy(outputs, masks)
        ious.append(iou.item())
        accs.append(acc.item())

        pbar.set_postfix({'loss': f'{loss.item():.3f}', 'iou': f'{iou:.3f}'})

    return total_loss / len(loader), sum(ious) / len(ious), sum(accs) / len(accs)

def val_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    ious, accs = [], []

    with torch.no_grad():
        pbar = tqdm(loader, desc="Validation")
        for images, masks, _ in pbar:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)

            total_loss += loss.item()
            iou = Metrics.iou(outputs, masks)
            acc = Metrics.accuracy(outputs, masks)
            ious.append(iou.item())
            accs.append(acc.item())

            pbar.set_postfix({'loss': f'{loss.item():.3f}', 'iou': f'{iou:.3f}'})

    return total_loss / len(loader), sum(ious) / len(ious), sum(accs) / len(accs)

def main():
    parser = argparse.ArgumentParser(description="SMP Segmentation Trainer")
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--output_dir', default='csvs')
    parser.add_argument('--model_name', default='unet')
    parser.add_argument('--encoder_name', default='resnet34')
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=Config.BATCH_SIZE)
    parser.add_argument('--use_amp', action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    train_loader, val_loader, _ = get_loaders(args.data_dir, args.output_dir, args.batch_size)

    model = get_model(args.model_name, args.encoder_name).to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=3, factor=0.5, verbose=True
    )

    dice_loss = smp.losses.DiceLoss(mode='multiclass', from_logits=True)
    ce_loss   = smp.losses.SoftCrossEntropyLoss(smooth_factor=0.1)
    criterion = lambda outputs, masks: dice_loss(outputs, masks) + ce_loss(outputs, masks)

    scaler = torch.cuda.amp.GradScaler(enabled=args.use_amp)

    writer = get_writer(args.output_dir, args.model_name)

    history = {'train_loss': [], 'val_loss': [], 'train_iou': [], 'val_iou': [],
               'train_acc': [], 'val_acc': []}
    best_iou = 0

    for epoch in range(args.epochs):
        start_time = time.time()

        train_loss, train_iou, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, scaler)
        val_loss, val_iou, val_acc = val_epoch(model, val_loader, criterion, device)

        history['train_loss'].append(train_loss)
        history['val_loss'].append(val_loss)
        history['train_iou'].append(train_iou)
        history['val_iou'].append(val_iou)
        history['train_acc'].append(train_acc)
        history['val_acc'].append(val_acc)

        if val_iou > best_iou:
            best_iou = val_iou
            save_model(model, args.model_name, epoch, val_iou, args.output_dir)

        scheduler.step(val_iou)

        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Train IoU: {train_iou:.3f}, Val IoU: {val_iou:.3f} | "
              f"Time: {elapsed:.1f}s")

        log_to_tensorboard(writer, history, epoch, val_loader, model, device)

    writer.close()
    plot_training_history({args.model_name: history}, args.output_dir)
    print_final_results({args.model_name: history})
    print(f"🏆 Training complete! Best IoU: {best_iou:.3f}")


if __name__ == "__main__":
    main()
