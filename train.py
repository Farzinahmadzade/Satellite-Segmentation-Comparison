import argparse
import torch
import torch.nn as nn
import time
from tqdm import tqdm
import numpy as np
from data_loader import get_loaders, Config
from model import get_model
from utils import Metrics, save_model, plot_training_history, print_final_results

def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    total_loss = 0
    ious, accs = [], []
    
    pbar = tqdm(loader, desc="Train")
    for images, masks, _ in pbar:
        images, masks = images.to(device), masks.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, masks)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        iou = Metrics.iou(outputs, masks)
        acc = Metrics.accuracy(outputs, masks)
        ious.append(iou)
        accs.append(acc)
        pbar.set_postfix({'loss': f'{loss.item():.3f}', 'iou': f'{iou:.3f}'})
    
    return total_loss/len(loader), np.mean(ious), np.mean(accs)

def val_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    ious, accs = [], []
    
    with torch.no_grad():
        pbar = tqdm(loader, desc="Val")
        for images, masks, _ in pbar:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            
            total_loss += loss.item()
            iou = Metrics.iou(outputs, masks)
            acc = Metrics.accuracy(outputs, masks)
            ious.append(iou)
            accs.append(acc)
            pbar.set_postfix({'loss': f'{loss.item():.3f}', 'iou': f'{iou:.3f}'})
    
    return total_loss/len(loader), np.mean(ious), np.mean(accs)

def main():
    parser = argparse.ArgumentParser(description="U-Net Training for Satellite Segmentation")
    parser.add_argument('--data_dir', required=True)
    parser.add_argument('--output_dir', default='csvs')
    parser.add_argument('--model_name', default='manual', choices=['manual', 'smp'])
    parser.add_argument('--encoder_name', default='resnet34')
    parser.add_argument('--epochs', type=int, default=2)
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=0.001)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Config.BATCH_SIZE = args.batch_size
    
    train_loader, val_loader, _ = get_loaders(args.data_dir, args.output_dir)
    model = get_model(args.model_name, args.encoder_name).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    history = {'train_loss': [], 'val_loss': [], 'train_iou': [], 'val_iou': [], 
               'train_acc': [], 'val_acc': []}
    best_iou = 0
    
    for epoch in range(args.epochs):
        start_time = time.time()
        train_loss, train_iou, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
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
        
        elapsed = time.time() - start_time
        print(f"Epoch {epoch+1}/{args.epochs} | "
              f"Loss: {train_loss:.4f}/{val_loss:.4f} | "
              f"IoU: {train_iou:.3f}/{val_iou:.3f} | "
              f"Time: {elapsed:.1f}s")
    
    plot_training_history({args.model_name: history}, args.output_dir)
    print_final_results({args.model_name: history})
    print(f"Training completed! Best IoU: {best_iou:.3f}")

if __name__ == "__main__":
    main()