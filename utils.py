import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
from torch.utils.tensorboard import SummaryWriter
import torchvision.utils as vutils
from datetime import datetime
from data_loader import Config

class Metrics:
    @staticmethod
    def accuracy(outputs, masks):
        preds = torch.argmax(outputs, dim=1)
        correct_pixels = (preds == masks).sum().item()
        total_pixels = torch.numel(masks)
        return correct_pixels / total_pixels

    @staticmethod
    def iou(outputs, masks, num_classes):
        preds = torch.argmax(outputs, dim=1)
        preds = preds.flatten()
        masks = masks.flatten()
        
        ious = []
        for c in range(num_classes):
            pred_c = (preds == c)
            mask_c = (masks == c)
            intersection = (pred_c & mask_c).sum().item()
            union = (pred_c | mask_c).sum().item()
            
            if union == 0:
                iou_c = np.nan
            else:
                iou_c = intersection / union
            ious.append(iou_c)
            
        m_iou = np.nanmean(ious)
        return m_iou, ious

def get_writer(output_dir, model_name):
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_dir = Path(output_dir) / 'logs' / f"{model_name}_{timestamp}"
    log_dir.mkdir(parents=True, exist_ok=True)
    return SummaryWriter(str(log_dir))

def save_model(model, model_name, epoch, iou, output_path):
    save_dir = Path(output_path) / 'checkpoints'
    save_dir.mkdir(exist_ok=True)
    filename = save_dir / f"{model_name}_epoch{epoch+1}_iou{iou:.4f}.pt"
    torch.save(model.state_dict(), filename)
    print(f"Model saved to {filename}")

def log_to_tensorboard(writer, history, epoch, loader, model, device):
    writer.add_scalar('Loss/Train', history['train_loss'][-1], epoch)
    writer.add_scalar('Loss/Validation', history['val_loss'][-1], epoch)
    writer.add_scalar('IoU/Train', history['train_iou'][-1], epoch)
    writer.add_scalar('IoU/Validation', history['val_iou'][-1], epoch)
    writer.add_scalar('Accuracy/Train', history['train_acc'][-1], epoch)
    writer.add_scalar('Accuracy/Validation', history['val_acc'][-1], epoch)
    
    model.eval()
    with torch.no_grad():
        images, masks_true, _ = next(iter(loader))
        images = images[:4].to(device)
        outputs = model(images)
        masks_pred = torch.argmax(outputs, dim=1)
        
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1).to(device)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1).to(device)
        images_vis = images * std + mean
        images_vis = torch.clamp(images_vis, 0, 1).cpu()
        
        colors = torch.tensor([
            [0.00, 0.00, 0.00],  # 0: bareland
            [0.50, 0.50, 0.00],  # 1: rangeland
            [0.80, 0.80, 0.80],  # 2: development
            [0.00, 0.00, 1.00],  # 3: road
            [0.00, 1.00, 0.00],  # 4: tree
            [0.00, 0.00, 0.80],  # 5: water
            [1.00, 1.00, 0.00],  # 6: agricultural
            [1.00, 0.00, 0.00],  # 7: building
            [0.80, 0.00, 0.80]   # 8: nodata
        ]).float().view(9, 3, 1, 1)
        
        masks_rgb = torch.zeros(4, 3, 256, 256)
        preds_rgb = torch.zeros(4, 3, 256, 256)
        
        masks_cpu = masks_true[:4].cpu()
        preds_cpu = masks_pred[:4].cpu()
        
        for i in range(4):
            for cls in range(Config.NUM_CLASSES):
                mask_cls = (masks_cpu[i] == cls).float().unsqueeze(0)
                pred_cls = (preds_cpu[i] == cls).float().unsqueeze(0)
                masks_rgb[i] += colors[cls] * mask_cls
                preds_rgb[i] += colors[cls] * pred_cls
        
        row1 = torch.cat([images_vis[0:1], masks_rgb[0:1], preds_rgb[0:1]], dim=2)
        row2 = torch.cat([images_vis[1:2], masks_rgb[1:2], preds_rgb[1:2]], dim=2)
        row3 = torch.cat([images_vis[2:3], masks_rgb[2:3], preds_rgb[2:3]], dim=2)
        row4 = torch.cat([images_vis[3:4], masks_rgb[3:4], preds_rgb[3:4]], dim=2)
        
        final_grid = torch.cat([row1, row2, row3, row4], dim=0)
        
        writer.add_image('Input | Label | Predict', 
                        vutils.make_grid(final_grid, nrow=3, normalize=False, padding=5), 
                        epoch)
    
    model.train()

def plot_training_history(history_dict, output_path):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    metrics = [('train_loss', 'val_loss', 'Loss'), 
               ('train_iou', 'val_iou', 'IoU'), 
               ('train_acc', 'val_acc', 'Accuracy')]
    
    for idx, (train_key, val_key, title) in enumerate(metrics):
        row, col = idx // 2, idx % 2
        ax = axes[row, col]
        
        for name, history in history_dict.items():
            ax.plot(history[train_key], label=f'{name}_train', linestyle='-')
            ax.plot(history[val_key], label=f'{name}_val', linestyle='--')
        
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(title)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = Path(output_path) / 'training_history.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.close()
    return str(plot_path)

def print_final_results(history_dict):
    df_data = []
    for name, history in history_dict.items():
        best_iou = max(history['val_iou'])
        final_iou = history['val_iou'][-1]
        final_acc = history['val_acc'][-1]
        
        df_data.append({
            'Model': name.upper(),
            'Best IoU': f"{best_iou:.3f}",
            'Final IoU': f"{final_iou:.3f}",
            'Final Acc': f"{final_acc:.3f}"
        })
    
    df = pd.DataFrame(df_data)
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(df.to_string(index=False))
    print("="*60)