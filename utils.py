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
    def iou(outputs, masks, num_classes=Config.NUM_CLASSES):
        preds = outputs.argmax(1)
        ious = []
        for cls in range(num_classes):
            pred_cls = (preds == cls)
            mask_cls = (masks == cls)
            intersection = (pred_cls & mask_cls).sum().float()
            union = (pred_cls | mask_cls).sum().float()
            if union == 0:
                ious.append(torch.tensor(1.0, device=outputs.device))
            else:
                ious.append(intersection / union)
        return torch.stack(ious).mean()

    @staticmethod
    def accuracy(outputs, masks):
        preds = outputs.argmax(1)
        return (preds == masks).float().mean()


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
    print(f"Model saved: {filename}")


def log_to_tensorboard(writer, history, epoch, loader, model, device):
    writer.add_scalar('Loss/Train', history['train_loss'][-1], epoch)
    writer.add_scalar('Loss/Validation', history['val_loss'][-1], epoch)
    writer.add_scalar('IoU/Train', history['train_iou'][-1], epoch)
    writer.add_scalar('IoU/Validation', history['val_iou'][-1], epoch)
    writer.add_scalar('Accuracy/Train', history['train_acc'][-1], epoch)
    writer.add_scalar('Accuracy/Validation', history['val_acc'][-1], epoch)

    model.eval()
    with torch.no_grad():
        try:
            batch = next(iter(loader))
            if batch[0].size(0) == 0:
                return

            images, masks_true, _ = batch
            actual_batch = min(4, images.size(0))
            images = images[:actual_batch].to(device)
            outputs = model(images)
            masks_pred = outputs.argmax(1)

            mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(3,1,1)
            std  = torch.tensor([0.229, 0.224, 0.225], device=device).view(3,1,1)
            images_vis = (images * std + mean).clamp(0,1).cpu()

            colors = torch.tensor([
                [0.00, 0.00, 0.00], [0.50, 0.50, 0.00], [0.80, 0.80, 0.80],
                [0.00, 0.00, 1.00], [0.00, 1.00, 0.00], [0.00, 0.00, 0.80],
                [1.00, 1.00, 0.00], [1.00, 0.00, 0.00], [0.80, 0.00, 0.80]
            ], device=device).float().view(Config.NUM_CLASSES,3,1,1)

            masks_rgb = torch.zeros(actual_batch,3,256,256)
            preds_rgb = torch.zeros(actual_batch,3,256,256)

            masks_cpu = masks_true[:actual_batch].cpu()
            preds_cpu = masks_pred[:actual_batch].cpu()

            for i in range(actual_batch):
                for cls in range(Config.NUM_CLASSES):
                    mask_cls = (masks_cpu[i]==cls).float().unsqueeze(0)
                    pred_cls = (preds_cpu[i]==cls).float().unsqueeze(0)
                    masks_rgb[i] += colors[cls].cpu() * mask_cls
                    preds_rgb[i] += colors[cls].cpu() * pred_cls

            grid_images = [torch.cat([images_vis[i:i+1], masks_rgb[i:i+1], preds_rgb[i:i+1]], dim=2) 
                          for i in range(actual_batch)]
            final_grid = torch.cat(grid_images, dim=0)

            writer.add_image('Input | Label | Predict',
                             vutils.make_grid(final_grid, nrow=3, normalize=False, padding=5), epoch)
        except Exception as e:
            print(f"TensorBoard skipped: {e}")
    model.train()


def plot_training_history(history_dict, output_path):
    fig, axes = plt.subplots(2,2,figsize=(12,10))
    metrics = [('train_loss','val_loss','Loss'),
               ('train_iou','val_iou','IoU'),
               ('train_acc','val_acc','Accuracy')]

    for idx,(train_key,val_key,title) in enumerate(metrics):
        ax = axes[idx//2, idx%2]
        for name, history in history_dict.items():
            epochs = range(1,len(history[train_key])+1)
            ax.plot(epochs, history[train_key], label=f'{name}_train', linestyle='-')
            ax.plot(epochs, history[val_key], label=f'{name}_val', linestyle='--')
        ax.set_title(title)
        ax.set_xlabel('Epoch')
        ax.set_ylabel(title)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = Path(output_path) / 'training_history.png'
    plt.savefig(plot_path,dpi=300,bbox_inches='tight')
    plt.close()
    print(f"Plot saved: {plot_path}")
    return str(plot_path)


def print_final_results(history_dict):
    df_data = []
    for name, history in history_dict.items():
        best_iou = max(history['val_iou'])
        best_epoch = history['val_iou'].index(best_iou)+1
        final_iou = history['val_iou'][-1]
        final_acc = history['val_acc'][-1]
        df_data.append({
            'Model':'SMP',
            'Best IoU':f"{best_iou:.3f}",
            'Best Epoch':best_epoch,
            'Final IoU':f"{final_iou:.3f}",
            'Final Acc':f"{final_acc:.3f}"
        })
    df = pd.DataFrame(df_data)
    print("\n"+"="*60)
    print("MAX RESULTS")
    print("="*60)
    print(df.to_string(index=False))
    print("="*60)