import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from data_loader import Config

class Metrics:
    @staticmethod
    def iou(pred, target, num_classes=Config.NUM_CLASSES):
        pred = pred.argmax(dim=1).cpu().numpy().flatten()
        target = target.cpu().numpy().flatten()
        ious = []
        for cls in range(num_classes):
            pred_cls = (pred == cls)
            target_cls = (target == cls)
            intersection = np.logical_and(pred_cls, target_cls).sum()
            union = np.logical_or(pred_cls, target_cls).sum()
            ious.append(1.0 if union == 0 else intersection / union)
        return np.mean(ious)

    @staticmethod
    def accuracy(pred, target):
        pred = pred.argmax(dim=1).cpu().numpy().flatten()
        target = target.cpu().numpy().flatten()
        return np.mean(pred == target)

def save_model(model, model_name, epoch, val_iou, output_dir):
    model_path = os.path.join(output_dir, f"{model_name}_best.pth")
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'val_iou': val_iou,
        'model_name': model_name
    }, model_path)

def load_model(model_name, output_dir, device):
    model_path = os.path.join(output_dir, f"{model_name}_best.pth")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    
    from model import get_model
    model = get_model(model_name).to(device)
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    return model

def plot_training_history(histories, output_dir):
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    metrics = ['loss', 'iou', 'acc']
    
    for i, metric in enumerate(metrics):
        ax = axes[i//2, i%2]
        for model_name, history in histories.items():
            ax.plot(history[f'train_{metric}'], label=f'{model_name} train')
            ax.plot(history[f'val_{metric}'], label=f'{model_name} val', linestyle='--')
        ax.set_title(metric.capitalize())
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'training_history.png'), dpi=300, bbox_inches='tight')
    plt.close()

def print_final_results(histories):
    df_data = []
    for model_name, history in histories.items():
        df_data.append({
            'Model': model_name.upper(),
            'Best IoU': f"{max(history['val_iou']):.3f}",
            'Final IoU': f"{history['val_iou'][-1]:.3f}",
            'Final Acc': f"{history['val_acc'][-1]:.3f}"
        })
    
    df = pd.DataFrame(df_data)
    print("\n" + "="*60)
    print("FINAL RESULTS")
    print("="*60)
    print(df.to_string(index=False))
    print("="*60)