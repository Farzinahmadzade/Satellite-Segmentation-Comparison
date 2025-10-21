import torch
import numpy as np
import matplotlib.pyplot as plt
import cv2
from sklearn.metrics import jaccard_score, accuracy_score, precision_score, recall_score, f1_score
import pandas as pd

class Visualizer:
    COLORS = np.array([
        [0, 0, 0],       # Background
        [0, 0, 255],     # Class1
        [255, 255, 0],   # Class2
        [0, 255, 0],     # Class3
        [255, 0, 0],     # Class4
    ], dtype=np.uint8)
    
    CLASS_NAMES = ['Background', 'Class1', 'Class2', 'Class3', 'Class4']
    
    @classmethod
    def mask_to_rgb(cls, mask):
        """Convert numeric mask to RGB color mask."""
        if isinstance(mask, torch.Tensor):
            mask = mask.cpu().numpy()
        
        h, w = mask.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        
        mask = np.clip(mask, 0, len(cls.COLORS)-1)
        
        for i, color in enumerate(cls.COLORS):
            rgb[mask == i] = color
            
        return rgb
    
    @classmethod
    def create_comparison_grid(cls, images, true_masks, pred_masks, filenames=None, num_samples=3):
        """Create side-by-side comparison grid."""
        fig, axes = plt.subplots(num_samples, 4, figsize=(20, 5 * num_samples))
        
        if num_samples == 1:
            axes = axes.reshape(1, -1)
        
        titles = ['Input Image', 'Ground Truth', 'Prediction', 'Overlay']
        
        for i in range(num_samples):
            img = images[i]
            if isinstance(img, torch.Tensor):
                img = img.cpu().numpy()
            if img.shape[0] == 3:
                img = img.transpose(1, 2, 0)
            
            mean = np.array([0.485, 0.456, 0.406])
            std = np.array([0.229, 0.224, 0.225])
            img = img * std + mean
            img = np.clip(img, 0, 1)
            
            axes[i, 0].imshow(img)
            axes[i, 0].set_title(titles[0], fontweight='bold', fontsize=12)
            axes[i, 0].axis('off')
            
            true_rgb = cls.mask_to_rgb(true_masks[i])
            axes[i, 1].imshow(true_rgb)
            axes[i, 1].set_title(titles[1], fontweight='bold', color='green', fontsize=12)
            axes[i, 1].axis('off')
            
            pred_rgb = cls.mask_to_rgb(pred_masks[i])
            axes[i, 2].imshow(pred_rgb)
            axes[i, 2].set_title(titles[2], fontweight='bold', color='blue', fontsize=12)
            axes[i, 2].axis('off')
            
            overlay = cv2.addWeighted((img * 255).astype(np.uint8), 0.7, pred_rgb, 0.3, 0)
            axes[i, 3].imshow(overlay)
            axes[i, 3].set_title(titles[3], fontweight='bold', color='red', fontsize=12)
            axes[i, 3].axis('off')
            
            if filenames and i < len(filenames):
                fig.text(0.1, 0.95 - i*0.2, f'File: {filenames[i]}', 
                        fontsize=10, ha='left', va='top', 
                        bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow"))
        
        plt.tight_layout()
        return fig

class MetricsCalculator:
    """Calculate segmentation metrics."""
    
    @staticmethod
    def calculate_metrics(preds, targets, num_classes=5):
        preds_flat = preds.cpu().numpy().flatten()
        targets_flat = targets.cpu().numpy().flatten()
        
        metrics = {}
        iou_scores = []
        for class_id in range(num_classes):
            iou = jaccard_score(targets_flat == class_id, preds_flat == class_id, zero_division=0)
            iou_scores.append(iou)
            metrics[f'IoU_Class_{class_id}'] = iou
        
        metrics['mIoU'] = np.mean(iou_scores)
        metrics['Accuracy'] = accuracy_score(targets_flat, preds_flat)
        metrics['Precision'] = precision_score(targets_flat, preds_flat, average='macro', zero_division=0)
        metrics['Recall'] = recall_score(targets_flat, preds_flat, average='macro', zero_division=0)
        metrics['F1-Score'] = f1_score(targets_flat, preds_flat, average='macro', zero_division=0)
        
        return metrics
    
    @staticmethod
    def create_metrics_dataframe(metrics_history):
        return pd.DataFrame(metrics_history)

def save_checkpoint(model, optimizer, epoch, metrics, filepath):
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'metrics': metrics
    }
    torch.save(checkpoint, filepath)

def load_checkpoint(model, optimizer, filepath):
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    return checkpoint['epoch'], checkpoint['metrics']