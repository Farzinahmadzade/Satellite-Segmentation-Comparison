import torch
import numpy as np

def iou_score(preds, labels, num_classes=9):
    preds = torch.argmax(preds, dim=1)
    ious = []
    for cls in range(num_classes):
        pred_inds = (preds == cls)
        target_inds = (labels == cls)
        intersection = (pred_inds & target_inds).sum().item()
        union = (pred_inds | target_inds).sum().item()
        if union == 0:
            ious.append(1.0)
        else:
            ious.append(intersection / union)
    return np.mean(ious)

def dice_score(preds, labels, num_classes=9):
    preds = torch.argmax(preds, dim=1)
    dices = []
    for cls in range(num_classes):
        pred_inds = (preds == cls)
        target_inds = (labels == cls)
        intersection = (pred_inds & target_inds).sum().item()
        total = pred_inds.sum().item() + target_inds.sum().item()
        if total == 0:
            dices.append(1.0)
        else:
            dices.append(2*intersection / total)
    return np.mean(dices)
