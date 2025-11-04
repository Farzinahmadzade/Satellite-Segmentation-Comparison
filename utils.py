import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

class Metrics:
    @staticmethod
    def compute_batch_metrics(outputs: torch.Tensor, masks: torch.Tensor, num_classes: int = 9):
        preds = torch.argmax(outputs, dim=1)
        device = preds.device
        preds_flat = preds.flatten()
        masks_flat = masks.flatten()

        tp = torch.zeros(num_classes, device=device)
        fp = torch.zeros(num_classes, device=device)
        fn = torch.zeros(num_classes, device=device)

        for c in range(num_classes):
            pred_c = (preds_flat == c)
            mask_c = (masks_flat == c)
            tp[c] = (pred_c & mask_c).sum()
            fp[c] = (pred_c & ~mask_c).sum()
            fn[c] = (~pred_c & mask_c).sum()

        eps = 1e-6
        precision = (tp / (tp + fp + eps)).mean().item()
        recall = (tp / (tp + fn + eps)).mean().item()
        f1 = (2 * precision * recall / (precision + recall + eps))
        iou = (tp / (tp + fp + fn + eps)).mean().item()
        acc = ((preds_flat == masks_flat).sum() / masks_flat.numel()).item()

        return {
            "loss": F.cross_entropy(outputs, masks, reduction='mean').item(),
            "accuracy": acc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "iou": iou}

    @staticmethod
    def pred_to_rgb(pred: np.ndarray, num_classes: int = 9) -> np.ndarray:
        cmap = plt.get_cmap('tab20', num_classes)
        normalized = pred.astype(np.float32) / (num_classes - 1)
        colored = cmap(normalized)[..., :3]
        return (colored * 255).astype(np.uint8)