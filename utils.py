import torch
import matplotlib.pyplot as plt


class Metrics:
    @staticmethod
    def compute_batch_metrics(outputs: torch.Tensor, masks: torch.Tensor, num_classes: int):
        """Compute IoU, Dice, and Accuracy for a batch."""
        preds = torch.argmax(outputs, dim=1)
        device = preds.device

        intersection = torch.zeros(num_classes, device=device)
        union = torch.zeros(num_classes, device=device)
        correct = (preds == masks).sum().item()
        total = masks.numel()

        for c in range(num_classes):
            pred_c = (preds == c)
            mask_c = (masks == c)
            intersection[c] = (pred_c & mask_c).sum().float()
            union[c] = (pred_c | mask_c).sum().float()

        iou = (intersection / (union + 1e-6)).mean().item()
        dice = (2 * intersection / (intersection + union + 1e-6)).mean().item()
        acc = correct / total
        return {"iou": iou, "dice": dice, "acc": acc}

    @staticmethod
    def visualize_sample(image, mask, output, idx, writer, epoch):
        """Visualize input, mask, and prediction in TensorBoard."""
        image = image.detach().cpu()
        mask = mask.squeeze(0).detach().cpu() if mask.dim() == 4 else mask.detach().cpu()
        out = output.squeeze(0).detach().cpu() if output.dim() == 4 else output.detach().cpu()
        pred = torch.argmax(out, dim=0)

        fig, axes = plt.subplots(1, 3, figsize=(9, 3))
        axes[0].imshow(image.permute(1, 2, 0))
        axes[0].set_title("Image")
        axes[1].imshow(mask, cmap='tab20', vmin=0, vmax=8)
        axes[1].set_title("Mask")
        axes[2].imshow(pred, cmap='tab20', vmin=0, vmax=8)
        axes[2].set_title("Prediction")
        for ax in axes:
            ax.axis("off")
        writer.add_figure(f"Sample_{idx}", fig, epoch)
        plt.close(fig)