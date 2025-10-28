import os
import torch
import argparse
import numpy as np
from tqdm import tqdm
from PIL import Image
import matplotlib.pyplot as plt
from config import Config
from model import get_model
from torch.utils.tensorboard import SummaryWriter
from torchvision.utils import make_grid


def load_image(path, image_size):
    img = Image.open(path).convert("RGB")
    img = img.resize(image_size)
    img = np.array(img) / 255.0
    img = np.transpose(img, (2, 0, 1))
    return torch.tensor(img, dtype=torch.float32).unsqueeze(0)


def load_mask(path, image_size):
    mask = Image.open(path).convert("L").resize(image_size, Image.NEAREST)
    return np.array(mask, dtype=np.uint8)


def pred_to_rgb(pred):

    colormap = plt.get_cmap('tab20', Config.NUM_CLASSES)
    normalized = pred.astype(np.float32) / max(1, (Config.NUM_CLASSES - 1))
    colored = colormap(normalized)[:, :, :3]
    colored = (colored * 255).astype(np.uint8)
    return colored

def save_composite(image, label, pred_rgb, path):

    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image)
    axes[0].set_title("Input")
    axes[1].imshow(label, cmap='tab20', vmin=0, vmax=Config.NUM_CLASSES-1)
    axes[1].set_title("Label")
    axes[2].imshow(pred_rgb)
    axes[2].set_title("Prediction")
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    fig.savefig(path)
    plt.close(fig)

def add_samples_to_tensorboard(writer, images_list, masks_list, preds_list, step):

    imgs_to_show = []
    for img, mask, pred in zip(images_list, masks_list, preds_list):
        img_tensor = torch.tensor(img).permute(2, 0, 1).float() / 255.0
        label_rgb = pred_to_rgb(mask)
        label_tensor = torch.tensor(label_rgb).permute(2, 0, 1).float() / 255.0
        pred_rgb = pred_to_rgb(pred)
        pred_tensor = torch.tensor(pred_rgb).permute(2, 0, 1).float() / 255.0
        combined = torch.cat([img_tensor, label_tensor, pred_tensor], dim=2)
        imgs_to_show.append(combined)

    n = len(imgs_to_show)
    nrow = min(n, 5)
    grid = make_grid(imgs_to_show, nrow=nrow, padding=4)
    writer.add_image("Predictions_Grid", grid, global_step=step)

def robust_load_state_dict(model, path, device):

    loaded = torch.load(path, map_location=device)
    if isinstance(loaded, dict) and "state_dict" in loaded and isinstance(loaded["state_dict"], dict):
        state_dict = loaded["state_dict"]
    else:
        state_dict = loaded
    model.load_state_dict(state_dict)

def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    os.makedirs(args.output_dir, exist_ok=True)

    tb_logdir = os.path.join("outputs", "logs")
    os.makedirs(tb_logdir, exist_ok=True)
    writer = SummaryWriter(log_dir=tb_logdir)

    image_files = sorted([f for f in os.listdir(args.image_dir) if f.lower().endswith(('.png', '.jpg', '.tif'))])
    print(f"Found {len(image_files)} images for prediction")

    model = get_model(args.model_name, args.encoder_name, classes=Config.NUM_CLASSES)
    robust_load_state_dict(model, args.model_path, device)
    model.to(device)
    model.eval()
    print(f"Loaded model from {args.model_path}")

    sample_imgs, sample_labels, sample_preds = [], [], []

    for idx, name in enumerate(tqdm(image_files, desc="Predicting")):
        img_path = os.path.join(args.image_dir, name)
        mask_path = os.path.join(args.mask_dir, name) if args.mask_dir else None

        try:
            img_tensor = load_image(img_path, Config.IMAGE_SIZE).to(device)
            with torch.no_grad():
                output = model(img_tensor)
                probs = torch.softmax(output, dim=1)
                pred = torch.argmax(probs, dim=1).cpu().squeeze().numpy()

            img_disp = np.array(Image.open(img_path).convert("RGB").resize(Config.IMAGE_SIZE))
            label_disp = load_mask(mask_path, Config.IMAGE_SIZE) if mask_path and os.path.exists(mask_path) else np.zeros_like(pred)

            pred_rgb = pred_to_rgb(pred)
            out_path = os.path.join(args.output_dir, f"{os.path.splitext(name)[0]}_comparison.png")
            save_composite(img_disp, label_disp, pred_rgb, out_path)

            if len(sample_imgs) < 10:
                sample_imgs.append(img_disp)
                sample_labels.append(label_disp)
                sample_preds.append(pred)

        except Exception as e:
            print(f"Skipping {name}: {e}")

    if len(sample_imgs) > 0:
        add_samples_to_tensorboard(writer, sample_imgs, sample_labels, sample_preds, step=0)

    writer.close()
    print(f"Predictions saved to {args.output_dir} and images logged to {tb_logdir}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--mask_dir", default=None)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--model_name", default="unet")
    parser.add_argument("--encoder_name", default="resnet34")
    parser.add_argument("--output_dir", required=True)
    args = parser.parse_args()
    main(args)