import os
import glob
import argparse
import torch
import numpy as np
import rasterio
from tqdm import tqdm
from model import get_model
from PIL import Image


MODEL_NAMES = [
    "unet", "unet++", "fpn", "pspnet", "deeplabv3",
    "deeplabv3+", "linknet", "manet", "pan", "upernet", "segformer"
]


def make_divisible_by_32(img: np.ndarray) -> tuple:
    """Pad image to make dimensions divisible by 32."""
    h, w = img.shape[:2]
    new_h = ((h - 1) // 32 + 1) * 32
    new_w = ((w - 1) // 32 + 1) * 32
    pad_h, pad_w = new_h - h, new_w - w
    img_padded = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
    return img_padded, (pad_h, pad_w)


def predict(model_name: str, model_dir: str, image_dir: str, output_dir: str, device: str = 'cuda'):
    model_path = os.path.join(model_dir, model_name, "best_model.pth")
    if not os.path.exists(model_path):
        print(f"Warning: Model weights not found: {model_path}")
        return

    model = get_model(model_name, encoder_name="resnet34", classes=9)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    pred_dir = os.path.join(output_dir, model_name)
    os.makedirs(pred_dir, exist_ok=True)

    image_paths = sorted(glob.glob(os.path.join(image_dir, "*.tif")))
    print(f"Predicting {len(image_paths)} images with {model_name}...")

    for img_path in tqdm(image_paths, desc=f"{model_name}"):
        with rasterio.open(img_path) as src:
            img = np.moveaxis(src.read(), 0, -1)
            profile = src.profile.copy()
            h, w = src.height, src.width

        img_padded, (pad_h, pad_w) = make_divisible_by_32(img)
        tensor = torch.tensor(img_padded).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        tensor = tensor.to(device)

        with torch.no_grad():
            pred = torch.argmax(model(tensor).squeeze(0), dim=0).cpu().numpy()
        pred = pred[:h, :w]

        basename = os.path.basename(img_path)
        out_path = os.path.join(pred_dir, basename.replace(".tif", f"_pred_{model_name}.tif"))
        profile.update(dtype=rasterio.uint8, count=1, compress='deflate')
        with rasterio.open(out_path, 'w', **profile) as dst:
            dst.write(pred.astype(rasterio.uint8), 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Predict with all trained models")
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--pred_dir", type=str, default="predictions_all_models")
    parser.add_argument("--model_dir", type=str, default="outputs_all_models")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.pred_dir, exist_ok=True)

    for model_name in MODEL_NAMES:
        print(f"\n{'='*50}")
        print(f" PREDICTING: {model_name.upper()} ")
        print(f"{'='*50}")
        predict(model_name, args.model_dir, args.image_dir, args.pred_dir, device)