import os
import glob
import torch
import numpy as np
import rasterio
from tqdm import tqdm
from model import get_model
from torchvision import transforms

MODEL_NAMES = [
    "unet", "unet++", "fpn", "pspnet", "deeplabv3",
    "deeplabv3+", "linknet", "manet", "pan", "upernet", "segformer"
    ]

NORMALIZE = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

def make_divisible_by_32(img: np.ndarray):
    """Pad image to be divisible by 32 (required by many encoders)."""
    h, w = img.shape[:2]
    new_h = ((h - 1) // 32 + 1) * 32
    new_w = ((w - 1) // 32 + 1) * 32
    pad_h, pad_w = new_h - h, new_w - w
    return np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant'), (pad_h, pad_w)


def predict(model_name, model_dir, image_dir, output_dir, device):
    model_path = os.path.join(model_dir, model_name, "best_model.pth")
    if not os.path.exists(model_path):
        return

    model = get_model(model_name, "resnet34", classes=9)
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.to(device)
    model.eval()

    pred_dir = os.path.join(output_dir, model_name)
    os.makedirs(pred_dir, exist_ok=True)
    images = sorted(glob.glob(os.path.join(image_dir, "*.tif")))

    for img_path in tqdm(images, desc=model_name):
        with rasterio.open(img_path) as src:
            img = np.moveaxis(src.read(), 0, -1)
            profile = src.profile.copy()
            h, w = src.height, src.width

        img_padded, (pad_h, pad_w) = make_divisible_by_32(img)
        tensor = torch.from_numpy(img_padded).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        tensor = NORMALIZE(tensor).to(device)

        with torch.no_grad():
            pred = torch.argmax(model(tensor).squeeze(0), dim=0).cpu().numpy()

        pred = pred[:h, :w]
        out_path = os.path.join(pred_dir,
                                os.path.basename(img_path).replace(".tif", f"_pred_{model_name}.tif"))
        profile.update(dtype=rasterio.uint8, count=1, compress='deflate')
        with rasterio.open(out_path, 'w', **profile) as dst:
            dst.write(pred.astype(rasterio.uint8), 1)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--pred_dir", default="predictions_all_models")
    parser.add_argument("--model_dir", default="outputs_all_models")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    os.makedirs(args.pred_dir, exist_ok=True)

    for m in MODEL_NAMES:
        print(f"\nPREDICTING: {m.upper()}")
        predict(m, args.model_dir, args.image_dir, args.pred_dir, device)