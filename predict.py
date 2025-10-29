import os
import torch
import argparse
import numpy as np
from tqdm import tqdm
import rasterio
from model import get_model


def make_divisible_by_32(img: np.ndarray) -> tuple:
    h, w = img.shape[:2]
    new_h = ((h - 1) // 32 + 1) * 32
    new_w = ((w - 1) // 32 + 1) * 32
    pad_h, pad_w = new_h - h, new_w - w
    img_padded = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant')
    return img_padded, (pad_h, pad_w)


def main(args):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs("predictions", exist_ok=True)

    model = get_model(args.model_name, args.encoder_name, classes=9)
    state = torch.load(args.model_path, map_location=device, weights_only=True)
    model.load_state_dict(state["state_dict"] if "state_dict" in state else state)
    model.to(device)
    model.eval()

    image_files = [f for f in os.listdir(args.image_dir) if f.lower().endswith('.tif')]
    print(f"Predicting {len(image_files)} images with {args.model_name}...")

    for name in tqdm(image_files):
        img_path = os.path.join(args.image_dir, name)
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

        out_path = os.path.join("predictions", name.replace(".tif", f"_pred_{args.model_name}.tif"))
        profile.update(dtype=rasterio.uint8, count=1, compress='deflate')
        with rasterio.open(out_path, 'w', **profile) as dst:
            dst.write(pred.astype(rasterio.uint8), 1)

    print(f"Done! Predictions saved to: predictions/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--model_name", default="unet")
    parser.add_argument("--encoder_name", default="resnet34")
    args = parser.parse_args()
    main(args)