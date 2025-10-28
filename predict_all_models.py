import os
import glob
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from tqdm import tqdm
from model import get_model

MODEL_NAMES = [
    "unet", "unet++", "fpn", "pspnet", "deeplabv3",
    "deeplabv3+", "linknet", "manet", "pan", "upernet", "segformer"]

def make_divisible_by_32(img):
    """Pad image to make height and width divisible by 32"""
    h, w = img.shape[:2]
    new_h = ((h - 1) // 32 + 1) * 32
    new_w = ((w - 1) // 32 + 1) * 32
    pad_h = new_h - h
    pad_w = new_w - w
    img_padded = np.pad(img, ((0, pad_h), (0, pad_w), (0, 0)), mode='constant', constant_values=0)
    return img_padded

def predict(model_name, model_dir, image_dir, output_dir, device='cuda'):
    model_path = os.path.join(model_dir, model_name, "best_model.pth")
    if not os.path.exists(model_path):
        print(f"Model weights not found: {model_path}")
        return

    model = get_model(model_name, encoder_name="resnet34", in_channels=3, classes=9)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()

    os.makedirs(os.path.join(output_dir, model_name), exist_ok=True)
    image_paths = sorted(glob.glob(os.path.join(image_dir, "*.tif")))

    for img_path in tqdm(image_paths, desc=f"Predicting {model_name}"):
        img = np.array(Image.open(img_path).convert("RGB"))
        img_resized = make_divisible_by_32(img)
        img_tensor = torch.tensor(img_resized).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        img_tensor = img_tensor.to(device)

        with torch.no_grad():
            output = model(img_tensor)
            pred = torch.argmax(output.squeeze(0), dim=0).cpu().numpy()
        pred_cropped = pred[:img.shape[0], :img.shape[1]]

        pred_img = Image.fromarray(pred_cropped.astype(np.uint8))
        basename = os.path.basename(img_path)
        pred_img.save(os.path.join(output_dir, model_name, basename.replace(".tif", "_pred.png")))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, required=True, help="Path to folder containing images for prediction")
    parser.add_argument("--pred_dir", type=str, default="outputs/predictions_all_models", help="Path to save predictions")
    parser.add_argument("--model_dir", type=str, default="outputs_all_models", help="Path to saved model weights")
    args = parser.parse_args()

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    for model_name in MODEL_NAMES:
        print(f"\n=== Predicting with {model_name} ===")
        predict(model_name, args.model_dir, args.image_dir, args.pred_dir, device=device)