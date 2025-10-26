import os
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from model import get_model
from config import Config
from tqdm import tqdm

CLASS_NAMES = ['bareland','rangeland','development','road','tree','water','agricultural','building','nodata']
CLASS_COLORS = [
    (210,180,140), (34,139,34), (220,20,60), (128,128,128),
    (0,100,0), (30,144,255), (255,215,0), (178,34,34), (0,0,0)]

def load_image(img_path, image_size):
    img = np.array(Image.open(img_path))
    if img.ndim == 2:
        img = np.stack([img]*3, axis=-1)
    img = Image.fromarray(img)
    if isinstance(image_size, int):
        img = img.resize((image_size, image_size), resample=Image.BILINEAR)
    elif isinstance(image_size, tuple):
        img = img.resize(image_size, resample=Image.BILINEAR)
    img_tensor = torch.tensor(np.array(img)/255.0, dtype=torch.float32).permute(2,0,1)
    return img_tensor.unsqueeze(0)

def load_mask(mask_path, image_size):
    mask = Image.open(mask_path)
    if isinstance(image_size, int):
        mask = mask.resize((image_size, image_size), resample=Image.NEAREST)
    elif isinstance(image_size, tuple):
        mask = mask.resize(image_size, resample=Image.NEAREST)
    return torch.from_numpy(np.array(mask)).long()

def colorize_mask(mask):
    h, w = mask.shape
    color_mask = np.zeros((h, w, 3), dtype=np.uint8)
    for idx, color in enumerate(CLASS_COLORS):
        color_mask[mask==idx] = color
    return Image.fromarray(color_mask)

def visualize_prediction(img_tensor, mask_tensor, pred_tensor, output_path):
    img = (img_tensor.permute(1,2,0).numpy()*255).astype(np.uint8)
    img_pil = Image.fromarray(img)

    mask_color = colorize_mask(mask_tensor.numpy())
    pred_color = colorize_mask(pred_tensor.numpy())

    total_width = img_pil.width*3
    total_height = img_pil.height
    new_img = Image.new('RGB', (total_width, total_height))
    new_img.paste(img_pil, (0,0))
    new_img.paste(mask_color, (img_pil.width,0))
    new_img.paste(pred_color, (img_pil.width*2,0))

    draw = ImageDraw.Draw(new_img)
    font = None
    try:
        font = ImageFont.truetype("arial.ttf", 16)
    except:
        font = ImageFont.load_default()
    for i, (name, color) in enumerate(zip(CLASS_NAMES, CLASS_COLORS)):
        y = i*20
        draw.rectangle([10, y+10, 30, y+30], fill=color)
        draw.text((35, y+10), name, fill=(255,255,255), font=font)

    new_img.save(output_path)

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_dir", type=str, required=True)
    parser.add_argument("--mask_dir", type=str, default=None)
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--model_name", type=str, default="unet")
    parser.add_argument("--encoder_name", type=str, default="resnet34")
    parser.add_argument("--output_dir", type=str, default="./predictions")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🧠 Using device: {device}")

    model = get_model(args.model_name, args.encoder_name, classes=Config.NUM_CLASSES)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device)
    model.eval()
    print(f"✅ Loaded model from {args.model_path}")

    img_files = sorted([os.path.join(args.image_dir,f) for f in os.listdir(args.image_dir) if f.lower().endswith('.tif')])
    mask_files = None
    if args.mask_dir:
        mask_files = sorted([os.path.join(args.mask_dir,f) for f in os.listdir(args.mask_dir) if f.lower().endswith('.tif')])

    for i, img_path in enumerate(tqdm(img_files, desc="Predicting")):
        try:
            img_tensor = load_image(img_path, Config.IMAGE_SIZE).to(device)
            with torch.no_grad():
                output = model(img_tensor)
                pred = torch.argmax(output, dim=1).squeeze(0).cpu()

            if mask_files:
                mask_tensor = load_mask(mask_files[i], Config.IMAGE_SIZE)
            else:
                mask_tensor = torch.zeros_like(pred)

            out_path = os.path.join(args.output_dir, f"{i}_pred.png")
            visualize_prediction(img_tensor.squeeze(0).cpu(), mask_tensor, pred, out_path)
        except Exception as e:
            print(f"⚠️ Skipping {img_path} due to error: {e}")

    print(f"\n✅ Predictions saved to {args.output_dir}")

if __name__ == "__main__":
    main()
