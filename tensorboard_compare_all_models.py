import os
import glob
import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

MODEL_NAMES = [
    "unet", "unet++", "fpn", "pspnet", "deeplabv3",
    "deeplabv3+", "linknet", "manet", "pan", "upernet", "segformer"
]

PRED_DIR = "outputs/predictions_all_models"
IMG_DIR = "K:/UNI/MAHBOD/OpenEarthMap/OpenEarthMap_wo_xBD/aachen/images"
GT_DIR = "K:/UNI/MAHBOD/OpenEarthMap/OpenEarthMap_wo_xBD/aachen/labels"

TB_LOG_DIR = "outputs/tb_comparison_full"
os.makedirs(TB_LOG_DIR, exist_ok=True)
writer = SummaryWriter(TB_LOG_DIR)

NUM_CLASSES = 9
NUM_SAMPLES = 10

def pred_to_rgb(pred):
    colormap = plt.get_cmap('tab20', NUM_CLASSES)
    normalized = pred.astype(np.float32) / max(1, NUM_CLASSES - 1)
    colored = colormap(normalized)[..., :3]
    return (colored * 255).astype(np.uint8)

def combine_images(img, gt, pred):
    h, w = 256, 256
    img = Image.fromarray(img).resize((w, h))
    gt = Image.fromarray(gt).resize((w, h), Image.NEAREST)
    pred = Image.fromarray(pred).resize((w, h), Image.NEAREST)

    img_np = np.array(img)
    gt_rgb = pred_to_rgb(np.array(gt))
    pred_rgb = pred_to_rgb(np.array(pred))

    combined = np.hstack([img_np, gt_rgb, pred_rgb])
    return combined

def add_comparison_images():
    print(f"Adding comparison images (Input | GT | Pred) to TensorBoard...")
    total_added = 0

    for model_name in MODEL_NAMES:
        model_pred_dir = os.path.join(PRED_DIR, model_name)
        if not os.path.exists(model_pred_dir):
            print(f"  [Warning] No predictions for {model_name}")
            continue

        pred_paths = sorted(glob.glob(os.path.join(model_pred_dir, "*_pred.png")))
        print(f"  [Info] {model_name}: {len(pred_paths)} predictions found")

        added_for_model = 0
        for pred_path in pred_paths[:NUM_SAMPLES]:
            try:
                base_name = os.path.basename(pred_path).replace("_pred.png", "")
                img_path = os.path.join(IMG_DIR, base_name + ".tif")
                gt_path = os.path.join(GT_DIR, base_name + ".png")

                if not os.path.exists(img_path):
                    img_path = os.path.join(IMG_DIR, base_name + ".png")
                if not os.path.exists(img_path):
                    continue

                if not os.path.exists(gt_path):
                    gt_path = os.path.join(GT_DIR, base_name + ".tif")
                if not os.path.exists(gt_path):
                    continue

                img = np.array(Image.open(img_path).convert("RGB"))
                gt = np.array(Image.open(gt_path).convert("L"))
                pred = np.array(Image.open(pred_path).convert("L"))

                combined = combine_images(img, gt, pred)
                combined_tensor = torch.tensor(combined).permute(2, 0, 1).float() / 255.0

                tag = f"Comparison/{model_name}_{added_for_model}"
                writer.add_image(tag, combined_tensor, global_step=0)
                added_for_model += 1
                total_added += 1

            except Exception as e:
                print(f"  [Error] Failed to process {pred_path}: {e}")

        print(f"  [Success] Added {added_for_model} samples for {model_name}")

    print(f"Total comparison images added: {total_added}")

def merge_training_logs():
    print("Merging training logs (Loss, IoU)...")
    merged = 0
    for model_name in MODEL_NAMES:
        log_dir = os.path.join("outputs", model_name, "logs")
        if not os.path.exists(log_dir):
            continue

        for event_file in glob.glob(os.path.join(log_dir, "events.out.tfevents.*")):
            try:
                ea = EventAccumulator(event_file)
                ea.Reload()
                for tag in ea.Tags().get("scalars", []):
                    for scalar in ea.Scalars(tag):
                        writer.add_scalar(f"{model_name}/{tag}", scalar.value, scalar.step)
                merged += 1
            except Exception as e:
                print(f"  [Error] Failed to read {event_file}: {e}")
    print(f"Merged {merged} event files.")

if __name__ == "__main__":
    print("Starting TensorBoard comparison pipeline...\n")
    add_comparison_images()
    print()
    merge_training_logs()
    writer.close()
    print(f"\nDone!")
    print(f"Run TensorBoard:")
    print(f"    tensorboard --logdir {TB_LOG_DIR}")
    print(f"    Then go to: http://localhost:6006")
    print(f"    Check 'Images' tab under 'Comparison/'")