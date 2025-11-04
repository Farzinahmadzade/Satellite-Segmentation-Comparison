import os
import shutil

MODEL_NAMES = [
    "unet", "unet++", "fpn", "pspnet", "deeplabv3",
    "deeplabv3+", "linknet", "manet", "pan", "upernet", "segformer"
]

TB_LOG_DIR = "outputs/tb_final_comparison"
os.makedirs(TB_LOG_DIR, exist_ok=True)

print("Merging all model logs into one TensorBoard...\n")

copied_count = 0
for model_name in MODEL_NAMES:
    src_dir = f"outputs_all_models/{model_name}/logs"
    if not os.path.exists(src_dir):
        print(f"  [Skip] No logs found for: {model_name}")
        continue

    model_run_dir = os.path.join(TB_LOG_DIR, model_name)
    os.makedirs(model_run_dir, exist_ok=True)

    event_files = [f for f in os.listdir(src_dir) if f.startswith("events.out.tfevents")]
    if not event_files:
        print(f"  [Skip] No event files in {model_name}")
        continue

    for file in event_files:
        src_file = os.path.join(src_dir, file)
        dst_file = os.path.join(model_run_dir, file)
        shutil.copy(src_file, dst_file)
        copied_count += 1

    print(f"  [Success] Copied logs for: {model_name}")

print(f"\nDone! {copied_count} event files copied.")
print(f"Run:\n  tensorboard --logdir {TB_LOG_DIR}")