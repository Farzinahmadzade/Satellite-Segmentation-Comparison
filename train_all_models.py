import os
import sys
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter

PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.append(PROJECT_DIR)

from train import train_model

MODELS = [
    "unet", "unet++", "fpn", "pspnet", "deeplabv3",
    "deeplabv3+", "linknet", "manet", "pan", "upernet", "segformer"
]


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Train all segmentation models")
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./outputs_all_models")
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--limit_samples", type=int, default=0)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for model_name in MODELS:
        print(f"\n{'='*60}")
        print(f" TRAINING: {model_name.upper()} ")
        print(f"{'='*60}")

        model_dir = os.path.join(args.output_dir, model_name)
        log_dir = os.path.join(model_dir, "logs")
        os.makedirs(log_dir, exist_ok=True)

        writer = SummaryWriter(log_dir=log_dir)

        writer.add_text(
            "Experiment Info",
            f"Model: {model_name}, Epochs: {args.epochs}, Started: {datetime.now():%Y-%m-%d %H:%M:%S}"
        )

        # --- Train model ---
        train_model(
            data_dir=args.data_dir,
            model_name=model_name,
            encoder_name="resnet34",
            output_dir=model_dir,
            limit_samples=args.limit_samples,
            epochs=args.epochs,
            writer=writer
        )

        # --- Flush and close writer to ensure logs are written ---
        writer.flush()
        writer.close()
        print(f"  [Success] TensorBoard logs saved for {model_name}")

    print(f"\nAll models trained! Logs are in: {args.output_dir}")


if __name__ == "__main__":
    main()