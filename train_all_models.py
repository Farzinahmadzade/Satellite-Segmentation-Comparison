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
    parser = argparse.ArgumentParser(description="Train all segmentation models on OpenEarthMap")
    parser.add_argument("--data_dir", type=str, required=True, help="Path to dataset root")
    parser.add_argument("--output_dir", type=str, default="./outputs_all_models", help="Output directory")
    parser.add_argument("--limit_samples", type=int, default=0, help="Limit training samples")
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for model_name in MODELS:
        print(f"\n{'='*60}")
        print(f" TRAINING: {model_name.upper()} ")
        print(f"{'='*60}")

        model_output_dir = os.path.join(args.output_dir, model_name)
        os.makedirs(model_output_dir, exist_ok=True)

        tb_writer = SummaryWriter(
            log_dir=os.path.join(model_output_dir, "logs"),
            name=f"{model_name}_resnet34",
            comment="OpenEarthMap Benchmark"
        )

        tb_writer.add_text(
            "Experiment Info",
            f"""
**Model**: `{model_name}`  
**Encoder**: `resnet34`  
**Dataset**: `OpenEarthMap`  
**Train/Val Split**: `80/20`  
**Started**: `{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}`
            """.strip(),
            global_step=0
        )

        train_model(
            data_dir=args.data_dir,
            model_name=model_name,
            encoder_name="resnet34",
            output_dir=model_output_dir,
            limit_samples=args.limit_samples,
            epochs=args.epochs,
            writer=tb_writer
        )

        tb_writer.close()


if __name__ == "__main__":
    main()