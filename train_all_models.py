import os, sys
PROJECT_DIR = os.path.dirname(os.path.abspath(__file__))
if PROJECT_DIR not in sys.path:
    sys.path.append(PROJECT_DIR)

if __name__ == "__main__":
    import argparse
    from train import train_model
    from torch.utils.tensorboard import SummaryWriter

    MODELS = [
        "unet", "unet++", "fpn", "pspnet", "deeplabv3",
        "deeplabv3+", "linknet", "manet", "pan", "upernet", "segformer"
    ]

    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./outputs_all_models")
    parser.add_argument("--limit_samples", type=int, default=0)
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    for model_name in MODELS:
        print(f"\n\n=== Training {model_name} ===")
        model_output_dir = os.path.join(args.output_dir, model_name)
        os.makedirs(model_output_dir, exist_ok=True)

        tb_writer = SummaryWriter(os.path.join(model_output_dir, "logs"))

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
