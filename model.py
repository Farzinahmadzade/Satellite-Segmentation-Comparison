import segmentation_models_pytorch as smp

def get_model(model_name="unet", encoder_name="resnet34", num_classes=9):
    if model_name.lower() == "unet":
        model = smp.Unet(encoder_name=encoder_name, encoder_weights="imagenet", classes=num_classes, activation=None)
    elif model_name.lower() == "fpn":
        model = smp.FPN(encoder_name=encoder_name, encoder_weights="imagenet", classes=num_classes, activation=None)
    else:
        raise ValueError(f"Unknown model: {model_name}")
    return model
