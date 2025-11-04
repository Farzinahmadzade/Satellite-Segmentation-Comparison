import segmentation_models_pytorch as smp

def get_model(model_name: str = 'unet', encoder_name: str = 'resnet34',
              in_channels: int = 3, classes: int = 9):
    name = model_name.lower()
    models = {
        'unet': smp.Unet,
        'unet++': smp.UnetPlusPlus,
        'fpn': smp.FPN,
        'pspnet': smp.PSPNet,
        'deeplabv3': smp.DeepLabV3,
        'deeplabv3+': smp.DeepLabV3Plus,
        'linknet': smp.Linknet,
        'manet': smp.MAnet,
        'pan': smp.PAN,
        'upernet': smp.UPerNet,
        'segformer': smp.Segformer,
    }
    if name not in models:
        raise NotImplementedError(f"Model '{model_name}' not supported.")
    return models[name](encoder_name=encoder_name, in_channels=in_channels, classes=classes)