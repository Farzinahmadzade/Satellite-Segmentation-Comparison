import segmentation_models_pytorch as smp

def get_model(model_name='unet', encoder_name='resnet34', in_channels=3, classes=9):
    name = model_name.lower()

    if name == 'unet':
        model = smp.Unet(encoder_name=encoder_name, in_channels=in_channels, classes=classes)
    elif name in ['unet++', 'unetplusplus', 'unetplus']:
        model = smp.UnetPlusPlus(encoder_name=encoder_name, in_channels=in_channels, classes=classes)
    elif name == 'fpn':
        model = smp.FPN(encoder_name=encoder_name, in_channels=in_channels, classes=classes)
    elif name == 'pspnet':
        model = smp.PSPNet(encoder_name=encoder_name, in_channels=in_channels, classes=classes)
    elif name == 'deeplabv3':
        model = smp.DeepLabV3(encoder_name=encoder_name, in_channels=in_channels, classes=classes)
    elif name in ['deeplabv3+', 'deeplabv3plus']:
        model = smp.DeepLabV3Plus(encoder_name=encoder_name, in_channels=in_channels, classes=classes)
    elif name == 'linknet':
        model = smp.Linknet(encoder_name=encoder_name, in_channels=in_channels, classes=classes)
    elif name == 'manet':
        model = smp.MAnet(encoder_name=encoder_name, in_channels=in_channels, classes=classes)
    elif name == 'pan':
        model = smp.PAN(encoder_name=encoder_name, in_channels=in_channels, classes=classes)
    elif name == 'upernet':
        model = smp.UPerNet(encoder_name=encoder_name, in_channels=in_channels, classes=classes)
    elif name == 'segformer':
        model = smp.Segformer(encoder_name=encoder_name, in_channels=in_channels, classes=classes)
    else:
        raise NotImplementedError(f"Model '{model_name}' not supported.")

    return model
