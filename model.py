import segmentation_models_pytorch as smp
from data_loader import Config

def get_model(model_name, encoder_name="resnet34", encoder_weights="imagenet"):
    """
    Return a segmentation model from segmentation_models_pytorch.
    Supports multiple architectures.
    """

    model_name = model_name.lower()

    if model_name == "unet":
        model = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            classes=Config.NUM_CLASSES,
            activation=None,
            decoder_attention_type="scse"
        )

    elif model_name in ["unet++", "unetplusplus", "unet_plus"]:
        model = smp.UnetPlusPlus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            classes=Config.NUM_CLASSES,
            activation=None
        )

    elif model_name == "fpn":
        model = smp.FPN(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            classes=Config.NUM_CLASSES,
            activation=None
        )

    elif model_name == "pspnet":
        model = smp.PSPNet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            classes=Config.NUM_CLASSES,
            activation=None
        )

    elif model_name == "deeplabv3":
        model = smp.DeepLabV3(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            classes=Config.NUM_CLASSES,
            activation=None
        )

    elif model_name in ["deeplabv3plus", "deeplabv3+"]:
        model = smp.DeepLabV3Plus(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            classes=Config.NUM_CLASSES,
            activation=None
        )

    elif model_name == "linknet":
        model = smp.Linknet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            classes=Config.NUM_CLASSES,
            activation=None
        )

    elif model_name == "manet":
        model = smp.MAnet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            classes=Config.NUM_CLASSES,
            activation=None
        )

    elif model_name == "pan":
        model = smp.PAN(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            classes=Config.NUM_CLASSES,
            activation=None
        )

    elif model_name == "upernet":
        model = smp.UPerNet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            classes=Config.NUM_CLASSES,
            activation=None
        )

    elif model_name == "segformer":
        if hasattr(smp, "Segformer"):
            model = smp.Segformer(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                classes=Config.NUM_CLASSES,
                activation=None
            )
        else:
            raise ImportError(
                "SegFormer is not available in this version of segmentation_models_pytorch. "
                "Please upgrade to version >=0.3.2."
            )

    elif model_name == "dpt":
        if hasattr(smp, "DPT"):
            model = smp.DPT(
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                classes=Config.NUM_CLASSES,
                activation=None
            )
        else:
            raise ImportError(
                "DPT is not available in this version of segmentation_models_pytorch. "
                "Please upgrade to version >=0.3.2."
            )

    else:
        raise ValueError(
            f"Unsupported model '{model_name}'. "
            f"Available: unet, unet++, fpn, pspnet, deeplabv3, deeplabv3+, "
            f"linknet, manet, pan, upernet, segformer, dpt."
        )

    print(f"Loaded model: {model_name.upper()} ({encoder_name})")
    return model