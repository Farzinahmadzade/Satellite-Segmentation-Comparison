import segmentation_models_pytorch as smp

class SegmentationModel:
    def __init__(self, model_name='unet', encoder_name='resnet34', num_classes=5, encoder_weights='imagenet'):
        self.model_name = model_name
        self.num_classes = num_classes
        
        if model_name == 'unet':
            self.model = smp.Unet(encoder_name=encoder_name, encoder_weights=encoder_weights,
                                  classes=num_classes, activation=None)
        elif model_name == 'deeplabv3plus':
            self.model = smp.DeepLabV3Plus(encoder_name=encoder_name, encoder_weights=encoder_weights,
                                           classes=num_classes, activation=None)
        elif model_name == 'pspnet':
            self.model = smp.PSPNet(encoder_name=encoder_name, encoder_weights=encoder_weights,
                                    classes=num_classes, activation=None)
        elif model_name == 'fpn':
            self.model = smp.FPN(encoder_name=encoder_name, encoder_weights=encoder_weights,
                                 classes=num_classes, activation=None)
        else:
            raise ValueError(f"Model {model_name} not supported")
    
    def to_device(self, device):
        self.model = self.model.to(device)
        return self
    
    def get_model(self):
        return self.model

def get_available_models():
    return ['unet', 'deeplabv3plus', 'pspnet', 'fpn']

def get_available_encoders():
    return ['resnet18', 'resnet34', 'resnet50', 'resnet101', 'efficientnet-b0', 'efficientnet-b1']