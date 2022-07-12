
from .xalexnet import XAlexNet
from .xdensenet121 import XDenseNet121
from .xmobinetv3 import XMobileNetV3
from .xresnet50 import XResNet50
from .xvgg16 import XVGG16

def load_model(model_name, class_num, pre_trained, layer_idx):

    if model_name == 'alexnet':
        return XAlexNet(class_num, pre_trained, layer_idx).model
    elif model_name == 'densenet121':
        return XDenseNet121(class_num, pre_trained, layer_idx).model
    elif model_name == 'mobilenetv3':
        return XMobileNetV3(class_num, pre_trained, layer_idx).model
    elif model_name == 'resnet50':
        return XResNet50(class_num, pre_trained, layer_idx).model
    elif model_name == 'vgg16':
        return XVGG16(class_num, pre_trained, layer_idx).model