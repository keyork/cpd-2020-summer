
import torch.nn as nn
import torchvision.models as models

class XResNet50:

    def __init__(self, class_num, pre_trained, layer_idx):
        
        self.class_num = class_num
        self.pre_trained = pre_trained
        self.layer_idx = layer_idx # 129
        self.model = models.resnet50(pretrained=self.pre_trained)
        for index, param in enumerate(self.model.parameters()):
            if index < self.layer_idx:
                param.requires_grad = False
        self.model.fc = nn.Linear(2048, self.class_num)