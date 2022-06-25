
import torch.nn as nn
import torchvision.models as models

class XDenseNet121:
    
    def __init__(self, class_num, pre_trained, layer_idx):
        
        self.class_num = class_num
        self.pre_trained = pre_trained
        self.layer_idx = layer_idx # 336
        self.model = models.densenet121(pretrained=self.pre_trained)
        for index, param in enumerate(self.model.parameters()):
            if index < self.layer_idx:
                param.requires_grad = False
        in_features = self.model.classifier.in_features
        self.model.classifier = nn.Linear(in_features, self.class_num)