'''
    MobileNetV3的封装
'''

import torch.nn as nn
import torchvision.models as models

class XMobileNetV3:
    
    def __init__(self, class_num, pre_trained, layer_idx):
        '''
        Args:
            - class_num: 类别数量
            - pre_trained: 是否使用ImageNet预训练模型
            - layer_idx: 从第几层开始允许更新参数(之前的全部冻结)
        '''
        self.class_num = class_num
        self.pre_trained = pre_trained
        self.layer_idx = layer_idx # 141
        self.model = models.mobilenet_v3_large(pretrained=self.pre_trained)
        if self.pre_trained:
            for index, param in enumerate(self.model.parameters()):
                if index < self.layer_idx:
                    param.requires_grad = False
        in_features = self.model.classifier[-1].in_features
        self.model.classifier[-1] = nn.Linear(in_features, self.class_num)
        self.name = 'MobileNetV3'