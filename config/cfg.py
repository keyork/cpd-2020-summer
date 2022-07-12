

class BaseConfig:

    def __init__(self):

        self.data_root = './data/processed/'
        self.class_num = 2
        self.layer_idx = 0
        self.weights_root = './weights/'
        
    def get_layer(self, model_name):
        
        if model_name == 'alexnet':
            self.layer_idx = 8
        elif model_name == 'densenet121':
            self.layer_idx = 336
        elif model_name == 'mobilenetv3':
            self.layer_idx = 141
        elif model_name == 'resnet50':
            self.layer_idx = 129
        elif model_name == 'vgg16':
            self.layer_idx = 18




