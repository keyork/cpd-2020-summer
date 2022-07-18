

import argparse
import torch.nn as nn
from config.cfg import BaseConfig
from utils.setlog import LOGGER
from utils.inputbox import str2bool
from utils.train import run_train
from utils.test import run_test
from model.loadmodel import load_model


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, default='train')
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--pretrained', type=str2bool, default=False)
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--tensorboard', type=str2bool, default=False)
    parser.add_argument('--cuda', type=str2bool, default=True)
    parser.add_argument('--mode', type=str, default='dataset')
    parser.add_argument('--milestones', type=int, nargs='+', default=[5])
    config = parser.parse_args()

    base_config = BaseConfig()
    base_config.get_layer(config.model)
    if config.task == 'train':
        model = load_model(config.model, base_config.class_num, config.pretrained, base_config.layer_idx)
        run_train(model, config, base_config)
    elif config.task == 'test':
        model = load_model(config.model, base_config.class_num, False, base_config.layer_idx)
        run_test(model, config, base_config)
