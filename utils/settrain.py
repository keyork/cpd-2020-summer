'''
    训练时候的一些参数
'''

import torch.nn as nn
import torch.optim as optim

def get_train_set(model, learning_rate, milestones):

    loss_f = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1, last_epoch=-1)

    return loss_f, optimizer, scheduler