'''
    训练模型
'''

import os
import time
import torch
from .loaddata import load_data
from .settrain import get_train_set
from torch.autograd import Variable
from .setlog import LOGGER
from tensorboardX import SummaryWriter

def run_train(model, config, base_config):

    save_path = os.path.join(base_config.weights_root, '{}.pth'.format(config.model))
    loss_f, optimizer, scheduler, dataloader = init_train(model, config.lr, config.milestones, base_config.data_root, config.batch_size)
    train(config, model, loss_f, optimizer, scheduler, config.epoch, config.tensorboard, dataloader, save_path, config.cuda, config.batch_size)

def init_train(model, learning_rate, milestones, data_root, batch_size):

    loss_f, optimizer, scheduler = get_train_set(model, learning_rate, milestones)
    dataloader = load_data(data_root, batch_size)

    return loss_f, optimizer, scheduler, dataloader


def train(config, model, loss_f, optimizer, scheduler, epoch_num, is_tensorboard, dataloader, save_path, is_cuda, batch_size):
    '''
    Args:
        - model: 模型
        - loss_f: 损失函数
        - optimizer: 优化器
        - scheduler: 学习率调整
        - epoch_num: 训练的总epoch数
        - is_tensorboard: 是否使用tensorboard可视化
        - dataloader: 数据加载
        - save_path: 模型保存路径
        - is_cuda: 是否使用GPU加速
        - batch_size: 参数batch size
    '''
    
    if is_tensorboard:

        exp_time = time.strftime("%Y%m%d-%H-%M-%S", time.localtime())
        model_writer = SummaryWriter('runs/' + exp_time + '/' + config.model) 

    if is_cuda:
        model = model.cuda()
    
    time_start = time.time()

    for epoch in range(epoch_num):

        LOGGER.info('Epoch {}/{}'.format(epoch + 1, epoch_num))

        for phase in ['train', 'val']:
            if phase == 'train':
                print('Train')
                model = model.train()
            else:
                print('Valid')
                model = model.eval()
            
            running_loss = 0.0
            running_corrects = 0

            for batch, data in enumerate(dataloader[phase], 1):
                X, y = data
                X, y = Variable(X), Variable(y)
                if is_cuda:
                    X, y = X.cuda(), y.cuda()
                y_pred = model(X)
                _, pred = torch.max(y_pred.data, 1)
                optimizer.zero_grad()
                loss = loss_f(y_pred, y)
                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                running_loss += loss.item()
                running_corrects += torch.sum(pred == y.data)

                if is_tensorboard:
                    if phase == 'train':
                        model_writer.add_scalar('training_loss', loss.item(), epoch*len(dataloader[phase])+batch)
                        model_writer.add_scalar('training_acc', torch.sum(pred == y.data)/batch_size, epoch*len(dataloader[phase])+batch)
                    if phase == 'val':
                        model_writer.add_scalar('val_loss', loss.item(), epoch*len(dataloader[phase])+batch)
                        model_writer.add_scalar('val_acc', torch.sum(pred == y.data)/batch_size, epoch*len(dataloader[phase])+batch)

                if batch%20 == 0 and phase == 'train':
                    print('Batch {}, Training Loss:{:.4f}, Train Acc:{:.4f}'.\
                    format(batch, running_loss/batch, float(100*running_corrects)/(batch_size*batch)))
                if batch == len(dataloader['val'])-1 and phase == 'val':
                    print('Validing Loss:{:.4f}, Valid Acc:{:.4f}'.\
                        format(running_loss/batch, float(100*running_corrects)/(batch_size*batch)))
        
        if phase == 'val':
            scheduler.step()
    
    time_end = time.time()
    print(time_end - time_start)

    if is_tensorboard:
        model_writer.close()
    
    torch.save(model.state_dict(), save_path)