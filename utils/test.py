
from .loaddata import load_data
from .settrain import get_train_set

import torch
from tqdm import tqdm
from torch.autograd import Variable

def run_test(model, config, base_config):

    if config.mode == 'dataset':
        test_dataset(model, loss_f, dataloader, save_path, is_cuda=config.cuda, batch_size=config.batch_size)

def init_test(model, learning_rate, milestones, data_root, batch_size):

    loss_f, optimizer, scheduler = get_train_set(model, learning_rate, milestones)
    dataloader = load_data(data_root, batch_size)

    return loss_f, optimizer, scheduler, dataloader

def test_dataset(model, loss_f, dataloader, save_path, is_cuda, batch_size):

    phase = 'test'
    model = model.eval()

    model.load_state_dict(torch.load(save_path))

    if is_cuda:
        model = model.cuda()

    running_loss = 0.0
    running_corrects = 0

    for batch, data in enumerate(tqdm(dataloader[phase]), 1):
        X, y = data
        X, y = Variable(X), Variable(y)
        if is_cuda:
            X, y = X.cuda(), y.cuda()
        y_pred = model(X)
        _, pred = torch.max(y_pred.data, 1)
        loss = loss_f(y_pred, y)
        running_loss += loss.item()
        running_corrects += torch.sum(pred == y.data)
    
    epoch_loss = running_loss*batch_size/len(dataloader[phase])
    epoch_acc = float(100*running_corrects)/len(dataloader[phase])
    print('{} Loss:{:.4f} Acc:{:.4f}%'.format(phase, epoch_loss, epoch_acc))