import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.autograd import Variable
from collections import OrderedDict
from torchvision import datasets, transforms, models
import torchvision
import numpy
from tqdm import tqdm
import os
import time
from tensorboardX import SummaryWriter
exp_time = time.strftime("%Y%m%d-%H-%M-%S", time.localtime())
model_writer = SummaryWriter('runs/' + exp_time + '/alexnet') 

BATCH_SIZE = 64
DATA_ROOT = './data/processed'
data_transform = transforms.Compose(
    [transforms.Resize([224, 224]),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
)

img_dataset = datasets.ImageFolder(
    root=DATA_ROOT,
    transform=data_transform
)

train_size = int(0.6 * len(img_dataset))
val_size = int(0.2 * len(img_dataset))
test_size = len(img_dataset) - train_size - val_size

train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(img_dataset, [train_size, val_size, test_size])

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory=True,
    num_workers=8
)

val_loader = DataLoader(
    dataset=val_dataset,
    batch_size=BATCH_SIZE,
    # shuffle=True,
    pin_memory=True,
    num_workers=8
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=BATCH_SIZE,
    # shuffle=True,
    pin_memory=True,
    num_workers=8
)

dataloader = {'train': train_loader, 'val': val_loader, 'test': test_loader}

model = models.resnet50(pretrained=True)
for index, param in enumerate(model.parameters()):
    if index < 129:
        param.requires_grad = False
    model.fc = nn.Linear(2048, 2)

print(model)

model = model.cuda()

loss_f = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr = 1e-4)
scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5], gamma=0.1, last_epoch=-1)

epoch_n = 10
time_start = time.time()

for epoch in range(epoch_n):

    print('Epoch {}/{}'.format(epoch + 1, epoch_n))

    for phase in ['train', 'val']:
        if phase == 'train':
            model = model.train()
        else:
            model = model.eval()

        running_loss = 0.0
        running_corrects = 0

        for batch, data in enumerate(dataloader[phase], 1):
            X, y = data
            X, y = Variable(X), Variable(y)
            if torch.cuda.is_available():
                X = X.cuda()
                y = y.cuda()
            y_pred = model(X)
            _, pred = torch.max(y_pred.data, 1)
            optimizer.zero_grad()
            loss = loss_f(y_pred, y)
            if phase == 'train':
                loss.backward()
                optimizer.step()
            running_loss += loss.item()
            running_corrects += torch.sum(pred == y.data)
            if phase == 'train':
                model_writer.add_scalar('training_loss', loss.item(), epoch*len(dataloader[phase])+batch)
                model_writer.add_scalar('training_acc', torch.sum(pred == y.data)/BATCH_SIZE, epoch*len(dataloader[phase])+batch)
            if phase == 'val':
                model_writer.add_scalar('val_loss', loss.item(), epoch*len(dataloader[phase])+batch)
                model_writer.add_scalar('val_acc', torch.sum(pred == y.data)/BATCH_SIZE, epoch*len(dataloader[phase])+batch)
            if batch%20 == 0 and phase == 'train':
                print('Batch {}, Training Loss:{:.4f}, Train Acc:{:.4f}'.\
                    format(batch, running_loss/batch, float(100*running_corrects)/(BATCH_SIZE*batch)))
            if batch == int(val_size/BATCH_SIZE)-1 and phase == 'val':
                print('Validing Loss:{:.4f}, Valid Acc:{:.4f}'.\
                    format(running_loss/batch, float(100*running_corrects)/(BATCH_SIZE*batch)))
    if phase == 'val':
        scheduler.step()
time_end = time.time()
print(time_end - time_start)
model_writer.close()

# test

phase = 'test'
model = model.eval()

loss_f = nn.CrossEntropyLoss()
BATCH_SIZE = 64

running_loss = 0.0
running_corrects = 0

for batch, data in enumerate(tqdm(dataloader[phase]), 1):
    X, y = data
    X, y = Variable(X), Variable(y)
    if torch.cuda.is_available():
        X = X.cuda()
        y = y.cuda()
    y_pred = model(X)
    _, pred = torch.max(y_pred.data, 1)
    loss = loss_f(y_pred, y)
    running_loss += loss.item()
    running_corrects += torch.sum(pred == y.data)
epoch_loss = running_loss*BATCH_SIZE/test_size
epoch_acc = float(100*running_corrects)/test_size
print('{} Loss:{:.4f} Acc:{:.4f}%'.format(phase, epoch_loss, epoch_acc))