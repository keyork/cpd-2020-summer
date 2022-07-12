'''
    将数据加载成Dataloader
'''

import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


def load_data(data_root, batch_size):

    data_transform = transforms.Compose(
        [transforms.Resize([224, 224]),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ]
    )

    img_dataset = datasets.ImageFolder(
        root=data_root,
        transform=data_transform
    )

    train_size = int(0.5 * len(img_dataset))
    val_size = int(0.3 * len(img_dataset))
    test_size = len(img_dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(img_dataset, [train_size, val_size, test_size])

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        pin_memory=True,
        num_workers=8,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=batch_size,
        # shuffle=True,
        pin_memory=True,
        num_workers=8,
    )

    test_loader = DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        # shuffle=True,
        pin_memory=True,
        num_workers=8,
    )

    dataloader = {'train': train_loader, 'val': val_loader, 'test': test_loader}

    return dataloader