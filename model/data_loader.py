
import torch
import torch.nn as nn
from torchvision import datasets, transforms
import torchvision

from lib.config import cfg
from torch.utils.data import Dataset, DataLoader

import matplotlib.pyplot as plt
import numpy as np




class RandomDataset(Dataset):

    def __init__(self, size, length):
        self.len = length
        self.data = torch.randn(length, *size)
        self.label = 0

    def __getitem__(self, index):
        return self.data[index], self.label

    def __len__(self):
        return self.len

        
def Random_DataLoader():
    input_size = [3, cfg.DATA_SET.H_IMG, cfg.DATA_SET.H_IMG]

    train_img_loader = DataLoader(dataset=RandomDataset(input_size, length = 1000),
                         batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=True)
    val_img_loader =  DataLoader(dataset=RandomDataset(input_size, length = 100),
                         batch_size=cfg.TRAIN.BATCH_SIZE, shuffle=False)

    return train_img_loader, val_img_loader


def get_DataLoader():
    """ Get torch data loaders for training and validation. The data loaders take a crop of the image,
    transform it into tensor, and normalize it."""
    img_data_transforms = {
        'train': transforms.Compose([
            transforms.RandomCrop((cfg.DATA_SET.H_IMG, cfg.DATA_SET.W_IMG), pad_if_needed=True),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ]),
        'val': transforms.Compose([
            transforms.CenterCrop((cfg.DATA_SET.H_IMG, cfg.DATA_SET.W_IMG)),
            transforms.ToTensor(),
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])
    }


    train_img = datasets.ImageFolder(cfg.DATA_LOADER.TRAIN_IMG_FOLDER, img_data_transforms['train'])
    train_img_loader = torch.utils.data.DataLoader(train_img, 
                                                batch_size=cfg.TRAIN.BATCH_SIZE, 
                                                shuffle=True,
                                                num_workers=cfg.DATA_LOADER.NUM_THREADS)

    val_img = datasets.ImageFolder(cfg.DATA_LOADER.VAL_IMG_FOLDER, img_data_transforms['val'])
    val_img_loader = torch.utils.data.DataLoader(val_img, 
                                                batch_size=cfg.TRAIN.BATCH_SIZE,
                                                shuffle=False, 
                                                num_workers=cfg.DATA_LOADER.NUM_THREADS)

    return train_img_loader, val_img_loader


def get_CIFARLoader():
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                            download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=cfg.TRAIN.BATCH_SIZE,
                                              shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                           download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=cfg.TRAIN.BATCH_SIZE,
                                             shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return trainloader, testloader
    # functions to show an image


def imshow_Loader(loader):
    dataiter = iter(loader)
    images, labels = dataiter.next()

    # show images
    imshow(torchvision.utils.make_grid(images))
    # print labels
    # print(' '.join('%5s' % classes[labels[j]] for j in range(4)))

def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.savefig('Loader_demo.png')
    # plt.show()
# get some random training images
