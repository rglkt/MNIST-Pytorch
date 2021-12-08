# reference https://blog.csdn.net/qq_34644203/article/details/104901786
from torchvision import transforms
import torchvision as tv
import torch
import os

def GetMNISTData(batch_size):
    DOWNLOAD = not os.path.exists('./data')

    transform = transforms.ToTensor()

    trainset = tv.datasets.MNIST(
        root='./data',
        train=True,
        download=DOWNLOAD,
        transform=transform
    )

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=batch_size,
        shuffle=True,
    )

    testset = tv.datasets.MNIST(
        root='./data/',
        train=False,
        download=DOWNLOAD,
        transform=transform,
    )

    testloader = torch.utils.data.DataLoader(
        testset,
        batch_size=batch_size,
        shuffle=False
    )
    return trainloader, testloader
