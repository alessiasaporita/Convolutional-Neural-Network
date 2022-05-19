import fiftyone as fiftyone
import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import *
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
from torchvision.datasets.imagenet import parse_devkit_archive, parse_train_archive, parse_val_archive


#DATABASE TRAINING

training_data = datasets.ImageFolder(
    root='./Data/archive',
    transform=transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    ]))


training_dataloader=DataLoader(training_data, batch_size=256, shuffle=True)
print(training_dataloader)


# DATABASE TESTING

test_data = datasets.ImageFolder(
    root='./Data/archive',
    transform=transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    ]))
test_dataloader=DataLoader(test_data, batch_size=256, shuffle=True)
