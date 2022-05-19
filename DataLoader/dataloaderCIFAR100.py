import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torchvision import transforms
import ssl
ssl._create_default_https_context = ssl._create_unverified_context

t = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


#DATABASE TRAINING
training_data=datasets.CIFAR100(
    root='./Data',
    train=True,
    download=True,
    transform=t
)

training_dataloader=DataLoader(training_data, batch_size=16, shuffle=True)
print(training_dataloader)


# DATABASE TESTING
test_data=datasets.CIFAR100(
    root='./Data',
    train=False,
    download=True,
    transform=t
)

test_dataloader=DataLoader(test_data, batch_size=16, shuffle=True)
