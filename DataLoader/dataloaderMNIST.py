import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from torchvision import transforms

t = transforms.Compose([transforms.RandomHorizontalFlip(),
                       transforms.ToTensor(),
                       transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))]
                       )


#DATABASE TRAINING
training_data=datasets.MNIST(
    root='./Data',
    train=True,
    download=True,
    transform=t
)

training_dataloader=DataLoader(training_data, batch_size=32, shuffle=True)



# DATABASE TESTING
test_data=datasets.MNIST(
    root='./Data',
    train=False,
    download=True,
    transform=t
)

test_dataloader=DataLoader(test_data, batch_size=32, shuffle=True)
