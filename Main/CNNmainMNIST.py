import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import Training as T
from DataLoader import dataloaderMNIST as DMNIST
from Architecture import CNNMNIST as AMNIST




loss_fn=nn.CrossEntropyLoss()
model = AMNIST.CNN().cuda()\
    if torch.cuda.is_available() else AMNIST.CNN()
optimizer=torch.optim.SGD(model.parameters(), lr=1e-2)

for name, param in model.named_parameters():

    print(name, param.shape)

T.train_loop(
    n_epochs = 50,
    optimizer = optimizer,
    model = model,
    loss_fn = loss_fn,
    train_loader=DMNIST.training_dataloader
)

PATH='../checkpointCNNMNIST'
torch.save(model.state_dict(), PATH)



