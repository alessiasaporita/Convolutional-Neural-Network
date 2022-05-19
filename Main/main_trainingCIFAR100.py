import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from Architecture import architectureCIFAR100 as AC100
from DataLoader import dataloaderCIFAR100 as DC100
import Training as T
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


loss_fn=nn.CrossEntropyLoss()
model = AC100.NeuralNetwork().cuda()\
    if torch.cuda.is_available() else AC100.NeuralNetwork()
optimizer=torch.optim.SGD(model.parameters(), lr=1e-2)

for name, param in model.named_parameters():

    print(name, param.shape)

T.train_loop(
    n_epochs = 50,
    optimizer = optimizer,
    model = model,
    loss_fn = loss_fn,
    train_loader=DC100.training_dataloader
)



PATH='../checkpointCIFAR100'
torch.save(model.state_dict(), PATH)



