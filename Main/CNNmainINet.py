import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import Training as T
from DataLoader import dataloaderINet as DINet
from Architecture import CNNINet as AINet


loss_fn=nn.CrossEntropyLoss()
model = AINet.CNN().cuda()\
    if torch.cuda.is_available() else AINet.CNN()
optimizer=torch.optim.SGD(model.parameters(), lr=1e-2)

for name, param in model.named_parameters():

    print(name, param.shape)

T.train_loop(
    n_epochs = 500,
    optimizer = optimizer,
    model = model,
    loss_fn = loss_fn,
    train_loader=DINet.training_dataloader
)

PATH='../checkpointINet'
torch.save(model.state_dict(), PATH)



