import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
from Architecture import CNNCIFAR100 as AC100
from DataLoader import dataloaderCIFAR100 as DC100
import Training as T
import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import argparse


parser = argparse.ArgumentParser()
parser.add_argument('-cid',"--chunk_id", type=int, help='Job chunk id',required=True) #obbligatori, required=True
parser.add_argument('-cdim',"--chunk_dim", type=int, help='Job chunk dimension',required=True)

args = parser.parse_args()

job_index=args.chunk_id
window_size=args.chunk_dim

loss_fn=nn.CrossEntropyLoss()
model = AC100.CNN3().cuda()\
    if torch.cuda.is_available() else AC100.CNN3()
optimizer=torch.optim.SGD(model.parameters(), lr=1e-2)
training_dataloader, test_dataloader=DC100.DataLoader_creation(window_size, job_index)

T.train_loop(
    n_epochs = 60,
    optimizer = optimizer,
    model = model,
    loss_fn = loss_fn,
    train_loader=training_dataloader
)

PATH=f'../checkpointCNNCIFAR1003_{job_index}'
torch.save(model.state_dict(), PATH)



