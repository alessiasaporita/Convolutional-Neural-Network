import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

loss_list=[]



#TRAINING LOOP
def train_loop(n_epochs, optimizer, model, loss_fn, train_loader):
    for epoch in range(n_epochs):
        for imgs, labels in train_loader:
            optimizer.zero_grad()
            imgs=imgs.cuda()
            model=model.cuda()
            labels=labels.cuda()
            t_out_train=model(imgs)
            loss_train=loss_fn(t_out_train, labels)
            loss_list.append(loss_train.cpu().item())
            loss_train.backward()
            optimizer.step()
        if epoch == 1 or epoch % 10 == 0:
            print(f"Epoch {epoch}, Training loss {loss_train.item():.4f}")
    plt.plot(loss_list)
    plt.show()