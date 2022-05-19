import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import torch.nn.functional as F
from Architecture import architectureMNIST as AMNIST
from DataLoader import dataloaderMNIST as DMNIST
from Test import Validazione as V
import numpy as np


model = AMNIST.NeuralNetwork().cuda()\
    if torch.cuda.is_available() else AMNIST.NeuralNetwork()

model.load_state_dict(torch.load('../checkpointMNIST', map_location=torch.device('cpu')))
V.validate(model, DMNIST.training_dataloader, DMNIST.test_dataloader)

for i in range (0, 10):
    t_out_test=F.softmax(model(DMNIST.test_data.data[i].unsqueeze(0).float()), dim=1)
    img = DMNIST.test_data.data[i]
    plt.imshow(img.squeeze().numpy().astype(np.uint8))
    print(DMNIST.test_data.classes[t_out_test.argmax()])
    plt.show()








