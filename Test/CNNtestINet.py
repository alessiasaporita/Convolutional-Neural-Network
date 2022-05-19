import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import torch.nn.functional as F
from Architecture import CNNMNIST as AINet
from DataLoader import dataloaderMNIST as DINet
from Test import Validazione as V
import numpy as np


model = AINet.CNN().cuda()\
    if torch.cuda.is_available() else AINet.CNN()

model.load_state_dict(torch.load('../checkpointCNNINet', map_location=torch.device('cpu')))
V.validate(model, DINet.training_dataloader, DINet.test_dataloader)

for i in range (0, 10):
    t_out_test=F.softmax(model(DINet.test_data.data[i].unsqueeze(0).float()), dim=1)
    img = DINet.test_data.data[i]
    plt.imshow(img.squeeze().numpy().astype(np.uint8))
    print(DINet.test_data.classes[t_out_test.argmax()])
    plt.show()








