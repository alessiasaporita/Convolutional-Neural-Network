import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt
import torch.nn.functional as F
from Architecture import architectureCIFAR100 as AC100
from DataLoader import dataloaderCIFAR100 as DC100
from Test import Validazione as V
import numpy as np


model = AC100.NeuralNetwork().cuda()\
    if torch.cuda.is_available() else AC100.NeuralNetwork()

#model.load_state_dict(torch.load('../checkpointCIFAR100', map_location=torch.device('cpu')))
V.validate(model, DC100.training_dataloader, DC100.test_dataloader)

for i in range (0, 10):
    t_out_test=F.softmax(model(torch.from_numpy(DC100.test_data.data[i]).unsqueeze(0).float()), dim=1)
    img = torch.Tensor(DC100.test_data.data[i]).permute(2, 0, 1)
    img = F.interpolate(img.unsqueeze(0), size=(512,512), mode='bilinear', align_corners=False)
    plt.imshow(img.squeeze().permute(1,2,0).numpy().astype(np.uint8))
    print(DC100.test_data.classes[t_out_test.argmax()])
    plt.show()








