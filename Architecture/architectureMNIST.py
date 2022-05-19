from torch import nn
from torchvision.transforms import ToTensor


#MODELLO
#Accuracy train: 0.
#Accuracy val: 0.

class NeuralNetwork(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.flatten = nn.Flatten()
        self.linear = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        y = self.flatten(x)
        out = self.linear(y)
        return out
