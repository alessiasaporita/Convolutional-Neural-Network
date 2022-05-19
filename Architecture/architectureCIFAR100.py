from torch import nn


#MODELLO
#Accuracy train: 0.
#Accuracy val: 0.

class NeuralNetwork(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.flatten=nn.Flatten()
        self.linear=nn.Sequential(
            nn.Linear(3072, 512),
            nn.ReLU(),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 100)
        )
    def forward(self, x):
        x=self.flatten(x)
        out=self.linear(x)
        return out


