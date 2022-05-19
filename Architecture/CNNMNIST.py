from torch import nn

#MODELLO
#Accuracy train: 0.98
#Accuracy val: 0.98

class CNN(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conv1=nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.act1=nn.ReLU()
        self.pool1=nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()
        self.pool3 = nn.AvgPool2d(7)
        self.fc1 = nn.Linear(64, 32)
        self.act4 = nn.ReLU()
        self.fc2 = nn.Linear(32, 10)
    def forward(self, x):
        out=self.pool1(self.act1(self.conv1(x)))
        out=self.pool2(self.act2(self.conv2(out)))
        out = self.pool3(self.act3(self.conv3(out)))
        out=out.view(-1, 64)
        out=self.act4(self.fc1(out))
        out=self.fc2(out)
        return out

