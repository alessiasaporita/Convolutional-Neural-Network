from torch import nn


#MODELLO
#Accuracy train: 0.
#Accuracy val: 0.
class CNN(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conv1=nn.Conv2d(3, 48, kernel_size=4, padding=1) #224
        self.act1=nn.ReLU()
        self.pool1=nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(48, 128, kernel_size=5, padding=1)#112
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)

        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)#56
        self.act3 = nn.ReLU()

        self.conv4 = nn.Conv2d(256, 256, kernel_size=3, padding=1)#56
        self.act4 = nn.ReLU()

        self.conv5 = nn.Conv2d(256, 128, kernel_size=3, padding=1)#56
        self.act5 = nn.ReLU()
        self.pool5 = nn.MaxPool2d(2)#28

        self.fc1 = nn.Linear(28*28*128, 2048)
        self.act5 = nn.ReLU()

        self.fc2 = nn.Linear(2048, 2048)
        self.act6 = nn.ReLU()

        self.fc3 = nn.Linear(2048, 1000)
    def forward(self, x):
        out=self.pool1(self.act1(self.conv1(x)))
        out=self.pool2(self.act2(self.conv2(out)))
        out = self.act3(self.conv3(out))
        out = self.act4(self.conv4(out))
        out = self.pool5(self.act5(self.conv5(out)))
        out=out.view(-1, 28*28*128)
        out=self.act5(self.fc1(out))
        out = self.act6(self.fc2(out))
        out=self.fc3(out)
        return out

