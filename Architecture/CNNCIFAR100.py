from torch import nn

#MODELLO
#MODELLO
#Accuracy train:0.62
#Accuracy val:0.4

class CNN(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conv1=nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self.act1=nn.ReLU()
        self.pool1=nn.MaxPool2d(2)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.act3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2)
        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.act4 = nn.ReLU()
        self.avg4 = nn.AvgPool2d(4)
        self.fc1 = nn.Linear(128, 32)
        self.act5 = nn.ReLU()
        self.fc2 = nn.Linear(32, 1024)
        self.act6 = nn.ReLU()
        self.fc3 = nn.Linear(1024, 100)
    def forward(self, x):
        out=self.pool1(self.act1(self.conv1(x)))
        out=self.pool2(self.act2(self.conv2(out)))
        out = self.pool3(self.act3(self.conv3(out)))
        out = self.avg4(self.act4(self.conv4(out)))
        out=out.view(-1, 128)
        out=self.act5(self.fc1(out))
        out = self.act6(self.fc2(out))
        out=self.fc3(out)
        return out


#MODELLO
#Accuracy train: 0.68
#Accuracy val: 0.4
class CNN2(nn.Module):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.conv1=nn.Conv2d(3, 32, kernel_size=3, padding=1)#32
        self.act1=nn.ReLU()
        self.pool1=nn.MaxPool2d(2)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)#16
        self.act2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(2)
        self.drop2 = nn.Dropout(0.25)

        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, padding=1)#8
        self.act3 = nn.ReLU()
        self.pool3 = nn.MaxPool2d(2)
        self.drop3 = nn.Dropout(0.25)

        self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)#4
        self.act4 = nn.ReLU()

        self.fc1 = nn.Linear(4*4*128, 32)
        self.act5 = nn.ReLU()

        self.fc2 = nn.Linear(32, 1024)
        self.act6 = nn.ReLU()

        self.fc3 = nn.Linear(1024, 100)

    def forward(self, x):
        out=self.pool1(self.act1(self.conv1(x)))
        out=self.drop2(self.pool2(self.act2(self.conv2(out))))
        out = self.drop3(self.pool3(self.act3(self.conv3(out))))
        out = self.act4(self.conv4(out))
        out=out.view(-1, 4*4*128)
        out=self.act5(self.fc1(out))
        out = self.act6(self.fc2(out))
        out=self.fc3(out)
        return out


# #MODELLO
# #Accuracy train: 0.9
# #Accuracy val: 0.34
# class CNN3(nn.Module):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.conv1=nn.Conv2d(3, 16, kernel_size=3, padding=1)
#         self.act1=nn.ReLU()
#         self.pool1=nn.MaxPool2d(2)
#         self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
#         self.act2 = nn.ReLU()
#         self.pool2 = nn.MaxPool2d(2)
#         self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
#         self.act3 = nn.ReLU()
#         self.pool3 = nn.MaxPool2d(2)
#         self.conv4 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
#         self.act4 = nn.ReLU()
#         self.pool4 = nn.MaxPool2d(2)
#         self.conv5 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
#         self.act5 = nn.ReLU()
#         self.avg5 = nn.AvgPool2d(2)
#         self.fc1 = nn.Linear(256*7*7, 32)
#         self.act6 = nn.ReLU()
#         self.fc2 = nn.Linear(32, 1024)
#         self.act7 = nn.ReLU()
#         self.fc3 = nn.Linear(1024, 100)
#     def forward(self, x):
#         out=self.pool1(self.act1(self.conv1(x)))
#         out=self.pool2(self.act2(self.conv2(out)))
#         out = self.pool3(self.act3(self.conv3(out)))
#         out = self.pool4(self.act4(self.conv4(out)))
#         out = self.avg5(self.act5(self.conv5(out)))
#         out=out.view(-1, 256*7*7)
#         out=self.act6(self.fc1(out))
#         out = self.act7(self.fc2(out))
#         out=self.fc3(out)
#         return out
#
# #MODELLO
# #Accuracy train: 0.98
# #Accuracy val: 0.34
# class CNN4(nn.Module):
#     def __init__(self, **kwargs):
#         super().__init__(**kwargs)
#         self.conv1=nn.Conv2d(3, 32, kernel_size=3, padding=1)
#         self.act1=nn.ReLU()
#
#         self.conv2 = nn.Conv2d(32, 32, kernel_size=3, padding=1)
#         self.act2 = nn.ReLU()
#         self.pool2 = nn.MaxPool2d(2)
#         self.drop2 = nn.Dropout(0.25)
#
#         self.conv3 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
#         self.act3 = nn.ReLU()
#
#         self.conv4 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
#         self.act4 = nn.ReLU()
#         self.pool4=nn.MaxPool2d(2)
#         self.drop4 = nn.Dropout(0.25)
#
#         self.fc1 = nn.Linear(8*8*64, 512)
#         self.act5 = nn.ReLU()
#         self.drop5 = nn.Dropout(0.25)
#
#         self.fc2 = nn.Linear(512, 100)
#
#     def forward(self, x):
#         out=self.act1(self.conv1(x))
#         out=self.drop2(self.pool2(self.act2(self.conv2(out))))
#         out = self.act3(self.conv3(out))
#         out = self.drop4(self.pool4(self.act4(self.conv4(out))))
#         out=out.view(-1, 8*8*64)
#         out=self.act5(self.fc1(out))
#         out=self.fc2(out)
#         return out
#
