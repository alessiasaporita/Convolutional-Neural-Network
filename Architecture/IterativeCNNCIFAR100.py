from torch import nn
import matplotlib

class CNN(nn.Module):
    def make_blocks (self, inplanes, planes, b, pool, drop, r, dilatation=1):
        for i in range(1, r):
            layers=nn.ModuleList()
            if (i % b == 0 and i > 1):
                inplanes = planes
                planes *= 2
            conv=nn.Conv2d(inplanes, planes, kernel_size=3, padding=dilatation)  # 32
            layers.append(conv)
            act=nn.ReLU()
            layers.append(act)

            if (i % pool == 0):
                pooling=nn.MaxPool2d(2)
                layers.append(pooling)
            if (i % drop == 0):
                dropout=nn.Dropout(0.25)
                layers.append(dropout)
        return layers


    def architecture(self, inplanes=3, planes=32, b=4, pool=5, drop=3, r=20):
        layers=self.make_blocks(self, inplanes, planes, b, pool, drop, r),
        net=nn.Sequential(
            nn.Sequential(*layers),
            nn.Linear( (int)(32/pow(2, (int)(r-1)/pool)) * (int)(32/pow(2, (int)(r-1)/pool)) * (int)(planes*pow(2, (int)(r-1)/b)), 32),
            nn.ReLU(),

            nn.Linear(32, 128),
            nn.ReLU(),

            nn.Linear(128, 100),
        )
        return net


    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.make_blocks(inplanes=3, planes=32, b=4, pool=5, drop=3, r=20, dilatation=1)
        self.architecture(inplanes=3, planes=32, b=4, pool=5, drop=3, r=20)



    def forward(self, x):
        out = self.architecture(inplanes=3, planes=32, b=4, pool=5, drop=3, r=20)
        return out

