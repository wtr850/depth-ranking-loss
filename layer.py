import torch.nn as nn

class Residual_Conv(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = nn.ReLU()
        self.conv = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x2 = self.relu(x)
        x2 = self.conv(x2)
        x2 = self.relu(x2)
        x2 = self.conv(x2)
        return x + x2

class Upsampling(nn.Module):
    def __init__(self, width_in):
        super().__init__()
        self.residual_conv = Residual_Conv() 
        self.upsample = nn.Upsample(size=(2*width_in, 2*width_in), mode="bilinear")

    def forward(self, x):
        x = self.residual_conv(x)
        x = self.upsample(x)
        return x

class Adaptive_Output(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(256, 128, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(128, 1, kernel_size=3, stride=1, padding=1)
        self.upsample = nn.Upsample(size=(384, 384), mode="bilinear")

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.upsample(x)
        return x

class Conv_Block_first(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 256, kernel_size=3, stride=2, padding=1)
        self.batchnorm = nn.BatchNorm2d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        return x

class Conv_Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1)
        self.batchnorm = nn.BatchNorm2d(256)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        x = self.batchnorm(x)
        x = self.relu(x)
        return x