import torch.nn as nn
from layer import Residual_Conv, Upsampling, Adaptive_Output, Conv_Block_first, Conv_Block

class Whole_Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.convlayer_first = Conv_Block_first()
        self.convlayer = Conv_Block()
        self.upsample0 = Upsampling(12)
        self.upsample1 = Upsampling(24)
        self.upsample2 = Upsampling(48)
        self.upsample3 = Upsampling(96)
        self.residual = Residual_Conv()
        self.adaptive = Adaptive_Output()

    def forward(self, x):
        x = self.convlayer_first(x)
        down1 = self.convlayer(x) # 96 * 96
        down2 = self.convlayer(down1) # 48 * 48
        down3 = self.convlayer(down2) # 24 * 24
        down4 = self.convlayer(down3) # 12 * 12
        
        up1 = self.upsample0(down4)
        up2 = self.upsample1(up1 + self.residual(down3))
        up3 = self.upsample2(up2 + self.residual(down2))
        up4 = self.upsample3(up3 + self.residual(down1))

        out = self.adaptive(up4)

        return out

