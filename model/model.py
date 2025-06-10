# Imports
import torch
import torch.nn as nn
import torch.nn.functional as F

# Helper blocks
class Conv(nn.Module): # the double conv class
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding = 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace = True),
            
            nn.Conv2d(out_ch, out_ch, 3, padding = 1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace = True),
        )
        
    def forward(self, x):
        return self.net(x)

class Down(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.net = nn.Sequential(
            nn.MaxPool2d(2),
            Conv(in_ch, out_ch)
        )
    def forward(self, x):
        return self.net(x)
    
class Up(nn.Module):
    def __init__(self, in_ch, out_ch):
        super().__init__()
        self.up = nn.Upsample(scale_factor = 2, mode = 'bilinear', align_corners = True)
        self.conv = Conv(in_ch, out_ch)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # pad x1 to match x2 (in case of odd dims)
        diffY = x2.size(2) - x1.size(2)
        diffX = x2.size(3) - x1.size(3)
        x1 = F.pad(x1, [diffX//2, diffX-diffX//2, diffY//2, diffY-diffY//2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

# the actual model class
class UNet(nn.Module):
    def __init__(self, in_channels=1, n_classes=313):
        super().__init__()
        self.inc   = Conv(in_channels, 64)
        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)
        self.down3 = Down(256, 512)
        self.down4 = Down(512, 512)
        self.up1 = Up(512+512, 256)
        self.up2 = Up(256+256, 128)
        self.up3 = Up(128+128, 64)
        self.up4 = Up( 64+ 64, 64)
        self.outc = nn.Conv2d(64, n_classes, kernel_size=1)

    def forward(self, x):
        x1 = self.inc(x) # 1 → 64
        x2 = self.down1(x1) # 64 → 128
        x3 = self.down2(x2) # 128 → 256
        x4 = self.down3(x3) # 256 → 512
        x5 = self.down4(x4) # 512 → 512
        x = self.up1(x5, x4) # 1024 → 256
        x = self.up2(x, x3) # 512 → 128
        x = self.up3(x, x2) # 256 → 64
        x = self.up4(x, x1) # 128 → 64
        return self.outc(x) # 64 → 313