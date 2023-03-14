import torch
import torch.nn as nn
import torch.nn.functional as F


class DepthAttention(nn.Module):
    def __init__(self, channel, depth=64):
        super(DepthAttention,self).__init__()
        reduction = channel
        self.avg_pool = nn.AdaptiveAvgPool3d((depth,1,1))
        self.fc = nn.Sequential(
            nn.Linear(channel*depth,(channel*depth) // reduction),
            nn.ReLU(inplace=True),
            nn.Linear((channel*depth)// reduction,channel*depth),
            nn.Sigmoid()
        )

    def forward(self,x):
        b,c,d,_,_ = x.size()
        y = self.avg_pool(x).view(b,c*d)
        y = self.fc(y).view(b,c,d,1,1)
        return x*y.expand_as(x)



class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _,_ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)



class DaDoubleConv3D(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, depth=None):
        super(DaDoubleConv3D,self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        
        self.conv1 = nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(mid_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.da = DepthAttention(out_channels,depth)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.da(out)
        out = self.relu(out)
        
        return out


class DaSeDoubleConv3D(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, depth=None):
        super(DaSeDoubleConv3D,self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        
        self.conv1 = nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(mid_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.da = DepthAttention(out_channels,depth)
        self.se = SELayer(out_channels,reduction=16)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.da(out)
        out = self.se(out)
        out = self.relu(out)
        
        return out


class SeDoubleConv3D(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, depth=None):
        super(SeDoubleConv3D,self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        
        self.conv1 = nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(mid_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.se = SELayer(out_channels,reduction=16)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.se(out)
        out = self.relu(out)
        
        return out


class ResDaSeDoubleConv3D(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, depth=None):
        super(ResDaSeDoubleConv3D,self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        
        self.conv1 = nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm3d(mid_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm3d(out_channels)
        self.da = DepthAttention(out_channels,depth)
        self.se = SELayer(out_channels,reduction=16)
        self.downsample = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.da(out)
        out = self.se(out)

        if residual.size() != out.size():
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)
        
        return out


class DoubleConv3D(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""

    def __init__(self, in_channels, out_channels, mid_channels=None, depth=None):
        super(DoubleConv3D,self).__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, mid_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv3d(mid_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

#-------------------------------------------

class Down3D(nn.Module):
    """Downscaling with maxpool then double conv"""

    def __init__(self, in_channels, out_channels, conv_builder, depth):
        super(Down3D,self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool3d(2),
            conv_builder(in_channels, out_channels, depth=depth)
        )

    def forward(self, x):
        return self.maxpool_conv(x)


#-------------------------------------------

class Up3D(nn.Module):
    """Upscaling then double conv"""

    def __init__(self, in_channels, out_channels, conv_builder, depth, bilinear=True):
        super(Up3D,self).__init__()

        # if bilinear, use the normal convolutions to reduce the number of channels
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='trilinear', align_corners=True)
            self.conv = conv_builder(in_channels, out_channels, in_channels // 2, depth=depth)
        else:
            self.up = nn.ConvTranspose3d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = conv_builder(in_channels, out_channels, depth=depth)


    def forward(self, x1, x2):
        x1 = self.up(x1)
        # input is CDHW
        diffD = x2.size()[2] - x1.size()[2]
        diffH = x2.size()[3] - x1.size()[3]
        diffW = x2.size()[4] - x1.size()[4]

        x1 = F.pad(x1, [diffD // 2, diffD - diffD // 2,
                        diffH // 2, diffH - diffH // 2,
                        diffW // 2, diffW - diffW // 2])

        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

#-------------------------------------------

class Tail3D(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Tail3D, self).__init__()
        self.conv = nn.Conv3d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

#-------------------------------------------

class DA_UNet(nn.Module):
    def __init__(self, stem, down, up, tail, width, depth, conv_builder, n_channels=1, n_classes=2, bilinear=True,dropout_flag=True):
        super(DA_UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes
        self.bilinear = bilinear
        self.width = width
        self.dropout_flag = dropout_flag
        factor = 2 if bilinear else 1

        self.inc = stem(n_channels, width[0], depth=depth[0])
        self.down1 = down(width[0], width[1], conv_builder, depth=depth[1])
        self.down2 = down(width[1], width[2], conv_builder, depth=depth[2])
        self.down3 = down(width[2], width[3], conv_builder, depth=depth[3])
        self.down4 = down(width[3], width[4] // factor, conv_builder, depth=depth[4])
        self.up1 = up(width[4], width[3] // factor, conv_builder, depth=depth[3], bilinear=self.bilinear)
        self.up2 = up(width[3], width[2]// factor, conv_builder, depth=depth[2], bilinear=self.bilinear)
        self.up3 = up(width[2], width[1] // factor, conv_builder, depth=depth[1], bilinear=self.bilinear)
        self.up4 = up(width[1], width[0], conv_builder, depth=depth[0], bilinear=self.bilinear)
        self.dropout = nn.Dropout(p=0.5)
        self.outc = tail(width[0], n_classes)

    def forward(self, x):
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)

        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        if self.dropout_flag:
            x = self.dropout(x)
        logits = self.outc(x)
        return logits

class UpConv(nn.Module):
    def __init__(self, in_channels, out_channels, scale=2):
        super().__init__()
        self.scale = scale
        self.double_conv = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm3d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.double_conv(x)
        x = F.interpolate(x, scale_factor=self.scale, mode='trilinear', align_corners=False)
        return x


def da_unet(init_depth=128,**kwargs):
    return DA_UNet(stem=DoubleConv3D,
                down=Down3D,
                up=Up3D,
                tail=Tail3D,
                width=[32,64,128,256,512],
                depth=[init_depth,init_depth//2,init_depth//4,init_depth//8,init_depth//16],
                conv_builder=DaDoubleConv3D,
                **kwargs)

def se_unet(init_depth=128,**kwargs):
    return DA_UNet(stem=DoubleConv3D,
                down=Down3D,
                up=Up3D,
                tail=Tail3D,
                width=[32,64,128,256,512],
                depth=[init_depth,init_depth//2,init_depth//4,init_depth//8,init_depth//16],
                conv_builder=SeDoubleConv3D,
                **kwargs)

def da_se_unet(init_depth=128,**kwargs):
    return DA_UNet(stem=DoubleConv3D,
                down=Down3D,
                up=Up3D,
                tail=Tail3D,
                width=[32,64,128,256,512],
                depth=[init_depth,init_depth//2,init_depth//4,init_depth//8,init_depth//16],
                conv_builder=DaSeDoubleConv3D,
                **kwargs)

def res_da_se_unet(init_depth=128,**kwargs):
    return DA_UNet(stem=DoubleConv3D,
                down=Down3D,
                up=Up3D,
                tail=Tail3D,
                width=[32,64,128,256,512],
                depth=[init_depth,init_depth//2,init_depth//4,init_depth//8,init_depth//16],
                conv_builder=ResDaSeDoubleConv3D,
                **kwargs)

