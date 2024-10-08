# --- Imports --- #
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.nn.utils import spectral_norm
from basicsr.utils.registry import ARCH_REGISTRY
from pdb import set_trace as stx
import math


##---------- pixel unshuffle ----------
def pixel_unshuffle(x, scale):
    """ Pixel unshuffle.

    Args:
        x (Tensor): Input feature with shape (b, c, hh, hw).
        scale (int): Downsample ratio.

    Returns:
        Tensor: the pixel unshuffled feature.
    """
    b, c, hh, hw = x.size()
    out_channel = c * (scale**2)
    assert hh % scale == 0 and hw % scale == 0
    h = hh // scale
    w = hw // scale
    x_view = x.view(b, c, h, scale, w, scale)
    return x_view.permute(0, 1, 3, 5, 2, 4).reshape(b, out_channel, h, w)


##---------- ue attention network ----------
class UNetDiscriminatorSN(nn.Module):
    """Defines a U-Net discriminator with spectral normalization (SN)

    It is used in Real-ESRGAN: Training Real-World Blind Super-Resolution with Pure Synthetic Data.

    Arg:
        num_in_ch (int): Channel number of inputs. Default: 3.
        num_feat (int): Channel number of base intermediate features. Default: 64.
        skip_connection (bool): Whether to use skip connections between U-Net. Default: True.
    """

    def __init__(self, num_in_ch, num_feat=64, skip_connection=True):
        super(UNetDiscriminatorSN, self).__init__()
        self.skip_connection = skip_connection
        norm = spectral_norm
        # the first convolution
        self.conv0 = nn.Conv2d(num_in_ch, num_feat, kernel_size=3, stride=1, padding=1)
        # downsample
        self.conv1 = norm(nn.Conv2d(num_feat, num_feat * 2, 4, 2, 1, bias=False))
        self.conv2 = norm(nn.Conv2d(num_feat * 2, num_feat * 4, 4, 2, 1, bias=False))
        self.conv3 = norm(nn.Conv2d(num_feat * 4, num_feat * 8, 4, 2, 1, bias=False))
        # upsample
        self.conv4 = norm(nn.Conv2d(num_feat * 8, num_feat * 4, 3, 1, 1, bias=False))
        self.conv5 = norm(nn.Conv2d(num_feat * 4, num_feat * 2, 3, 1, 1, bias=False))
        self.conv6 = norm(nn.Conv2d(num_feat * 2, num_feat, 3, 1, 1, bias=False))
        # extra convolutions
        self.conv7 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv8 = norm(nn.Conv2d(num_feat, num_feat, 3, 1, 1, bias=False))
        self.conv9 = nn.Conv2d(num_feat, 1, 3, 1, 1)

    def forward(self, x):
        # downsample
        x0 = F.leaky_relu(self.conv0(x), negative_slope=0.2, inplace=True)
        x1 = F.leaky_relu(self.conv1(x0), negative_slope=0.2, inplace=True)
        x2 = F.leaky_relu(self.conv2(x1), negative_slope=0.2, inplace=True)
        x3 = F.leaky_relu(self.conv3(x2), negative_slope=0.2, inplace=True)

        # upsample
        x3 = F.interpolate(x3, scale_factor=2, mode='bilinear', align_corners=False)
        x4 = F.leaky_relu(self.conv4(x3), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x4 = x4 + x2
        x4 = F.interpolate(x4, scale_factor=2, mode='bilinear', align_corners=False)
        x5 = F.leaky_relu(self.conv5(x4), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x5 = x5 + x1
        x5 = F.interpolate(x5, scale_factor=2, mode='bilinear', align_corners=False)
        x6 = F.leaky_relu(self.conv6(x5), negative_slope=0.2, inplace=True)

        if self.skip_connection:
            x6 = x6 + x0

        # extra convolutions
        out = F.leaky_relu(self.conv7(x6), negative_slope=0.2, inplace=True)
        out = F.leaky_relu(self.conv8(out), negative_slope=0.2, inplace=True)
        out = self.conv9(out)
        out = torch.sigmoid(out)
        return out


##---------- Selective Kernel Feature Fusion (SKFF) ----------
class SKFF(nn.Module):
    def __init__(self, in_channels, height=3,reduction=8,bias=False):
        super(SKFF, self).__init__()

        self.height = height
        d = max(int(in_channels/reduction),4)

        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(nn.Conv2d(in_channels, d, 1, padding=0, bias=bias), nn.LeakyReLU(0.2))

        self.fcs = nn.ModuleList([])
        for i in range(self.height):
            self.fcs.append(nn.Conv2d(d, in_channels, kernel_size=1, stride=1,bias=bias))

        self.softmax = nn.Softmax(dim=1)

    def forward(self, inp_feats):
        batch_size = inp_feats[0].shape[0]
        n_feats =  inp_feats[0].shape[1]


        inp_feats = torch.cat(inp_feats, dim=1)
        inp_feats = inp_feats.view(batch_size, self.height, n_feats, inp_feats.shape[2], inp_feats.shape[3])

        feats_U = torch.sum(inp_feats, dim=1)
        feats_S = self.avg_pool(feats_U)
        feats_Z = self.conv_du(feats_S)

        attention_vectors = [fc(feats_Z) for fc in self.fcs]
        attention_vectors = torch.cat(attention_vectors, dim=1)
        attention_vectors = attention_vectors.view(batch_size, self.height, n_feats, 1, 1)
        # stx()
        attention_vectors = self.softmax(attention_vectors)

        feats_V = torch.sum(inp_feats*attention_vectors, dim=1)

        return feats_V



class ContextBlock(nn.Module):

    def __init__(self, n_feat, bias=False):
        super(ContextBlock, self).__init__()

        self.conv_mask = nn.Conv2d(n_feat, 1, kernel_size=1, bias=bias)
        self.softmax = nn.Softmax(dim=2)

        self.channel_add_conv = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias),
            nn.LeakyReLU(0.2),
            nn.Conv2d(n_feat, n_feat, kernel_size=1, bias=bias)
        )

    def modeling(self, x):
        batch, channel, height, width = x.size()
        input_x = x
        # [N, C, H * W]
        input_x = input_x.view(batch, channel, height * width)
        # [N, 1, C, H * W]
        input_x = input_x.unsqueeze(1)
        # [N, 1, H, W]
        context_mask = self.conv_mask(x)
        # [N, 1, H * W]
        context_mask = context_mask.view(batch, 1, height * width)
        # [N, 1, H * W]
        context_mask = self.softmax(context_mask)
        # [N, 1, H * W, 1]
        context_mask = context_mask.unsqueeze(3)
        # [N, 1, C, 1]
        context = torch.matmul(input_x, context_mask)
        # [N, C, 1, 1]
        context = context.view(batch, channel, 1, 1)

        return context

    def forward(self, x):
        # [N, C, 1, 1]
        context = self.modeling(x)

        # [N, C, 1, 1]
        channel_add_term = self.channel_add_conv(context)
        x = x + channel_add_term

        return x

### --------- Residual Context Block (RCB) ----------
class RCB(nn.Module):
    def __init__(self, n_feat, kernel_size=3, reduction=8, bias=False, groups=1):
        super(RCB, self).__init__()

        act = nn.LeakyReLU(0.2)

        self.body = nn.Sequential(
            nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=1, padding=1, bias=bias, groups=groups),
            act,
            nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=1, padding=1, bias=bias, groups=groups)
        )

        self.act = act

        self.gcnet = ContextBlock(n_feat, bias=bias)

    def forward(self, x):
        res = self.body(x)
        res = self.act(self.gcnet(res))
        res += x
        return res


##---------- Resizing Modules ----------
class Down(nn.Module):
    def __init__(self, in_channels, chan_factor, bias=False):
        super(Down, self).__init__()

        self.bot = nn.Sequential(
            nn.AvgPool2d(2, ceil_mode=True, count_include_pad=False),
            nn.Conv2d(in_channels, int(in_channels*chan_factor), 1, stride=1, padding=0, bias=bias)
            )

    def forward(self, x):
        return self.bot(x)

class DownSample(nn.Module):
    def __init__(self, in_channels, scale_factor, chan_factor=2, kernel_size=3):
        super(DownSample, self).__init__()
        self.scale_factor = int(np.log2(scale_factor))

        modules_body = []
        for i in range(self.scale_factor):
            modules_body.append(Down(in_channels, chan_factor))
            in_channels = int(in_channels * chan_factor)

        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        x = self.body(x)
        return x

class Up(nn.Module):
    def __init__(self, in_channels, chan_factor, bias=False):
        super(Up, self).__init__()

        self.bot = nn.Sequential(
            nn.Conv2d(in_channels, int(in_channels//chan_factor), 1, stride=1, padding=0, bias=bias),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=bias)
            )

    def forward(self, x):
        return self.bot(x)

class UpSample(nn.Module):
    def __init__(self, in_channels, scale_factor, chan_factor=2, kernel_size=3):
        super(UpSample, self).__init__()
        self.scale_factor = int(np.log2(scale_factor))

        modules_body = []
        for i in range(self.scale_factor):
            modules_body.append(Up(in_channels, chan_factor))
            in_channels = int(in_channels // chan_factor)

        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        x = self.body(x)
        return x

##---------- Multi-Scale Resiudal Block (MRB) ----------
class MRB(nn.Module):
    def __init__(self, n_feat, height, width, chan_factor, bias,groups):
        super(MRB, self).__init__()

        self.n_feat, self.height, self.width = n_feat, height, width

        self.dau_top = RCB(int(n_feat*chan_factor**0), bias=bias, groups=groups)
        self.dau_mid = RCB(int(n_feat*chan_factor**1), bias=bias, groups=groups)
        self.dau_bot = RCB(int(n_feat*chan_factor**2), bias=bias, groups=groups)

        self.down2 = DownSample(int((chan_factor**0)*n_feat),2,chan_factor)
        self.down4 = nn.Sequential(
            DownSample(int((chan_factor**0)*n_feat),2,chan_factor),
            DownSample(int((chan_factor**1)*n_feat),2,chan_factor)
        )

        self.up21_1 = UpSample(int((chan_factor**1)*n_feat),2,chan_factor)
        self.up21_2 = UpSample(int((chan_factor**1)*n_feat),2,chan_factor)
        self.up32_1 = UpSample(int((chan_factor**2)*n_feat),2,chan_factor)
        self.up32_2 = UpSample(int((chan_factor**2)*n_feat),2,chan_factor)

        self.conv_out = nn.Conv2d(n_feat, n_feat, kernel_size=1, padding=0, bias=bias)

        # only two inputs for SKFF
        self.skff_top = SKFF(int(n_feat*chan_factor**0), 2)
        self.skff_mid = SKFF(int(n_feat*chan_factor**1), 2)

    def forward(self, x):
        x_top = x.clone()
        x_mid = self.down2(x_top)
        x_bot = self.down4(x_top)

        x_top = self.dau_top(x_top)
        x_mid = self.dau_mid(x_mid)
        x_bot = self.dau_bot(x_bot)

        x_mid = self.skff_mid([x_mid, self.up32_1(x_bot)])
        x_top = self.skff_top([x_top, self.up21_1(x_mid)])

        x_top = self.dau_top(x_top)
        x_mid = self.dau_mid(x_mid)
        x_bot = self.dau_bot(x_bot)

        x_mid = self.skff_mid([x_mid, self.up32_2(x_bot)])
        x_top = self.skff_top([x_top, self.up21_2(x_mid)])

        out = self.conv_out(x_top)
        out = out + x

        return out


##---------- Recursive Residual Group (RRG) ----------
class RRG(nn.Module):
    def __init__(self, n_feat, n_MRB, height, width, chan_factor, bias=False, groups=1):
        super(RRG, self).__init__()
        modules_body = [MRB(n_feat, height, width, chan_factor, bias, groups) for _ in range(n_MRB)]
        modules_body.append(nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=1, padding=1, bias=bias))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


class CFMLayer(nn.Module):
    def __init__(self, n_feat):
        super(CFMLayer, self).__init__()
        self.n_feat = n_feat
        self.CFM_scale_conv0 = nn.Conv2d(n_feat, n_feat, 1)
        self.CFM_scale_conv1 = nn.Conv2d(n_feat, n_feat, 1)
        self.CFM_shift_conv0 = nn.Conv2d(n_feat, n_feat, 1)
        self.CFM_shift_conv1 = nn.Conv2d(n_feat, n_feat, 1)

    def forward(self, x):
        scale = self.CFM_scale_conv1(F.leaky_relu(self.CFM_scale_conv0(x[:,self.n_feat:,:,:]), 0.1, inplace=True))
        shift = self.CFM_shift_conv1(F.leaky_relu(self.CFM_shift_conv0(x[:,self.n_feat:,:,:]), 0.1, inplace=True))
        return torch.cat((x[:,:self.n_feat,:,:] * (scale + 1) + shift, x[:,self.n_feat:,:,:]), dim=1)


class MRB_douinp(nn.Module):
    def __init__(self, n_feat, height, width, chan_factor, bias, groups):
        super(MRB_douinp, self).__init__()

        self.n_feat, self.height, self.width = n_feat, height, width

        self.dau_top = RCB(int(n_feat*chan_factor**0), bias=bias, groups=groups)
        self.dau_mid = RCB(int(n_feat*chan_factor**1), bias=bias, groups=groups)
        self.dau_bot = RCB(int(n_feat*chan_factor**2), bias=bias, groups=groups)

        self.down2 = DownSample(int((chan_factor**0)*n_feat),2,chan_factor)
        self.down4 = nn.Sequential(
            DownSample(int((chan_factor**0)*n_feat),2,chan_factor),
            DownSample(int((chan_factor**1)*n_feat),2,chan_factor)
        )

        self.up21_1 = UpSample(int((chan_factor**1)*n_feat),2,chan_factor)
        self.up21_2 = UpSample(int((chan_factor**1)*n_feat),2,chan_factor)
        self.up32_1 = UpSample(int((chan_factor**2)*n_feat),2,chan_factor)
        self.up32_2 = UpSample(int((chan_factor**2)*n_feat),2,chan_factor)

        self.conv_out = nn.Conv2d(n_feat, n_feat, kernel_size=1, padding=0, bias=bias)

        # only two inputs for SKFF
        self.skff_top = SKFF(int(n_feat*chan_factor**0), 2)
        self.skff_mid = SKFF(int(n_feat*chan_factor**1), 2)

    def forward(self, x):
        x_top = x[:,:self.n_feat,:,:].clone()
        x_mid = self.down2(x_top)
        x_bot = self.down4(x_top)

        x_top = self.dau_top(x_top)
        x_mid = self.dau_mid(x_mid)
        x_bot = self.dau_bot(x_bot)

        x_mid = self.skff_mid([x_mid, self.up32_1(x_bot)])
        x_top = self.skff_top([x_top, self.up21_1(x_mid)])

        x_top = self.dau_top(x_top)
        x_mid = self.dau_mid(x_mid)
        x_bot = self.dau_bot(x_bot)

        x_mid = self.skff_mid([x_mid, self.up32_2(x_bot)])
        x_top = self.skff_top([x_top, self.up21_2(x_mid)])

        out = self.conv_out(x_top)
        out = out + x[:,:self.n_feat,:,:]
        out = torch.cat((out, x[:,self.n_feat:,:,:]), dim=1)
        return out


class CFMRRG(nn.Module):
    def __init__(self, n_feat, n_MRB, height, width, chan_factor, bias=False, groups=1):
        super(CFMRRG, self).__init__()
        self.n_feat = n_feat
        modules_body = []
        for i in range(2*n_MRB):
            if (i%2) == 0:
                modules_body.append(CFMLayer(n_feat))
            else:
                modules_body.append(MRB_douinp(n_feat, height, width, chan_factor, bias, groups))
        self.body = nn.Sequential(*modules_body)
        self.conv = nn.Conv2d(n_feat, n_feat, kernel_size=3, stride=1, padding=1, bias=bias)

    def forward(self, x):
        res = self.body(x)
        res = self.conv(res[:,:self.n_feat,:,:])
        res += x[:,:self.n_feat,:,:]
        return torch.cat((res, x[:,self.n_feat:,:,:]), dim=1)



@ARCH_REGISTRY.register()
class Netv2ueatten(nn.Module):
    def __init__(self,
        inp_channels=3,
        out_channels=3,
        n_feat=80,
        chan_factor=1.5,
        n_RRG=4,
        n_MRB=2,
        height=3,
        width=2,
        scale=1,
        bias=False,
        task= None
    ):
        super(Netv2ueatten, self).__init__()

        kernel_size=3
        self.n_feat = n_feat
        self.task = task

        self.scale = scale
        if scale == 2:
            inp_channels = inp_channels * 4
        elif scale == 1:
            inp_channels = inp_channels * 16

        self.conv_in = nn.Conv2d(inp_channels, n_feat, kernel_size=3, padding=1, bias=bias)

        self.Condition_process = nn.Sequential(
            nn.Conv2d(inp_channels//3, n_feat, 3, 1, 1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(n_feat, n_feat, 1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(n_feat, n_feat, 1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(n_feat, n_feat, 1),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(n_feat, n_feat, 1)
        )

        modules_body = []

        modules_body.append(CFMRRG(n_feat, n_MRB, height, width, chan_factor, bias, groups=1))
        modules_body.append(CFMRRG(n_feat, n_MRB, height, width, chan_factor, bias, groups=2))
        modules_body.append(CFMRRG(n_feat, n_MRB, height, width, chan_factor, bias, groups=4))
        modules_body.append(CFMRRG(n_feat, n_MRB, height, width, chan_factor, bias, groups=4))

        self.body = nn.Sequential(*modules_body)

        # upsample
        self.conv_up1 = nn.Conv2d(n_feat, n_feat, 3, 1, 1)
        self.conv_up2 = nn.Conv2d(n_feat, n_feat, 3, 1, 1)
        self.conv_hr = nn.Conv2d(n_feat, n_feat, 3, 1, 1)
        self.conv_out = nn.Conv2d(n_feat, out_channels, kernel_size=3, padding=1, bias=bias)

        self.lrelu = nn.LeakyReLU(negative_slope=0.2, inplace=True)

        # Attention Network
        self.atten_network = UNetDiscriminatorSN(3)

    def forward(self, inp_img):
        atten = self.atten_network(inp_img)

        if self.scale == 2:
            feat_inp = pixel_unshuffle(inp_img, scale=2)
            feat_atten = pixel_unshuffle(atten, scale=2)
        elif self.scale == 1:
            feat_inp = pixel_unshuffle(inp_img, scale=4)
            feat_atten = pixel_unshuffle(atten, scale=4)
        else:
            feat_inp = inp_img
            feat_atten = atten

        shallow_feats_inp = self.conv_in(feat_inp)
        shallow_feats_atten = self.Condition_process(feat_atten)
        shallow_feats = torch.cat((shallow_feats_inp, shallow_feats_atten), dim=1)
        deep_feats = self.body(shallow_feats)[:,:self.n_feat,:,:]

        if self.task == 'defocus_deblurring':
            deep_feats += shallow_feats
            deep_feats = self.lrelu(self.conv_up1(F.interpolate(deep_feats, scale_factor=2, mode='nearest')))
            deep_feats = self.lrelu(self.conv_up2(F.interpolate(deep_feats, scale_factor=2, mode='nearest')))
            out_img = self.conv_out(deep_feats)

        else:
            deep_feats = self.lrelu(self.conv_up1(F.interpolate(deep_feats, scale_factor=2, mode='nearest')))
            deep_feats = self.lrelu(self.conv_up2(F.interpolate(deep_feats, scale_factor=2, mode='nearest')))
            out_img = self.conv_out(deep_feats)

        return out_img, atten