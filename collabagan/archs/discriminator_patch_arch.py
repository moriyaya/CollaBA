from basicsr.utils.registry import ARCH_REGISTRY
from torch import nn as nn
from torch.nn import functional as F
from torch.nn.utils import spectral_norm
import numpy as np


@ARCH_REGISTRY.register()
class NoNormDiscriminator(nn.Module):
    def __init__(self, num_in_ch, num_feat=64, n_layers=3, use_sigmoid=False):
        super(NoNormDiscriminator, self).__init__()

        kw = 4
        padw = int(np.ceil((kw-1)/2))
        sequence = [
            nn.Conv2d(num_in_ch, num_feat, kernel_size=kw, stride=2, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2**n, 8)
            sequence += [
                nn.Conv2d(num_feat * nf_mult_prev, num_feat * nf_mult,
                          kernel_size=kw, stride=2, padding=padw),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2**n_layers, 8)
        sequence += [
            nn.Conv2d(num_feat * nf_mult_prev, num_feat * nf_mult,
                      kernel_size=kw, stride=1, padding=padw),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(num_feat * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        
        return self.model(input)