#!/usr/bin/env python
# encoding: utf-8
import torch
from torch import nn
from FCDenseNet import FCDenseNetSeg


def weights_init(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform(m.weight)
        if m.bias is not None:
            nn.init.constant(m.bias, 0.01)

    if isinstance(m, nn.BatchNorm2d):
        nn.init.normal(m.weight, 1.0, 0.02)
        nn.init.constant(m.bias, 0.01)

    if isinstance(m, nn.ConvTranspose2d):
        nn.init.kaiming_uniform(m.weight)
        if m.bias is not None:
            nn.init.constant(m.bias, 0.01)


class NetG(nn.Module):
    def __init__(self):
        super(NetG, self).__init__()
        self.op = FCDenseNetSeg(1)

    def forward(self, input):
        return self.op(input)


class NetD(nn.Module):
    def __init__(self, channel_dim=1, ndf=64):
        super(NetD, self).__init__()
        self.convblock1 = nn.Sequential(
            nn.Conv2d(channel_dim, ndf, 7, 2, 3, bias=False),
            nn.LeakyReLU(0.2, inplace=True))

        self.convblock2 = nn.Sequential(
            nn.Conv2d(ndf * 1, ndf * 2, 5, 2, 2, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.convblock3 = nn.Sequential(
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.convblock4 = nn.Sequential(
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.convblock5 = nn.Sequential(
            nn.Conv2d(ndf * 8, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )
        self.convblock6 = nn.Sequential(
            nn.Conv2d(ndf * 8, ndf * 8, 3, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True)
        )

    def forward(self, x):

        batchsize = x.size()[0]
        out1 = self.convblock1(x)

        out2 = self.convblock2(out1)

        out3 = self.convblock3(out2)

        out4 = self.convblock4(out3)

        out5 = self.convblock5(out4)

        out6 = self.convblock6(out5)

        output = torch.cat((x.view(batchsize, -1), 1 * out1.view(batchsize, -1),
                            2 * out2.view(batchsize, -1), 2 * out3.view(batchsize, -1),
                            2 * out4.view(batchsize, -1), 2 * out5.view(batchsize, -1),
                            4 * out6.view(batchsize, -1)), 1)

        return output
