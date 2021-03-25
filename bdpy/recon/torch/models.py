'''PyTorch models.'''


import torch
import torch.nn as nn


class AlexNetRelu7Generator(nn.Module):

    def __init__(self, input_size=4096, n_out_channel=3, device=None):

        super(AlexNetRelu7Generator, self).__init__()

        if device is None:
            self.__device0 = 'cpu'
            self.__device1 = 'cpu'
        elif isinstance(device, str):
            self.__device0 = device
            self.__device1 = device
        else:
            self.__device0 = device[0]
            self.__device1 = device[1]

        self.defc7 = nn.Linear(input_size, 4096)
        self.relu_defc7 = nn.LeakyReLU(0.3,inplace=True)

        self.defc6 = nn.Linear(4096, 4096)
        self.relu_defc6 = nn.LeakyReLU(0.3,inplace=True)

        self.defc5 = nn.Linear(4096, 4096)
        self.relu_defc5 = nn.LeakyReLU(0.3,inplace=True)

        self.deconv5 = nn.ConvTranspose2d(256, 256, kernel_size=4, stride=2, padding=1, bias=True)
        self.relu_deconv5 = nn.LeakyReLU(0.3,inplace=True)

        self.conv5_1 = nn.ConvTranspose2d(256, 512, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu_conv5_1 = nn.LeakyReLU(0.3,inplace=True)

        self.deconv4 = nn.ConvTranspose2d(512, 256, kernel_size=4, stride=2, padding=1, bias=True)
        self.relu_deconv4 = nn.LeakyReLU(0.3,inplace=True)

        self.conv4_1 = nn.ConvTranspose2d(256, 256, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu_conv4_1 = nn.LeakyReLU(0.3,inplace=True)

        self.deconv3 = nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1, bias=True)
        self.relu_deconv3 = nn.LeakyReLU(0.3,inplace=True)

        self.conv3_1 = nn.ConvTranspose2d(128, 128, kernel_size=3, stride=1, padding=1, bias=True)
        self.relu_conv3_1 = nn.LeakyReLU(0.3,inplace=True)

        self.deconv2 = nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1, bias=True)
        self.relu_deconv2 = nn.LeakyReLU(0.3,inplace=True)

        self.deconv1 = nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1, bias=True)
        self.relu_deconv1 = nn.LeakyReLU(0.3,inplace=True)

        self.deconv0 = nn.ConvTranspose2d(32, n_out_channel, kernel_size=4, stride=2, padding=1, bias=True)

        self.defc = nn.Sequential(
            self.defc7,
            self.relu_defc7,
            self.defc6,
            self.relu_defc6,
            self.defc5,
            self.relu_defc5,
        ).to(self.__device0)

        self.deconv = nn.Sequential(
            self.deconv5,
            self.relu_deconv5,
            self.conv5_1,
            self.relu_conv5_1,
            self.deconv4,
            self.relu_deconv4,
            self.conv4_1,
            self.relu_conv4_1,
            self.deconv3,
            self.relu_deconv3,
            self.conv3_1,
            self.relu_conv3_1,
            self.deconv2,
            self.relu_deconv2,
            self.deconv1,
            self.relu_deconv1,
            self.deconv0,
        ).to(self.__device1)

    def forward(self, z):

        f = self.defc(z)
        f = f.view(-1, 256, 4, 4)
        g = self.deconv(f)

        return g
