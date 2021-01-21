import torchvision.transforms as tf
import torch.nn as nn
import torch.nn.functional as F
from utils.layer import conv_layer
from utils.layer import res_block
from utils.layer import deconv_layer


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv1 = conv_layer(3, 48, 4, stride=2)
        self.conv2 = conv_layer(48, 96, 4, stride=2)
        self.conv3 = conv_layer(96, 192, 4, stride=2)
        self.conv4 = conv_layer(129, 384, 4)
        self.conv5 = conv_layer(384, 1, 4, activation=nn.Sigmoid())

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x


class Generator(nn.Module):
    def __init__(self, k0, k1, k2, k3):
        super(Generator, self).__init__()

        self.k0 = k0
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3

        self.mse = nn.BCELoss()
        self.smooth = nn.SmoothL1Loss()

        self.conv1 = conv_layer(3, 32, 9)
        self.conv2 = conv_layer(32, 64, 3)
        self.conv3 = conv_layer(64, 128, 3)

        self.res1 = res_block(128, 3)
        self.res2 = res_block(128, 3)
        self.res3 = res_block(128, 3)

        self.deconv1 = deconv_layer(128, 64, 3, (128, 128))
        self.deconv2 = deconv_layer(64, 32, 3, (256, 256))
        self.deconv3 = deconv_layer(32, 3, 3, activation=nn.Tanh())

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x += self.res1(x)
        x += self.res2(x)
        x += self.res3(x)

        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        return x

    # TODO: loss shouldn't be in the model class
    # k0 * Adversarial loss +
    # k1 * Pixel loss +
    # k2 * Feature loss +
    # k3 * Smooth loss
    def gen_loss(self, adv_loss, y, t):
        pix_loss = self.mce(y, t)
        fea_loss = self.mce(y, t)
        # TODO: what is smooth loss?
        smo_loss = lambda x: 0
        return (
            self.k0 * adv_loss
            + self.k1 * pix_loss
            + self.k2 * fea_loss
            + self.k3 * smo_loss
        )
