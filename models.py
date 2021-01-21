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
    def __init__(self):
        super(Generator, self).__init__()

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


def shifted(img):
    pad = nn.ZeroPad2d((0, 1, 1, 0))
    return pad(img)[:-1, 1:]


def vgg_prediction(vgg_model, img):
    # model = models.vgg16(pretrained=True, progress=True).cuda().features[:3]
    return vgg_model(img)


# k0 * Adversarial loss +
# k1 * Pixel loss +
# feat_loss_factor * Feature loss +
# k3 * Smooth loss
class GeneratorLoss(nn.Module):
    def __init__(
        self,
        vgg_model,
        disc_loss_factor,
        pix_loss_factor,
        feat_loss_factor,
        smooth_loss_factor,
    ):
        super(GeneratorLoss, self).__init__()
        self.vgg_model = vgg_model
        self.disc_loss_factor = disc_loss_factor
        self.pix_loss_factor = pix_loss_factor
        self.feat_loss_factor = feat_loss_factor
        self.smooth_loss_factor = smooth_loss_factor

    def forward(self, disc_loss, y, t):
        mse = nn.MSELoss()
        features = lambda x: vgg_prediction(self.vgg_model, x)

        pix_loss = mse(y, t)
        fea_loss = mse(features(y), features(t))
        smo_loss = mse(shifted(y), y)
        return (
            self.disc_loss_factor * disc_loss
            + self.pix_loss_factor * pix_loss
            + self.feat_loss_factor * fea_loss
            + self.smooth_loss_factor * smo_loss
        )
