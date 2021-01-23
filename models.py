import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import models
import torchvision.transforms as tf
from utils import shifted


def conv_layer(
    in_ch, out_ch, kernel, activation=nn.LeakyReLU(), stride=1, padding="same"
):
    """Convolutional block, composed by Conv2D, BatchNorm and non-linearity.

    Args:
        in_ch (int): Number of input channels for the convolution.
        out_ch (int): Number of output channels for the convolution.
        kernel (int): Filter size for the convolution.
        activation (nn.activation, optional): Non-linearity after the convolutional layer.
                                              Defaults to nn.LeakyReLU().
        stride (int, optional): Stride used in the convolutional layer. Defaults to 1.
        padding (str, optional): Zero-padding. If 'same', dimensions are kept.
                                 Defaults to "same".

    Returns:
        func: Convolutional block.
    """
    if padding == "same":
        padding = kernel // 2
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=padding),
        nn.BatchNorm2d(out_ch),
        activation,
    )


def res_block(channels, kernel):
    """Residual block, used to keep skip connections.
    Applies a convolutional layer with non-linearity.

    Args:
        channels (int): Input and output channels (channels are kept).
        kernel (int): Filter size for the convolution.

    Returns:
        func: Residual block.
    """
    return nn.Sequential(
        conv_layer(channels, channels, kernel), conv_layer(channels, channels, kernel)
    )


def deconv_layer(in_ch, out_ch, kernel, new_size=None, activation=nn.LeakyReLU()):
    """Deconvolutional layer used in the generator. Applies convolution
    with activation. If `new_size` is given, image is also resized accordingly.

    Args:
        in_ch (int): Number of input channels for the convolution.
        out_ch (int): Number of output channels for the convolution.
        kernel (int): Filter size for the convolution.
        new_size (int, optional): New size of the image after the resize. Defaults to None.
        activation (nn.activation, optional): Non-linearity after the convolutional layer.
                                              Defaults to nn.LeakyReLU().

    Returns:
        func: Deconvolutional layer.
    """
    if new_size:
        return nn.Sequential(
            tf.Resize(new_size, interpolation=3),
            conv_layer(in_ch, out_ch, kernel, activation),
        )
    return conv_layer(in_ch, out_ch, kernel, activation)


def maxpool(kernel=2):
    return nn.MaxPool2d(kernel_size=kernel)


def upsample(scale_factor=2):
    return nn.Upsample(scale_factor=scale_factor)


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

        x = x + self.res1(x)
        x = x + self.res2(x)
        x = x + self.res3(x)

        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        return x


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.conv0 = conv_layer(in_ch=3, out_ch=48, kernel=3, stride=1)
        self.conv1 = conv_layer(in_ch=48, out_ch=48, kernel=3, stride=1)
        self.maxpool0 = maxpool()
        self.conv2 = conv_layer(in_ch=48, out_ch=48, kernel=3, stride=1)
        self.maxpool1 = maxpool()
        self.conv3 = conv_layer(in_ch=48, out_ch=48, kernel=3, stride=1)
        self.maxpool2 = maxpool()
        self.conv4 = conv_layer(in_ch=48, out_ch=48, kernel=3, stride=1)
        self.maxpool3 = maxpool()
        self.conv5 = conv_layer(in_ch=48, out_ch=48, kernel=3, stride=1)
        self.maxpool4 = maxpool()
        self.conv6 = conv_layer(in_ch=48, out_ch=48, kernel=3, stride=1)
        self.upsample5 = upsample()
        # concat output of pool4 on channel dimension
        self.dec_conv5a = conv_layer(in_ch=48, out_ch=96, kernel=3, stride=1)
        self.dec_conv5b = conv_layer(in_ch=96, out_ch=96, kernel=3, stride=1)
        self.upsample4 = upsample()
        # concat output of pool3 on channel dimension
        self.dec_conv4a = conv_layer(in_ch=144, out_ch=96, kernel=3, stride=1)
        self.dec_conv4b = conv_layer(in_ch=96, out_ch=96, kernel=3, stride=1)
        self.upsample3 = upsample()
        # concat output of pool2 on channel dimension
        self.dec_conv3a = conv_layer(in_ch=144, out_ch=96, kernel=3, stride=1)
        self.dec_conv3b = conv_layer(in_ch=96, out_ch=96, kernel=3, stride=1)
        self.upsample2 = upsample()
        # concat output of pool1 on channel dimension
        self.dec_conv2a = conv_layer(in_ch=144, out_ch=96, kernel=3, stride=1)
        self.dec_conv2b = conv_layer(in_ch=96, out_ch=96, kernel=3, stride=1)
        self.upsample1 = upsample()
        # concat output of pool0 on input
        self.dec_conv1a = conv_layer(in_ch=99, out_ch=64, kernel=3, stride=1)
        self.dec_conv1b = conv_layer(in_ch=64, out_ch=32, kernel=3, stride=1)
        self.dec_conv1c = conv_layer(
            in_ch=32, out_ch=3, kernel=3, stride=1, activation=nn.Identity()
        )

    def forward(self, x):
        concats = [x]
        output = self.conv0(x)
        output = self.conv1(output)
        output = self.maxpool0(output)
        output = self.conv2(output)
        output = self.maxpool1(output)
        concats.append(output)
        output = self.conv3(output)
        output = self.maxpool2(output)
        concats.append(output)
        output = self.conv4(output)
        output = self.maxpool3(output)
        concats.append(output)
        output = self.conv5(output)
        output = self.maxpool4(output)
        concats.append(output)
        output = self.conv6(output)
        output = self.upsample5(output)
        output = torch.cat((output, concats.pop()), dim=1)
        # concat output of pool4 on channel dimension
        output = self.dec_conv5a(output)
        output = self.dec_conv5b(output)
        output = self.upsample4(output)
        output = torch.cat((output, concats.pop()), dim=1)
        # concat output of pool3 on channel dimension
        output = self.dec_conv4a(output)
        output = self.dec_conv4b(output)
        output = self.upsample3(output)
        output = torch.cat((output, concats.pop()), dim=1)
        # concat output of pool2 on channel dimension
        output = self.dec_conv3a(output)
        output = self.dec_conv3b(output)
        output = self.upsample2(output)
        output = torch.cat((output, concats.pop()), dim=1)
        # concat output of pool1 on channel dimension
        output = self.dec_conv2a(output)
        output = self.dec_conv2b(output)
        output = self.upsample1(output)
        output = torch.cat((output, concats.pop()), dim=1)
        # concat input
        output = self.dec_conv1a(output)
        output = self.dec_conv1b(output)
        output = self.dec_conv1c(output)
        return output


class GeneratorLoss(nn.Module):
    """Custom loss for the generator model of the GAN.
    This loss is composed by the weighted sum of 4 losses:
    1) Adversarial loss: loss of the discriminator.
    2) Pixel loss: MSE between generated image and groundtruth.
    3) Feature loss: MSE between VGG16 Conv2 layer of the generated image
    and VGG16 of the groundtruth.
    4) Smooth loss: MSE between generated image and one-unit (left and bottom)
    copy of the generated image.

    Args:
        vgg_model (nn.Module): VGG16 pretrained model used to compute feature spaces.
        disc_loss_factor (float): Weight for the adversarial (discriminator) loss.
        pix_loss_factor (float): Weight for the pixel loss.
        feat_loss_factor (float): Weight for the feature loss.
        smooth_loss_factor (float): Weight for the smooth loss.
    """

    def __init__(
        self,
        pix_loss_factor,
        feat_loss_factor,
        smooth_loss_factor,
    ):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        super(GeneratorLoss, self).__init__()
        self.vgg_model = (
            torchvision.models.vgg16(pretrained=True).features[:3].to(device)
        )
        self.pix_loss_factor = pix_loss_factor
        self.feat_loss_factor = feat_loss_factor
        self.smooth_loss_factor = smooth_loss_factor
        self.vgg_model = models.vgg16(pretrained=True).features[:3]
        if torch.cuda.is_available():
            self.vgg_model = self.vgg_model.cuda()

    def features(self, x):
        self.vgg_model(x)

    def forward(self, y, t):
        mse = F.mse_loss
        features = self.vgg_model

        pix_loss = mse(y, t)
        fea_loss = mse(features(y), features(t))
        smo_loss = mse(shifted(y), y)

        return (
            self.pix_loss_factor * pix_loss
            + self.feat_loss_factor * fea_loss
            + self.smooth_loss_factor * smo_loss
        )
