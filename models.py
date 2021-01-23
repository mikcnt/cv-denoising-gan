import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import models
import torchvision.transforms as tf

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
    """Deconvolutional layer. Applies convolutionwith activation.
    If `new_size` is given, image is also resized accordingly.

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


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        # 3x256x256
        self.conv0 = conv_layer(in_ch=3, out_ch=48, kernel=3, stride=1)
        # 48x256x256
        self.conv1 = conv_layer(in_ch=48, out_ch=48, kernel=3, stride=1)
        # 48x256x256
        self.maxpool1 = maxpool()
        # 48x128x128
        self.conv2 = conv_layer(in_ch=48, out_ch=48, kernel=3, stride=1)
        # 48x128x128
        self.maxpool2 = maxpool()
        # 48x64x64
        self.conv3 = conv_layer(in_ch=48, out_ch=48, kernel=3, stride=1)
        # 48x64x64
        self.maxpool3 = maxpool()
        # 48x32x32
        self.conv4 = conv_layer(in_ch=48, out_ch=48, kernel=3, stride=1)
        # 48x32x32
        self.maxpool4 = maxpool()
        # 48x16x16
        self.conv5 = conv_layer(in_ch=48, out_ch=48, kernel=3, stride=1)
        # 48x16x16
        self.maxpool5 = maxpool()
        # 48x8x8
        self.conv6 = conv_layer(in_ch=48, out_ch=48, kernel=3, stride=1)
        # 48x8x8
        self.upsample5 = upsample()
        # 48x16x16
        # concat output of pool4 on channel dimension
        self.dec_conv5a = conv_layer(in_ch=96, out_ch=96, kernel=3, stride=1)
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
        output = self.maxpool1(output)
        concats.append(output)
        output = self.conv2(output)
        output = self.maxpool2(output)
        concats.append(output)
        output = self.conv3(output)
        output = self.maxpool3(output)
        concats.append(output)
        output = self.conv4(output)
        output = self.maxpool4(output)
        concats.append(output)
        output = self.conv5(output)
        output = self.maxpool5(output)
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


class RelativeMSE(nn.Module):
    def __init__(self):
        super(RelativeMSE, self).__init__()

    def forward(self, y, t):
        return torch.mean((y - t) ** 2 / ((y + 0.001) ** 2))
