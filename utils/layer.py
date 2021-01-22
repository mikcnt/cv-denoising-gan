import torchvision.transforms as tf
import torch.nn as nn
import torch.nn.functional as F

def conv_layer(in_ch, out_ch, kernel, activation=nn.LeakyReLU(), stride=1, padding = None):

    if padding == None:
        padding = kernel // 2
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel, stride=stride, padding=padding),
        nn.BatchNorm2d(out_ch),
        activation,
    )


def res_block(channels, kernel):

    return nn.Sequential(
        conv_layer(channels, channels, kernel), conv_layer(channels, channels, kernel)
    )


def deconv_layer(in_ch, out_ch, kernel, new_size=None, activation=nn.LeakyReLU()):

    if new_size:
        return nn.Sequential(
            tf.Resize(new_size, interpolation=3),
            conv_layer(in_ch, out_ch, kernel, activation),
        )
    return conv_layer(in_ch, out_ch, kernel, activation)
