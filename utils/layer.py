import torchvision.transforms as tf
import torch.nn as nn
import torch.nn.functional as F

def conv_layer(
    in_ch,
    out_ch,
    kernel,
    activation = nn.LeakyReLU(),
    stride = 1,
    batch_size = 5):
    
    return nn.Sequential(
        nn.Conv2d(in_ch, out_ch, kernel, stride = stride),
        nn.BatchNorm2d(batch_size),
        activation
    )

def res_block(channels, kernel, batch_size = 5):
    return nn.Sequential(
        conv_layer(channels, channels, kernel, batch_size = batch_size),
        conv_layer(channels, channels, kernel, batch_size = batch_size)
    )

def deconv_layer(
    in_ch,
    out_ch,
    kernel,
    new_size = None,
    activation = nn.LeakyReLU(),
    batch_size = 5):
    
    if new_size:
        return nn.Sequential(
            tf.Resize(new_size, interpolation=3),
            conv_layer(in_ch, out_ch, kernel, activation, batch_size = batch_size)
        )
    return conv_layer(in_ch, out_ch, kernel, activation, batch_size = batch_size)