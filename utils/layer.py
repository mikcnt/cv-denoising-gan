from torch import nn
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
        # nn.BatchNorm2d(out_ch),
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
