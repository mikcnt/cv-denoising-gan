import torchvision.transforms as tf
import torch.nn as nn
import torch.nn.functional as F


def shifted(img):
    """Shift image one unit down and one unit left.
    Zero-pad the top and right sides to keep the dimension intact.

    Args:
        img (Tensor): 4D mini-batch Tensor of shape (B x C x H x W).

    Returns:
        Tensor: Tensor image shifted.
    """
    pad = nn.ZeroPad2d((0, 1, 1, 0))
    return pad(img)[:, :, :-1, 1:]