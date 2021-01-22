import torchvision.transforms as tf
import torch.nn as nn
import torch.nn.functional as F


def shifted(img):
    pad = nn.ZeroPad2d((0, 1, 1, 0))
    return pad(img)[:, :, :-1, 1:]