import numpy as np
import torch
import torch.optim as optim
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
    
    self.conv1 = nn.Sequential(
        nn.Conv2d(in_channels=3, out_channels=32, kernel_size=9, stride=1, padding=4),
        nn.BatchNorm2d(32),
        nn.LeakyReLU(0.1)
    )
    self.conv2 = nn.Sequential(
        nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(64),
        nn.LeakyReLU(0.1)
    )
    self.conv3 = nn.Sequential(
        nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(128),
        nn.LeakyReLU(0.1)
    )
    self.conv4 = nn.Sequential(
        nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(128),
        nn.LeakyReLU(0.1),
        nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(128),
        nn.LeakyReLU(0.1)
    )
    self.conv5 = nn.Sequential(
        nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(128),
        nn.LeakyReLU(0.1),
        nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(128),
        nn.LeakyReLU(0.1)
    )
    self.conv6 = nn.Sequential(
        nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(128),
        nn.LeakyReLU(0.1),
        nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
        nn.BatchNorm2d(128),
        nn.LeakyReLU(0.1)
    )
    self.dconv1 = nn.Sequential(
        # resize img
        nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size=3),
        nn.BatchNorm2d(64),
        nn.LeakyReLU(0.1)
    )
    self.dconv2 = nn.Sequential(
        # resize img
        nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=3),
        nn.BatchNorm2d(32),
        nn.LeakyReLU(0.1)
    )
    self.dconv3 = nn.Sequential(
        nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size=3),
        nn.BatchNorm2d(64),
        nn.Tanh()
    )