import torchvision.transforms as tf
import torch.nn as nn
import torch.nn.functional as F
import dataset
from utils.layer import conv_layer

class Discriminator(nn.Module, batch_size = 5):
    def __init__(self):
        super(Discriminator, self).__init__()
        
        self.conv1 = conv_layer(3, 48, 4, stride = 2, batch_size = batch_size)
        self.conv2 = conv_layer(48, 96, 4, stride = 2, batch_size = batch_size)
        self.conv3 = conv_layer(96, 192, 4, stride = 2, batch_size = batch_size)
        self.conv4 = conv_layer(129, 384, 4, batch_size = batch_size)
        self.conv5 = conv_layer(384, 1, 4, activation = nn.Sigmoid(), batch_size = batch_size)
        
    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x