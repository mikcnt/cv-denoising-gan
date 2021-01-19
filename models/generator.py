import torchvision.transforms as tf
import torch.nn as nn
import torch.nn.functional as F
from utils.layer import conv_layer
from utils.layer import res_block
from utils.layer import deconv_layer

class Generator(nn.Module):
    def __init__(self, k0, k1, k2, k3, batch_size = 5):
        super(Generator, self).__init__()
        
        self.k0 = k0
        self.k1 = k1
        self.k2 = k2
        self.k3 = k3
        
        self.mse = nn.BCELoss()
        self.smooth = nn.SmoothL1Loss()
        
        self.conv1 = conv_layer(3, 32, 9, batch_size = batch_size)
        self.conv2 = conv_layer(32, 64, 3, batch_size = batch_size)
        self.conv3 = conv_layer(64, 128, 3, batch_size = batch_size)
        
        self.res1 = res_block(128, 3, batch_size = batch_size)
        self.res2 = res_block(128, 3, batch_size = batch_size)
        self.res3 = res_block(128, 3, batch_size = batch_size)
        
        self.deconv1 = deconv_layer(128, 64, 3, (128, 128), batch_size = batch_size)
        self.deconv2 = deconv_layer(64, 32, 3, (256, 256), batch_size = batch_size)
        self.deconv3 = deconv_layer(32, 3, 3, activation = nn.Tanh(), batch_size = batch_size)
        
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
    
    # k0 * Adversarial loss +
    # k1 * Pixel loss +
    # k2 * Feature loss +
    # k3 * Smooth loss
    def loss(self, adv_loss, y, t):
        pix_loss = self.mce(y, t)
        fea_loss = self.mce(y, t)
        return self.k0 * adv_loss + self.k1 * pix_loss + self.k2 * fea_loss + self.k3 * smo_loss
    
# Test
gen = Generator(0,0,0,0)