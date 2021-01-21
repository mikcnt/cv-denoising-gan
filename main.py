from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm

import torchvision.transforms as tf
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

from models import Discriminator, Generator, GeneratorLoss
import dataset

# Hyperparameters
num_epochs = 100

# Initialize vgg for feature loss
vgg = models.vgg16(pretrained=True).cuda().features[:3]

# Initialization of models
discriminator = Discriminator()
generator = Generator()

disc_opt = optim.Adam(discriminator.parameters(), lr=0.01)
gen_opt = optim.Adam(generator.parameters(), lr=0.01)

train_dataset = dataset.ImageDataset("data/train")
test_dataset = dataset.ImageDataset("data/test")
train_loader = DataLoader(train_dataset, batch_size=5, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=5, shuffle=False)

# TODO: maybe we should be using custom losses for gen and disc
disc_criterion = nn.BCELoss()
gen_criterion = GeneratorLoss(
    vgg,
    disc_loss_factor=0.5,
    pix_loss_factor=1,
    feat_loss_factor=1,
    smooth_loss_factor=1,
)
for epoch in range(num_epochs):
    for data, target in tqdm(train_loader):
        fake = generator(data)
        p = discriminator(data)  # probability of classifying real data
        q = discriminator(fake)  # probability of misclassifying fake data

        disc_loss = disc_criterion(p, 1 - q)
        gen_loss = gen_criterion(disc_loss, fake, target)

        disc_loss.backward()
        gen_loss.backward()

        disc_opt.step()
        disc_opt.zero_grad()

        gen_opt.step()
        gen_opt.zero_grad()