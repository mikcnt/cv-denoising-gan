from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm
import os

import torchvision.transforms as tf
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision import transforms

from models import Discriminator, Generator, GeneratorLoss
import dataset

# Create checkpoints directories
os.makedirs("checkpoints/discriminator", exist_ok=True)
os.makedirs("checkpoints/generator", exist_ok=True)

# Hyperparameters
NUM_EPOCHS = 100
DISC_LR = 0.1
GEN_LR = 0.1
BATCH_SIZE = 5

DISC_LOSS_FACTOR = 0.5
PIX_LOSS_FACTOR = 1
FEAT_LOSS_FACTOR = 1
SMOOTH_LOSS_FACTOR = 1


# Initialize vgg for feature loss
vgg = models.vgg16(pretrained=True).cuda().features[:3]

# Initialization of models
discriminator = Discriminator()
generator = Generator()

# Data
transform = transforms.Compose([
    transforms.ToTensor(),
])

train_dataset = dataset.ImageDataset("data/train", transform=transform)
test_dataset = dataset.ImageDataset("data/test", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Initialization of optimizers and losses
disc_opt = optim.Adam(discriminator.parameters(), lr=DISC_LR)
gen_opt = optim.Adam(generator.parameters(), lr=GEN_LR)

disc_criterion = nn.BCELoss()
gen_criterion = GeneratorLoss(
    vgg,
    DISC_LOSS_FACTOR,
    PIX_LOSS_FACTOR,
    FEAT_LOSS_FACTOR,
    SMOOTH_LOSS_FACTOR,
)

# Fit
for epoch in range(NUM_EPOCHS):
    for data, target in tqdm(train_loader):
        fake = generator(data)
        p_real = discriminator(data)
        p_fake = discriminator(fake)

        disc_loss = disc_criterion(p_real, 1 - p_fake)
        gen_loss = gen_criterion(disc_loss, fake, target)

        disc_loss.backward()
        gen_loss.backward()

        disc_opt.step()
        disc_opt.zero_grad()

        gen_opt.step()
        gen_opt.zero_grad()

    # Save checkpoints
    discriminator_checkpoint = {
        "model_state_dict": discriminator.state_dict(),
        "optimizer_state_dict": disc_opt.state_dict(),
        "epoch": epoch,
        "loss": disc_loss,
    }

    generator_checkpoint = {
        "model_state_dict": generator.state_dict(),
        "optimizer_state_dict": gen_opt.state_dict(),
        "epoch": epoch,
        "loss": gen_loss,
    }

    torch.save(discriminator_checkpoint, "checkpoints/discriminator")
    torch.save(generator_checkpoint, "checkpoints/generator")
