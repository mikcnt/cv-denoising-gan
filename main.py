import torch
from torch.utils.data import DataLoader
from torch import optim
from tqdm import tqdm
import os

import torch.nn as nn
import torch.nn.functional as F
from torchvision import models
from torchvision import transforms

from models import Discriminator, Generator, GeneratorLoss
import dataset

# Create checkpoints directories
os.makedirs("checkpoints/discriminator", exist_ok=True)
os.makedirs("checkpoints/generator", exist_ok=True)
os.makedirs("outputs", exist_ok=True)

# Hyperparameters
NUM_EPOCHS = 100
DISC_LR = 0.1
GEN_LR = 0.1
BATCH_SIZE = 2

DISC_LOSS_FACTOR = 0.5
PIX_LOSS_FACTOR = 1
FEAT_LOSS_FACTOR = 1
SMOOTH_LOSS_FACTOR = 1

device = "cuda" if torch.cuda.is_available() else "cpu"


# Initialize vgg for feature loss
vgg = models.vgg16(pretrained=True).cuda().features[:3].to(device)

# Initialization of models
disc = Discriminator().to(device)
gen = Generator().to(device)

# Data
transform = transforms.Compose(
    [
        transforms.ToTensor(),
    ]
)

train_dataset = dataset.ImageDataset("data/train", transform=transform)
test_dataset = dataset.ImageDataset("data/test", transform=transform)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Initialization of optimizers and losses
opt_disc = optim.Adam(disc.parameters(), lr=DISC_LR)
opt_gen = optim.Adam(gen.parameters(), lr=GEN_LR)

criterion = nn.BCELoss()
gen_criterion = GeneratorLoss(
    vgg,
    DISC_LOSS_FACTOR,
    PIX_LOSS_FACTOR,
    FEAT_LOSS_FACTOR,
    SMOOTH_LOSS_FACTOR,
)


# Fit
for epoch in range(NUM_EPOCHS):
    for batch_idx, (noise, real) in enumerate(tqdm(train_loader)):

        noise = noise.to(device)
        real = real.to(device)
        fake = gen(noise)

        # Train Discriminator: max log(D(x)) + log(1 - D(G(z)))
        disc_real = disc(real).reshape(-1)
        loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
        disc_fake = disc(fake.detach()).reshape(-1)
        loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
        loss_disc = (loss_disc_real + loss_disc_fake) / 2
        disc.zero_grad()
        loss_disc.backward()
        opt_disc.step()

        # Train Generator: min log(1 - D(G(z))) <-> max log(D(G(z))
        output = disc(fake).reshape(-1)
        adv_loss = criterion(output, torch.ones_like(output))

        loss_gen = gen_criterion(adv_loss, fake, real)
        gen.zero_grad()
        loss_gen.backward()
        opt_gen.step()

    # Print losses
    print(
        f"Epoch [{epoch}/{NUM_EPOCHS}] Loss D: {loss_disc:.4f}, loss G: {loss_gen:.4f}"
    )

    with torch.no_grad():
        real_images = []
        gen_images = []
        for test_noise, test_real in test_loader:
            real_images = real_images.append(test_real)
            gen_images = gen_images.append(gen(test_noise))

        real_images = torch.cat(real_images, 0)
        gen_images = torch.cat(gen_images, 0)

        img_grid_fake = torchvision.utils.make_grid(gen_images, nrow=8)
        img_grid_real = torchvision.utils.make_grid(real_images, nrow=8)
        torchvision.utils.save_image(img_grid_fake, "outputs/fake_{}.png".format(epoch))
        torchvision.utils.save_image(img_grid_real, "outputs/real_{}.png".format(epoch))

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