import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch import optim
import torchvision

from models import Discriminator, Generator, GeneratorLoss
from dataset import ImageDataset
from parser import main_parser
from utils import load_checkpoint, save_checkpoint


def main():
    # Load arguments from the parser
    parser = main_parser()
    args = parser.parse_args()

    checkpoints_path = {
        "discriminator": "checkpoints/discriminator",
        "generator": "checkpoints/generator",
    }

    # Create checkpoints directories
    os.makedirs(checkpoints_path["discriminator"], exist_ok=True)
    os.makedirs(checkpoints_path["generator"], exist_ok=True)

    os.makedirs("outputs", exist_ok=True)

    # Hyperparameters
    RESUME_LAST = args.resume_last
    GENERATOR_CHECKPOINT = args.generator_checkpoint
    DISCRIMINATOR_CHECKPOINT = args.discriminator_checkpoint

    BATCH_SIZE = args.batch_size
    NUM_EPOCHS = args.epochs
    LEARNING_RATE = args.learning_rate
    DATA_PATH = args.data_path
    TRAIN_DATA_PATH = os.path.join(DATA_PATH, "train")
    TEST_DATA_PATH = os.path.join(DATA_PATH, "test")

    DISC_LOSS_FACTOR = 0.5
    PIX_LOSS_FACTOR = 1
    FEAT_LOSS_FACTOR = 1
    SMOOTH_LOSS_FACTOR = 1

    # Select device for training (gpu if available)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Starting epoch
    gen_epoch = 0

    # Initialize vgg for feature loss
    vgg = torchvision.models.vgg16(pretrained=True).features[:3].to(device)

    # Initialization of models
    disc = Discriminator().to(device)
    gen = Generator().to(device)

    # Instantiate losses
    disc_loss = {}
    gen_loss = {}

    test_disc_loss = {}
    test_gen_loss = {}

    # Load data
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor(),
        ]
    )

    train_dataset = ImageDataset(TRAIN_DATA_PATH, transform=transform)
    test_dataset = ImageDataset(TEST_DATA_PATH, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize optimizers and losses
    opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE)
    opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE)

    criterion = nn.BCELoss()
    gen_criterion = GeneratorLoss(
        vgg,
        DISC_LOSS_FACTOR,
        PIX_LOSS_FACTOR,
        FEAT_LOSS_FACTOR,
        SMOOTH_LOSS_FACTOR,
    )

    # Resume last checkpoints
    if RESUME_LAST:
        disc_checkpoints = sorted(os.listdir(checkpoints_path["discriminator"]))
        gen_checkpoints = sorted(os.listdir(checkpoints_path["generator"]))
        if disc_checkpoints and gen_checkpoints:
            DISCRIMINATOR_CHECKPOINT = os.path.join(
                checkpoints_path["discriminator"], disc_checkpoints[-1]
            )
            GENERATOR_CHECKPOINT = os.path.join(
                checkpoints_path["generator"], gen_checkpoints[-1]
            )

    # Resume specific checkpoints for generator and discriminator
    if DISCRIMINATOR_CHECKPOINT:
        disc, opt_disc, disc_epoch, disc_loss, test_disc_loss = load_checkpoint(
            disc, opt_disc, DISCRIMINATOR_CHECKPOINT
        )
        print("Finished loading discriminator checkpoint.")
        print("Resuming training from epoch {}.".format(disc_epoch))
        
    if GENERATOR_CHECKPOINT:
        gen, opt_gen, gen_epoch, gen_loss, test_gen_loss = load_checkpoint(
            gen, opt_gen, GENERATOR_CHECKPOINT
        )
        print("Finished loading generator checkpoint.")
        print("Resuming training from epoch {}.".format(gen_epoch))

    saved_images = False

    # Fit GAN
    for epoch in range(gen_epoch + 1, NUM_EPOCHS + 1):
        epoch_d_loss = 0
        epoch_g_loss = 0
        for noise, real in tqdm(
            train_loader, ncols=70, desc="Epoch {}/{}".format(epoch, NUM_EPOCHS)
        ):

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

            epoch_d_loss += loss_disc.item()
            epoch_g_loss += loss_gen.item()

        # Print losses
        print(
            f"Epoch [{epoch}/{NUM_EPOCHS}] Loss D: {epoch_d_loss:.4f}, loss G: {epoch_g_loss:.4f}"
        )

        # Update losses dictionaries
        disc_loss[epoch] = epoch_d_loss
        gen_loss[epoch] = epoch_g_loss

        # Validation
        print("Starting validation for epoch {}.".format(epoch))
        # Save outputs of the generator each epoch
        with torch.no_grad():
            gen_images = []
            test_d_loss = 0
            test_g_loss = 0
            for test_noise, test_real in test_loader:
                test_real = test_real.to(device)
                test_noise = test_noise.to(device)
                test_fake = gen(test_noise)
                gen_images.append(test_fake)

                # Losses for test
                disc_real = disc(test_real).reshape(-1)
                loss_disc_real = criterion(disc_real, torch.ones_like(disc_real))
                disc_fake = disc(test_fake.detach()).reshape(-1)
                loss_disc_fake = criterion(disc_fake, torch.zeros_like(disc_fake))
                loss_disc = (loss_disc_real + loss_disc_fake) / 2
                output = disc(test_fake).reshape(-1)
                adv_loss = criterion(output, torch.ones_like(output))
                loss_gen = gen_criterion(adv_loss, test_fake, test_real)

                test_d_loss += loss_disc
                test_g_loss += loss_gen

            gen_images = torch.cat(gen_images, 0)

            img_grid_fake = torchvision.utils.make_grid(gen_images, nrow=8)
            torchvision.utils.save_image(
                img_grid_fake, "outputs/{}_fake.png".format(str(epoch).zfill(3))
            )

            test_disc_loss[epoch] = test_d_loss
            test_gen_loss[epoch] = test_g_loss

        # Save real and noisy image during first epoch
        if not saved_images:
            noise_images = []
            real_images = []
            for test_noise, test_real in test_loader:
                real_images.append(test_real)
                noise_images.append(test_noise)
            noise_images = torch.cat(noise_images, 0)
            real_images = torch.cat(real_images, 0)

            img_grid_noise = torchvision.utils.make_grid(noise_images, nrow=8)
            img_grid_real = torchvision.utils.make_grid(real_images, nrow=8)

            torchvision.utils.save_image(img_grid_noise, "outputs/noise.png")
            torchvision.utils.save_image(img_grid_real, "outputs/real.png")

            saved_images = True

        # Save checkpoints

        disc_check_path = os.path.join(
            checkpoints_path["discriminator"],
            "disc-{}.pth".format(str(epoch).zfill(3)),
        )
        gen_check_path = os.path.join(
            checkpoints_path["generator"], "gen-{}.pth".format(str(epoch).zfill(3))
        )

        save_checkpoint(disc, opt_disc, epoch, disc_loss, test_d_loss, disc_check_path)
        save_checkpoint(gen, opt_gen, epoch, gen_loss, test_g_loss, gen_check_path)


if __name__ == "__main__":
    main()