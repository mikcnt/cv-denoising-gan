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
        if os.path.isfile(DISCRIMINATOR_CHECKPOINT):
            checkpoint = torch.load(
                DISCRIMINATOR_CHECKPOINT, map_location=lambda storage, loc: storage
            )
            disc_epoch = checkpoint["epoch"]
            disc.load_state_dict(checkpoint["model_state_dict"])
            opt_disc.load_state_dict(checkpoint["optimizer_state_dict"])
            disc_loss = checkpoint["loss"]

            print("Finished loading discriminator checkpoint.")
            print(
                "Resuming training of the discriminator from epoch {}.".format(
                    disc_epoch
                )
            )
        else:
            print("Discriminator checkpoint filepath incorrect.")
            return

    if GENERATOR_CHECKPOINT:
        if os.path.isfile(GENERATOR_CHECKPOINT):
            print(
                "Loading checkpoint {} of the generator...".format(GENERATOR_CHECKPOINT)
            )
            checkpoint = torch.load(
                GENERATOR_CHECKPOINT, map_location=lambda storage, loc: storage
            )
            gen_epoch = checkpoint["epoch"]
            gen.load_state_dict(checkpoint["model_state_dict"])
            opt_gen.load_state_dict(checkpoint["optimizer_state_dict"])
            gen_loss = checkpoint["loss"]

            print("Finished loading generator checkpoint.")
            print("Resuming training of the generator from epoch {}.".format(gen_epoch))
        else:
            print("Generator checkpoint filepath incorrect.")
            return

    # Fit GAN
    for epoch in range(gen_epoch + 1, NUM_EPOCHS + 1):
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

        # Print losses
        print(
            f"Epoch [{epoch}/{NUM_EPOCHS}] Loss D: {loss_disc:.4f}, loss G: {loss_gen:.4f}"
        )

        # Update losses dictionaries
        disc_loss[epoch] = loss_disc
        gen_loss[epoch] = loss_gen

        # Save checkpoints
        discriminator_checkpoint = {
            "model_state_dict": disc.state_dict(),
            "optimizer_state_dict": opt_disc.state_dict(),
            "epoch": epoch,
            "loss": disc_loss,
        }

        generator_checkpoint = {
            "model_state_dict": gen.state_dict(),
            "optimizer_state_dict": opt_gen.state_dict(),
            "epoch": epoch,
            "loss": gen_loss,
        }

        disc_check_path = os.path.join(
            checkpoints_path["discriminator"],
            "disc-{}.pth".format(str(epoch).zfill(3)),
        )
        gen_check_path = os.path.join(
            checkpoints_path["generator"], "gen-{}.pth".format(str(epoch).zfill(3))
        )

        torch.save(discriminator_checkpoint, disc_check_path)
        torch.save(generator_checkpoint, gen_check_path)

        with torch.no_grad():
            real_images = []
            gen_images = []
            for test_noise, test_real in test_loader:
                test_noise = test_noise.to(device)
                test_real = test_real.to(device)
                real_images.append(test_real)
                gen_images.append(gen(test_noise))

            real_images = torch.cat(real_images, 0)
            gen_images = torch.cat(gen_images, 0)

            img_grid_fake = torchvision.utils.make_grid(gen_images, nrow=8)
            img_grid_real = torchvision.utils.make_grid(real_images, nrow=8)
            torchvision.utils.save_image(
                img_grid_fake, "outputs/{}_fake.png".format(str(epoch).zfill(3))
            )
            torchvision.utils.save_image(
                img_grid_real, "outputs/{}_real.png".format(str(epoch).zfill(3))
            )


if __name__ == "__main__":
    main()