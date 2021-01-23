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
from utils import load_checkpoint, save_checkpoint, DCGAN


def main():
    # Load arguments from the parser
    parser = main_parser()
    args = parser.parse_args()

    checkpoints_path = {
        "discriminator": "checkpoints/discriminator",
        "generator": "checkpoints/generator",
    }

    # Create checkpoints and outputs directories
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

    # Initialization of models
    disc = Discriminator().to(device)
    gen = Generator().to(device)

    # Instantiate losses
    disc_loss = {}
    gen_loss = {}

    test_disc_loss = {}
    test_gen_loss = {}

    # Load data
    transform = torchvision.transforms.ToTensor()

    train_dataset = ImageDataset(TRAIN_DATA_PATH, transform=transform)
    test_dataset = ImageDataset(TEST_DATA_PATH, transform=transform)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize optimizers and losses
    opt_disc = optim.Adam(disc.parameters(), lr=LEARNING_RATE)
    opt_gen = optim.Adam(gen.parameters(), lr=LEARNING_RATE)

    criterion = nn.BCELoss()
    gen_criterion = GeneratorLoss(
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

    gan = DCGAN(disc, gen, opt_disc, opt_gen, criterion, gen_criterion, device)

    # Fit GAN
    for epoch in range(gen_epoch + 1, NUM_EPOCHS + 1):
        # Train both generator and discriminator
        train_d_loss, train_g_loss = gan.train_epoch(train_loader, epoch)
        # Print losses and update dictionaries
        print(f"Loss D: {train_d_loss:.4f}, loss G: {train_g_loss:.4f}")

        disc_loss[epoch] = train_d_loss
        gen_loss[epoch] = train_g_loss

        # Validation and outputs
        print("Starting validation for epoch {}.".format(epoch))
        test_d_loss, test_g_loss = gan.evaluate_epoch(test_loader, epoch)

        test_disc_loss[epoch] = test_d_loss
        test_gen_loss[epoch] = test_g_loss

        # Save checkpoints
        disc_losses = {"train": disc_loss, "test": test_disc_loss}
        gen_lossess = {"train": gen_loss, "test": test_gen_loss}
        gan.save_checkpoints(epoch, disc_losses, gen_lossess)


if __name__ == "__main__":
    main()
