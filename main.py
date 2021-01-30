import os
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
import torchvision

from models import Generator, GeneratorLoss
from models import Discriminator, DiscriminatorLoss
from dataset import ImageDataset
from parser import main_parser
from utils.checkpoint import Checkpoint
from utils.output import Output


def main():
    # Load arguments from the parser
    parser = main_parser()
    args = parser.parse_args()

    # Hyperparameters
    RESUME_LAST = args.resume_last
    GENERATOR_CHECKPOINT = args.generator_checkpoint
    DISCRIMINATOR_CHECKPOINT = args.discriminator_checkpoint
    h, w = 256, 256
    VAL_IMAGES = 40
    BATCH_SIZE = args.batch_size
    NUM_EPOCHS = args.epochs
    LEARNING_RATE = args.learning_rate
    BETAS = (0.5, 0.999)
    DATA_PATH = args.data_path
    TRAIN_DATA_PATH = os.path.join(DATA_PATH, "train")
    TEST_DATA_PATH = os.path.join(DATA_PATH, "test")
    
    DISC_LOSS_FACTOR = 0.5
    PIX_LOSS_FACTOR = 1
    FEAT_LOSS_FACTOR = 1
    SMOOTH_LOSS_FACTOR = 0.0001
    
    # Select device for training (gpu if available)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Starting epoch
    epoch = 0

    noise_output = Output("outputs/noise.png", VAL_IMAGES, overwrite = False)
    clean_output = Output("outputs/clean.png", VAL_IMAGES, overwrite = False)
    gen_output = Output("outputs", VAL_IMAGES, overwrite = True)

    # Resume last checkpoints
    checkpoints_path = {
        "discriminator": "checkpoints/discriminator",
        "generator": "checkpoints/generator",
    }

    gen_path = checkpoints_path['generator']
    dis_path = checkpoints_path['discriminator']

    if GENERATOR_CHECKPOINT:
        id_gen_path = GENERATOR_CHECKPOINT
    else:
        id_gen_path = ''
    if DISCRIMINATOR_CHECKPOINT:
        id_dis_path = DISCRIMINATOR_CHECKPOINT
    else:
        id_dis_path = ''

    try:
        gen_checkpoint = Checkpoint(gen_path, RESUME_LAST)
        dis_checkpoint = Checkpoint(dis_path, RESUME_LAST)

        generator = Generator().to(device)
        discriminator = Discriminator().to(device)

        dis_opt = optim.Adam(discriminator.parameters(), lr=LEARNING_RATE, betas=BETAS)
        gen_opt = optim.Adam(generator.parameters(), lr=LEARNING_RATE, betas=BETAS)

        gen_criterion = GeneratorLoss(DISC_LOSS_FACTOR,
                                      PIX_LOSS_FACTOR,
                                      FEAT_LOSS_FACTOR,
                                      SMOOTH_LOSS_FACTOR)
        dis_criterion = DiscriminatorLoss()

        generator, gen_opt, epoch, gen_train_losses, gen_test_losses = gen_checkpoint.load(generator, gen_opt, id_gen_path)
        discriminator, dis_opt, epoch, dis_train_losses, dis_test_losses = dis_checkpoint.load(discriminator, dis_opt, id_dis_path)
        print("Models loaded from checkpoints.")
        print("Starting training from epoch {}.".format(epoch))
    except RuntimeError:
        print("No checkpoints, so the models are new!")
        gen_train_losses = {}
        gen_test_losses = {}
        dis_train_losses = {}
        dis_test_losses = {}

    # Load data
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    # Noise parameters
    g_min = 0.05
    g_max = 0.09
    p_min = 0.75
    p_max = 0.15
    s_min = 0.03
    s_max = 0.05

    train_dataset = ImageDataset(
        TRAIN_DATA_PATH,
        transform=transform,
        g_min=g_min,
        g_max=g_max,
        p_min=p_min,
        p_max=p_max,
        s_min=s_min,
        s_max=s_max,
    )
    test_dataset = ImageDataset(
        TEST_DATA_PATH,
        transform=transform,
        g_min=g_max,
        g_max=g_max,
        p_min=p_max,
        p_max=p_max,
        s_min=s_max,
        s_max=s_max,
    )

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    for epoch in range(epoch + 1, NUM_EPOCHS + 1):
        generator.train()
        discriminator.train()

        gen_train_loss_epoch = 0
        dis_train_loss_epoch = 0
        gen_test_loss_epoch = 0
        dis_test_loss_epoch = 0

        for noise, clean in tqdm(train_loader, ncols=70, desc="Epoch {}".format(epoch)):
            noise = noise.to(device)
            clean = clean.to(device)

            fake = generator(noise)
            prediction_real = discriminator(clean).reshape(-1)
            prediction_fake = discriminator(fake.detach()).reshape(-1)
            ones = torch.ones_like(prediction_real)
            zeros = torch.zeros_like(prediction_fake)

            # Train Discriminator
            dis_loss = dis_criterion(prediction_real, ones, prediction_fake, zeros)
            dis_opt.zero_grad()
            dis_loss.backward()
            dis_opt.step()

            # Train generator
            disc_fake_predictions = discriminator(fake).reshape(-1)
            ones_gen = torch.ones_like(disc_fake_predictions)
            gen_loss = gen_criterion(disc_fake_predictions, ones_gen, fake, clean)
            gen_opt.zero_grad()
            gen_loss.backward()
            gen_opt.step()

            # Update train losses of the epoch
            gen_train_loss_epoch += gen_loss.item()
            dis_train_loss_epoch += dis_loss.item()

        with torch.no_grad():
            num_batches = VAL_IMAGES // BATCH_SIZE + 1

            for batch_idx, (noise_test, clean_test) in enumerate(
                tqdm(test_loader, ncols=70, desc="Validation")
            ):
                noise_test = noise_test.to(device)
                clean_test = clean_test.to(device)
                fake_test = generator(noise_test)
                
                prediction_real_test = discriminator(clean_test)
                prediction_fake_test = discriminator(fake_test)

                ones_test = torch.ones_like(prediction_real_test)
                zeros_test = torch.zeros_like(prediction_fake_test)

                # Train Discriminator
                dis_loss_test = dis_criterion(prediction_real_test, ones_test, prediction_fake_test, zeros_test)

                # Train generator
                gen_loss_test = gen_criterion(prediction_fake_test, ones_test, noise_test, clean_test)

                # Update test losses of the epoch
                gen_test_loss_epoch += gen_loss_test.item()
                dis_test_loss_epoch += dis_loss_test.item()

                # Store images for visual feedbacks
                if batch_idx < num_batches:
                    noise_output.append(noise_test)
                    clean_output.append(clean_test)
                    gen_output.append(fake_test)

        # Store losses of the epoch in dictionaries
        gen_train_losses[epoch] = gen_train_loss_epoch
        dis_train_losses[epoch] = dis_train_loss_epoch
        gen_test_losses[epoch] = gen_test_loss_epoch
        dis_test_losses[epoch] = dis_test_loss_epoch

        print(
            "G. Train loss = {:.4f} \t D. Train loss = {:.4f}".format(
                gen_train_loss_epoch, dis_train_loss_epoch
            )
        )
        print(
            "G. Test loss = {:.4f} \t D. Test loss = {:.4f}".format(
                gen_test_loss_epoch, dis_test_loss_epoch
            )
        )

        # Save images
        noise_output.save()
        clean_output.save()
        gen_output.save(filename = "{}_fake.png".format(str(epoch).zfill(3)))
        
        # Save checkpoints
        gen_checkpoint.save(generator, gen_opt, epoch, gen_train_losses, gen_test_losses)
        dis_checkpoint.save(discriminator, dis_opt, epoch, dis_train_losses, dis_test_losses)


if __name__ == "__main__":
    main()
