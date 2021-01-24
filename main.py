import os
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch import optim
import torchvision

from models import AutoEncoder, RelativeMSE
from dataset import ImageDataset
from parser import main_parser
from utils import load_checkpoint, save_checkpoint


def main():
    # Load arguments from the parser
    parser = main_parser()
    args = parser.parse_args()

    # Create checkpoints and outputs directories
    os.makedirs("checkpoints", exist_ok=True)

    os.makedirs("outputs", exist_ok=True)

    # Hyperparameters
    RESUME_LAST = args.resume_last
    MODEL_CHECKPOINT = args.model_checkpoint
    h, w = 256, 256
    VAL_IMAGES = 40
    BATCH_SIZE = args.batch_size
    NUM_EPOCHS = args.epochs
    LEARNING_RATE = args.learning_rate
    DATA_PATH = args.data_path
    TRAIN_DATA_PATH = os.path.join(DATA_PATH, "train")
    TEST_DATA_PATH = os.path.join(DATA_PATH, "test")

    # Select device for training (gpu if available)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Starting epoch
    epoch = 0

    # Initialization of models
    model = AutoEncoder().to(device)

    # Instantiate losses
    train_losses = {}

    test_losses = {}

    # Load data
    transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

    g_min = 0.25
    g_max = 0.3
    p_min = 0.35
    p_max = 0.5

    train_dataset = ImageDataset(
        TRAIN_DATA_PATH,
        transform=transform,
        g_min=g_min,
        g_max=g_max,
        p_min=p_min,
        p_max=p_max,
    )
    test_dataset = ImageDataset(
        TEST_DATA_PATH,
        transform=transform,
        g_min=g_max,
        g_max=g_max,
        p_min=p_max,
        p_max=p_max,
    )
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    # Initialize optimizers and losses
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

    criterion = nn.MSELoss()

    # Resume last checkpoints
    if RESUME_LAST:
        checkpoints = sorted(os.listdir("checkpoints"))
        if checkpoints:
            MODEL_CHECKPOINT = os.path.join("checkpoints", checkpoints[-1])

    # Resume specific checkpoint
    if MODEL_CHECKPOINT:
        model, optimizer, epoch, train_losses, test_losses = load_checkpoint(
            model, optimizer, MODEL_CHECKPOINT
        )
        print("Finished loading checkpoint.")
        print("Resuming training from epoch {}.".format(epoch))

    for epoch in range(epoch + 1, NUM_EPOCHS + 1):
        model.train()
        train_loss = 0
        test_loss = 0
        for x, t in tqdm(train_loader, ncols=70, desc="Epoch {}".format(epoch)):
            x = x.to(device)
            t = t.to(device)

            y = model(x)

            optimizer.zero_grad()
            loss = criterion(y, t)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        with torch.no_grad():
            noise_images = []
            clean_images = []
            gen_images = []
            num_batches = VAL_IMAGES // BATCH_SIZE + 1
            for batch_idx, (x, t) in enumerate(test_loader):
                x = x.to(device)
                t = t.to(device)
                y = model(x)

                loss = criterion(y, t)
                test_loss += loss.item()

                if batch_idx < num_batches:
                    noise_images.append(x)
                    clean_images.append(t)
                    gen_images.append(y)

        train_losses[epoch] = train_loss
        test_losses[epoch] = test_loss

        print("Train loss = {:.4f} \t Test loss = {:.4f}".format(train_loss, test_loss))

        noise_path = "outputs/noise.png"
        real_path = "outputs/real.png"
        gen_path = "outputs/{}_fake.png".format(str(epoch).zfill(3))

        noise_images = torch.cat(noise_images, dim=0)[:VAL_IMAGES, ...]
        clean_images = torch.cat(clean_images, dim=0)[:VAL_IMAGES, ...]
        gen_images = torch.cat(gen_images, dim=0)[:VAL_IMAGES, ...]

        img_grid_noise = torchvision.utils.make_grid(noise_images, nrow=8)
        img_grid_clean = torchvision.utils.make_grid(clean_images, nrow=8)
        img_grid_gen = torchvision.utils.make_grid(gen_images, nrow=8)

        torchvision.utils.save_image(img_grid_noise, noise_path)
        torchvision.utils.save_image(img_grid_clean, real_path)
        torchvision.utils.save_image(img_grid_gen, gen_path)

        check_path = "checkpoints/{}.pth".format(str(epoch).zfill(3))

        save_checkpoint(model, optimizer, epoch, train_losses, test_losses, check_path)


if __name__ == "__main__":
    main()
