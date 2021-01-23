import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision


def shifted(img):
    """Shift image one unit down and one unit left.
    Zero-pad the top and right sides to keep the dimension intact.

    Args:
        img (Tensor): 4D mini-batch Tensor of shape (B x C x H x W).

    Returns:
        Tensor: Tensor image shifted.
    """
    pad = nn.ZeroPad2d((0, 1, 1, 0))
    return pad(img)[:, :, :-1, 1:]


def load_checkpoint(model, optimizer, path):
    if os.path.isfile(path):
        checkpoint = torch.load(path, map_location=lambda storage, loc: storage)
        epoch = checkpoint["epoch"]
        model.load_state_dict(checkpoint["model_state_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        train_loss = checkpoint["train_loss"]
        test_loss = checkpoint["test_loss"]
    else:
        raise FileNotFoundError("Checkpoint '{}' doesn't exist.".format(path))

    return (model, optimizer, epoch, train_loss, test_loss)


def save_checkpoint(model, optimizer, epoch, train_loss, test_loss, path):
    model_checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "train_loss": train_loss,
        "test_loss": test_loss,
    }

    torch.save(model_checkpoint, path)


class DCGAN(object):
    def __init__(self, gen, opt_gen, criterion, gen_criterion, device):
        self.gen = gen
        self.opt_gen = opt_gen
        self.criterion = criterion
        self.gen_criterion = gen_criterion
        self.device = device
        self.saved_images = False
        self.ones = torch.tensor([])
        self.zeros = torch.tensor([])

    def train_gen(self, noise, real):
        fake = self.gen(noise)

        loss_gen = self.gen_criterion(fake, real)
        self.gen.zero_grad()
        loss_gen.backward()
        self.opt_gen.step()

        return loss_gen

    def train_iteration(self, noise, real):
        loss_gen = self.train_gen(noise, real)

        return loss_gen

    def train_epoch(self, train_loader, epoch):
        self.gen.train()
        epoch_g_loss = 0
        for noise, real in tqdm(train_loader, ncols=70, desc="Epoch {}".format(epoch)):
            noise = noise.to(self.device)
            real = real.to(self.device)

            loss_gen = self.train_iteration(noise, real)

            epoch_g_loss += loss_gen.item()

        return epoch_g_loss

    def evaluate_iteration(self, noise, real):
        fake = self.gen(noise)
        loss_gen = self.gen_criterion(fake, real)
        return loss_gen

    def evaluate_epoch(self, test_loader, epoch):
        noise_path = "outputs/noise.png"
        real_path = "outputs/real.png"
        gen_path = "outputs/{}_fake.png".format(str(epoch).zfill(3))
        gen_images = []
        noise_images = []
        real_images = []

        self.gen.eval()
        with torch.no_grad():
            test_d_loss = 0
            test_g_loss = 0
            for noise, real in test_loader:
                noise = noise.to(self.device)
                real = real.to(self.device)
                fake = self.gen(noise)

                loss_gen = self.evaluate_iteration(noise, real)

                test_g_loss += loss_gen.item()

                gen_images.append(fake)
                if not self.saved_images:
                    noise_images.append(noise)
                    real_images.append(real)

        gen_images = torch.cat(gen_images, 0)
        img_grid_fake = torchvision.utils.make_grid(gen_images, nrow=8)
        torchvision.utils.save_image(img_grid_fake, gen_path)

        if not self.saved_images:
            noise_images = torch.cat(noise_images, 0)
            real_images = torch.cat(real_images, 0)
            img_grid_noise = torchvision.utils.make_grid(noise_images, nrow=8)
            img_grid_real = torchvision.utils.make_grid(real_images, nrow=8)
            torchvision.utils.save_image(img_grid_noise, noise_path)
            torchvision.utils.save_image(img_grid_real, real_path)
            self.saved_images = True

        return test_d_loss, test_g_loss

    def save_checkpoints(self, epoch, gen_losses):
        gen_train_loss = gen_losses["train"]
        gen_test_loss = gen_losses["test"]

        gen_check_path = "checkpoints/generator/gen-{}.pth".format(str(epoch).zfill(3))

        save_checkpoint(
            self.gen, self.opt_gen, epoch, gen_train_loss, gen_test_loss, gen_check_path
        )
