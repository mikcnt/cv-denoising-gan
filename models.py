from tqdm import tqdm
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torchvision import models
import torchvision.transforms as tf

from utils.layer import conv_layer, deconv_layer, res_block

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv1 = conv_layer(3, 48, 4, stride=2, padding=1)
        self.conv2 = conv_layer(48, 96, 4, stride=2, padding=1)
        self.conv3 = conv_layer(96, 192, 4, stride=2, padding=1)
        self.conv4 = conv_layer(192, 384, 4)
        self.conv5 = conv_layer(384, 1, 4, activation=nn.Sigmoid())

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        x = self.conv4(x)
        x = self.conv5(x)
        return x


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()

        self.conv1 = conv_layer(3, 32, 9)
        self.conv2 = conv_layer(32, 64, 3)
        self.conv3 = conv_layer(64, 128, 3)

        self.res1 = res_block(128, 3)
        self.res2 = res_block(128, 3)
        self.res3 = res_block(128, 3)

        self.deconv1 = deconv_layer(128, 64, 3, (128, 128))
        self.deconv2 = deconv_layer(64, 32, 3, (256, 256))
        self.deconv3 = deconv_layer(32, 3, 3, activation=nn.Tanh())

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)

        x = x + self.res1(x)
        x = x + self.res2(x)
        x = x + self.res3(x)

        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        return x


class DiscriminatorLoss(nn.Module):
    def __init__(self):
        super(DiscriminatorLoss, self).__init__()
        self.criterion = nn.BCELoss()

    def forward(self, prediction_real, ones, prediction_fake, zeros):
        real_loss = self.criterion(prediction_real, ones)
        fake_loss = self.criterion(prediction_fake, zeros)
        return (real_loss + fake_loss) / 2


class GeneratorLoss(nn.Module):
    """Custom loss for the generator model of the GAN.
    This loss is composed by the weighted sum of 4 losses:
    1) Adversarial loss: loss of the discriminator.
    2) Pixel loss: MSE between generated image and groundtruth.
    3) Feature loss: MSE between VGG16 Conv2 layer of the generated image
    and VGG16 of the groundtruth.
    4) Smooth loss: MSE between generated image and one-unit (left and bottom)
    copy of the generated image.

    Args:
        vgg_model (nn.Module): VGG16 pretrained model used to compute feature spaces.
        disc_loss_factor (float): Weight for the adversarial (discriminator) loss.
        pix_loss_factor (float): Weight for the pixel loss.
        feat_loss_factor (float): Weight for the feature loss.
        smooth_loss_factor (float): Weight for the smooth loss.
    """

    def __init__(
        self,
        disc_loss_factor,
        pix_loss_factor,
        feat_loss_factor,
        smooth_loss_factor,
    ):
        device = "cuda" if torch.cuda.is_available() else "cpu"
        super(GeneratorLoss, self).__init__()
        self.vgg_model = (
            torchvision.models.vgg16(pretrained=True).features[:3].to(device)
        )
        self.disc_loss_factor = disc_loss_factor
        self.pix_loss_factor = pix_loss_factor
        self.feat_loss_factor = feat_loss_factor
        self.smooth_loss_factor = smooth_loss_factor
        self.vgg_model = models.vgg16(pretrained=True).features[:3].to(device)
        
        self.adversarial = nn.BCELoss()
        self.mse = nn.MSELoss()

    def features(self, x):
        self.vgg_model(x)

    def forward(self, fake_predictions, ones, y, t):
        features = self.vgg_model

        disc_loss = self.adversarial(fake_predictions, ones)
        pix_loss = self.mse(y, t)
        fea_loss = self.mse(features(y), features(t))
        smo_loss = self.mse(shifted(y), y)

        return (
            self.disc_loss_factor * disc_loss
            + self.pix_loss_factor * pix_loss
            + self.feat_loss_factor * fea_loss
            + self.smooth_loss_factor * smo_loss
        )


class DCGAN(object):
    def __init__(self, disc, gen, opt_disc, opt_gen, criterion, gen_criterion, device):
        self.disc = disc
        self.gen = gen
        self.opt_disc = opt_disc
        self.opt_gen = opt_gen
        self.criterion = criterion
        self.gen_criterion = gen_criterion
        self.device = device
        self.saved_images = False
        self.ones = torch.tensor([])
        self.zeros = torch.tensor([])

    def train_disc(self, noise, real):
        fake = self.gen(noise)
        disc_real = self.disc(real).reshape(-1)
        if self.ones.shape != disc_real.shape:
            self.ones = torch.ones_like(disc_real)
        loss_disc_real = self.criterion(disc_real, self.ones)

        disc_fake = self.disc(fake.detach()).reshape(-1)
        if self.zeros.shape != disc_fake.shape:
            self.zeros = torch.zeros_like(disc_fake)
        loss_disc_fake = self.criterion(disc_fake, self.zeros)

        loss_disc = (loss_disc_real + loss_disc_fake) / 2
        self.disc.zero_grad()
        loss_disc.backward()
        self.opt_disc.step()

        return loss_disc

    def train_gen(self, noise, real):
        fake = self.gen(noise)
        output = self.disc(fake).reshape(-1)
        if self.ones.shape != output.shape:
            self.ones = torch.ones_like(output)

        adv_loss = self.criterion(output, self.ones)
        loss_gen = self.gen_criterion(adv_loss, fake, real)
        self.gen.zero_grad()
        loss_gen.backward()
        self.opt_gen.step()

        return loss_gen

    def train_iteration(self, noise, real):
        loss_disc = self.train_disc(noise, real)
        loss_gen = self.train_gen(noise, real)

        return loss_disc, loss_gen

    def train_epoch(self, train_loader, epoch):
        self.gen.train()
        self.disc.train()
        epoch_d_loss = 0
        epoch_g_loss = 0
        for noise, real in tqdm(train_loader, ncols=70, desc="Epoch {}".format(epoch)):
            noise = noise.to(self.device)
            real = real.to(self.device)

            loss_disc, loss_gen = self.train_iteration(noise, real)

            epoch_d_loss += loss_disc.item()
            epoch_g_loss += loss_gen.item()

        return epoch_d_loss, epoch_g_loss

    def evaluate_iteration(self, noise, real):
        fake = self.gen(noise)

        disc_real = self.disc(real).reshape(-1)
        if self.ones.shape != disc_real.shape:
            self.ones = torch.ones_like(disc_real)
        loss_disc_real = self.criterion(disc_real, self.ones)
        disc_fake = self.disc(fake.detach()).reshape(-1)
        if self.zeros.shape != disc_fake.shape:
            self.zeros = torch.zeros_like(disc_fake)
        loss_disc_fake = self.criterion(disc_fake, self.zeros)
        loss_disc = (loss_disc_real + loss_disc_fake) / 2
        output = self.disc(fake).reshape(-1)
        adv_loss = self.criterion(output, self.ones)
        loss_gen = self.gen_criterion(adv_loss, fake, real)

        return loss_disc, loss_gen

    def evaluate_epoch(self, test_loader, epoch):
        noise_path = "outputs/noise.png"
        real_path = "outputs/real.png"
        gen_path = "outputs/{}_fake.png".format(str(epoch).zfill(3))
        gen_images = []
        noise_images = []
        real_images = []

        self.disc.eval()
        self.gen.eval()
        with torch.no_grad():
            test_d_loss = 0
            test_g_loss = 0
            for noise, real in test_loader:
                noise = noise.to(self.device)
                real = real.to(self.device)
                fake = self.gen(noise)

                loss_disc, loss_gen = self.evaluate_iteration(noise, real)

                test_d_loss += loss_disc.item()
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

    def save_checkpoints(self, epoch, disc_losses, gen_losses):
        disc_train_loss = disc_losses["train"]
        disc_test_loss = disc_losses["test"]
        gen_train_loss = gen_losses["train"]
        gen_test_loss = gen_losses["test"]

        disc_check_path = "checkpoints/discriminator/disc-{}.pth".format(
            str(epoch).zfill(3)
        )
        gen_check_path = "checkpoints/generator/gen-{}.pth".format(str(epoch).zfill(3))

        save_checkpoint(
            self.disc,
            self.opt_disc,
            epoch,
            disc_train_loss,
            disc_test_loss,
            disc_check_path,
        )
        save_checkpoint(
            self.gen, self.opt_gen, epoch, gen_train_loss, gen_test_loss, gen_check_path
        )


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