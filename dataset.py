import os
import random

import cv2
import numpy as np
from skimage.color import rgb2lab
from torch.utils.data import Dataset

random.seed(42069)

# Utils
def pepper_noise(img, threshold=0.1, amount=0.5):
    h, w, _ = img.shape
    img_lab = rgb2lab(img)
    img_l = (
        img_lab[..., 0].reshape(h, w) / 100
    )  # Normalize the luminosity between 0 and 1
    rand_img = np.random.rand(h, w)
    thresh_img = img_l < threshold
    probability_mask = rand_img <= amount
    black_mask = thresh_img & probability_mask
    out = img.copy()
    out[black_mask, :] = 0
    return out

def gaussian_noise(img, amount=0.2, calibration=0.05):
    h, w, ch = img.shape
    noise = np.random.normal(0, amount, (h, w, ch)) - calibration
    out = img.copy() / 255
    out = out + noise
    out = np.clip(out, 0, 1)
    return out.astype(np.float32)


# A class defining the dataset
class ImageDataset(Dataset):
    def __init__(
        self,
        images_folder,
        g_min=0.08,
        g_max=0.15,
        p_min=0.1,
        p_max=0.2,
        transform=None,
    ):
        super().__init__()
        files = os.listdir(images_folder)
        self.image_paths = [images_folder + "/" + file for file in files if file.endswith(('.jpg', '.png'))]
        self.g_min = g_min
        self.g_max = g_max
        self.p_min = p_min
        self.p_max = p_max
        self.transform = transform

    # Returns the number of samples, it is used for iteration porpuses
    def __len__(self):
        return len(self.image_paths)

    # Returns a random sample for training(generally)
    def __getitem__(self, idx):
        # Load RANDOM clean image into memory...
        image_path = self.image_paths[idx]
        clean_image = cv2.imread(image_path)
        noisy_image = clean_image
        noisy_image = pepper_noise(
            noisy_image, threshold=0.5, amount=random.uniform(self.p_min, self.p_max)
        )
        noisy_image = gaussian_noise(
            noisy_image, amount=random.uniform(self.g_min, self.g_max)
        )

        noisy_image = cv2.cvtColor(noisy_image, cv2.COLOR_BGR2RGB)
        clean_image = cv2.cvtColor(clean_image, cv2.COLOR_BGR2RGB)

        if self.transform:
            noisy_image = self.transform(noisy_image)
            clean_image = self.transform(clean_image)

        return noisy_image, clean_image
