import os
import random

import cv2
import numpy as np
from torch.utils.data import Dataset

from utils import noise

random.seed(42069)

# A class defining the dataset
class ImageDataset(Dataset):
    def __init__(
        self,
        images_folder,
        g_min=0.08,
        g_max=0.15,
        p_min=0.1,
        p_max=0.2,
        s_min=0.1,
        s_max=0.2,
        transform=None,
    ):
        super().__init__()
        files = os.listdir(images_folder)
        self.image_paths = [
            images_folder + "/" + file
            for file in files
            if file.endswith((".jpg", ".png"))
        ]
        self.g_min = g_min
        self.g_max = g_max
        self.p_min = p_min
        self.p_max = p_max
        self.s_min = s_min
        self.s_max = s_max
        self.transform = transform

    # Returns the number of samples, it is used for iteration porpuses
    def __len__(self):
        return len(self.image_paths)

    # Returns a random sample for training(generally)
    def __getitem__(self, idx):
        # Load RANDOM clean image into memory...
        image_path = self.image_paths[idx]
        clean_image = cv2.imread(image_path) / 255
        noisy_image = clean_image
        noisy_image = noise.pepper(
            noisy_image, threshold=0.5, amount=random.uniform(self.p_min, self.p_max)
        )
        noisy_image = noise.gaussian(
            noisy_image, amount=random.uniform(self.g_min, self.g_max)
        )
        noisy_image = noise.salt(
            noisy_image, amount=random.uniform(self.s_min, self.s_max)
        )

        noisy_image = (cv2.cvtColor(noisy_image, cv2.COLOR_BGR2RGB)).astype(np.float32)
        clean_image = (cv2.cvtColor(clean_image, cv2.COLOR_BGR2RGB)).astype(np.float32)

        if self.transform:
            noisy_image = self.transform(noisy_image)
            clean_image = self.transform(clean_image)

        return noisy_image, clean_image
