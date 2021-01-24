import argparse
import numpy as np
import torch
import torchvision
from torch import optim
import os
from PIL import Image
import matplotlib.pyplot as plt
import random
import cv2

from models import AutoEncoder
from dataset import pepper_noise
from dataset import gaussian_noise

parser = argparse.ArgumentParser(description="Arguments parser")

parser.add_argument('--noise', dest='noise', action='store_true',
                    help='use this flag to add noise to the image before the generation')

parser.add_argument(
        "--img",
        default="",
        type=str,
        help="img path",
    )

parser.add_argument(
    "--model",
    default="",
    type=str,
    help="path for the model",
)

device = "cuda" if torch.cuda.is_available() else "cpu"

args = parser.parse_args()

NOISE = args.noise

IMG_PATH = args.img
to_tensor = torchvision.transforms.ToTensor()
img = np.array(Image.open(IMG_PATH))

h, w, c = img.shape

new_h = int(h / 32) * 32
new_w = int(w / 32) * 32

img = cv2.resize(img, (new_w, new_h))

g_min=0.08,
g_max=0.15,
p_min=0.1,
p_max=0.2,

if NOISE:
    img = pepper_noise(
            img, threshold=0.5, amount=p_max
        )
    img = gaussian_noise(
        img, amount=g_max
    )

img_tensor = to_tensor(img).unsqueeze(0)


MODEL_CHECKPOINT = args.model
gen = AutoEncoder()


if MODEL_CHECKPOINT:
    if os.path.isfile(MODEL_CHECKPOINT):
        print(
            "Loading checkpoint {} of the generator...".format(MODEL_CHECKPOINT)
        )
        checkpoint = torch.load(
            MODEL_CHECKPOINT, map_location=lambda storage, loc: storage
        )
        gen.load_state_dict(checkpoint["model_state_dict"])

        print("Generator correctly loaded.")
    else:
        print("Generator checkpoint filepath incorrect.")
        exit()

print(img_tensor.shape)
gen.eval()
generated = gen(img_tensor).squeeze()
print(generated.shape)

generated = generated.permute(1, 2, 0).detach().cpu().numpy()

plt.figure()
plt.imshow(img)
plt.show()

plt.figure()
plt.imshow(generated)
plt.show()