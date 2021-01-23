import argparse
import numpy as np
import torch
import torchvision
from torch import optim
import os
from PIL import Image
import matplotlib.pyplot as plt

from models import Generator

parser = argparse.ArgumentParser(description="Arguments parser")

parser.add_argument(
        "--img_path",
        default="",
        type=str,
        help="img path",
    )

parser.add_argument(
    "--model_path",
    default="",
    type=str,
    help="path for the generator model",
)

args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"

IMG_PATH = args.img_path
to_tensor = torchvision.transforms.ToTensor()
img = np.array(Image.open(IMG_PATH))

img_tensor = to_tensor(img).unsqueeze(0).to(device)


GENERATOR_CHECKPOINT = args.model_path
gen = Generator().to(device)


if GENERATOR_CHECKPOINT:
    if os.path.isfile(GENERATOR_CHECKPOINT):
        print(
            "Loading checkpoint {} of the generator...".format(GENERATOR_CHECKPOINT)
        )
        checkpoint = torch.load(
            GENERATOR_CHECKPOINT, map_location=lambda storage, loc: storage
        )
        gen.load_state_dict(checkpoint["model_state_dict"])

        print("Generator correctly loaded.")
    else:
        print("Generator checkpoint filepath incorrect.")
        exit()


generated = gen(img_tensor).reshape(3, 256, 256)
print(generated.shape)

generated = generated.permute(1, 2, 0).detach().cpu().numpy()

plt.figure()
plt.imshow(generated)
plt.show()