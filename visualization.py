import os
import PySimpleGUI as sg
from PIL import Image, ImageTk
import numpy as np
import cv2
import torch
import torch.nn as nn
import torchvision

from models import AutoEncoder
import utils.noise as noise

img_size = (350, 350)
img_box_size = (800, 350)
image_orig_str = "-IMAGE_ORIG-"
image_pred_str = "-IMAGE-PRED-"


def get_img(path, noises):
    """ Generate png image from jpg """
    img = np.array(Image.open(path)) / 255
    img = noise.pepper(img, amount=noises["pepper"], threshold=1)
    img = noise.salt(img, amount=noises["salt"])
    img = noise.gaussian(img, amount=noises["gaussian"])
    return img.astype(np.float32)


def get_img_prediction(model, img):
    to_tensor = torchvision.transforms.ToTensor()
    h, w, _ = img.shape

    new_h = int(h / 32) * 32
    new_w = int(w / 32) * 32

    img = cv2.resize(img, (new_w, new_h))

    img_tensor = to_tensor(img).unsqueeze(0)
    model.eval()
    generated = model(img_tensor).squeeze()
    generated = generated.permute(1, 2, 0).detach().cpu().numpy()
    generated = np.clip(generated, 0, 1)
    return generated


def to_tk(img, img_size=img_size):
    img = (img * 255).astype(np.uint8)
    img = Image.fromarray(img).resize(img_size)
    return ImageTk.PhotoImage(img)


layout = [[sg.Text("Image Denoiser")]]

file_list_column = [
    [
        sg.Text("Select model"),
        sg.In(size=(25, 1), enable_events=True, key="-MODEL-", disabled=False),
        sg.FileBrowse(disabled=False, key="-MODEL_BROWSE-"),
    ],
    [
        sg.Text("Last activation"),
        sg.DropDown(
            ["Sigmoid", "Identity"],
            key="-ACTIVATION-",
            enable_events=True,
            disabled=True,
        ),
    ],
    [
        sg.Text("Image Folder"),
        sg.In(size=(25, 1), enable_events=True, key="-FOLDER-"),
        sg.FolderBrowse(),
    ],
    [
        sg.Text("Gaussian noise\t"),
        sg.Slider(
            range=(0.0, 1.0),
            default_value=0.0,
            resolution=0.01,
            orientation="horizontal",
            key="-GNOISE-",
        ),
    ],
    [
        sg.Text("Pepper noise\t"),
        sg.Slider(
            range=(0.0, 1.0),
            default_value=0.0,
            resolution=0.01,
            orientation="horizontal",
            key="-PNOISE-",
        ),
    ],
    [
        sg.Text("Salt noise\t"),
        sg.Slider(
            range=(0.0, 1.0),
            default_value=0.0,
            resolution=0.01,
            orientation="horizontal",
            key="-SNOISE-",
        ),
    ],
    [sg.Listbox(values=[], enable_events=True, size=(40, 20), key="-FILE LIST-")],
    [sg.Text("", key="-LOG-", size=(40, 2))],
]

image_viewer_column_original = [
    [sg.Text("Input image")],
    [sg.Image(size=img_size, key=image_orig_str)],
]

image_viewer_column_pred = [
    [sg.Text("Denoised image")],
    [sg.Image(size=img_size, key=image_pred_str)],
]

# Full layout
layout = [
    [
        sg.Column(file_list_column),
        sg.VSeperator(),
        sg.Column(image_viewer_column_original),
        sg.Column(image_viewer_column_pred),
    ]
]

window = sg.Window("Image Viewer", layout, font="Courier 12")

model_loaded = False
# Run the Event Loop
while True:
    event, values = window.read()
    if event == "Exit" or event == sg.WIN_CLOSED:
        break
    # Folder name was filled in, make a list of files in the folder
    if event == "-FOLDER-":
        folder = values["-FOLDER-"]
        try:
            # Get list of files in folder
            file_list = sorted(os.listdir(folder))
        except:
            file_list = []

        fnames = [
            f
            for f in file_list
            if os.path.isfile(os.path.join(folder, f))
            and f.lower().endswith((".jpg", ".png", ".gif"))
        ]
        window["-FILE LIST-"].update(fnames)
    elif event == "-MODEL-":
        checkpoint = values["-MODEL-"]
        if checkpoint == "":
            continue

        checkpoint = torch.load(checkpoint, map_location=lambda storage, loc: storage)
        model = AutoEncoder()
        try:
            model.load_state_dict(checkpoint["model_state_dict"])
            window["-LOG-"].update("Model correctly loaded.")
        except:
            window["-LOG-"].update("Error loading model.")

        window["-ACTIVATION-"].update(disabled=False)
        model_loaded = True
    elif event == "-FILE LIST-":  # A file was chosen from the listbox
        filename = os.path.join(values["-FOLDER-"], values["-FILE LIST-"][0])

        noises = {
            "gaussian": values["-GNOISE-"],
            "pepper": values["-PNOISE-"],
            "salt": values["-SNOISE-"],
        }

        img = get_img(filename, noises=noises)
        img_tk = to_tk(img)
        window[image_orig_str].update(data=img_tk)

        if model_loaded:
            img_pred = get_img_prediction(model, img)
            img_pred_tk = to_tk(img_pred)

            window[image_pred_str].update(data=img_pred_tk)
    elif event == "-ACTIVATION-":
        model.last_activation = (
            nn.Sigmoid() if values["-ACTIVATION-"] == "Sigmoid" else nn.Identity()
        )
        window["-LOG-"].update(
            "{} loaded as last activation function.".format(values["-ACTIVATION-"])
        )

window.close()
