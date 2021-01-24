import PySimpleGUI as sg
import numpy as np
import os
from models import AutoEncoder
import torch
import torchvision
from PIL import Image, ImageTk
import cv2

img_size = (350, 350)
img_box_size = (800, 350)
image_orig_str = "-IMAGE_ORIG-"
image_pred_str = "-IMAGE-PRED-"

device = "cuda" if torch.cuda.is_available() else "cpu"


def get_img(filename, img_size=img_size, noise=False):
    """ Generate png image from jpg """
    img = Image.open(filename).resize(img_size)
    return ImageTk.PhotoImage(img)


def get_img_prediction(model, pathname, img_size=img_size):
    to_tensor = torchvision.transforms.ToTensor()
    img = np.array(Image.open(pathname))
    h, w, _ = img.shape

    new_h = int(h / 32) * 32
    new_w = int(w / 32) * 32

    img = cv2.resize(img, (new_w, new_h))

    img_tensor = to_tensor(img).unsqueeze(0)
    model.eval()
    generated = model(img_tensor).squeeze()
    generated = generated.permute(1, 2, 0).detach().cpu().numpy()
    generated = np.clip(generated, 0, 1) * 255
    generated = generated.astype(np.uint8)
    img_pred = Image.fromarray(generated).resize(img_size)
    return ImageTk.PhotoImage(image=img_pred)


layout = [[sg.Text("Image Denoiser")]]

file_list_column = [
    [
        sg.Text("Select model"),
        sg.In(size=(25, 1), enable_events=True, key="-MODEL-", disabled=False),
        sg.FileBrowse(disabled=False, key="-MODEL_BROWSE-"),
    ],
    [
        sg.Text("Image Folder"),
        sg.In(size=(25, 1), enable_events=True, key="-FOLDER-"),
        sg.FolderBrowse(),
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

# ----- Full layout -----
layout = [
    [
        sg.Column(file_list_column),
        sg.VSeperator(),
        sg.Column(image_viewer_column_original),
        sg.Column(image_viewer_column_pred),
    ]
]

window = sg.Window("Image Viewer", layout)

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
        # a model has been selected
        # load model
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
    elif event == "-FILE LIST-":  # A file was chosen from the listbox
        # try:
        filename = os.path.join(values["-FOLDER-"], values["-FILE LIST-"][0])
        window[image_orig_str].update(data=get_img(filename))

        img_pred_tk = get_img_prediction(model, filename)
        window[image_pred_str].update(data=img_pred_tk)

window.close()
