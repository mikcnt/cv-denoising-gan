import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision


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