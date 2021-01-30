import os
from tqdm import tqdm
import torch
import torch.nn as nn
import torchvision

# If resume, load last checkpoint
# Otherwise, pass the complete path as parameter
class Checkpoint:
    def __init__(self, path, resume=False):
        self.path = path
        os.makedirs(path, exist_ok=True)
        self.resume = resume

    def load(self, model, optimizer, id_path=''):
        if (not self.resume) and id_path == '':
            raise RuntimeError()
        if self.resume:
            id_path = sorted(os.listdir(self.path))[-1]
        self.checkpoint = torch.load(os.path.join(self.path, id_path), map_location=lambda storage, loc: storage)
        if self.checkpoint == None:
            raise RuntimeError("Checkpoint empty!")
        epoch = self.checkpoint["epoch"]
        model.load_state_dict(self.checkpoint["model_state_dict"])
        optimizer.load_state_dict(self.checkpoint["optimizer_state_dict"])
        train_loss = self.checkpoint["train_loss"]
        test_loss = self.checkpoint["test_loss"]
        return (model, optimizer, epoch, train_loss, test_loss)

    def save(self, model, optimizer, epoch, train_loss, test_loss):
        model_checkpoint = {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epoch,
            "train_loss": train_loss,
            "test_loss": test_loss,
        }
        checkpoint_name = '{}.pth'.format(str(epoch).zfill(3))
        complete_path = os.path.join(self.path, checkpoint_name)
        torch.save(model_checkpoint, complete_path)
        return checkpoint_name
