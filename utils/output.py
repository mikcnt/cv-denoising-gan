import os
import torch
import torchvision


class Output:
    def __init__(self, path, num, overwrite=True):
        self.path = path
        self.path_is_file = True if self.path.endswith((".png", ".jpg")) else False
        if self.path_is_file:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
        else:
            os.makedirs(self.path, exist_ok=True)
        self.num = num
        self.overwrite = True
        self.imgs = []

    def append(self, img):
        if self.path_is_file and (not self.overwrite) and os.path.isfile(self.path):
            return False

        self.imgs.append(img)
        return True

    def save(self, filename="", imgs=[]):
        # If the filename is empty then we will be using the path as the destination
        # Otherwise we will append the filename to the path.
        destination = self.path if (not filename) else os.path.join(self.path, filename)
        # If the images are give through parameters, then we will save them instead.
        imgs = imgs if imgs != [] else self.imgs

        # Validation checks
        if self.path_is_file and filename:
            raise RuntimeError(
                "The image path is already given! No need to pass another file destination!"
            )
        if self.path_is_file and (not self.overwrite) and os.path.isfile(self.path):
            return False
        if (not self.overwrite) and os.path.isfile(destination):
            return False

        # Saving
        imgs = torch.cat(imgs, dim=0)[: self.num, ...]
        imgs_grid = torchvision.utils.make_grid(imgs, nrow=8)
        torchvision.utils.save_image(imgs_grid, destination)

        # Clear the cache of images
        self.imgs = []

        return True
