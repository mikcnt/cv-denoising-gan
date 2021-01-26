from skimage.color import rgb2lab
import numpy as np


def pepper(img, threshold=1, amount=0.5):
    h, w, _ = img.shape
    img_lab = rgb2lab(img)
    img_l = (
        img_lab[..., 0].reshape(h, w) / 100
    )  # Normalize the luminosity between 0 and 1
    rand_img = np.random.rand(h, w)
    thresh_img = img_l < threshold
    probability_mask = rand_img <= amount
    black_mask = thresh_img & probability_mask
    out = img

    luminosity = np.random.rand(*out.shape[:-1]) * 0.7

    luminosity = np.dstack([luminosity] * 3)
    out[black_mask, :] = np.multiply(out[black_mask, :], luminosity[black_mask, :])
    return out


def salt(img, amount=0.5):
    _range = 0.7
    h, w, _ = img.shape
    rand_img = np.random.rand(h, w)
    white_mask = rand_img <= amount
    out = img

    luminosity = (np.random.rand(*out.shape[:-1]) * _range) + 1 - _range

    luminosity = np.dstack([luminosity] * 3)
    out[white_mask, :] = out[white_mask, :] + luminosity[white_mask, :]
    return out


def gaussian(img, amount=0.2, calibration=0.05):
    h, w, ch = img.shape
    noise = np.random.normal(0, amount, (h, w, ch)) - calibration
    out = img
    out = out + noise
    out = np.clip(out, 0, 1)
    return out
