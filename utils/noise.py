from skimage.color import rgb2lab
import numpy as np

def pepper(img, threshold=0.1, amount=0.5):
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

def gaussian(img, amount=0.2, calibration=0.05):
    h, w, ch = img.shape
    noise = np.random.normal(0, amount, (h, w, ch)) - calibration
    out = img.copy() / 255
    out = out + noise
    out = np.clip(out, 0, 1)
    return out.astype(np.float32)
