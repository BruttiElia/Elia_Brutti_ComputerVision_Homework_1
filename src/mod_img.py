
#  this module contains functions to modify images using OpenCV. 


import cv2
import numpy as np

ALPHA = 0.7
SIGMA = 10
SIZE = 256

def resize_image(image, size=SIZE):
    """
    Resize the image so that its shorter side equals `size`,
    then crop the center to obtain a size x size image.
    Preserves the aspect ratio (no distortion).
    """
    h, w = image.shape[:2]

    # scale factor: the shorter side becomes `size`
    scale = size / min(h, w)
    new_w = max(size, round(w * scale))
    new_h = max(size, round(h * scale))
    resized = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # center crop
    y0 = (new_h - size) // 2
    x0 = (new_w - size) // 2
    return resized[y0:y0 + size, x0:x0 + size]

def reducing_contrast(image, alpha=ALPHA):
    """
    Reduce the contrast of the image while preserving its mean brightness.
    alpha < 1 compresses pixel values toward the image mean.
    """
    mean = np.mean(image)
    beta = mean * (1 - alpha)
    reduced_contrast_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return reduced_contrast_image

def gaussian_noise(image, sigma=SIGMA):
    """
    Add zero-mean Gaussian noise to the image.
    sigma controls the noise intensity (standard deviation).
    """
    noise = np.random.normal(0, sigma, image.shape)
    noisy_image = image.astype(np.float32) + noise
    noisy_image = np.clip(noisy_image, 0, 255)
    return noisy_image.astype(np.uint8)

def modify_image(image):
    """
    Apply a series of modifications to the image: reduce contrast and add Gaussian noise.
    """
    reduced_contrast_image = reducing_contrast(image)
    final_image = gaussian_noise(reduced_contrast_image)
    
    return final_image
