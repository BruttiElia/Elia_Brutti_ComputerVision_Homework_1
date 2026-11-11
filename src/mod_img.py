
"""
Data preparation: resizing and synthetic degradation of the input images
(contrast reduction + Gaussian noise).
"""

import cv2
import numpy as np


def resize_image(image, size):
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

def reducing_contrast(image, alpha):
    """
    Reduce the contrast of the image while preserving its mean brightness:
    out = mean + alpha * (I - mean), with the global mean over all channels.
    alpha < 1 compresses pixel values toward the mean.
    """
    mean = np.mean(image)
    beta = mean * (1 - alpha)
    reduced_contrast_image = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return reduced_contrast_image

def gaussian_noise(image, sigma):
    """
    Add zero-mean Gaussian noise to the image.
    sigma controls the noise intensity (standard deviation).
    """
    noise = np.random.normal(0, sigma, image.shape)   # independent noise per pixel and channel
    noisy_image = image.astype(np.float32) + noise    # float avoids uint8 overflow/wrap-around
    noisy_image = np.clip(noisy_image, 0, 255)        # back to the valid range
    return noisy_image.astype(np.uint8)

def modify_image(image, alpha, sigma):
    """
    Synthetic degradation: first reduce the contrast, then add Gaussian noise.
    """
    reduced_contrast_image = reducing_contrast(image, alpha)
    final_image = gaussian_noise(reduced_contrast_image, sigma)
    
    return final_image
