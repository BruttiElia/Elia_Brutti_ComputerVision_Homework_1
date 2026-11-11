
"""
Quality metrics: MSE, PSNR and SSIM between a reference image and a test image.
"""

import numpy as np
from skimage.metrics import structural_similarity


def metric_ssim(reference, test):
    """
    Compute the Structural Similarity Index (SSIM) between two color images.
    SSIM is computed on local windows (Gaussian-weighted) and averaged over
    the image and over the three channels. Range: [-1, 1], higher is better (1 = identical).
    """
    if reference.shape != test.shape:
        raise ValueError("Input images must have the same dimensions.")

    return structural_similarity(
        reference, test,
        data_range=255,              # uint8 images: values go from 0 to 255
        channel_axis=2,              # channels are on the third axis (H, W, C)
        gaussian_weights=True,       # Gaussian window, as in the original SSIM paper
        sigma=1.5,                   # std of the Gaussian window (11x11 window)
        use_sample_covariance=False, # population covariance, as in the original paper
    )


def metric_mse(reference, test):
    """
    Compute the Mean Squared Error (MSE) between two images.
    MSE is the average of the squared differences between pixel values.
    """
    if reference.shape != test.shape:
        raise ValueError("Input images must have the same dimensions.")
    
    # float avoids uint8 wrap-around when subtracting (e.g. 10 - 20 = 246 in uint8)
    mse = np.mean((reference.astype(np.float32) - test.astype(np.float32)) ** 2)
    return mse

def metric_psnr(reference, test, max_val=255):
    """
    Compute the Peak Signal-to-Noise Ratio (PSNR) between two images, in dB:
    PSNR = 10 * log10(max_val^2 / MSE). Higher is better.
    Returns inf when the images are identical (MSE = 0).
    """
    mse = metric_mse(reference, test)
    if mse == 0:
        return float('inf')
    psnr = 10 * np.log10((max_val ** 2) / mse)
    return psnr
