
import numpy as np

def metric_mse(references, test):
    """
    Compute the Mean Squared Error (MSE) between two images.
    MSE is the average of the squared differences between pixel values.
    """
    if references.shape != test.shape:
        raise ValueError("Input images must have the same dimensions.")
    
    mse = np.mean((references.astype(np.float32) - test.astype(np.float32)) ** 2)
    return mse

def metric_psnr(references, test, max_val=255):
    """
    Compute the Peak Signal-to-Noise Ratio (PSNR) between two images.
    PSNR is a measure of the quality of reconstruction of an image.
    """
    mse = metric_mse(references, test)
    if mse == 0:
        return float('inf')
    psnr = 10 * np.log10((max_val ** 2) / mse)
    return psnr

