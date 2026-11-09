import cv2

def denoising_bilateral(image, d, sigma_color, sigma_space):
    """
    Apply bilateral filtering to denoise the image.

    Parameters:
    - image: Input image (numpy array).
    - d: Diameter of each pixel neighborhood.
    - sigma_color: Filter sigma in color space.
    - sigma_space: Filter sigma in coordinate space.

    Returns:
    - Denoised image (numpy array).
    """
    denoised_image = cv2.bilateralFilter(image, d, sigma_color, sigma_space)
    return denoised_image

def denoising_non_local_means(image, h, template_window_size, search_window_size):
    """
    Apply Non-Local Means Denoising to the image.

    Parameters:
    - image: Input image (numpy array).
    - h: Parameter regulating filter strength.
    - template_window_size: Size of the template patch.
    - search_window_size: Size of the window used to compute weighted average.

    Returns:
    - Denoised image (numpy array).
    """
    denoised_image = cv2.fastNlMeansDenoisingColored(image, None, h, h, template_window_size, search_window_size)
    return denoised_image