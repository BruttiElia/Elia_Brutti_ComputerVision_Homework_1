"""
Method 3: histogram equalization on the V channel of HSV.
"""

import cv2


def equalization_HSV(image):
    """
    Equalize the histogram of the V (value) channel of a BGR image,
    leaving hue (H) and saturation (S) unchanged.
    V = max(B, G, R) does not follow perceived brightness, so saturated
    colors can change more than expected.

    Parameters:
    - image: input BGR image (uint8).

    Returns:
    - equalized BGR image (uint8).
    """
    # Convert BGR to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    # Equalize only V: spreads the brightness values over the full range [0, 255]
    hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])

    # Convert back to BGR
    equalized_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

    return equalized_image