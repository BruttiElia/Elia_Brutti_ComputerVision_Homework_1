import cv2
import numpy as np


def equalization_HSV(image):
    # Convert BGR to HSV
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Equalize the V channel
    hsv[:, :, 2] = cv2.equalizeHist(hsv[:, :, 2])
    
    # Convert back to BGR
    equalized_image = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    return equalized_image