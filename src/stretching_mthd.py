
"""
Method 2: linear contrast stretching, applied to each BGR channel separately.
"""

import numpy as np

def stretching_canalBGR (image, a, b , PERCENTILE_LOW, PERCENTILE_HIGH): 

    """
    Stretch the pixel values of a BGR image to a specified range [a, b].
    This function applies linear contrast stretching to each channel independently.
    
    Parameters:
    - image: Input BGR image (numpy array).
    - a: Lower bound of the desired output range.
    - b: Upper bound of the desired output range.
    - PERCENTILE_LOW: percentile used as the lower limit of the input range (e.g. 0.5).
    - PERCENTILE_HIGH: percentile used as the upper limit of the input range (e.g. 99.5).
    
    Returns:
    - Stretched BGR image (numpy array) with pixel values in the range [a, b].
    """
    # Ensure the input image is in float format for calculations
    image_float = image.astype(np.float32)
    
    # Initialize an empty array for the stretched image
    stretched_image = np.zeros_like(image_float)
    
    # Process each channel independently
    for channel in range(3):  # Assuming the image has 3 channels (B,G,R)
        channel_data = image_float[:, :, channel]

        # Robust min/max: percentiles ignore a few extreme pixels (e.g. noise outliers)
        min_val = np.percentile(channel_data, PERCENTILE_LOW) 
        max_val = np.percentile(channel_data, PERCENTILE_HIGH)  
        
        # Apply linear contrast stretching formula
        # out = (I - low) * (b - a) / (high - low) + a
        stretched_channel = (channel_data - min_val) * ((b - a) / (max_val - min_val)) + a
        
        # Clip values to ensure they are within [a, b]
        stretched_channel = np.clip(stretched_channel, a, b)
        
        # Assign the stretched channel back to the output image
        stretched_image[:, :, channel] = stretched_channel
    
    # Convert back to uint8 format for display or saving
    return stretched_image.astype(np.uint8)