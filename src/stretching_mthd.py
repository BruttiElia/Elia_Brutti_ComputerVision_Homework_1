import numpy as np

def stretching_canalBGR (image, a, b , PERCENTILE_LOW, PERCENTILE_HIGH): 

    """
    Stretch the pixel values of an BGR image to a specified range [a, b].
    This function applies linear contrast stretching to each channel independently.
    
    Parameters:
    - image: Input BGR image (numpy array).
    - a: Lower bound of the desired output range.
    - b: Upper bound of the desired output range.
    
    Returns:
    - Stretched BGR image (numpy array) with pixel values in the range [a, b].
    """
    # Ensure the input image is in float format for calculations
    image_float = image.astype(np.float32)
    
    # Initialize an empty array for the stretched image
    stretched_image = np.zeros_like(image_float)
    
    # Process each channel independently
    for channel in range(3):  # Assuming the image has 3 channels (R, G, B)
        channel_data = image_float[:, :, channel]
        
        # Find the minimum and maximum pixel values in the current channel
        min_val = np.percentile(channel_data, PERCENTILE_LOW) 
        max_val = np.percentile(channel_data, PERCENTILE_HIGH)  
        
        # Apply linear contrast stretching formula
        stretched_channel = (channel_data - min_val) * ((b - a) / (max_val - min_val)) + a
        
        # Clip values to ensure they are within [a, b]
        stretched_channel = np.clip(stretched_channel, a, b)
        
        # Assign the stretched channel back to the output image
        stretched_image[:, :, channel] = stretched_channel
    
    # Convert back to uint8 format for display or saving
    return stretched_image.astype(np.uint8)