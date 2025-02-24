import cv2
import numpy as np
import os

# Function to process images
def convert_to_16bit(input_dir, output_dir, filename):
    input_path = os.path.join(input_dir, filename)
    output_path = os.path.join(output_dir, filename)

    # Read the image
    img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)

    if img is None:
        print(f"Error: Could not read image {input_path}.")
        return

    # Check if the image is already 16-bit
    if img.dtype != np.uint16:
        # Convert to 16-bit by scaling 8-bit values
        img = np.uint16(img) * 256  # Scale values to 16-bit range

    # Save the image to the output directory
    cv2.imwrite(output_path, img)
    print(f"Processed and saved: {output_path}")

# Main logic
def process_images(input_dir, output_dir):
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Process each file in the input directory
    for filename in os.listdir(input_dir):
        if filename.endswith('.tif') or filename.endswith('.tiff'):  # Process only TIFF files
            convert_to_16bit(input_dir, output_dir, filename)

# Example usage
input_directory = '/home/greenbaumgpu/Reuben/js_annotation/uploads'
output_directory = '/home/greenbaumgpu/Reuben/js_annotation/input'

process_images(input_directory, output_directory)

