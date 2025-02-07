import os
import numpy as np
import tifffile as tiff

def split_tiff(input_dir, output_dir, tile_size=(512, 512)):
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Get all TIFF files (both .tiff and .tif) in the input directory
    input_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.tiff', '.tif'))]

    all_split_images = []

    # Process each TIFF file in the input directory
    for input_file in input_files:
        input_path = os.path.join(input_dir, input_file)
        img = tiff.imread(input_path)

        # Check if the image has multiple channels or pages
        if img.ndim == 3:
            img_height, img_width, num_channels = img.shape
        elif img.ndim == 2:
            img_height, img_width = img.shape
            num_channels = 1
        else:
            raise ValueError("Unexpected image format.")

        # Split the image into smaller tiles
        for top in range(0, img_height, tile_size[0]):
            for left in range(0, img_width, tile_size[1]):
                # Calculate the bottom and right corners of the tile
                bottom = min(top + tile_size[0], img_height)
                right = min(left + tile_size[1], img_width)
                
                # Extract the tile from the image
                tile = img[top:bottom, left:right]
                
                # Save the tile as a new TIFF image in the output directory
                tile_filename = f"{os.path.splitext(input_file)[0]}_tile_{top}_{left}.tiff"
                tile_path = os.path.join(output_dir, tile_filename)
                tiff.imwrite(tile_path, tile)
                all_split_images.append(tile_path)
    
    return all_split_images

# Example usage
input_dir = '/home/greenbaumgpu/Reuben/js_annotation/input'
output_dir = '/home/greenbaumgpu/Reuben/js_annotation/images'
split_images = split_tiff(input_dir, output_dir)
print(f"Created {len(split_images)} tiles.")
