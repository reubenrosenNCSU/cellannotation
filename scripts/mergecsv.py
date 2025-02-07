import os
import csv

def adjust_bounding_boxes(input_csv_dir, output_csv_file, tile_size=(512, 512)):
    # Ensure the output CSV directory exists
    os.makedirs(os.path.dirname(output_csv_file), exist_ok=True)
    
    # Collect all CSV files in the input directory
    input_csv_files = [f for f in os.listdir(input_csv_dir) if f.lower().endswith('.csv')]
    
    # List to store all adjusted annotations
    adjusted_annotations = []

    # To calculate full image dimensions, we need to know the number of tiles in the x and y directions
    # Assuming all tiles are the same size and cover the entire image
    image_width = 0
    image_height = 0

    for input_csv_file in input_csv_files:
        input_csv_path = os.path.join(input_csv_dir, input_csv_file)
        
        # Read the CSV file
        with open(input_csv_path, mode='r') as file:
            reader = csv.reader(file)
            for row in reader:
                # Extract the tile filename and bounding box coordinates
                tile_filename = row[0]
                x_min = float(row[1])
                y_min = float(row[2])
                x_max = float(row[3])
                y_max = float(row[4])
                
                # Extract the tile position (top, left) from the filename
                filename_parts = tile_filename.split('_')
                top = int(filename_parts[-2])  # Extract the 'top' value from the filename
                left = int(filename_parts[-1].split('.')[0])  # Extract the 'left' value from the filename
                
                # Track the maximum width and height to calculate full image dimensions
                image_width = max(image_width, left + tile_size[1])
                image_height = max(image_height, top + tile_size[0])
                
                # Adjust bounding box coordinates based on the tile's top and left position
                adjusted_x_min = x_min + left
                adjusted_y_min = y_min + top
                adjusted_x_max = x_max + left
                adjusted_y_max = y_max + top
                
                # Append the adjusted annotation with the full image filename
                adjusted_annotations.append(['detections.png', adjusted_x_min, adjusted_y_min, adjusted_x_max, adjusted_y_max] + row[5:])
    
    # Write the adjusted annotations to a new CSV file
    with open(output_csv_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        #writer.writerow(['filename', 'x_min', 'y_min', 'x_max', 'y_max', 'class', 'confidence', 'area', 'class_id_number'])
        writer.writerows(adjusted_annotations)

    # Optionally, print out the full image dimensions
    print(f"Full image dimensions: {image_width} x {image_height}")
    return image_width, image_height

# Example usage
input_csv_dir = '/home/greenbaumgpu/Reuben/js_annotation/output/output_csv'  # Path to the folder containing your CSV files
output_csv_file = '/home/greenbaumgpu/Reuben/js_annotation/finaloutput/annotations.csv'  # Path for the output merged CSV file
tile_size = (512, 512)  # The size of each tile (you can adjust this)
image_width, image_height = adjust_bounding_boxes(input_csv_dir, output_csv_file, tile_size)
print(f"Adjusted annotations saved to {output_csv_file}")