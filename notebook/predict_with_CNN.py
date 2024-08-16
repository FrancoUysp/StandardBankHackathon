import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image


to_predict = pd.DataFrame()

def get_annotations(filename):
    data = []
    with open(filename, 'r') as file:
        for line in file:
            # Split the line into its components
            class_id, x_center, y_center, width, height = line.strip().split()
            
            # Append the data to the list
            data.append({
                'image_name': 1,
                'class_id': int(class_id),
                'x_center': float(x_center),
                'y_center': float(y_center),
                'width': float(width),
                'height': float(height)
            })
    return data

def yolo_to_bbox(x_center, y_center, width, height, img_width, img_height):
    x_min = (x_center - width / 2) * img_width
    x_max = (x_center + width / 2) * img_width
    y_min = (y_center - height / 2) * img_height
    y_max = (y_center + height / 2) * img_height
    return x_min, y_min, x_max, y_max

# Calculate the diagonal length of the bounding box for `class_id` 1
def calculate_stick_size(row, img_width, img_height):
    if (row['class_id'] != 1):
        return 0
    x_min, y_min, x_max, y_max = yolo_to_bbox(
        row['x_center'], row['y_center'], row['width'], row['height'], img_width, img_height
    )
    return np.sqrt((x_max - x_min) ** 2 + (y_max - y_min) ** 2)

# Example function to load an image and get its dimensions
def get_image_size(image_path):
    with Image.open(image_path) as img:
        return img.size


to_predict = get_annotations("INSERT_TEXT_FILE.txt")
# Add a column with stick sizes
to_predict['stick_size'] = to_predict.apply(
    lambda row: calculate_stick_size(row, *get_image_size(f'../data/train_all/{row["img_file"]}')),
    axis=1)

to_predict['stick_size'] = to_predict.groupby('pothole_number')['stick_size'].transform(
    lambda x: x.replace(0, x.max())
)

# If stick_size is still zero, populate it with the mean of non-zero stick_size values
mean_stick_size = to_predict['stick_size'].replace(0, np.nan).mean()
to_predict['stick_size'] = to_predict['stick_size'].replace(0, mean_stick_size)

to_predict['scale_factor'] = (400 / to_predict['stick_size'])



def resize_image(image_path, scale_factor, output_path):
    with Image.open(image_path) as img:
        # Calculate new size
        width, height = img.size
        new_size = (int(width * scale_factor), int(height * scale_factor))
        
        # Resize and save image
        img_resized = img.resize(new_size, Image.LANCZOS)
        img_resized.save(output_path)  # Save resized image

def resize_and_pad_image(image_path, scale_factor, output_path, target_size=(512, 512), max_image_size=None):
    with Image.open(image_path) as img:

        width, height = img.size
        new_size = (int(width * scale_factor), int(height * scale_factor))
        
        # Resize and save image
        img = img.resize(new_size, Image.LANCZOS)
        # Calculate scaling factor based on the largest image
        if max_image_size is None:
            max_image_size = max(img.size)
        
        scale_factor = target_size[0] / max_image_size
        
        # Resize the image while keeping aspect ratio
        new_size = (int(img.width * scale_factor), int(img.height * scale_factor))
        img_resized = img.resize(new_size, Image.LANCZOS)
        
        # Create a new image with a black background
        new_img = Image.new("RGB", target_size, (0, 0, 0))
        
        # Calculate padding
        padding_x = (target_size[0] - new_size[0]) // 2
        padding_y = (target_size[1] - new_size[1]) // 2
        
        # Paste the resized image onto the black background
        new_img.paste(img_resized, (padding_x, padding_y))
        
        # Save the new image
        new_img.save(output_path)