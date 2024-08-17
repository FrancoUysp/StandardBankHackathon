import tensorflow as tf
import numpy as np
import pandas as pd
from PIL import Image
import argparse
from tensorflow.keras.preprocessing.image import img_to_array, load_img
import random

def add_randomness_to_colors_tf(image):
    color_shift_range = tf.random.uniform([], 0, 15, dtype=tf.float32)

    # Convert the image to a float32 tensor and normalize the values to [0, 1]
    img_tensor = tf.image.convert_image_dtype(image, tf.float32)

    # Generate random shifts for each color channel
    random_shift = tf.random.uniform(tf.shape(img_tensor), -color_shift_range/255.0, color_shift_range/255.0)

    # Add the random shift and clip values to stay within valid range [0, 1]
    img_tensor = tf.clip_by_value(img_tensor + random_shift, 0.0, 1.0)

    # Convert back to the original data type
    img_tensor = tf.image.convert_image_dtype(img_tensor, tf.float32)

    return img_tensor

class Augment(tf.keras.layers.Layer):
    def __init__(self, contrast_range=[0.4, 1.5], 
                 brightness_delta=[-0.4, 0.3],
                 hue_delta=[-0.1, 0.1],
                 jpeg_qual = [40,100],
                 **kwargs):
        super(Augment, self).__init__(**kwargs)
        self.contrast_range = contrast_range
        self.brightness_delta = brightness_delta
        self.hue_delta = hue_delta
        self.jpeg_qual = jpeg_qual
    
    def ensure_rank_4(self, images):
        """Ensure that the images tensor has rank 4."""
        if len(images.shape) == 3:
            # If the image is rank 3 (height, width, channels), add a batch dimension
            images = tf.expand_dims(images, axis=0)
        elif len(images.shape) == 5:
            # If the image is rank 5 (batch_size, height, width, channels, extra), squeeze out the extra dimension
            images = tf.squeeze(images, axis=-1)
        return images
    
    def call(self, images, training=None):
        if not training:
            return images
        
        images = self.ensure_rank_4(images)

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

def resize_image(image_path, scale_factor, output_path):
    with Image.open(image_path) as img:
        # Calculate new size
        width, height = img.size
        new_size = (int(width * scale_factor), int(height * scale_factor))
        
        # Resize and save image
        img_resized = img.resize(new_size, Image.LANCZOS)
        img_resized.save(output_path)  # Save resized image

def resize_and_pad_image(image_path, scale_factor, target_size=(512, 512), max_image_size=None):
    with Image.open(image_path) as img:

        width, height = img.size
        new_size = (int(width * scale_factor), int(height * scale_factor))
        
        # Resize and save image
        img_resized = img.resize(new_size, Image.LANCZOS)
        # Calculate scaling factor based on the largest image
        scale_factor_res = target_size[0] / max_image_size
        
        # Resize the image while keeping aspect ratio
        new_size = (int(img_resized.width * scale_factor_res), int(img_resized.height * scale_factor_res))
        img_cropped = img_resized.resize(new_size, Image.LANCZOS)
        
        # Create a new image with a black background
        new_img = Image.new("RGB", target_size, (0, 0, 0))
        
        # Calculate padding
        padding_x = (target_size[0] - new_size[0]) // 2
        padding_y = (target_size[1] - new_size[1]) // 2
        
        # Paste the resized image onto the black background
        new_img.paste(img_cropped, (padding_x, padding_y))
        
        # Save the new image
        return new_img
def preprocess_image(image_path):
    img_width, img_height = 256, 256
    img = load_img(image_path, target_size=(img_width, img_height))
    img_array = img_to_array(img)
    img_array = img_array / 255.0  # Normalize to [0, 1]
    return img_array


def predict_CNN(image_path,text_path):

    to_predict = get_annotations(text_path)
    to_predict = pd.DataFrame(to_predict)
    # Add a column with stick sizes
    to_predict['stick_size'] = to_predict.apply(
        lambda row: calculate_stick_size(row, *get_image_size(image_path)),
        axis=1)

    to_predict['stick_size'] = to_predict['stick_size'].transform(
        lambda x: x.replace(0, x.max())
    )
    # If stick_size is still zero, populate it with the mean of non-zero stick_size values
    to_predict['stick_size'] = to_predict['stick_size'].replace(0, 300)

    scale_factor  = (400 / to_predict['stick_size'][0])
    
    # Resize Image and Pad black arounf the edges
    input_img = resize_and_pad_image(image_path,scale_factor,(512,512),2886)

    #Getting input ready for the model to predict
    img_width, img_height = 256, 256
    input_img_2 = input_img.resize((img_width, img_height))
    img_array = img_to_array(input_img_2)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    #Load the previously trained model
    loaded_model = tf.keras.models.load_model('../model_weights/CNN_padded_images.keras', custom_objects={'Augment': Augment})

    #Isolate the second to last layer and make it it's own layer
    second_to_last_layer = loaded_model.layers[-2].output
    new_model = tf.keras.Model(inputs=loaded_model.inputs, outputs=second_to_last_layer)

    #Predict with this to get the last layers output
    second_to_last_layer_output = new_model.predict(img_array)
    
    return second_to_last_layer_output


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(description="Resize and pad an image to a target size.")
    parser.add_argument("input_image", type=str, help="Path to the input image file.")
    parser.add_argument("input_text", type=str, help="Path to the input image file.")

    args = parser.parse_args()

    INPUT_IMAGE = args.input_image
    INPUT_TEXT = args.input_text
    predict_CNN(INPUT_IMAGE,INPUT_TEXT)

if __name__ == "__main__":
    main()
    