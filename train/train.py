import os
import sys

# Add the parent directory of MiDaS to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
midas_dir = os.path.join(current_dir, 'MiDaS')
sys.path.append(midas_dir)

import cv2
import numpy as np
import torch
from torchvision.transforms import Compose
from ultralytics import YOLO
import matplotlib.pyplot as plt

# Commenting out MiDaS imports
# from MiDaS.midas.model_loader import load_model
# from MiDaS.midas.transforms import Resize, NormalizeImage, PrepareForNet

class PotholeYOLO:
    def __init__(self, yolo_model_path=None):
        self.model = None
        if yolo_model_path:
            self.load_model(yolo_model_path)

    def load_model(self, yolo_model_path):
        # Load the YOLOv8 model using the correct method
        self.model = YOLO(yolo_model_path)

    def fine_tune(self, data_yaml_path, epochs=500, project_path="./runs"):
        # Fine-tuning the YOLOv8 model on the custom dataset
        #
        self.model.add_callback("on_train_start", self.freeze_layer)
        self.model.train(data=data_yaml_path, epochs=epochs, project=project_path)
        print("Finished Model Training")

    def validate(self):
        # Validate the model and print results
        results = self.model.val()
        print(results)

    def predict(self, image_path):
        # Predicting bounding boxes for an image
        result = self.model.predict(image_path)
        return result

    def freeze_layer(self, trainer):
        model = trainer.model

        # Get a list of all layers
        layers = list(model.named_parameters())
        total_layers = len(layers)

        # Calculate the index after which we want to unfreeze the layers
        freeze_until = total_layers - (total_layers - 10)

        print(f"Freezing all but the last 2 layers")

        # Freeze all layers except the last 2
        for i, (k, v) in enumerate(layers):
            if i < freeze_until:
                v.requires_grad = False
                print(f"freezing {k}")
            else:
                v.requires_grad = True

        print(f"All but the last 2 layers are frozen.")

    def predict_and_show(self, image_path):
        # Predict bounding boxes for the image
        result = self.predict(image_path)
        image = cv2.imread(".jpg")

        # Draw bounding boxes on the image
        for box in result[0].boxes:
            x1, y1, x2, y2 = map(
                int, box.xyxy[0]
            )  # Assuming box.xyxy returns the bounding box coordinates
            confidence = box.conf[0]  # Confidence score
            label = (
                f"{box.cls[0]}: {confidence:.2f}"  # Class label and confidence score
            )
            cv2.rectangle(
                image, (x1, y1), (x2, y2), (0, 255, 0), 2
            )  # Green bounding box
            cv2.putText(
                image,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        # Show the image
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.show()

        # Display the image with bounding boxes
        img = result[0].plot()  # Plot the result
        plt.imshow(img)
        plt.axis("off")
        plt.show()

class PotholeFeatureExtractor:
    def __init__(self, yolo_model, midas_model_type="DPT_Large"):
        self.yolo_model = yolo_model
        
        # Commenting out MiDaS initialization
        # # Initialize MiDaS model
        # self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # midas_model_path = self.download_midas_model(midas_model_type)
        
        # # Load the MiDaS model
        # self.midas_model = load_model(midas_model_path)
        # self.midas_model.to(self.device)
        # self.midas_model.eval()
        
        # # MiDaS transform
        # self.midas_transform = Compose(
        #     [
        #         Resize(
        #             384,
        #             384,
        #             resize_target=None,
        #             keep_aspect_ratio=True,
        #             ensure_multiple_of=32,
        #             resize_method="upper_bound",
        #             image_interpolation_method=cv2.INTER_CUBIC,
        #         ),
        #         NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        #         PrepareForNet(),
        #     ]
        # )

    # Commenting out MiDaS download method
    # def download_midas_model(self, model_type):
    #     model_url = {
    #         "DPT_Large": "https://github.com/intel-isl/DPT/releases/download/1_0/dpt_large-midas-2f21e586.pt",
    #         "DPT_Hybrid": "https://github.com/intel-isl/DPT/releases/download/1_0/dpt_hybrid-midas-501f0c75.pt",
    #         "MiDaS_small": "https://github.com/AlexeyAB/MiDaS/releases/download/midas_dpt/midas_v21_small-70d6b9c8.pt"
    #     }
        
    #     if model_type not in model_url:
    #         raise ValueError(f"Invalid model type. Choose from {list(model_url.keys())}")

    #     model_path = os.path.join(os.path.dirname(__file__), "MiDaS", "weights", f"{model_type}.pt")
    #     os.makedirs(os.path.dirname(model_path), exist_ok=True)

    #     if not os.path.exists(model_path):
    #         print(f"Downloading {model_type} model...")
    #         torch.hub.download_url_to_file(model_url[model_type], model_path)
    #         print("Download completed.")

    #     return model_path

    def read_yolo_annotation(self, file_path):
        with open(file_path, 'r') as file:
            lines = file.readlines()
        annotations = []
        for line in lines:
            parts = line.strip().split()
            label = int(parts[0])
            x_center, y_center, width, height = map(float, parts[1:])
            annotations.append((label, x_center, y_center, width, height))
        return annotations

    def pixel_to_mm_conversion(self, annotations, image_width, image_height):
        l1_annotation = next((ann for ann in annotations if ann[0] == 1), None)
        
        if l1_annotation:
            _, _, _, width, height = l1_annotation
            pixel_width = width * image_width
            pixel_height = height * image_height
            
            # L1 true length is 500mm
            mm_per_pixel_width = 500 / pixel_width
            mm_per_pixel_height = 500 / pixel_height
            
            # Use the average of width and height conversion factors
            mm_per_pixel = (mm_per_pixel_width + mm_per_pixel_height) / 2
            
            return mm_per_pixel
        else:
            return None

    def calculate_area(self, annotations, image_width, image_height):
        areas = []
        for annotation in annotations:
            _, x_center, y_center, width, height = annotation
            x_min = (x_center - width / 2) * image_width
            x_max = (x_center + width / 2) * image_width
            y_min = (y_center - height / 2) * image_height
            y_max = (y_center + height / 2) * image_height
            area = (x_max - x_min) * (y_max - y_min)
            areas.append(area)
        return areas

    def extract_color_info(self, image, annotations):
        colors = []
        image_height, image_width, _ = image.shape
        for annotation in annotations:
            _, x_center, y_center, width, height = annotation
            x_min = int((x_center - width / 2) * image_width)
            x_max = int((x_center + width / 2) * image_width)
            y_min = int((y_center - height / 2) * image_height)
            y_max = int((y_center + height / 2) * image_height)
            roi = image[y_min:y_max, x_min:x_max]
            roi_mean = np.mean(roi, axis=(0, 1))
            colors.append(roi_mean)
        return colors

    # Commenting out depth estimation method
    # def estimate_depth(self, image):
    #     # Prepare image for MiDaS
    #     img = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
    #     img_input = self.midas_transform({"image": img})["image"]

    #     # Compute depth
    #     with torch.no_grad():
    #         sample = torch.from_numpy(img_input).to(self.device).unsqueeze(0)
    #         prediction = self.midas_model.forward(sample)
    #         prediction = torch.nn.functional.interpolate(
    #             prediction.unsqueeze(1),
    #             size=img.shape[:2],
    #             mode="bicubic",
    #             align_corners=False,
    #         ).squeeze()

    #     depth = prediction.cpu().numpy()
    #     return depth

    def extract_features(self, image_path, annotation_path=None):
        # Read image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Error: Unable to read image at {image_path}")
            return []
        
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_height, image_width, _ = image.shape
        
        print(f"Image shape: {image.shape}")

        # Use YOLO model to detect potholes with a lower confidence threshold
        results = self.yolo_model.predict(image_path, conf=0.25)  # Lower confidence threshold
        
        print(f"Number of detections: {len(results[0].boxes)}")

        features_list = []

        for result in results:
            for box in result.boxes:
                features = {}
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                width = x2 - x1
                height = y2 - y1
                confidence = box.conf.item()

                print(f"Detection: x1={x1}, y1={y1}, x2={x2}, y2={y2}, confidence={confidence:.2f}")

                features['pothole_width_pixels'] = width
                features['pothole_height_pixels'] = height
                features['pothole_area_pixels'] = width * height
                features['aspect_ratio'] = width / height
                features['relative_size'] = (width * height) / (image_width * image_height)
                features['confidence'] = confidence

                # Extract color information
                roi = image_rgb[y1:y2, x1:x2]
                avg_color = np.mean(roi, axis=(0, 1))
                features['avg_color_r'], features['avg_color_g'], features['avg_color_b'] = avg_color

                features_list.append(features)

        return features_list

    def extract_batch(self, image_dir):
        all_features = []
        for image_file in os.listdir(image_dir):
            if image_file.endswith(('.jpg', '.jpeg', '.png')):
                image_path = os.path.join(image_dir, image_file)
                features = self.extract_features(image_path)
                for feature in features:
                    feature['image_file'] = image_file
                all_features.extend(features)
        
        return all_features


# Usage example:
# extractor = PotholeFeatureExtractor('path/to/your/yolo/model.pt')
# features = extractor.extract_features('path/to/image.jpg', 'path/to/annotation.txt')
# print(features)

# For batch processing:
# all_features = extractor.extract_batch('path/to/image/directory', 'path/to/annotation/directory')
# import pandas as pd
# df = pd.DataFrame(all_features)
# print(df)

if __name__ == "__main__":
    # Print the current working directory
    print(f"Current working directory: {os.getcwd()}")

    # Get the directory of the current script
    current_dir = os.path.dirname(os.path.abspath(__file__))
    print(f"Script directory: {current_dir}")
    
    # Construct the full path to data.yaml
    data_yaml_path = os.path.join(current_dir, "data.yaml")
    
    # Print the path to verify
    print(f"Looking for data.yaml at: {data_yaml_path}")

    # Check if the file exists
    if not os.path.exists(data_yaml_path):
        print(f"Error: data.yaml not found at {data_yaml_path}")
        exit(1)

    # Usage example:
    yolo_model_path = os.path.join(current_dir, 'yolov8n.pt')
    yolo_model = PotholeYOLO(yolo_model_path)
    extractor = PotholeFeatureExtractor(yolo_model.model)
    
    image_path = os.path.join(current_dir, 'data', 'images', 'train', 'p101.jpg')
    annotation_path = os.path.join(current_dir, 'data', 'labels', 'train', 'p101.txt')
    
    if not os.path.exists(image_path):
        print(f"Error: Image not found at {image_path}")
        exit(1)
    
    if not os.path.exists(annotation_path):
        print(f"Error: Annotation not found at {annotation_path}")
        exit(1)
    
    features = extractor.extract_features(image_path, annotation_path)
    if not features:
        print("No features extracted. The model might not have detected any potholes.")
    else:
        print(f"Number of features extracted: {len(features)}")
        print(features)

    # Optionally, you can add this to visualize the detections:
    yolo_model.predict_and_show(image_path)
