import os
from pathlib import Path
from ultralytics import YOLO
from PIL import Image

# Define directories
model_path = "../runs/detect/train/weights/best.pt"
images_dir = "../data/Test/test_images"
output_dir = "predictions"

# Create output subdirectories
images_output_dir = os.path.join(output_dir, "images")
annotations_output_dir = os.path.join(output_dir, "annotations")
os.makedirs(images_output_dir, exist_ok=True)
os.makedirs(annotations_output_dir, exist_ok=True)

# Load the YOLOv8 model
model = YOLO(model_path)

# Iterate over all JPG images in the images directory
for img_file in Path(images_dir).glob("*.jpg"):
    # Load the image
    img = Image.open(img_file)

    # Perform prediction
    results = model.predict(source=str(img_file), save=False)

    # Save the raw image (without bounding boxes)
    image_output_path = os.path.join(images_output_dir, img_file.name)
    img.save(image_output_path)

    # Save annotations in YOLO format
    annotation_output_path = os.path.join(
        annotations_output_dir, img_file.stem + ".txt"
    )

    # Track classes that have already been written
    seen_classes = set()

    with open(annotation_output_path, "w") as f:
        for detection in results[0].boxes.data.tolist():
            # YOLO format: class_id x_center y_center width height
            class_id = int(detection[5])  # Assuming the class ID is the last element

            # Skip this detection if we've already written this class
            if class_id in seen_classes:
                continue

            # Mark this class as seen
            seen_classes.add(class_id)

            # Calculate the YOLO bounding box format
            bbox = detection[:4]
            x_center = (bbox[0] + bbox[2]) / 2 / img.width
            y_center = (bbox[1] + bbox[3]) / 2 / img.height
            width = (bbox[2] - bbox[0]) / img.width
            height = (bbox[3] - bbox[1]) / img.height

            # Write the first instance of the class to the annotation file
            f.write(f"{class_id} {x_center} {y_center} {width} {height}\n")

print(f"Raw images and annotations saved in {output_dir}")
