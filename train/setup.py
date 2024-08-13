import os
import shutil
import random
import pandas as pd
import yaml


def setup_directories(base_dir):
    """Create base directories for images and labels, and subdirectories for train and val."""
    dirs = ["images/train", "images/val", "labels/train", "labels/val"]
    for directory in dirs:
        dir_path = os.path.join(base_dir, directory)
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
        os.makedirs(dir_path, exist_ok=True)


def save_annotations_and_images(
    data_dir, pothole_ids, split_dir_images, split_dir_labels
):
    """Copy images and annotation files to the appropriate directories."""
    images_dir = os.path.join(data_dir, "train_images")
    annotations_dir = os.path.join(data_dir, "train_annotations")

    for pothole_id in pothole_ids:
        image_name = f"p{pothole_id}.jpg"
        annotation_name = f"p{pothole_id}.txt"

        # Copy image
        shutil.copy(
            os.path.join(images_dir, image_name),
            os.path.join(split_dir_images, image_name),
        )

        # Copy annotation
        shutil.copy(
            os.path.join(annotations_dir, annotation_name),
            os.path.join(split_dir_labels, annotation_name),
        )


def split_data(data_dir, train_ratio=0.85, val_ratio=0.15):
    """Split data into train and validation sets."""
    images_dir = os.path.join(data_dir, "train_images")
    annotations_dir = os.path.join(data_dir, "train_annotations")

    image_files = [
        f.split(".")[0][1:] for f in os.listdir(images_dir) if f.endswith(".jpg")
    ]
    annotation_files = [
        f.split(".")[0][1:] for f in os.listdir(annotations_dir) if f.endswith(".txt")
    ]

    # Find common files that have both image and annotation
    common_files = list(set(image_files).intersection(annotation_files))
    random.shuffle(common_files)

    # Split into training and validation sets
    train_end = int(train_ratio * len(common_files))
    train_files = common_files[:train_end]
    val_files = common_files[train_end:]

    return train_files, val_files


def prepare_dataset(
    data_dir,
    base_output_dir,
    train_ratio=0.85,
    val_ratio=0.15,
):
    """Main function to prepare dataset, split data, and create directories."""
    # Setup directories
    setup_directories(base_output_dir)

    # Split data into train and validation sets
    train_files, val_files = split_data(data_dir, train_ratio, val_ratio)

    # Save annotations and images
    save_annotations_and_images(
        data_dir,
        train_files,
        os.path.join(base_output_dir, "images/train"),
        os.path.join(base_output_dir, "labels/train"),
    )
    save_annotations_and_images(
        data_dir,
        val_files,
        os.path.join(base_output_dir, "images/val"),
        os.path.join(base_output_dir, "labels/val"),
    )

    # Create YAML file for YOLO
    yaml_path = os.path.join(base_output_dir, "data.yaml")
    create_yaml_file(yaml_path, base_output_dir)

    print(f"Data preparation complete. YAML file created at {yaml_path}")
    return yaml_path


def create_yaml_file(
    yaml_path, base_output_dir, nc=3, names=["pothole", "l1_stick", "l2_stick"]
):
    """Create a YAML configuration file for YOLO."""
    data = {
        "path": base_output_dir,
        "train": os.path.join(base_output_dir, "images/train"),
        "val": os.path.join(base_output_dir, "images/val"),
        "nc": nc,
        "names": names,
    }
    with open(yaml_path, "w") as file:
        yaml.dump(data, file, default_flow_style=False)


def check_pothole_completeness(data_dir, csv_file):
    # Load the CSV file
    csv_path = os.path.join(data_dir, csv_file)
    df = pd.read_csv(csv_path)

    # Create sets for images, annotations, and labels
    images_dir = os.path.join(data_dir, "train_images")
    annotations_dir = os.path.join(data_dir, "train_annotations")
    image_set = set(
        [f.split(".")[0][1:] for f in os.listdir(images_dir) if f.endswith(".jpg")]
    )
    annotation_set = set(
        [f.split(".")[0][1:] for f in os.listdir(annotations_dir) if f.endswith(".txt")]
    )
    label_set = set(df["Pothole number"].astype(str))

    # Initialize counters
    count_all_three = 0
    count_annotations_only = 0
    count_labels_only = 0

    # Union of all pothole IDs
    all_potholes = image_set.union(annotation_set).union(label_set)

    # Check each pothole and categorize based on presence of components
    for pothole in all_potholes:
        has_image = pothole in image_set
        has_annotation = pothole in annotation_set
        has_label = pothole in label_set

        if has_image and has_annotation and has_label:
            count_all_three += 1
        elif has_annotation and not has_image and not has_label:
            count_annotations_only += 1
        elif has_label and not has_image and not has_annotation:
            count_labels_only += 1

    # Output the results
    print(f"Total potholes with all three components: {count_all_three}")
    print(f"Total potholes with annotations only: {count_annotations_only}")
    print(f"Total potholes with labels only: {count_labels_only}")


# Example usage
if __name__ == "__main__":
    data_dir = "../data"
    csv_file = "train_labels.csv"
    base_output_dir = "data"  # Updated to avoid double nesting
    check_pothole_completeness(data_dir, csv_file)

    # Prepare dataset for YOLO fine-tuning
    yaml_path = prepare_dataset(data_dir, base_output_dir)
