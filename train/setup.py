import os
import shutil
import random
import pandas as pd
import yaml
import sys


def save_annotations_and_images(
    images_dir,
    annotations_dir,
    pothole_ids,
    split_dir_images,
    split_dir_labels,
    split_dir_annotations=None,
    class_to_modify=2,
    target_class=1,
):
    """Copy images and annotation files to the appropriate directories, modifying class type 2 to class type 1."""
    for pothole_id in pothole_ids:
        image_name = f"p{pothole_id}.jpg"
        annotation_name = f"p{pothole_id}.txt"

        # Copy image
        shutil.copy(
            os.path.join(images_dir, image_name),
            os.path.join(split_dir_images, image_name),
        )

        with open(os.path.join(annotations_dir, annotation_name), "r") as infile:
            lines = infile.readlines()

        # Modify lines corresponding to class_to_modify
        modified_lines = [
            line.replace(f"{class_to_modify} ", f"{target_class} ")
            if line.startswith(f"{class_to_modify} ")
            else line
            for line in lines
        ]

        # Save the modified annotations to the destination
        with open(os.path.join(split_dir_labels, annotation_name), "w") as outfile:
            outfile.writelines(modified_lines)

        if split_dir_annotations:
            shutil.copy(
                os.path.join(split_dir_labels, annotation_name),
                os.path.join(split_dir_annotations, annotation_name),
            )


def create_yaml_file(yaml_path, base_output_dir):
    """Create a YAML configuration file for YOLO."""
    data = {
        "path": os.path.abspath(base_output_dir),
        "train": "images/train",
        "val": "images/val",
        "names": {0: "pothole", 1: "l1_stick"},
    }
    with open(yaml_path, "w") as file:
        yaml.dump(data, file, default_flow_style=False)


def setup_directories(base_dir, pp_mode=False):
    """Create base directories for images, annotations, and labels."""
    dirs = (
        ["images", "annotations", "labels"]
        if pp_mode
        else [
            "images/train",
            "images/val",
            "images/test",
            "labels/train",
            "labels/val",
            "labels/test",
            "annotations/train",
            "annotations/val",
            "annotations/test",
        ]
    )

    for directory in dirs:
        dir_path = os.path.join(base_dir, directory)
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
        os.makedirs(dir_path, exist_ok=True)


def save_annotations_and_images_pp(
    images_dir, annotations_dir, labels_df, base_output_dir
):
    """Copy images, annotations, and labels to the appropriate directories in pp mode."""
    image_dest_dir = os.path.join(base_output_dir, "images")
    annotation_dest_dir = os.path.join(base_output_dir, "annotations")
    label_dest_dir = os.path.join(base_output_dir, "labels")

    # Copy all images
    for image_file in os.listdir(images_dir):
        if image_file.endswith(".jpg"):
            shutil.copy(
                os.path.join(images_dir, image_file),
                os.path.join(image_dest_dir, image_file),
            )

    # Copy all annotations
    for annotation_file in os.listdir(annotations_dir):
        if annotation_file.endswith(".txt"):
            shutil.copy(
                os.path.join(annotations_dir, annotation_file),
                os.path.join(annotation_dest_dir, annotation_file),
            )

    # Copy the CSV file to the labels directory
    labels_csv_dest = os.path.join(label_dest_dir, "labels.csv")
    shutil.copy(labels_df, labels_csv_dest)


def split_data(
    images_dir,
    annotations_dir,
    labels_df=None,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    mode="od",
    seed=None,
):
    """Split data into train, validation, and test sets based on mode."""
    if seed is not None:
        random.seed(seed)

    if mode == "pp":
        # No need to split, return empty lists as we don't perform any splitting in PP mode
        return [], [], []

    image_files = set(
        [f.split(".")[0][1:] for f in os.listdir(images_dir) if f.endswith(".jpg")]
    )
    annotation_files = set(
        [f.split(".")[0][1:] for f in os.listdir(annotations_dir) if f.endswith(".txt")]
    )

    if mode == "od":
        # In `od` mode, consider only images that have annotations
        common_files = list(image_files.intersection(annotation_files))
    elif mode == "pp":
        # In `pp` mode, consider only images that have both annotations and labels
        label_set = set(labels_df["Pothole number"].astype(str))
        common_files = list(
            image_files.intersection(annotation_files).intersection(label_set)
        )
    else:
        raise ValueError("Invalid mode. Use 'od' or 'pp'.")

    random.shuffle(common_files)

    # Split into training, validation, and test sets
    train_end = int(train_ratio * len(common_files))
    val_end = train_end + int(val_ratio * len(common_files))
    train_files = common_files[:train_end]
    val_files = common_files[train_end:val_end]
    test_files = common_files[val_end:]

    return train_files, val_files, test_files


def prepare_dataset(
    images_dir,
    annotations_dir,
    base_output_dir,
    labels_df=None,
    mode="od",
    seed=None,
):
    """Main function to prepare dataset and create directories."""
    pp_mode = mode == "pp"

    # Setup directories based on mode
    setup_directories(base_output_dir, pp_mode=pp_mode)

    if pp_mode:
        # For PP mode, copy everything to the respective directories without splitting
        save_annotations_and_images_pp(
            images_dir,
            annotations_dir,
            labels_df,  # Pass the path to the CSV file instead of the DataFrame
            base_output_dir,
        )
    else:
        # Split data into train, validation, and test sets
        train_files, val_files, test_files = split_data(
            images_dir,
            annotations_dir,
            labels_df,
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            mode=mode,
            seed=seed,
        )

        # Save annotations and images for train, validation, and test
        save_annotations_and_images(
            images_dir,
            annotations_dir,
            train_files,
            os.path.join(base_output_dir, "images/train"),
            os.path.join(base_output_dir, "labels/train"),
            os.path.join(base_output_dir, "annotations/train"),
        )
        save_annotations_and_images(
            images_dir,
            annotations_dir,
            val_files,
            os.path.join(base_output_dir, "images/val"),
            os.path.join(base_output_dir, "labels/val"),
            os.path.join(base_output_dir, "annotations/val"),
        )

        save_annotations_and_images(
            images_dir,
            annotations_dir,
            test_files,
            os.path.join(base_output_dir, "images/test"),
            os.path.join(base_output_dir, "labels/test"),
            os.path.join(base_output_dir, "annotations/test"),
        )

        # Create YAML file for YOLO only in `od` mode
        if mode == "od":
            yaml_path = os.path.join(os.path.dirname(base_output_dir), "data.yaml")
            create_yaml_file(yaml_path, base_output_dir)
            print(f"YAML file created at {yaml_path}")

    print(f"Data preparation complete. Files created in {base_output_dir}.")
    return


if __name__ == "__main__":
    arg = sys.argv[1]  # Expecting 'od' or 'pp'
    seed = 42  # Set your seed here for reproducibility

    base_dir = os.path.join("..", "data")  # Base directory for the data
    data_dir = os.path.join(base_dir, "Training Data")  # Path to the Training Data
    base_output_dir = "data"

    if arg == "od":
        # Define data directories
        images_dir = os.path.join(data_dir, "train_images")
        annotations_dir = os.path.join(data_dir, "train_annotations")

        # Prepare dataset for YOLO fine-tuning using directories
        prepare_dataset(
            images_dir,
            annotations_dir,
            base_output_dir,
            mode="od",
            seed=seed,
        )

    elif arg == "pp":
        # In PP mode, we will use the existing data sources to create the splits in a new 'data' directory

        # Define data directories
        images_dir = os.path.join(data_dir, "train_images")
        annotations_dir = os.path.join(data_dir, "train_annotations")
        labels_dir = os.path.join(
            data_dir, "train_labels.csv"
        )  # Corrected the path to look for labels.csv directly

        # Prepare dataset without train/val/test splits in `pp` mode
        prepare_dataset(
            images_dir=images_dir,
            annotations_dir=annotations_dir,
            base_output_dir=base_output_dir,  # Write splits to the 'data' directory
            labels_df=labels_dir,  # Pass the path of labels CSV file
            mode="pp",
            seed=seed,
        )

    else:
        print(
            "Invalid argument. Use 'od' for original dataset split or 'pp' for full dataset preparation without split."
        )
