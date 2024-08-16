import os
import shutil
import random
import pandas as pd
import yaml
import sys


def setup_directories(base_dir, split_dirs, include_test=False):
    """Create base directories for images, labels, and subdirectories for train, val, and test (if needed)."""
    dirs = []
    for split_dir in split_dirs:
        dirs.append(f"images/{split_dir}")
        dirs.append(f"labels/{split_dir}")
        if include_test:
            dirs.append(f"annotations/{split_dir}")

    for directory in dirs:
        dir_path = os.path.join(base_dir, directory)
        if os.path.exists(dir_path):
            shutil.rmtree(dir_path)
        os.makedirs(dir_path, exist_ok=True)


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
    split_dirs,
    train_ratio=0.7,
    val_ratio=0.15,
    test_ratio=0.15,
    include_annotations=False,
    labels_df=None,
    mode="od",
    seed=None,
):
    """Main function to prepare dataset, split data, and create directories."""
    # Setup directories
    setup_directories(base_output_dir, split_dirs, include_test=include_annotations)

    # Split data into train, validation, and test sets
    train_files, val_files, test_files = split_data(
        images_dir,
        annotations_dir,
        labels_df,
        train_ratio,
        val_ratio,
        test_ratio,
        mode,
        seed,
    )

    # Save annotations and images for train, validation, and test
    save_annotations_and_images(
        images_dir,
        annotations_dir,
        train_files,
        os.path.join(base_output_dir, "images/train"),
        os.path.join(base_output_dir, "labels/train"),
        os.path.join(base_output_dir, "annotations/train")
        if include_annotations
        else None,
    )
    save_annotations_and_images(
        images_dir,
        annotations_dir,
        val_files,
        os.path.join(base_output_dir, "images/val"),
        os.path.join(base_output_dir, "labels/val"),
        os.path.join(base_output_dir, "annotations/val")
        if include_annotations
        else None,
    )

    if include_annotations:
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

    print(f"Data preparation complete. Split files created in {base_output_dir}.")
    return


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
            ["train", "val"],
            train_ratio=0.85,
            val_ratio=0.15,
            test_ratio=0.0,  # No test set for `od` mode
            include_annotations=False,  # No separate annotations directory in `od` mode
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

        # Load the labels
        labels_df = pd.read_csv(labels_dir)

        # Prepare dataset with train, val, and test splits in `pp` mode
        prepare_dataset(
            images_dir=images_dir,
            annotations_dir=annotations_dir,
            base_output_dir=base_output_dir,  # Write splits to the 'data' directory
            split_dirs=["train", "val", "test"],
            train_ratio=0.7,
            val_ratio=0.15,
            test_ratio=0.15,
            include_annotations=True,  # Include annotations directory in `pp` mode
            labels_df=labels_df,
            mode="pp",
            seed=seed,
        )

    else:
        print(
            "Invalid argument. Use 'od' for original dataset split or 'pp' for train/val/test split."
        )
