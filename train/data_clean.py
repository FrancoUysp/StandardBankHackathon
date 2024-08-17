import os
import pandas as pd


def check_data_integrity(images_dir, annotations_dir, labels_csv_path):
    # Set up file sets
    image_files = set(
        f.split(".")[0][1:] for f in os.listdir(images_dir) if f.endswith(".jpg")
    )  # Remove the "p" prefix
    annotation_files = set(
        f.split(".")[0][1:] for f in os.listdir(annotations_dir) if f.endswith(".txt")
    )  # Remove the "p" prefix
    labels_df = pd.read_csv(labels_csv_path)
    label_files = set(
        labels_df["Pothole number"].astype(str)
    )  # Assume "Pothole number" is the relevant column in the CSV

    # Calculate the intersections and differences
    all_three = image_files.intersection(annotation_files).intersection(label_files)
    images_only_labels = label_files.difference(annotation_files)
    images_only_annotations = annotation_files.difference(label_files)

    # Print the results
    print(f"Total images: {len(image_files)}")
    print(f"Images with all three (annotations, labels, and images): {len(all_three)}")
    print(f"Images with labels only (no annotations): {len(images_only_labels)}")
    print(f"Images with annotations only (no labels): {len(images_only_annotations)}")

    return {
        "total_images": len(image_files),
        "all_three": len(all_three),
        "images_only_labels": len(images_only_labels),
        "images_only_annotations": len(images_only_annotations),
    }


if __name__ == "__main__":
    # Define data directories
    base_dir = os.path.join("..", "data", "Training Data")
    images_dir = os.path.join(base_dir, "train_images")
    annotations_dir = os.path.join(base_dir, "train_annotations")
    labels_csv_path = os.path.join(base_dir, "train_labels.csv")

    # Run the check
    integrity_report = check_data_integrity(
        images_dir, annotations_dir, labels_csv_path
    )

    # Print out the report
    print("\nData Integrity Report:")
    print(f"Total Images: {integrity_report['total_images']}")
    print(
        f"Images with all three (annotations, labels, and images): {integrity_report['all_three']}"
    )
    print(
        f"Images with labels only (no annotations): {integrity_report['images_only_labels']}"
    )
    print(
        f"Images with annotations only (no labels): {integrity_report['images_only_annotations']}"
    )
