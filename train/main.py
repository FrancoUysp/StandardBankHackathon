import os
import cv2
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.svm import SVR
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA
from pretrained import YOLOv8Model  # Import your custom YOLOv8 model class
from feature_extractor import (
    PotholeFeatureExtractor,
)  # Import the PotholeFeatureExtractor class
from sklearn.impute import SimpleImputer


def load_data_tuples(images_dir, annotations_dir, labels_csv_path):
    # Load labels from the CSV
    labels_df = pd.read_csv(labels_csv_path)
    labels_dict = labels_df.set_index("Pothole number").to_dict()[
        "Bags used "
    ]  # Assume target variable is the column name

    data_tuples = []
    skipped_due_to_missing_annotations = 0
    skipped_due_to_missing_classes = 0
    possible_tuples = 0

    for image_file in os.listdir(images_dir):
        if image_file.endswith(".jpg"):
            possible_tuples += 1

            # Extract the numeric part of the filename (e.g., 'p232.jpg' -> '232')
            image_id = image_file.split(".")[0][1:]

            image_path = os.path.join(images_dir, image_file)
            image = cv2.imread(image_path)

            annotation_file = f"p{image_id}.txt"
            annotation_path = os.path.join(annotations_dir, annotation_file)

            if not os.path.exists(annotation_path):
                skipped_due_to_missing_annotations += 1
                continue

            with open(annotation_path, "r") as f:
                annotations = [line.strip() for line in f.readlines()]

            # Ensure both 'pothole' and 'L1' classes are present
            classes_present = [ann.split()[0] for ann in annotations]
            if "0" not in classes_present or "1" not in classes_present:
                skipped_due_to_missing_classes += 1
                continue

            target_variable = labels_dict.get(float(image_id), None)

            if target_variable is not None and image is not None and annotations:
                data_tuples.append((target_variable, image, annotations))

    # Print the summary of skipped tuples
    print(f"Total possible tuples: {possible_tuples}")
    print(
        f"Tuples skipped due to missing annotations: {skipped_due_to_missing_annotations}"
    )
    print(
        f"Tuples skipped due to missing classes ('pothole' or 'L1'): {skipped_due_to_missing_classes}"
    )
    print(f"Total successful tuples: {len(data_tuples)}")

    return data_tuples


def split_data_tuples(data_tuples, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    # Ensure the split ratios sum to 1
    assert train_ratio + val_ratio + test_ratio == 1.0, "Ratios must sum to 1"

    train_tuples, temp_tuples = train_test_split(
        data_tuples, test_size=(1 - train_ratio), random_state=42
    )
    val_tuples, test_tuples = train_test_split(
        temp_tuples, test_size=(test_ratio / (test_ratio + val_ratio)), random_state=42
    )

    return train_tuples, val_tuples, test_tuples


def convert_to_feature_extractor_format(data_tuple):
    target_variable, image, annotations = data_tuple

    # Convert annotations to the expected format for the PotholeFeatureExtractor
    formatted_annotations = []
    for annotation in annotations:
        components = annotation.split()
        if len(components) == 5:
            class_id, x_center, y_center, width, height = components
            if class_id == "0":
                class_label = "pothole"
            elif class_id == "1":
                class_label = "L1"
            else:
                continue
            formatted_annotations.append(
                {
                    "class": class_label,
                    "x": float(x_center),
                    "y": float(y_center),
                    "width": float(width),
                    "height": float(height),
                }
            )

    return target_variable, image, formatted_annotations


def extract_features(data_tuples):
    extractor = PotholeFeatureExtractor()
    features = []
    targets = []

    for i, data_tuple in enumerate(data_tuples):
        print(f"Processing {i + 1}/{len(data_tuples)} data tuples...")
        target_variable, image, formatted_annotations = (
            convert_to_feature_extractor_format(data_tuple)
        )
        feature_vector = extractor.extract(image, formatted_annotations)
        features.append(feature_vector)
        targets.append(target_variable)

    # Convert list of dicts to a DataFrame
    features_df = pd.DataFrame(features)

    # Use SimpleImputer to fill missing values with the mean
    imputer = SimpleImputer(strategy="mean")
    features_df_imputed = pd.DataFrame(
        imputer.fit_transform(features_df), columns=features_df.columns
    )

    return features_df_imputed, np.array(targets)


def main():
    # Paths to the directories and CSV file
    base_output_dir = "data"  # Assuming the setup was run and output is in 'data'
    images_dir = os.path.join(base_output_dir, "images")
    annotations_dir = os.path.join(base_output_dir, "annotations")
    labels_csv_path = os.path.join(base_output_dir, "labels", "labels.csv")

    # Load data tuples
    data_tuples = load_data_tuples(images_dir, annotations_dir, labels_csv_path)

    if not data_tuples:
        print("No valid data tuples found. Please check your dataset.")
        return

    train_tuples, val_tuples, test_tuples = split_data_tuples(data_tuples)

    X_train, y_train = extract_features(train_tuples)
    X_val, y_val = extract_features(val_tuples)
    X_test, y_test = extract_features(test_tuples)

    pca = PCA(n_components=2)
    X_train_pca = pca.fit_transform(X_train)
    X_val_pca = pca.transform(X_val)
    X_test_pca = pca.transform(X_test)
    print("here")

    # Perform grid search on validation set
    param_grid = {
        "C": [0.1, 1, 10],
        "epsilon": [0.01, 0.1, 0.2],
        "kernel": ["rbf"],
    }

    svr = SVR()
    grid_search = GridSearchCV(svr, param_grid, cv=5, scoring="r2")
    grid_search.fit(X_train_pca, y_train)

    best_svr = grid_search.best_estimator_

    # Evaluate on the validation set
    y_val_pred = best_svr.predict(X_val_pca)
    r2_val = r2_score(y_val, y_val_pred)
    print(f"R^2 on the validation set with best model: {r2_val}")

    # Evaluate on the test set
    y_test_pred = best_svr.predict(X_test_pca)
    r2_test = r2_score(y_test, y_test_pred)
    print(f"R^2 on the test set: {r2_test}")

    print("Best SVR model parameters found by grid search:")
    print(grid_search.best_params_)


if __name__ == "__main__":
    # Call the main function
    main()
