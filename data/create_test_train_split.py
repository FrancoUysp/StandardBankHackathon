import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import shutil

# Set the random seed for reproducibility
np.random.seed(42)

# Define paths
csv_file = "train_labels.csv"
source_folder = "train_images"
train_folder = "training"
test_folder = "testing"
val_folder = "validation"

# Read the CSV file
df = pd.read_csv(csv_file, names=["Pothole", "numberBags"])

# Split the data
train_data, temp_data = train_test_split(df, test_size=0.3, random_state=42)
test_data, val_data = train_test_split(temp_data, test_size=0.5, random_state=42)

# Create destination folders if they don't exist
for folder in [train_folder, test_folder, val_folder]:
    os.makedirs(folder, exist_ok=True)


# Function to move files
def move_files(data, destination):
    for image_name in data["Pothole"]:
        src = os.path.join(source_folder, f"p{image_name}.jpg")
        dst = os.path.join(destination, f"p{image_name}.jpg")
        if os.path.exists(src):
            shutil.move(src, dst)
        else:
            print(f"Warning: {src} not found")


# Move files to their respective folders
move_files(train_data, train_folder)
move_files(test_data, test_folder)
move_files(val_data, val_folder)

print("Data split and files moved successfully!")
print(f"Training samples: {len(train_data)}")
print(f"Testing samples: {len(test_data)}")
print(f"Validation samples: {len(val_data)}")
