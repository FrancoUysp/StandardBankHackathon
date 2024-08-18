
# Pothole Prediction

![Pothole Prediction](assets/readme.gif)

## Overview

This project is focused on predicting the number of tar bags required to fill potholes using images. The repository includes the necessary code, data, and scripts to set up the environment, perform data analysis, train models, and deploy the PatchPerfect web application.

## Setup Instructions

### Prerequisites

- Python 3.x must be installed on your system.
- `git` should be installed (for cloning the repository and version control).

### Step 1: Clone the Repository

Clone the repository to your local machine using the following commands:

```bash
git clone https://github.com/your-username/pothole-prediction.git
cd pothole-prediction
```

### Step 2: Initialize the Environment

To set up the virtual environment and install all necessary dependencies, run the `startup.sh` script located in the root of the repository:

```bash
./startup.sh
```

This will initialize the virtual environment, and you'll be ready to proceed with the training and prediction steps.

### Step 3: Training the YOLOv8 Model

Navigate to the `train` directory and run `setup.py` to prepare the training and target variables:

- **Training Preparation**: To prepare the data for fine-tuning the YOLOv8 model, run the following command:

  ```bash
  python setup.py od
  ```

- **Training the Model**: Once the data is prepared, you can initiate the training or fine-tuning of the YOLOv8 model with:

  ```bash
  python train.py
  ```

The training results, including model weights and logs, are stored in the `runs` directory in the root of the repository.

### Step 4: Generating Target Data

To generate data for the actual target variable, run `setup.py` in `pp` mode:

```bash
python setup.py pp
```

This command moves all the data to a local directory called `data`, which contains labels, annotations, and images.

### Step 5: Making Predictions

To generate predictions or bounding boxes for the test data, run the following script:

```bash
python predict_test.py
```

This will create a `predictions` directory with results in YOLO format.

## Additional Information

- **Combination.ipynb**: This notebook was used for experimenting with various machine learning models and the data. It is located in the root directory.

- **Models Directory**: The `models` directory within the `train` directory stores pickled versions of some of the experimental models. These are not the main models used in the final solution.

- **Notebook Directory**: The main solution was generated using `main.ipynb`, located in the `notebook` directory within the root of the repository.

- **Docs Folder**: Contains the slides and the solution statement that documents the project’s objectives, approach, and outcomes.

- **Model Weights**: Includes the weights for the CNN model used in the final solution.

- **Front-End**: The `front-end` directory contains the PatchPerfect web application, which serves as a proof of concept for a larger scale, community-based application. Developed using React and Firebase, this application allows users to upload images of potholes, which are automatically annotated with their location and predicted material requirements.

