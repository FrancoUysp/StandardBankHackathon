# PatchPerfect: pothole detection and asphalt estimation

![PatchPerfect demo](assets/readme.gif)

Team ComSigh's entry for the **Standard Bank and Motus Data School Hackathon, August 2024**. The task was to estimate, from a photo, how many bags of asphalt a pothole needs. We finished **3rd of 68 teams**.

Team: Steffan Schoonbee, Gunther Tonitz, Dirk Hoffmann, Shriyan Singh and Franco Uys. My own work in this repository is the data preparation and training pipeline under `train/`, the YOLOv8 fine-tuning on an AWS EC2 instance, and the modelling notebooks. The PatchPerfect app under `Frontend/` was built by my teammates.

> **Status: archived snapshot.** This is the code as it stood at the end of the hackathon, rebuilt in September 2026 without the dataset, model weights and backend (see below). Dependencies date from August 2024 and are unmaintained. Read the code by all means, but do not deploy it as-is.

## What it does

1. **Detection.** A YOLOv8-medium model, fine-tuned on the hackathon images, finds potholes and a reference object (a stick of known length) in each photo. On the validation set it reached 95% precision, 96% recall and 97% mAP@50.
2. **Estimation.** A custom convolutional network (two convolutional layers with max pooling, two dense layers with dropout) takes the detected region, scales it using the reference stick, pads it to a fixed size so that physical size survives as a feature, and predicts the number of bags of asphalt. A custom TensorFlow `Augment` layer applies orientation, hue, contrast and brightness perturbations during training to limit overfitting.
3. **App.** PatchPerfect is a React (Expo) app backed by Firebase. Users photograph a pothole, the location is captured automatically, and a serverless function runs the model so that a municipality or partner sees the pothole, its position and the estimated material on a live map.

The full write-up is in [`docs/Report.pdf`](docs/Report.pdf) and the pitch deck in [`docs/presentation.pdf`](docs/presentation.pdf).

## Repository layout

| Path | Contents |
|---|---|
| `train/` | Data cleaning (`data_clean.py`), YOLO dataset setup (`setup.py`), YOLOv8 training (`train.py`), feature extraction and CNN experiments (`feature_extractor.py`, `main.py`, `combination.ipynb`), test-set prediction (`predict_test.py`). |
| `notebook/main.ipynb` | The notebook that produced the final submission. |
| `Frontend/Tar/` | The PatchPerfect Expo app. Firebase settings are read from environment variables; copy `.env.example` to `.env` to run it. |
| `docs/` | Report and presentation. |
| `assets/` | Demo GIF. |

## Running the training code

```bash
./startup.sh                 # creates a virtual environment and installs requirements.txt
cd train
python setup.py od           # prepare the YOLO dataset (needs the hackathon images, see below)
python train.py              # fine-tune YOLOv8
python setup.py pp           # build the CNN training set
python predict_test.py       # write YOLO-format predictions for the test images
```

`requirements.txt` lists the packages used in August 2024 with version pins removed. Expect to resolve incompatibilities if you install it today.

## What was removed in the 2026 rebuild, and why

- **The hackathon dataset** (`data/`, `Data_CNN/`) was supplied by the organisers and is not ours to redistribute.
- **Model weights and YOLO training runs** (`model_weights/`, `runs/`, `*.pt`, `*.h5`, `*.keras`) added several hundred megabytes to every clone.
- **The Firebase Cloud Functions backend** (`Frontend/Tar/functions/`) pinned Python packages with many published vulnerabilities. It was a hackathon prototype and should be rewritten before any deployment.
- **The Firebase web configuration** was hardcoded and now comes from environment variables. The original Firebase project is no longer in use.

The git history was rewritten to drop these files from every commit; authorship and commit messages are otherwise unchanged.
