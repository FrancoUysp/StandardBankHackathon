# MiDaS Pre-trained Model Installation Guide

This guide provides step-by-step instructions for installing and using the MiDaS pre-trained model for depth estimation in our pothole detection project.

## Prerequisites

- Python 3.7 or higher
- pip (Python package installer)
- Git

## Installation Steps

1. Clone the MiDaS repository:
   ```
   git clone https://github.com/isl-org/MiDaS.git
   ```

2. Navigate to the MiDaS directory:
   ```
   cd MiDaS
   ```

3. Install the required dependencies:
   ```
   pip install torch torchvision timm
   ```

4. Install the MiDaS package:
   ```
   pip install -e .
   ```

5. Return to your project directory:
   ```
   cd ..
   ```

## Using MiDaS in Your Project

1. In your Python script, import the necessary modules:
   ```python
   import os
   import sys
   import torch
   from torchvision.transforms import Compose

   # Add the MiDaS directory to the Python path
   midas_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'MiDaS')
   sys.path.append(midas_dir)

   from midas.model_loader import load_model
   from midas.transforms import Resize, NormalizeImage, PrepareForNet
   ```

2. Initialize the MiDaS model in your code:
   ```python
   def initialize_midas_model(model_type="DPT_Large"):
       model_path = os.path.join(midas_dir, "weights", f"{model_type}.pt")
       
       # Download the model if it doesn't exist
       if not os.path.exists(model_path):
           os.makedirs(os.path.dirname(model_path), exist_ok=True)
           torch.hub.download_url_to_file(
               f"https://github.com/intel-isl/DPT/releases/download/1_0/{model_type.lower()}-midas-2f21e586.pt",
               model_path
           )

       device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
       model = load_model(model_path)
       model.to(device)
       model.eval()

       return model, device
   ```

3. Create the MiDaS transform:
   ```python
   midas_transform = Compose(
       [
           Resize(
               384,
               384,
               resize_target=None,
               keep_aspect_ratio=True,
               ensure_multiple_of=32,
               resize_method="upper_bound",
               image_interpolation_method=cv2.INTER_CUBIC,
           ),
           NormalizeImage(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
           PrepareForNet(),
       ]
   )
   ```

4. Use the model for depth estimation:
   ```python
   def estimate_depth(model, device, image):
       image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) / 255.0
       img_input = midas_transform({"image": image})["image"]

       with torch.no_grad():
           sample = torch.from_numpy(img_input).to(device).unsqueeze(0)
           prediction = model.forward(sample)
           prediction = torch.nn.functional.interpolate(
               prediction.unsqueeze(1),
               size=image.shape[:2],
               mode="bicubic",
               align_corners=False,
           ).squeeze()

       return prediction.cpu().numpy()
   ```

## Troubleshooting

- If you encounter import errors, ensure that the MiDaS directory is correctly added to your Python path.
- Make sure you're running your script from the correct working directory.
- If the model fails to download automatically, manually download it from the MiDaS GitHub repository and place it in the `MiDaS/weights` directory.

## Note

This installation process assumes that you're using the MiDaS repository as a subdirectory in your project. Adjust paths accordingly if your project structure differs.