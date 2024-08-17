import os
import cv2


class YoloSynthetic:
    def __init__(self, base_dir):
        """
        Initializes the synthetic YOLO model by setting the base directory where data is stored.

        :param base_dir: Path to the base directory containing 'images' and 'labels' directories.
        """
        self.base_dir = base_dir

    def read_annotation(self, annotation_path):
        """
        Reads the YOLO-style annotation file and converts it into the required format.

        :param annotation_path: Path to the annotation file (.txt)
        :return: List of dictionaries with 'class', 'x', 'y', 'width', 'height'
        """
        annotations = []
        with open(annotation_path, "r") as file:
            lines = file.readlines()
            for line in lines:
                parts = line.strip().split()
                class_id = int(parts[0])
                x_center, y_center, width, height = map(float, parts[1:])
                annotations.append(
                    {
                        "class": "pothole" if class_id == 0 else "L1",
                        "x": x_center,
                        "y": y_center,
                        "width": width,
                        "height": height,
                    }
                )
        return annotations

    def predict(self, image_path):
        """
        Simulates the prediction process by reading the image and its corresponding annotation file.

        :param image_path: Path to the image file (.jpg)
        :return: Dictionary containing the original image and annotations
        """
        # Load the image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Error: Unable to load image at {image_path}")

        # Determine corresponding annotation file path
        image_name = os.path.basename(image_path)
        annotation_name = os.path.splitext(image_name)[0] + ".txt"
        annotation_path = os.path.join(self.base_dir, "labels", annotation_name)

        if not os.path.exists(annotation_path):
            raise FileNotFoundError(f"Annotation file not found for {image_path}")

        # Read the annotation file
        annotations = self.read_annotation(annotation_path)

        return {
            "image": cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
            "annotations": annotations,
        }


# Example usage
if __name__ == "__main__":
    base_dir = "path_to_your_local_data_directory"  # Update with the correct path
    yolo_synthetic = YoloSynthetic(base_dir)

    image_path = os.path.join(base_dir, "images", "val", "p102.jpg")
    prediction_result = yolo_synthetic.predict(image_path)

    # This prediction_result can now be fed into your PotholeFeatureExtractor
    print("Prediction result:")
    print(prediction_result["annotations"])
