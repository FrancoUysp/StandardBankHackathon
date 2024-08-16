import cv2
import matplotlib.pyplot as plt


class YOLOv8Model:
    def __init__(self, model_file):
        """
        Initializes the YOLOv8 model using the specified model file.

        :param model_file: Path to the YOLOv8 model file (e.g., .pt file)
        """
        self.model = YOLO(model_file)

    def predict(self, image_path):
        """
        Predicts objects in the given image using the YOLOv8 model.

        :param image_path: Path to the image file to be predicted
        :return: A dictionary containing the original image and annotations
        """
        # Load the image
        image = cv2.imread(image_path)
        image_rgb = cv2.cvtColor(
            image, cv2.COLOR_BGR2RGB
        )  # Convert to RGB for visualization

        # Run prediction
        results = self.model.predict(image_path)

        # Extract annotations in the required format
        annotations = []
        for r in results:
            for box in r.boxes:
                class_id = int(box.cls)
                class_name = r.names[class_id]
                x_center, y_center, width, height = box.xywh.tolist()[0]
                annotations.append(
                    {
                        "class": class_name,
                        "x": x_center,
                        "y": y_center,
                        "width": width,
                        "height": height,
                    }
                )

        # Return the image and annotations
        return {"image": image_rgb, "annotations": annotations}

    def show_image_with_annotations(self, prediction_result):
        """
        Displays the image with annotations overlaid.

        :param prediction_result: The dictionary output from the predict() method
        """
        image = prediction_result["image"]
        annotations = prediction_result["annotations"]

        plt.figure(figsize=(10, 10))
        plt.imshow(image)
        plt.axis("off")

        for annotation in annotations:
            x = annotation["x"] - annotation["width"] / 2
            y = annotation["y"] - annotation["height"] / 2
            rect = plt.Rectangle(
                (x, y),
                annotation["width"],
                annotation["height"],
                fill=False,
                color="red",
                linewidth=2,
            )
            plt.gca().add_patch(rect)
            plt.text(
                x,
                y,
                annotation["class"],
                color="white",
                verticalalignment="top",
                bbox={"color": "red", "pad": 0},
            )

        plt.show()


if __name__ == "__main__":
    model = YOLOv8Model("path/to/your/fine-tuned/model.pt")
    result = model.predict("path/to/image.jpg")
    print(result["annotations"])
    model.show_image_with_annotations(result)
