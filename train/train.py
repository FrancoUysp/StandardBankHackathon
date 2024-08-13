import os
import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt


class PotholeYOLO:
    def __init__(self, yolo_model_path=None):
        self.model = None
        if yolo_model_path:
            self.load_model(yolo_model_path)

    def load_model(self, yolo_model_path):
        # Load the YOLOv8 model using the correct method
        self.model = YOLO(yolo_model_path)

    def fine_tune(self, data_yaml_path, epochs=500, project_path="./runs"):
        # Fine-tuning the YOLOv8 model on the custom dataset
        #
        self.model.add_callback("on_train_start", self.freeze_layer)
        self.model.train(data=data_yaml_path, epochs=epochs, project=project_path)
        print("Finished Model Training")

    def validate(self):
        # Validate the model and print results
        results = self.model.val()
        print(results)

    def predict(self, image_path):
        # Predicting bounding boxes for an image
        result = self.model.predict(image_path)
        return result

    def freeze_layer(self, trainer):
        model = trainer.model

        # Get a list of all layers
        layers = list(model.named_parameters())
        total_layers = len(layers)

        # Calculate the index after which we want to unfreeze the layers
        freeze_until = total_layers - (total_layers - 10)

        print(f"Freezing all but the last 2 layers")

        # Freeze all layers except the last 2
        for i, (k, v) in enumerate(layers):
            if i < freeze_until:
                v.requires_grad = False
                print(f"freezing {k}")
            else:
                v.requires_grad = True

        print(f"All but the last 2 layers are frozen.")

    def predict_and_show(self, image_path):
        # Predict bounding boxes for the image
        result = self.predict(image_path)
        image = cv2.imread(".png")

        # Draw bounding boxes on the image
        for box in result[0].boxes:
            x1, y1, x2, y2 = map(
                int, box.xyxy[0]
            )  # Assuming box.xyxy returns the bounding box coordinates
            confidence = box.conf[0]  # Confidence score
            label = (
                f"{box.cls[0]}: {confidence:.2f}"  # Class label and confidence score
            )
            cv2.rectangle(
                image, (x1, y1), (x2, y2), (0, 255, 0), 2
            )  # Green bounding box
            cv2.putText(
                image,
                label,
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 255, 0),
                2,
            )

        # Show the image
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        plt.show()

        # Display the image with bounding boxes
        img = result[0].plot()  # Plot the result
        plt.imshow(img)
        plt.axis("off")
        plt.show()


if __name__ == "__main__":
    # Update the paths based on the new directory structure
    base_dir = os.path.dirname(os.path.abspath(__file__))  # Get the current directory
    data_yaml_path = os.path.join(
        base_dir, "split_dataset", "data.yaml"
    )  # Absolute path to your data.yaml
    yolo_model_path = "yolov8n.pt"  # Path to your YOLOv8 model

    # Load YOLOv8 model and fine-tune
    yolo = PotholeYOLO(yolo_model_path=yolo_model_path)
    yolo.fine_tune(
        data_yaml_path=data_yaml_path, epochs=1, project_path=base_dir
    )  # Use current directory as project path

    yolo.validate()

    test_image_path = os.path.join(
        base_dir, "..", "data/test_images/p104.jpg"
    )  # Path to your test image
    yolo.predict_and_show(test_image_path)
