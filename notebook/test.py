import os
import cv2
from ultralytics import YOLO
import matplotlib.pyplot as plt


def predict(chosen_model, img):
    results = chosen_model.predict(img)
    return results


def predict_and_detect(chosen_model, img, rectangle_thickness=2, text_thickness=1):
    # Get prediction results
    results = predict(chosen_model, img)
    print("Prediction results:", results)  # Debugging: Print the results to inspect

    # Ensure the results contain detections
    if not results or not results[0].boxes:
        print("No detections found.")
        return img, results

    # Draw bounding boxes and labels on the image
    for result in results:
        for box in result.boxes:
            print(
                f"Box coordinates: {box.xyxy}, Class: {result.names[int(box.cls[0])]}"
            )  # Debugging: Check box details

            cv2.rectangle(
                img,
                (int(box.xyxy[0][0]), int(box.xyxy[0][1])),
                (int(box.xyxy[0][2]), int(box.xyxy[0][3])),
                (255, 0, 0),
                rectangle_thickness,
            )
            cv2.putText(
                img,
                f"{result.names[int(box.cls[0])]}",
                (int(box.xyxy[0][0]), int(box.xyxy[0][1]) - 10),
                cv2.FONT_HERSHEY_PLAIN,
                1,
                (255, 0, 0),
                text_thickness,
            )
    return img, results


if __name__ == "__main__":
    # Load the YOLOv10 model (use the correct model path)
    model = YOLO("yolov10x.pt")

    # Define the path to the image
    img_path = "../data/test_images/p103.jpg"

    # Read the image using OpenCV
    img = cv2.imread(img_path)
    img_rgb = cv2.cvtColor(
        img, cv2.COLOR_BGR2RGB
    )  # Convert BGR to RGB for display in matplotlib

    # Perform object detection on the image
    result_img, results = predict_and_detect(model, img_rgb)

    # Display the result if detections exist
    if results and results[0].boxes:
        plt.imshow(result_img)
        plt.axis("off")
        plt.show()

        # Save the result image (optional)
        output_path = "../data/test_images/p103_detected.jpg"
        cv2.imwrite(
            output_path, cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR)
        )  # Save in BGR format for OpenCV
    else:
        print("No bounding boxes to display.")
