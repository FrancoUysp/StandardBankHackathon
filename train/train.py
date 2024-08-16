#from setup import check_pothole_completeness, prepare_dataset

from ultralytics import YOLO  # Assuming you're using the Ultralytics YOLOv8 package

if __name__ == "__main__":
    #data_dir = "../data"
    #csv_file = "train_labels.csv"
    #base_output_dir = "data"  # Updated to avoid double nesting
    #check_pothole_completeness(data_dir, csv_file)

    #yaml_path = prepare_dataset(data_dir, base_output_dir)
    model = YOLO("yolov8m.pt")  # load a pretrained model (recommended for training)

    model.train(data="data.yaml", epochs=200, patience=25, imgsz=256, freeze=15, save_period=10)
    model.val()
