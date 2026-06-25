#  train_accident_model.py
# ======================================
# Train YOLOv8 model for Accident Detection
# ======================================

import matplotlib.pyplot as plt
from ultralytics import YOLO

# ⿡ Load the medium YOLOv8 model
model = YOLO("yolov8m.pt")


# ⿢ Train the model with data augmentation
results = model.train(
    data=r"D:\data final project\Accident dataset\Accident_frames\data.yaml",  # Path to dataset YAML
    epochs=15,               # Number of training epochs
    imgsz=512,               # Image input size
    batch=4,                 # Batch size (adjust if you face memory issues)
    name="accident_yolov8m_augmented",  # Run name for saving results
    workers=2,               # Number of CPU workers for data loading
    device="cpu",            # Change to "cuda" if you have a GPU

    #  Data Augmentation Settings
    hsv_h=0.015,             # Hue variation
    hsv_s=0.7,               # Saturation variation
    hsv_v=0.4,               # Brightness variation
    degrees=5,               # Random rotation
    translate=0.1,           # Random translation
    scale=0.5,               # Random scaling
    shear=0.1,               # Random shearing
    fliplr=0.5,              # Horizontal flip
    mosaic=1.0,              # Mosaic augmentation
    mixup=0.2                # Mixup augmentation
)

#  Validate the trained model
metrics = model.val()

#  Print evaluation metrics
print("\n Model Evaluation Metrics:")
print(f"Precision (mAP@0.5): {metrics.box.map50:.4f}")
print(f"Recall (mAP@0.75): {metrics.box.map75:.4f}")
print(f"mAP[0.5:0.95]: {metrics.box.map:.4f}")

try:
    curves = metrics.curves  
    if len(curves) >= 2:
        x, y = curves[0], curves[1]
        plt.figure(figsize=(6, 4))
        plt.plot(x, y, label="Precision-Recall Curve", color='blue')
        plt.xlabel("Recall")
        plt.ylabel("Precision")
        plt.title("Precision vs Recall Curve")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.savefig(r"D:\data final project\Accident dataset\runs\precision_recall_curve_yolov8m.png")
        plt.show()
    else:
        print(" Unable to extract PR curve data.")
except Exception as e:
    print(f" Could not plot PR curve: {e}")

print("\n Training complete using YOLOv8m with augmentation!") 