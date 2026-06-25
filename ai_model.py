
'''from ultralytics import YOLO
import torch
import cv2
import base64
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from alert_engine import fire_alert, accident_alert, sos_alert

# =====================
# Load Models
# =====================
fire_model = YOLO(r"D:/data final project/Emergency feature/integration with back\best_fire.pt")
accident_model = YOLO(r"D:/data final project/Emergency feature/integration with back/best_acc.pt")

sos_model = AutoModelForSequenceClassification.from_pretrained(r"D:/data final project/Emergency feature/integration with back/sos_classifier_model")
sos_tokenizer = AutoTokenizer.from_pretrained(r"D:/data final project/Emergency feature/integration with back/sos_classifier_model")


# =====================
# Fire Detection
# =====================
def detect_fire(frame):
    results = fire_model(frame, verbose=False)
    img = results[0].plot()
    count = len(results[0].boxes) if results[0].boxes else 0

    alert = fire_alert(count)

    return {
        "image": img,
        "fire_count": count,
        "alert": alert
    }


# =====================
# Accident Detection
# =====================
def detect_accident(frame):
    results = accident_model(frame, verbose=False)
    img = results[0].plot()
    count = len(results[0].boxes) if results[0].boxes else 0

    alert = accident_alert(count)

    return {
        "image": img,
        "accident_count": count,
        "alert": alert
    }


# =====================
# SOS Text
# =====================
def detect_sos(text):
    inputs = sos_tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    with torch.no_grad():
        logits = sos_model(**inputs).logits

    label_id = torch.argmax(logits, dim=1).item()
    label = sos_model.config.id2label[label_id]

    alert = sos_alert(label)

    return {
        "label": label,
        "alert": alert
    }'''
    
    
    
import torch
from ultralytics import YOLO
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch.nn.functional as F

# ================= FIRE MODEL =================
fire_model = YOLO(r"D:\data final project\Emergency feature\integration with back\best_fire.pt")

def detect_fire(image_path):
    results = fire_model(image_path)
    fire_count = len(results[0].boxes)
    return fire_count


# ================= ACCIDENT MODEL =================
accident_model = YOLO(r"D:\data final project\Emergency feature\integration with back\best_acc.pt")

def detect_accident(image_path):
    results = accident_model(image_path)
    accident_count = len(results[0].boxes)
    return accident_count


# ================= SOS TEXT MODEL =================
tokenizer = AutoTokenizer.from_pretrained(r"D:\data final project\Emergency feature\integration with back\sos_classifier_model")
sos_model = AutoModelForSequenceClassification.from_pretrained(r"D:\data final project\Emergency feature\integration with back\sos_classifier_model")

def classify_sos_text(text):
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
    outputs = sos_model(**inputs)
    probs = F.softmax(outputs.logits, dim=1)
    label_id = torch.argmax(probs, dim=1).item()
    label = sos_model.config.id2label[label_id]
    confidence = probs[0][label_id].item()

    return label, confidence