from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List
import uvicorn
import cv2
import numpy as np
from datetime import datetime
import tempfile
import whisper
import re
import torch  
torch.device("cpu")

from ai_model import detect_fire, detect_accident


app = FastAPI(
    title="SmartCity AI Service",
    description="AI-powered incident detection",
    version="1.0.0"
)

'''# ===== LOAD SPEECH MODEL =====
whisper_model = whisper.load_model("base")

# ================= CORS =================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= Models =================
class DetectionResponse(BaseModel):
    detected: bool
    confidence: float
    incident_type: str
    alert_level: str
    description: str
    recommended_unit_type: str
    coordinates: Optional[dict] = None


class SOSClassificationResponse(BaseModel):
    is_emergency: bool
    confidence: float
    incident_type: str
    alert_level: str
    keywords_detected: List[str]
    recommended_unit_type: str


# ================= Speech To Text =================
def speech_to_text(audio_path):
    result = whisper_model.transcribe(
        audio_path,
        language="ar",
        task="transcribe",
        temperature=0.2,
        best_of=3
    )
    return result["text"]


# ================= Text Cleaning =================
def clean_text(text: str):
    text = text.lower()
    text = re.sub(r"[^\u0600-\u06FF\s]", " ", text)
    text = re.sub(r"(.)\1{2,}", r"\1", text)

    stop_words = ["يعني", "اه", "امم", "الو", "طيب", "بس", "يا"]
    for word in stop_words:
        text = text.replace(word, "")

    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ================= Helper =================
def read_image_or_sampled_video(file: UploadFile, sample_fps: int = 1):
    contents = file.file.read()

    image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    if image is not None:
        return "image", image

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(contents)
        temp_path = temp_video.name

    cap = cv2.VideoCapture(temp_path)
    if not cap.isOpened():
        return None, None

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_interval = max(int(video_fps // sample_fps), 1)

    frames = []
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            frames.append(frame)
        count += 1
    cap.release()

    if len(frames) == 0:
        return None, None

    return "video", frames


# ================= Health =================
@app.get("/")
async def root():
    return {"status": "running"}


@app.get("/health")
async def health():
    return {"status": "healthy"}


# ================= Fire =================
@app.post("/detect/fire", response_model=DetectionResponse)
async def fire_detection(file: UploadFile = File(...)):
    file_type, data = read_image_or_sampled_video(file, sample_fps=1)

    if file_type is None:
        raise HTTPException(status_code=400, detail="Invalid file type")

    if file_type == "image":
        confidence = detect_fire(data)
    else:
        confidences = [detect_fire(frame) for frame in data]
        confidence = float(np.mean(confidences))

    detected = confidence > 0.6

    alert_level = (
        "HIGH" if confidence >= 0.8
        else "MEDIUM" if confidence >= 0.6
        else "LOW"
    )

    return DetectionResponse(
        detected=detected,
        confidence=confidence,
        incident_type="Fire",
        alert_level=alert_level,
        description="Fire detection result",
        recommended_unit_type="Fire"
    )


# ================= Accident =================
@app.post("/detect/accident", response_model=DetectionResponse)
async def accident_detection(file: UploadFile = File(...)):
    file_type, data = read_image_or_sampled_video(file, sample_fps=1)

    if file_type is None:
        raise HTTPException(status_code=400, detail="Invalid file type")

    if file_type == "image":
        confidence = detect_accident(data)
    else:
        confidences = [detect_accident(frame) for frame in data]
        confidence = float(np.mean(confidences))

    detected = confidence > 0.6

    alert_level = (
        "HIGH" if confidence >= 0.8
        else "MEDIUM" if confidence >= 0.6
        else "LOW"
    )

    return DetectionResponse(
        detected=detected,
        confidence=confidence,
        incident_type="Medical",
        alert_level=alert_level,
        description="Accident detection result",
        recommended_unit_type="Medical"
    )


# ================= TEXT + AUDIO =================
@app.post("/classify/text", response_model=SOSClassificationResponse)
async def classify_text(
    text: Optional[str] = None,
    file: UploadFile = File(None)
):
    # ===== لو صوت =====
    if file is not None:
        contents = await file.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio.write(contents)
            temp_path = temp_audio.name
        text = speech_to_text(temp_path)

    # ===== لو نص =====
    elif text is not None:
        text = text.strip().lower()
    else:
        raise HTTPException(status_code=400, detail="No input provided")

    # ===== تنظيف النص =====
    text = clean_text(text)
    text = text.replace("ة", "ه").replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")

    if not text or len(text) < 3:
        raise HTTPException(status_code=400, detail="Input not clear")

    # ===== كلمات مفتاحية Rule-Based =====
    keywords_mapping = {
        "Fire": ["حريق", "دخان", "نار", "مولع", "ولع", "بيولع", "اشتعل", "البيت بيولع", "السياره مولعه"],
        "Accident": ["حادث", "تصادم", "اصطدام", "سقوط", "انفجار", "دخلوا في بعض", "اصطدموا", "اصطدام بين عربيات", "خبطوا"],
        "Assault": ["ضرب", "خناقه", "تهديد", "اعتداء", "معتدي", "تخانق", "اعتدى"],
        "Theft": ["سرقه", "حرامي", "مفقود", "نشل", "سرق", "اتسرق"],
        "Emergency": ["طارئ", "مستعجل", "استغاثه", "اسعاف", "مساعدة", "ساعدوني"]
    }

    incident = "Normal"
    detected_keywords = []

    # ===== البحث عن الكلمات المفتاحية =====
    for category, keywords in keywords_mapping.items():
        for kw in keywords:
            if kw in text:
                incident = category
                detected_keywords.append(kw)
        if detected_keywords:
            break

    confidence = 0.95 if detected_keywords else 0.3

    # ===== Alert Level =====
    alert_level = (
        "HIGH" if confidence >= 0.7
        else "MEDIUM" if confidence >= 0.5
        else "LOW"
    )

    return SOSClassificationResponse(
        is_emergency=True if incident != "Normal" else False,
        confidence=confidence,
        incident_type=incident,
        alert_level=alert_level,
        keywords_detected=detected_keywords,
        recommended_unit_type=incident
    )'''
    
    
    
    # ===== LOAD SPEECH MODEL =====
whisper_model = whisper.load_model("base")

# ================= CORS =================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ================= Models =================
class DetectionResponse(BaseModel):
    detected: bool
    confidence: float
    incident_type: str
    alert_level: str
    description: str
    recommended_unit_type: str
    coordinates: Optional[dict] = None


class SOSClassificationResponse(BaseModel):
    is_emergency: bool
    confidence: float
    incident_type: str
    alert_level: str
    keywords_detected: List[str]
    recommended_unit_type: str


# ================= Speech To Text =================
def speech_to_text(audio_path):
    result = whisper_model.transcribe(
        audio_path,
        language="ar",
        task="transcribe",
        temperature=0.2,
        best_of=3
    )
    return result["text"]


# ================= Text Cleaning =================
def clean_text(text: str):
    text = text.lower()
    text = re.sub(r"[^\u0600-\u06FF\s]", " ", text)
    text = re.sub(r"(.)\1{2,}", r"\1", text)

    stop_words = ["يعني", "اه", "امم", "الو", "طيب", "بس", "يا"]
    for word in stop_words:
        text = text.replace(word, "")

    text = re.sub(r"\s+", " ", text)
    return text.strip()


# ================= Helper =================
def read_image_or_sampled_video(file: UploadFile, sample_fps: int = 1):
    contents = file.file.read()

    image = cv2.imdecode(np.frombuffer(contents, np.uint8), cv2.IMREAD_COLOR)
    if image is not None:
        return "image", image

    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as temp_video:
        temp_video.write(contents)
        temp_path = temp_video.name

    cap = cv2.VideoCapture(temp_path)
    if not cap.isOpened():
        return None, None

    video_fps = cap.get(cv2.CAP_PROP_FPS) or 25
    frame_interval = max(int(video_fps // sample_fps), 1)

    frames = []
    count = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        if count % frame_interval == 0:
            frames.append(frame)
        count += 1
    cap.release()

    if len(frames) == 0:
        return None, None

    return "video", frames


# ================= Health =================
@app.get("/")
async def root():
    return {"status": "running"}


@app.get("/health")
async def health():
    return {"status": "healthy"}

'''
# ================= Fire =================
@app.post("/detect/fire", response_model=DetectionResponse)
async def fire_detection(file: UploadFile = File(...)):
    file_type, data = read_image_or_sampled_video(file, sample_fps=1)

    if file_type is None:
        raise HTTPException(status_code=400, detail="Invalid file type")

    if file_type == "image":
        confidence = detect_fire(data)
    else:
        confidences = [detect_fire(frame) for frame in data]
        confidence = float(np.mean(confidences))

    detected = confidence > 0.6

    alert_level = (
        "HIGH" if confidence >= 0.8
        else "MEDIUM" if confidence >= 0.6
        else "LOW"
    )

    return DetectionResponse(
        detected=detected,
        confidence=confidence,
        incident_type="Fire",
        alert_level=alert_level,
        description="Fire detection result",
        recommended_unit_type="Fire"
    )


# ================= Accident =================
@app.post("/detect/accident", response_model=DetectionResponse)
async def accident_detection(file: UploadFile = File(...)):
    file_type, data = read_image_or_sampled_video(file, sample_fps=1)

    if file_type is None:
        raise HTTPException(status_code=400, detail="Invalid file type")

    if file_type == "image":
        confidence = detect_accident(data)
    else:
        confidences = [detect_accident(frame) for frame in data]
        confidence = float(np.percentile(confidences, 90))

    detected = confidence > 0.6

    alert_level = (
        "HIGH" if confidence >= 0.8
        else "MEDIUM" if confidence >= 0.6
        else "LOW"
    )

    return DetectionResponse(
        detected=detected,
        confidence=confidence,
        incident_type="Medical",
        alert_level=alert_level,
        description="Accident detection result",
        recommended_unit_type="Medical"
    )

'''


# ================= Fire =================
@app.post("/detect/fire", response_model=DetectionResponse)
async def fire_detection(file: UploadFile = File(...)):
    file_type, data = read_image_or_sampled_video(file, sample_fps=1)

    if file_type is None:
        raise HTTPException(status_code=400, detail="Invalid file type")

    if file_type == "image":
        confidence = detect_fire(data)
        detected = confidence > 0.3
    else:
        confidences = [detect_fire(frame) for frame in data]
        detected = any(c > 0.3 for c in confidences)
        confidence = float(np.max(confidences))     

    alert_level = (
        "HIGH" if confidence >= 0.8
        else "MEDIUM" if confidence >= 0.5
        else "LOW"
    )

    return DetectionResponse(
        detected=detected,
        confidence=confidence,
        incident_type="Fire",
        alert_level=alert_level,
        description="Fire detection result",
        recommended_unit_type="Fire"
    )

# ================= Accident =================
@app.post("/detect/accident", response_model=DetectionResponse)
async def accident_detection(file: UploadFile = File(...)):
    file_type, data = read_image_or_sampled_video(file, sample_fps=1)

    if file_type is None:
        raise HTTPException(status_code=400, detail="Invalid file type")

    if file_type == "image":
        confidence = detect_accident(data)
        detected = confidence > 0  # أي قيمة >0 تعتبر حادثة
    else:
        confidences = [detect_accident(frame) for frame in data]
        confidence = float(np.max(confidences))
        detected = any(conf > 0 for conf in confidences)

    alert_level = (
        "HIGH" if confidence >= 0.8
        else "MEDIUM" if confidence >= 0.5
        else "LOW"
    )

    return DetectionResponse(
        detected=detected,
        confidence=confidence,
        incident_type="Medical",
        alert_level=alert_level,
        description="Accident detection result",
        recommended_unit_type="Medical"
    )
# ================= TEXT + AUDIO =================
@app.post("/classify/text", response_model=SOSClassificationResponse)
async def classify_text(
    text: Optional[str] = None,
    file: UploadFile = File(None)
):
    if file is not None:
        contents = await file.read()

        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_audio:
            temp_audio.write(contents)
            temp_path = temp_audio.name

        text = speech_to_text(temp_path)

        text = clean_text(text)

  
        if not text or len(text) < 3:
            raise HTTPException(status_code=400, detail="Audio not clear")


    elif text is not None:
        text = text.strip().lower()

        if not text:
            raise HTTPException(status_code=400, detail="Text empty")

        text = clean_text(text)

    else:
        raise HTTPException(status_code=400, detail="No input provided")

    text = text.replace("ة", "ه").replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")

    if not text or len(text) < 3:
        raise HTTPException(status_code=400, detail="Input not clear")


    keywords_mapping = {
        "Fire": ["حريق", "دخان", "نار", "مولع", "ولع", "بيولع", "اشتعل", "البيت بيولع", "السياره مولعه"],
        "Accident": ["حادث", "تصادم", "اصطدام", "سقوط", "انفجار", "دخلوا في بعض", "اصطدموا", "اصطدام بين عربيات", "خبطوا"],
        "Assault": ["ضرب", "خناقه", "تهديد", "اعتداء", "معتدي", "تخانق", "اعتدى"],
        "Theft": ["سرقه", "حرامي", "مفقود", "نشل", "سرق", "اتسرق"],
        "Emergency": ["طارئ", "مستعجل", "استغاثه", "اسعاف", "مساعدة", "ساعدوني"]
    }

    incident = "Normal"
    detected_keywords = []

    for category, keywords in keywords_mapping.items():
        for kw in keywords:
            if kw in text:
                incident = category
                detected_keywords.append(kw)
        if detected_keywords:
            break

    confidence = 0.95 if detected_keywords else 0.3

    alert_level = (
        "HIGH" if confidence >= 0.7
        else "MEDIUM" if confidence >= 0.5
        else "LOW"
    )

    return SOSClassificationResponse(
        is_emergency=True if incident != "Normal" else False,
        confidence=confidence,
        incident_type=incident,
        alert_level=alert_level,
        keywords_detected=detected_keywords,
        recommended_unit_type=incident
    )


# ====== Run Server ======
if __name__ == "__main__":
    print("=" * 60)
    print(" Starting SmartCity AI Service")
    print("=" * 60)
    print(" Running on: http://0.0.0.0:5000")
    print(" API Docs: http://localhost:5000/docs")
    print("=" * 60)

    uvicorn.run(
        app,
        host="0.0.0.0",
        port=5000,
        log_level="info"
    )