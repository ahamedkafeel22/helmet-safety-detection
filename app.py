from fastapi import FastAPI, UploadFile, File
from ultralytics import YOLO
import cv2
import numpy as np
import shutil
import os
from datetime import datetime

app = FastAPI(
    title="Helmet Safety Detection API",
    description="YOLOv8-powered PPE compliance detection for construction sites",
    version="1.0"
)

MODEL_PATH = r"C:\Users\syedk\Documents\Self Projects\Project 4\best.pt"
model = YOLO(MODEL_PATH)
CLASS_NAMES = {0: 'head', 1: 'helmet', 2: 'person'}

@app.get("/")
def home():
    return {
        "message": "Helmet Safety Detection API is running!",
        "model": "YOLOv8n",
        "classes": ["head (no helmet)", "helmet", "person"],
        "mAP50_helmet": 0.982,
        "mAP50_head": 0.964
    }

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # Save uploaded image
    temp_path = f"temp_{file.filename}"
    with open(temp_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    # Run detection
    results = model(temp_path, conf=0.5)[0]

    helmet_count = 0
    head_count = 0
    person_count = 0
    detections = []

    for box in results.boxes:
        cls = int(box.cls[0])
        conf = float(box.conf[0])
        x1, y1, x2, y2 = map(int, box.xyxy[0])

        if cls == 0: head_count += 1
        elif cls == 1: helmet_count += 1
        else: person_count += 1

        detections.append({
            "class": CLASS_NAMES[cls],
            "confidence": round(conf, 3),
            "bbox": [x1, y1, x2, y2]
        })

    total_workers = helmet_count + head_count
    compliance = round((helmet_count / total_workers * 100), 2) if total_workers > 0 else 100.0
    status = "SAFE" if compliance == 100 else "VIOLATION"

    os.remove(temp_path)

    return {
        "status": status,
        "compliance_%": compliance,
        "helmet_count": helmet_count,
        "violations": head_count,
        "person_count": person_count,
        "total_detections": len(detections),
        "detections": detections,
        "timestamp": datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }

@app.get("/stats")
def stats():
    return {
        "total_images_processed": 706,
        "total_helmets_detected": 1855,
        "total_violations": 704,
        "overall_compliance": 72.49,
        "model_performance": {
            "helmet_mAP50": 0.982,
            "head_mAP50": 0.964,
            "epochs_trained": 50
        }
    }

@app.get("/health")
def health():
    return {"status": "healthy", "model_loaded": True}