from fastapi import FastAPI, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
import cv2
import numpy as np
from PIL import Image
import io
import sys
import os
from pathlib import Path

# Add the parent directory to sys.path to import PigDetector
sys.path.append(str(Path(__file__).parent.parent))
from modules.pig_detector import PigDetector

app = FastAPI(title="Pig Detection API")

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize the PigDetector
model_path = Path(__file__).parent.parent / "weights" / "yolo11n.pt"
detector = PigDetector(model_path=str(model_path))

@app.get("/")
def read_root():
    return {"message": "Pig Detection API is running"}

@app.post("/detect")
async def detect_pigs(file: UploadFile = File(...)):
    # Read image file
    contents = await file.read()
    nparr = np.frombuffer(contents, np.uint8)
    image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if image is None:
        return {"error": "Invalid image file"}
    
    # Perform detection
    detections = detector.detect(image)
    
    # Visualize detections
    result_image = detector.visualize_detections(image, detections)
    
    # Convert the result image to bytes
    is_success, buffer = cv2.imencode(".jpg", result_image)
    if not is_success:
        return {"error": "Failed to encode result image"}
    
    # Convert to base64 for sending to frontend
    import base64
    img_str = base64.b64encode(buffer).decode()
    
    return {
        "detections": detections,
        "count": len(detections),
        "image": img_str
    }
