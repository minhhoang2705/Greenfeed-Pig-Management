from fastapi import FastAPI, File, UploadFile, Depends
from fastapi.middleware.cors import CORSMiddleware
from celery import Celery
import os
from modules.pig_detector import PigDetector
import cv2
import numpy as np
import tempfile
from uuid import uuid4

# Configuration
class Config:
    RESULTS_DIR = 'results/detections'
    PROCESSED_VIDEOS_DIR = os.path.join(RESULTS_DIR, 'processed_videos')

# Initialize FastAPI app
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Celery configuration
celery = Celery(
    __name__,
    broker='redis://localhost:6379/0',
    backend='redis://localhost:6379/0'
)

# Ensure results directory exists
os.makedirs(Config.PROCESSED_VIDEOS_DIR, exist_ok=True)

# Image detection endpoint
@app.post('/detect/image')
async def detect_image(file: UploadFile = File(...)):
    contents = await file.read()
    np_arr = np.frombuffer(contents, np.uint8)
    img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
    detector = PigDetector()
    detections, tracked_objects = detector.process_frame(img)
    detection_results = []
    for det in detections:
        detection_results.append({
            'bbox': det['bbox'],
            'confidence': det['confidence'],
            'class': det['class'],
            'track_id': det.get('track_id', None)
        })
    total_count = detector.total_count
    return {'success': True, 'detections': detection_results, 'total_count': total_count}

# Video processing task
@celery.task(name='api.app.process_video_task')
def process_video_task(video_path, output_dir):
    detector = PigDetector()
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    output_path = os.path.join(output_dir, 'output_video.mp4')
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frame_width, frame_height))
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        detections, tracked_objects = detector.process_frame(frame)
        result_frame = detector.visualize_detections(frame, detections, tracked_objects)
        out.write(result_frame)
    cap.release()
    out.release()
    return output_path

# Video detection endpoint
@app.post('/detect/video')
async def detect_video(file: UploadFile = File(...)):
    temp_dir = tempfile.mkdtemp()
    video_path = os.path.join(temp_dir, file.filename)
    with open(video_path, 'wb') as buffer:
        buffer.write(await file.read())
    task_id = str(uuid4())
    output_dir = os.path.join(Config.PROCESSED_VIDEOS_DIR, task_id)
    os.makedirs(output_dir, exist_ok=True)
    task = process_video_task.delay(video_path, output_dir)
    return {'success': True, 'task_id': task_id}

# Task status endpoint
@app.get('/task/{task_id}')
def get_task_status(task_id: str):
    from celery.result import AsyncResult
    task = AsyncResult(task_id, app=celery)
    if task.state == 'PENDING':
        response = {
            'state': task.state,
            'status': 'Task is pending.'
        }
    elif task.state != 'FAILURE':
        response = {
            'state': task.state,
            'status': 'Task is processing.',
            'download_url': f'/download/{task_id}' if task.state == 'SUCCESS' else None
        }
    else:
        response = {
            'state': task.state,
            'status': str(task.info),
        }
    return response

# Download processed video endpoint
@app.get('/download/{task_id}')
def download_processed_video(task_id: str):
    output_dir = os.path.join(Config.PROCESSED_VIDEOS_DIR, task_id)
    output_path = os.path.join(output_dir, 'output_video.mp4')
    if os.path.exists(output_path):
        return {'success': True, 'file_path': output_path}
    else:
        return {'success': False, 'error': 'File not found'}

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host='0.0.0.0', port=8000)
