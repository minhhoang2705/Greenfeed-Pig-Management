"""
FastAPI backend for pig detection and tracking system.

This module implements RESTful API endpoints for image and video processing,
following SOLID principles and providing comprehensive error handling.
"""

import os
from typing import List, Optional
from pathlib import Path
from datetime import datetime
import logging
import shutil
from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from pydantic import BaseModel, Field
import cv2
import numpy as np

from modules.pig_detector import PigDetector

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/api.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Pig Detection API",
    description="API for detecting and tracking pigs in images and videos",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace with specific origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize detector
pig_detector = PigDetector()

# Create necessary directories
UPLOAD_DIR = Path("uploads")
RESULTS_DIR = Path("results")
for dir_path in [UPLOAD_DIR, RESULTS_DIR]:
    dir_path.mkdir(exist_ok=True)


class ProcessingResponse(BaseModel):
    """Response model for processing endpoints."""
    status: str
    message: str
    result_path: Optional[str] = None
    detection_count: Optional[int] = None
    processing_time: float = Field(...,
                                   description="Processing time in seconds")
    tracked_objects: Optional[dict] = None


class ProcessingError(BaseModel):
    """Error response model."""
    error: str
    detail: Optional[str] = None


def cleanup_old_files(directory: Path, max_age_hours: int = 24):
    """Clean up files older than specified hours."""
    current_time = datetime.now().timestamp()
    for file_path in directory.glob("*"):
        if file_path.is_file():
            file_age = current_time - file_path.stat().st_mtime
            if file_age > max_age_hours * 3600:
                try:
                    file_path.unlink()
                    logger.info(f"Cleaned up old file: {file_path}")
                except Exception as e:
                    logger.error(
                        f"Error cleaning up file {file_path}: {str(e)}")


@app.post("/api/v1/detect/image", response_model=ProcessingResponse)
async def process_image(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
) -> ProcessingResponse:
    """
    Process an uploaded image for pig detection.

    Args:
        file: Uploaded image file
        background_tasks: FastAPI background tasks handler

    Returns:
        ProcessingResponse: Processing result including status and detection details

    Raises:
        HTTPException: If file format is invalid or processing fails
    """
    try:
        # Validate file format
        is_valid_image = False
        if file.content_type and file.content_type.startswith('image/'):
            is_valid_image = True
        else:
            # Fallback to file extension check if content_type is not available
            file_ext = os.path.splitext(file.filename)[1].lower()
            valid_image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
            if file_ext in valid_image_extensions:
                is_valid_image = True

        if not is_valid_image:
            raise HTTPException(
                status_code=400,
                detail="Invalid file format. Please upload an image file (supported formats: JPG, JPEG, PNG, BMP)."
            )

        start_time = datetime.now()

        # Read and validate image data
        contents = await file.read()
        if not contents:
            raise HTTPException(
                status_code=400,
                detail="Empty file received. Please upload a valid image file."
            )

        # Log basic info about the received data
        logger.info(
            f"Received image data: {len(contents)} bytes, content type: {file.content_type}")

        try:
            # Convert to numpy array and validate
            nparr = np.frombuffer(contents, np.uint8)
            if nparr.size == 0:
                logger.error(
                    "Failed to convert image data to numpy array - empty buffer")
                raise HTTPException(
                    status_code=400,
                    detail="Invalid image data - could not convert to numpy array"
                )

            # Additional validation of numpy array
            if nparr.nbytes < 100:  # Minimum expected size for an image
                logger.error(
                    f"Invalid image data size: {nparr.nbytes} bytes")
                raise HTTPException(
                    status_code=400,
                    detail="Invalid image data - file size too small"
                )

            # Decode image with error handling
            try:
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if image is None:
                    logger.error(
                        f"Failed to decode image. File size: {len(contents)} bytes, numpy array size: {nparr.size}")
                    raise HTTPException(
                        status_code=400,
                        detail="Could not decode image. Please ensure it's a valid image file."
                    )
            except Exception as decode_error:
                logger.error(f"Image decoding failed: {str(decode_error)}")
                raise HTTPException(
                    status_code=400,
                    detail=f"Image decoding failed: {str(decode_error)}"
                )
            if image is None:
                logger.error(
                    f"Failed to decode image. File size: {len(contents)} bytes, numpy array size: {nparr.size}")
                raise HTTPException(
                    status_code=400,
                    detail="Could not decode image. Please ensure it's a valid image file."
                )

            # Log successful decoding
            logger.info(
                f"Successfully decoded image. Dimensions: {image.shape}, numpy array size: {nparr.size}")
        except Exception as decode_error:
            logger.error(f"Image decoding failed: {str(decode_error)}")
            raise HTTPException(
                status_code=400,
                detail=f"Image decoding failed: {str(decode_error)}"
            )

        # Process image using detector
        detections, tracked_objects = pig_detector.process_single_frame(image)

        # Visualize results
        result_image = pig_detector.visualize_detections(
            image, detections, tracked_objects)

        # Save result
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        result_path = str(RESULTS_DIR / f"result_image_{timestamp}.jpg")
        cv2.imwrite(result_path, result_image)

        processing_time = (datetime.now() - start_time).total_seconds()

        # Clean up old results in background
        if background_tasks:
            background_tasks.add_task(cleanup_old_files, RESULTS_DIR)

        return ProcessingResponse(
            status="success",
            message=f"Successfully processed image. Detected {len(tracked_objects)} pigs.",
            result_path=result_path,
            detection_count=len(tracked_objects),
            processing_time=processing_time,
            tracked_objects=tracked_objects
        )

    except Exception as e:
        logger.error(f"Error processing image: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )


@app.post("/api/v1/detect/video", response_model=ProcessingResponse)
async def process_video(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
) -> ProcessingResponse:
    """
    Process an uploaded video for pig detection and tracking.

    Args:
        file: Uploaded video file
        background_tasks: FastAPI background tasks handler

    Returns:
        ProcessingResponse: Processing result including status and video path

    Raises:
        HTTPException: If file format is invalid or processing fails
    """
    try:
        # Validate file format using extension
        file_ext = os.path.splitext(file.filename)[1].lower()
        valid_video_extensions = ['.mp4', '.avi', '.mov', '.mkv']
        if file_ext not in valid_video_extensions:
            raise HTTPException(
                status_code=400,
                detail="Invalid file format. Please upload a video file (supported formats: MP4, AVI, MOV, MKV)."
            )

        start_time = datetime.now()

        # Save uploaded video temporarily
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        temp_path = UPLOAD_DIR / f"temp_video_{timestamp}.mp4"
        result_path = RESULTS_DIR / f"result_video_{timestamp}.mp4"

        with temp_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Process video
        cap = cv2.VideoCapture(str(temp_path))
        if not cap.isOpened():
            raise HTTPException(
                status_code=400,
                detail="Could not open video file."
            )

        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        # Initialize video writer with MP4V codec for better compatibility
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(str(result_path), fourcc, fps, (width, height))

        frame_count = 0
        total_detections = 0

        # Process frames
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame
            detections, tracked_objects = pig_detector.process_single_frame(
                frame)

            # Visualize and save frame
            result_frame = pig_detector.visualize_detections(
                frame, detections, tracked_objects)
            out.write(result_frame)

            frame_count += 1
            total_detections += len(tracked_objects)

        # Clean up
        cap.release()
        out.release()
        temp_path.unlink()  # Remove temporary file

        processing_time = (datetime.now() - start_time).total_seconds()

        # Clean up old files in background
        if background_tasks:
            background_tasks.add_task(cleanup_old_files, RESULTS_DIR)
            background_tasks.add_task(cleanup_old_files, UPLOAD_DIR)

        return ProcessingResponse(
            status="success",
            message=f"Successfully processed video. Processed {frame_count} frames.",
            result_path=str(result_path),
            detection_count=total_detections,
            processing_time=processing_time
        )

    except Exception as e:
        logger.error(f"Error processing video: {str(e)}", exc_info=True)
        # Clean up temporary files if they exist
        if 'temp_path' in locals():
            temp_path.unlink(missing_ok=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing video: {str(e)}"
        )


@app.get("/api/v1/results/{filename}")
async def get_result(filename: str):
    """
    Retrieve a processed result file.

    Args:
        filename: Name of the result file

    Returns:
        FileResponse: The requested file

    Raises:
        HTTPException: If file is not found
    """
    file_path = RESULTS_DIR / filename
    if not file_path.exists():
        raise HTTPException(
            status_code=404,
            detail="Result file not found"
        )
    try:
        # Verify file is readable
        if not os.access(file_path, os.R_OK):
            logger.error(f"File {filename} exists but is not readable")
            raise HTTPException(
                status_code=500,
                detail="File exists but is not accessible"
            )

        # Set appropriate content type and headers for proper streaming
        content_type = "video/mp4" if file_path.suffix.lower() == ".mp4" else "image/jpeg"

        # For video files, ensure they can be properly streamed
        headers = {
            "Accept-Ranges": "bytes",
            "Connection": "keep-alive",
            "Cache-Control": "public, max-age=3600"
        }

        return FileResponse(
            path=str(file_path),
            media_type=content_type,
            headers=headers,
            filename=filename  # Ensures proper filename in download
        )
    except Exception as e:
        logger.error(f"Error serving file {filename}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error accessing result file: {str(e)}"
        )


@app.get("/api/v1/health")
async def health_check():
    """Check API health status."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "detector_status": "initialized" if pig_detector else "not initialized"
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
