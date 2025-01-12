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

        # Save raw contents to debug file
        debug_path = Path("debug_upload.bin")
        with debug_path.open("wb") as f:
            f.write(contents)
        logger.info(f"Saved raw upload data to {debug_path}")

        try:
            # Convert to numpy array and validate
            nparr = np.frombuffer(contents, np.uint8)
            logger.info(f"Created numpy array of size {nparr.size} bytes")
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

            # Validate numpy array before decoding
            if nparr.size < 100:  # Minimum expected size for an image
                logger.error(
                    f"Invalid image data size: {nparr.size} bytes")
                raise HTTPException(
                    status_code=400,
                    detail="Invalid image data - file size too small"
                )

            # Decode image with detailed error handling
            try:
                logger.info(
                    f"Attempting to decode image with OpenCV, buffer size: {nparr.size} bytes")
                image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                if image is None:
                    logger.error(
                        f"Failed to decode image. File size: {len(contents)} bytes, numpy array size: {nparr.size}")
                    raise HTTPException(
                        status_code=400,
                        detail="Could not decode image. The file may be corrupted or in an unsupported format."
                    )

                # Validate decoded image dimensions
                if image.size == 0 or image.shape[0] == 0 or image.shape[1] == 0:
                    logger.error(
                        f"Invalid image dimensions after decoding: {image.shape}")
                    raise HTTPException(
                        status_code=400,
                        detail="Invalid image dimensions after decoding. The file may be corrupted."
                    )

                logger.info(
                    f"Successfully decoded image with dimensions: {image.shape}")
            except cv2.error as cv_error:
                logger.error(f"OpenCV decoding error: {str(cv_error)}")
                raise HTTPException(
                    status_code=400,
                    detail=f"Image decoding failed: {str(cv_error)}"
                )
            except Exception as decode_error:
                logger.error(f"Unexpected decoding error: {str(decode_error)}")
                raise HTTPException(
                    status_code=400,
                    detail=f"Unexpected error during image decoding: {str(decode_error)}"
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
        detections, tracked_objects = pig_detector.process_frame(image)

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
            processing_time=processing_time
        )

    except Exception as e:
        logger.error(f"Error processing image: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image: {str(e)}"
        )


@app.post("/api/v1/detect/video", response_model=ProcessingResponse)
@app.post("/api/v1/detect/video", response_model=ProcessingResponse)
async def process_video(
    file: UploadFile = File(...),
    background_tasks: BackgroundTasks = None
) -> ProcessingResponse:
    """
    Process an uploaded video for pig detection and tracking.
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
        temp_path = UPLOAD_DIR / f"temp_video_{timestamp}{file_ext}"
        result_path = RESULTS_DIR / f"result_video_{timestamp}{file_ext}"

        with temp_path.open("wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        cap = cv2.VideoCapture(str(temp_path))
        if not cap.isOpened():
            raise HTTPException(
                status_code=400,
                detail="Could not open video file."
            )

        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # IMPORTANT: Use float here, not int
        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            # Some videos might report 0 or negative for FPS
            # Fallback to something standard or raise an error
            fps = 30.0

        frame_count_input = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count_input / fps if fps > 0 else 0

        logger.info(f"Input video properties - Width: {width}, Height: {height}, "
                    f"FPS: {fps}, Frame count: {frame_count_input}, "
                    f"Duration: {duration:.2f}s")

        # Try H264 codec first, fallback to MP4V if unavailable
        try:
            fourcc = cv2.VideoWriter_fourcc(*'H264')
            writer = cv2.VideoWriter(str(result_path), fourcc, fps, (width, height))
            if not writer.isOpened():
                logger.warning("H264 codec not available, falling back to MP4V")
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                writer = cv2.VideoWriter(str(result_path), fourcc, fps, (width, height))
                if not writer.isOpened():
                    raise HTTPException(
                        status_code=500,
                        detail="Failed to initialize video writer with both H264 and MP4V codecs."
                    )
        except Exception as e:
            logger.error(f"Video writer initialization error: {str(e)}")
            raise HTTPException(
                status_code=500,
                detail=f"Video writer initialization failed: {str(e)}"
            )

        frame_count = 0
        unique_pig_ids = set()
        frame_detections = {}

        # Process frames
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Detect + track
            detections, tracked_objects = pig_detector.process_frame(frame)

            # Update stats
            frame_detections[frame_count] = {
                'count': len(tracked_objects),
                'ids': set(tracked_objects.keys()) if isinstance(tracked_objects, dict) else set()
            }
            if isinstance(tracked_objects, dict):
                unique_pig_ids.update(tracked_objects.keys())

            # Visualize results
            result_frame = pig_detector.visualize_detections(
                frame, detections, tracked_objects
            )

            # Just write the frame (do NOT check writer.write(...) in an if-condition)
            writer.write(result_frame)

            frame_count += 1

        total_unique_pigs = len(unique_pig_ids)
        avg_detections_per_frame = 0
        if frame_count > 0:
            avg_detections_per_frame = sum(
                f['count'] for f in frame_detections.values()
            ) / frame_count

        logger.info("Video Processing Statistics:")
        logger.info(f"Total frames processed: {frame_count}")
        logger.info(f"Total unique pigs detected: {total_unique_pigs}")
        logger.info(f"Average detections per frame: {avg_detections_per_frame:.2f}")

        # Clean up resources
        cap.release()
        writer.release()

        # Verify output
        if not result_path.exists():
            raise HTTPException(
                status_code=500,
                detail="Failed to create output video file."
            )
        file_size = result_path.stat().st_size
        if file_size < 1024:  # Arbitrary check for suspiciously small
            result_path.unlink(missing_ok=True)
            raise HTTPException(
                status_code=500,
                detail="Output video file is invalid or corrupted."
            )

        logger.info(
            f"Successfully created output video: {result_path} ({file_size} bytes)"
        )
        temp_path.unlink(missing_ok=True)

        processing_time = (datetime.now() - start_time).total_seconds()

        # Schedule old file cleanup in background
        if background_tasks:
            background_tasks.add_task(cleanup_old_files, RESULTS_DIR)
            background_tasks.add_task(cleanup_old_files, UPLOAD_DIR)

        return ProcessingResponse(
            status="success",
            message=f"Successfully processed video. Processed {frame_count} frames.",
            result_path=str(result_path),
            detection_count=total_unique_pigs,
            processing_time=processing_time
        )

    except Exception as e:
        logger.error(f"Error processing video: {str(e)}", exc_info=True)
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
            filename=filename
        )
    except Exception as e:
        logger.error(f"Error serving file {filename}: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Error accessing result file: {str(e)}"
        )
