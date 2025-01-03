import torch
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import os
import time
from collections import defaultdict


class PigDetector:
    def __init__(self, model_path='/home/minhtranh/works/pig_detection/pig_detection_project/experiment_24/weights/best.pt', confidence_threshold=0.5, iou_threshold=0.4):
        """
        Initialize the PigDetector with YOLO model and ByteTrack
        Args:
            model_path: Path to the trained YOLO model weights
            confidence_threshold: Minimum confidence threshold for detections
            iou_threshold: IoU threshold for NMS
        """
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.model = self._load_model(model_path)

        # Tracking history
        self.tracking_history = defaultdict(list)
        self.total_count = 0
        self.active_tracks = set()

    def _load_model(self, model_path):
        """
        Load the YOLO model with ByteTrack
        Args:
            model_path: Path to the model weights
        Returns:
            YOLO model instance
        """
        try:
            if Path(model_path).exists():
                print(f"Loading custom YOLO model from {model_path}...")
                model = YOLO(model_path)
            else:
                print("Custom model not found, loading pretrained YOLO model...")
                model = YOLO('weights/yolov8s.pt')

            # Configure model settings
            model.conf = self.confidence_threshold
            model.iou = self.iou_threshold
            # Enable ByteTrack tracking
            model.tracker = "/home/minhtranh/works/pig_detection/configs/pig_tracker.yaml"
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None

    def detect(self, images):
        """
        Detect pigs in a batch of images
        Args:
            images: List of input images (BGR format)
        Returns:
            List of detection lists, each containing bbox coordinates, confidence, and class
        """
        if self.model is None:
            print("Model not loaded.")
            return [[] for _ in images]

        # Filter out invalid images
        valid_images = [img for img in images if img is not None and img.size > 0]
        if not valid_images:
            print("No valid images in batch")
            return [[] for _ in images]

        try:
            # Run batch inference
            batch_results = self.model(valid_images, verbose=False)
            if not batch_results:
                print("No results from model inference")
                return [[] for _ in images]

            all_detections = []
            for result in batch_results:
                detections = []
                # Check if boxes attribute exists and has data
                if hasattr(result, 'boxes') and len(result.boxes) > 0:
                    # Process results
                    for r in result.boxes.data.tolist():
                        if len(r) >= 6:  # Ensure we have all required values
                            x1, y1, x2, y2, conf, cls = r
                            if conf >= self.confidence_threshold:
                                detections.append({
                                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                    'confidence': float(conf),
                                    'class': int(cls)
                                })
                all_detections.append(detections)

            # Return detections in same order as input images
            return all_detections
        except Exception as e:
            print(f"Error during detection: {str(e)}")
            return []

    def process_frame(self, image):
        """
        Process a single frame with detection and ByteTrack tracking
        Args:
            image: Input image (numpy array)
        Returns:
            Tuple of (detections, tracked_objects) where:
                detections: List of detection dictionaries
                tracked_objects: Dictionary of tracked objects with IDs as keys
        Raises:
            ValueError: If input image is invalid
        """
        if not isinstance(image, np.ndarray) or image.size == 0:
            raise ValueError("Invalid input image")
            
        try:
            # Process single frame using sequential processing
            results = self.process_frames_sequential([image])
            if results:
                return results[0]
            return [], {}
        except Exception as e:
            print(f"Error processing frame: {str(e)}")
            return [], {}

    def process_frames_sequential(self, images):
        """
        Process multiple frames sequentially with detection and ByteTrack tracking
        Args:
            images: List of input images
        Returns:
            List of tuples (detections, tracked_objects)
        """
        if self.model is None or not images:
            return [([], {}) for _ in images]

        # Filter valid images
        valid_images = [img for img in images if img is not None and img.size > 0]
        if not valid_images:
            return [([], {}) for _ in images]

        # Run batch inference with tracking
        batch_results = self.model.track(
            valid_images, persist=True, verbose=False, tracker="bytetrack.yaml")
        if not batch_results:
            return [([], {}) for _ in images]

        all_results = []
        for result in batch_results:
            detections = []
            tracked_objects = {}

            # Process tracked objects
            if hasattr(result, 'boxes') and len(result.boxes) > 0:
                for box in result.boxes:
                    # Get box data
                    box_data = box.data[0]
                    x1, y1, x2, y2, conf, cls = map(float, box_data[:6])

                    # Get tracking ID
                    track_id = int(box.id[0]) if box.id is not None else None

                    detection = {
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': float(conf),
                        'class': int(cls),
                        'track_id': track_id
                    }
                    detections.append(detection)

                    if track_id is not None:
                        tracked_objects[track_id] = detection

                        # Update tracking history
                        self.tracking_history[track_id].append({
                            'frame_time': time.time(),
                            'bbox': detection['bbox'],
                            'confidence': detection['confidence']
                        })

                        # Update total count if new track
                        if track_id not in self.active_tracks:
                            self.total_count += 1
                            self.active_tracks.add(track_id)

                # Remove lost tracks
                lost_tracks = set(self.active_tracks) - set(tracked_objects.keys())
                self.active_tracks -= lost_tracks

            all_results.append((detections, tracked_objects))

        return all_results

    def process_frames_parallel(self, images, max_workers=4):
        """
        Process multiple frames in parallel using ThreadPoolExecutor
        Args:
            images: List of input images
            max_workers: Maximum number of parallel workers
        Returns:
            List of tuples (detections, tracked_objects)
        """
        from concurrent.futures import ThreadPoolExecutor

        # Split images into chunks for parallel processing
        chunk_size = max(1, len(images) // max_workers)
        image_chunks = [images[i:i + chunk_size] 
                       for i in range(0, len(images), chunk_size)]

        results = []
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # Process each chunk in parallel
            futures = [executor.submit(self.process_frames, chunk) 
                      for chunk in image_chunks]
            
            # Collect results
            for future in futures:
                try:
                    results.extend(future.result())
                except Exception as e:
                    print(f"Error processing frame chunk: {str(e)}")
                    # Return empty results for failed chunks
                    results.extend([([], {}) for _ in range(len(chunk))])

        return results

    def visualize_detections(self, image, detections, tracked_objects=None):
        """
        Draw bounding boxes, labels, and tracking history on the image
        Args:
            image: Input image
            detections: List of detections
            tracked_objects: Dictionary of tracked objects with their IDs
        Returns:
            Annotated image
        """
        img_copy = image.copy()

        if tracked_objects is not None:
            # Draw tracking history paths
            for track_id, history in self.tracking_history.items():
                if track_id in tracked_objects:
                    # Draw path
                    points = []
                    for entry in history[-30:]:  # Last 30 frames
                        x1, y1, x2, y2 = entry['bbox']
                        center = ((x1 + x2) // 2, (y1 + y2) // 2)
                        points.append(center)

                    if len(points) > 1:
                        # Draw path with decreasing opacity
                        for i in range(1, len(points)):
                            alpha = i / len(points)
                            color = (0, int(255 * (1 - alpha)),
                                     int(255 * alpha))
                            thickness = max(1, int(3 * (1 - alpha)))
                            cv2.line(
                                img_copy, points[i-1], points[i], color, thickness)

            # Draw tracked objects with IDs
            for track_id, detection in tracked_objects.items():
                bbox = detection['bbox']
                conf = detection['confidence']

                # Draw bounding box
                color = (0, 255, 0)  # Green for active tracks
                if track_id not in self.active_tracks:
                    color = (0, 0, 255)  # Red for lost tracks

                cv2.rectangle(img_copy,
                              (bbox[0], bbox[1]),
                              (bbox[2], bbox[3]),
                              color, 2)

                # Add label with ID and confidence score
                label = f"Pig #{track_id} ({conf:.2f})"
                (label_width, label_height), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

                # Draw label background
                cv2.rectangle(img_copy,
                              (bbox[0], bbox[1] - 25),
                              (bbox[0] + label_width, bbox[1]),
                              color, -1)

                # Draw label text
                cv2.putText(img_copy, label,
                            (bbox[0], bbox[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        # Add total count and active count
        active_count = len(tracked_objects) if tracked_objects else 0
        cv2.putText(img_copy, f'Active Pigs: {active_count}',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return img_copy


def demo(video_path=None, batch_size=16, max_workers=4):
    """
    Run a demo of the pig detector on images or video with optimized processing
    Args:
        video_path: Path to video file (optional)
        batch_size: Number of frames to process in each batch
        max_workers: Maximum number of parallel workers for image processing
    """
    # Create results directory
    results_dir = 'results/detections'
    os.makedirs(results_dir, exist_ok=True)

    # Initialize detector with adjusted confidence threshold
    detector = PigDetector(confidence_threshold=0.5)

    if video_path:
        # Process video
        if not os.path.exists(video_path):
            print(f"Video file not found: {video_path}")
            return

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Could not open video: {video_path}")
            return

        frame_count = 0
        output_path = os.path.join(results_dir, 'output_video.avi')  # Using .avi extension for XVID codec
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        fps = cap.get(cv2.CAP_PROP_FPS)
        frame_size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                      int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        out = cv2.VideoWriter(output_path, fourcc, fps, frame_size)

        # Buffer for batch processing
        frame_buffer = []
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                # Process remaining frames in buffer
                if frame_buffer:
                    try:
                        # Process batch
                        batch_results = detector.process_frames_sequential(frame_buffer)
                        for i, (detections, tracked_objects) in enumerate(batch_results):
                            # Visualize and write each frame
                            result_frame = detector.visualize_detections(
                                frame_buffer[i], detections, tracked_objects)
                            out.write(result_frame)
                    except Exception as e:
                        print(f"Error processing final batch: {str(e)}")
                break

            frame_buffer.append(frame)
            frame_count += 1

            if len(frame_buffer) >= batch_size:
                try:
                    # Process batch
                    batch_results = detector.process_frames_sequential(frame_buffer)
                    for i, (detections, tracked_objects) in enumerate(batch_results):
                        # Visualize and write each frame
                        result_frame = detector.visualize_detections(
                            frame_buffer[i], detections, tracked_objects)
                        out.write(result_frame)
                    
                    # Clear buffer
                    frame_buffer = []
                    print(f"Processed frames {frame_count - batch_size + 1} to {frame_count}")
                except Exception as e:
                    print(f"Error processing batch: {str(e)}")
                    continue

        cap.release()
        out.release()
        cv2.destroyAllWindows()
        print(f"Saved detection results to: {output_path}")

    else:
        # Process images
        image_dir = 'data/test/images'
        if not os.path.exists(image_dir):
            print(f"Image directory not found: {image_dir}")
            return

        # Get all image paths
        image_files = [f for f in sorted(os.listdir(image_dir)) 
                      if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if not image_files:
            print("No valid images found in directory")
            return

        # Process images in parallel
        try:
            # Read all images
            images = [cv2.imread(os.path.join(image_dir, f)) for f in image_files]
            
            # Process in parallel
            results = detector.process_frames_parallel(images, max_workers=max_workers)
            
            # Save results
            for i, (detections, tracked_objects) in enumerate(results):
                image_file = image_files[i]
                print(f"\nProcessing {image_file}...")
                print(f"Number of pigs detected: {len(tracked_objects)}")

                # Print detection details
                for object_id, bbox in tracked_objects.items():
                    conf = next((d['confidence']
                                for d in detections if d['bbox'] == bbox), 0.0)
                    print(f"Pig #{object_id}: Confidence: {conf:.2f}, BBox: {bbox}")

                # Visualize detections
                result_image = detector.visualize_detections(
                    images[i], detections, tracked_objects)

                # Save result
                output_path = os.path.join(
                    results_dir, f'detected_{image_file}')
                cv2.imwrite(output_path, result_image)
                print(f"Saved detection result to: {output_path}")

        except Exception as e:
            print(f"Error processing images: {str(e)}")


if __name__ == '__main__':
    # Run demo on video if available, otherwise on images
    video_path = 'pg5.mp4' if os.path.exists('pg5.mp4') else None
    demo(video_path=video_path)
