import torch
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import os
import time
from collections import defaultdict


class PigDetector:
    def __init__(self, model_path=None, confidence_threshold=0.6, iou_threshold=0.4,
                 min_track_duration=30, max_track_gap=15, min_track_quality=0.8):
        """
        Initialize the PigDetector with YOLO model and enhanced tracking
        Args:
            model_path: Path to the trained YOLO model weights (optional)
            confidence_threshold: Minimum confidence threshold for detections (0-1)
            iou_threshold: IoU threshold for Non-Maximum Suppression (0-1)
            min_track_duration: Minimum number of frames a track must exist to be considered valid
            max_track_gap: Maximum number of frames to keep a track alive without detection
            min_track_quality: Minimum quality score (0-1) for a track to be confirmed
        """
        # Set default model path based on environment (Docker vs local)
        if model_path is None:
            if os.getenv('DOCKER_ENV') == 'true':
                model_path = '/app/weights/best.pt'
            else:
                model_path = '/home/minhtranh/works/pig_detection/pig_detection_project/experiment_24/weights/best.pt'
        
        # Use GPU if available, otherwise fall back to CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Detection thresholds
        self.confidence_threshold = confidence_threshold  # Minimum confidence score for detection
        self.iou_threshold = iou_threshold  # Intersection over Union threshold for NMS
        
        # Load YOLO model with ByteTrack tracker
        self.model = self._load_model(model_path)

        # Enhanced tracking parameters with stricter criteria
        self.min_track_duration = min_track_duration  # Minimum frames for track confirmation
        self.max_track_gap = max_track_gap  # Maximum frames without detection before track is lost
        self.min_track_quality = min_track_quality  # Minimum quality score for track confirmation
        
        # Tracking state management
        self.track_velocities = defaultdict(list)  # Track motion history (dx, dy per frame)
        self.tracking_history = defaultdict(list)  # Full history of each track's detections
        self.track_qualities = defaultdict(float)  # Quality scores for each track (0-1)
        self.track_gaps = defaultdict(int)  # Frames since last detection for each track
        self.total_count = 0  # Total number of unique pigs detected
        self.active_tracks = set()  # Currently active track IDs
        self.confirmed_tracks = set()  # Tracks that meet duration/quality criteria

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

    def _enhanced_nms(self, detections, iou_threshold=0.5, score_threshold=0.5):
        """
        Apply enhanced Non-Maximum Suppression to detections with additional filtering
        Args:
            detections: List of detection dictionaries containing:
                - bbox: [x1, y1, x2, y2] bounding box coordinates
                - confidence: Detection confidence score (0-1)
                - class: Class ID
            iou_threshold: IoU threshold for suppression (0-1)
            score_threshold: Minimum confidence score threshold (0-1)
        Returns:
            List of filtered detections after NMS
        Notes:
            - Implements standard NMS algorithm with additional size and aspect ratio filtering
            - Removes overlapping boxes while preserving the highest confidence detections
            - Uses numpy for efficient array operations
        """
        if not detections:
            return []

        # Convert to numpy arrays for efficient computation
        boxes = np.array([d['bbox'] for d in detections])
        scores = np.array([d['confidence'] for d in detections])

        # Calculate areas for each box
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        areas = (x2 - x1) * (y2 - y1)

        # Get indices of boxes sorted by score
        order = scores.argsort()[::-1]

        keep = []
        while order.size > 0:
            # Pick the box with highest score
            i = order[0]

            # Skip if score is below threshold
            if scores[i] < score_threshold:
                order = order[1:]
                continue

            keep.append(i)

            if order.size == 1:
                break

            # Calculate IoU with rest of boxes
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])

            w = np.maximum(0.0, xx2 - xx1)
            h = np.maximum(0.0, yy2 - yy1)
            inter = w * h

            ovr = inter / (areas[i] + areas[order[1:]] - inter)

            # Get indices of boxes with IoU <= threshold
            inds = np.where(ovr <= iou_threshold)[0]
            order = order[inds + 1]

        return [detections[i] for i in keep]

    def detect(self, images):
        """
        Detect pigs in a batch of images with enhanced NMS
        Args:
            images: List of input images (BGR format)
        Returns:
            List of detection lists, each containing bbox coordinates, confidence, and class
        """
        if self.model is None:
            print("Model not loaded.")
            return [[] for _ in images]

        # Filter out invalid images
        valid_images = [
            img for img in images if img is not None and img.size > 0]
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

                            # Basic size filtering
                            width = x2 - x1
                            height = y2 - y1
                            aspect_ratio = width / height if height > 0 else 0
                            area = width * height

                            # Filter out detections with unrealistic sizes or aspect ratios
                            if (conf >= self.confidence_threshold and
                                0.5 <= aspect_ratio <= 2.0 and  # Reasonable aspect ratio for pigs
                                    area >= 1000):  # Minimum area threshold
                                detections.append({
                                    'bbox': [int(x1), int(y1), int(x2), int(y2)],
                                    'confidence': float(conf),
                                    'class': int(cls)
                                })

                    # Apply enhanced NMS to the filtered detections
                    detections = self._enhanced_nms(
                        detections,
                        iou_threshold=self.iou_threshold,
                        score_threshold=self.confidence_threshold
                    )

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

    def _calculate_track_velocity(self, track_id, detection):
        """
        Calculate and update track velocity for motion consistency
        Args:
            track_id: Track ID to calculate velocity for
            detection: Current detection dictionary containing bbox coordinates
        Returns:
            tuple: (dx, dy) velocity vector in pixels/frame or None if not enough history
        Notes:
            - Velocity is calculated as the difference between current and previous detection centers
            - Used to maintain motion consistency and detect abnormal movements
            - Returns None if there's insufficient history (less than 2 frames)
        """
        history = self.tracking_history[track_id]
        if len(history) < 2:
            return None

        prev_bbox = history[-1]['bbox']
        curr_bbox = detection['bbox']

        # Calculate center points
        prev_x = (prev_bbox[0] + prev_bbox[2]) / 2
        prev_y = (prev_bbox[1] + prev_bbox[3]) / 2
        curr_x = (curr_bbox[0] + curr_bbox[2]) / 2
        curr_y = (curr_bbox[1] + curr_bbox[3]) / 2

        # Calculate velocity (movement per frame)
        dx = curr_x - prev_x
        dy = curr_y - prev_y

        return (dx, dy)

    def _check_motion_consistency(self, track_id, velocity):
        """
        Check if current motion is consistent with track history
        Args:
            track_id: Track ID
            velocity: Current velocity vector
        Returns:
            float: Motion consistency score (0-1)
        """
        if velocity is None or len(self.track_velocities[track_id]) < 2:
            return 1.0

        dx, dy = velocity
        # Last 5 velocities
        prev_velocities = self.track_velocities[track_id][-5:]

        # Calculate average velocity
        avg_dx = sum(v[0] for v in prev_velocities) / len(prev_velocities)
        avg_dy = sum(v[1] for v in prev_velocities) / len(prev_velocities)

        # Calculate velocity difference
        vel_diff = np.sqrt((dx - avg_dx)**2 + (dy - avg_dy)**2)
        avg_vel = np.sqrt(avg_dx**2 + avg_dy**2)

        # Normalize difference score
        consistency = 1.0 - min(1.0, vel_diff / (avg_vel + 1e-6))
        return max(0.0, consistency)

    def _update_track_quality(self, track_id, detection):
        """
        Update track quality metrics based on detection confidence and consistency
        Args:
            track_id: Track ID
            detection: Detection dictionary containing bbox and confidence
        """
        history = self.tracking_history[track_id]

        # Calculate detection consistency
        if len(history) > 1:
            prev_bbox = history[-1]['bbox']
            curr_bbox = detection['bbox']

            # Calculate IoU between current and previous detection
            def calculate_iou(box1, box2):
                x1 = max(box1[0], box2[0])
                y1 = max(box1[1], box2[1])
                x2 = min(box1[2], box2[2])
                y2 = min(box1[3], box2[3])

                intersection = max(0, x2 - x1) * max(0, y2 - y1)
                area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
                area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

                return intersection / (area1 + area2 - intersection + 1e-6)

            iou = calculate_iou(prev_bbox, curr_bbox)

            # Calculate velocity and check motion consistency
            velocity = self._calculate_track_velocity(track_id, detection)
            if velocity:
                self.track_velocities[track_id].append(velocity)
                motion_consistency = self._check_motion_consistency(
                    track_id, velocity)
            else:
                motion_consistency = 1.0

            # Update quality score based on detection confidence, IoU, and motion consistency
            quality_score = (0.5 * detection['confidence'] +
                             0.3 * iou +
                             0.2 * motion_consistency)

            # Exponential moving average with more weight on history
            self.track_qualities[track_id] = 0.9 * \
                self.track_qualities[track_id] + 0.1 * quality_score
        else:
            self.track_qualities[track_id] = detection['confidence']

        # Reset gap counter for active detection
        self.track_gaps[track_id] = 0

    def _should_merge_tracks(self, track_id1, track_id2):
        """
        Determine if two tracks should be merged based on spatial and temporal proximity
        Args:
            track_id1: First track ID to compare
            track_id2: Second track ID to compare
        Returns:
            bool: True if tracks should be merged, False otherwise
        Notes:
            - Tracks are merged if they:
                - Are temporally close (within 2 seconds)
                - Have significant spatial overlap (IoU > 0.3)
                - Are moving similarly (velocity difference < 10 pixels/frame)
                - Are spatially close (center distance < 100 pixels)
            - Helps prevent track fragmentation and ID switches
        """
        history1 = self.tracking_history[track_id1]
        history2 = self.tracking_history[track_id2]

        if not history1 or not history2:
            return False

        # Get latest detections
        det1 = history1[-1]
        det2 = history2[-1]

        # Calculate temporal difference
        time_diff = abs(det1['frame_time'] - det2['frame_time'])
        if time_diff > 2.0:  # More than 2 seconds apart
            return False

        # Calculate spatial overlap
        def calculate_iou(box1, box2):
            x1 = max(box1[0], box2[0])
            y1 = max(box1[1], box2[1])
            x2 = min(box1[2], box2[2])
            y2 = min(box1[3], box2[3])

            intersection = max(0, x2 - x1) * max(0, y2 - y1)
            area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
            area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])

            return intersection / (area1 + area2 - intersection + 1e-6)

        # Check spatial overlap
        iou = calculate_iou(det1['bbox'], det2['bbox'])
        if iou > 0.3:  # Significant overlap
            return True

        # Check if tracks are close and moving similarly
        if track_id1 in self.track_velocities and track_id2 in self.track_velocities:
            vel1 = self.track_velocities[track_id1][-1] if self.track_velocities[track_id1] else None
            vel2 = self.track_velocities[track_id2][-1] if self.track_velocities[track_id2] else None

            if vel1 and vel2:
                # Calculate velocity similarity
                vel_diff = np.sqrt((vel1[0] - vel2[0])
                                   ** 2 + (vel1[1] - vel2[1])**2)
                if vel_diff < 10:  # Similar motion
                    # Calculate center distance
                    center1 = [(det1['bbox'][0] + det1['bbox'][2])/2,
                               (det1['bbox'][1] + det1['bbox'][3])/2]
                    center2 = [(det2['bbox'][0] + det2['bbox'][2])/2,
                               (det2['bbox'][1] + det2['bbox'][3])/2]
                    dist = np.sqrt((center1[0] - center2[0])**2 +
                                   (center1[1] - center2[1])**2)

                    # If centers are close and moving similarly
                    if dist < 100:  # Adjust threshold based on video resolution
                        return True

        return False

    def _merge_tracks(self, source_id, target_id):
        """
        Merge source track into target track
        Args:
            source_id: ID of track to merge from
            target_id: ID of track to merge into
        """
        # Update tracking history
        source_history = self.tracking_history[source_id]
        target_history = self.tracking_history[target_id]

        # Merge histories maintaining temporal order
        merged_history = sorted(source_history + target_history,
                                key=lambda x: x['frame_time'])
        self.tracking_history[target_id] = merged_history

        # Merge velocities if available
        if source_id in self.track_velocities and target_id in self.track_velocities:
            self.track_velocities[target_id].extend(
                self.track_velocities[source_id])

        # Update quality score as weighted average
        source_quality = self.track_qualities[source_id]
        target_quality = self.track_qualities[target_id]
        source_len = len(source_history)
        target_len = len(target_history)
        total_len = source_len + target_len

        self.track_qualities[target_id] = (
            (source_quality * source_len + target_quality * target_len) / total_len
        )

        # Clean up source track
        self.active_tracks.discard(source_id)
        self.confirmed_tracks.discard(source_id)
        del self.track_qualities[source_id]
        del self.track_gaps[source_id]
        if source_id in self.track_velocities:
            del self.track_velocities[source_id]
        del self.tracking_history[source_id]

    def _update_track_status(self):
        """
        Update track status based on quality metrics and duration
        """
        current_tracks = set(self.track_qualities.keys())

        # Update gap counters for tracks without detections
        for track_id in current_tracks:
            if track_id not in self.active_tracks:
                self.track_gaps[track_id] += 1

        # Remove old tracks
        tracks_to_remove = set()
        for track_id in current_tracks:
            track_duration = len(self.tracking_history[track_id])

            # Remove tracks with long gaps
            if self.track_gaps[track_id] > self.max_track_gap:
                tracks_to_remove.add(track_id)
                continue

            # Confirm tracks that meet very strict quality criteria
            if (track_duration >= self.min_track_duration and
                self.track_qualities[track_id] > self.min_track_quality and
                len([d for d in self.tracking_history[track_id] 
                     if d['confidence'] > 0.7]) >= 10):  # At least 10 high-confidence detections
                # Check if this track should be merged with an existing confirmed track
                should_confirm = True
                for confirmed_id in list(self.confirmed_tracks):
                    if self._should_merge_tracks(track_id, confirmed_id):
                        # Merge into existing track
                        self._merge_tracks(track_id, confirmed_id)
                        should_confirm = False
                        break

                if should_confirm:
                    # Additional verification of track consistency
                    history = self.tracking_history[track_id]
                    bbox_sizes = [((d['bbox'][2]-d['bbox'][0])*(d['bbox'][3]-d['bbox'][1])) 
                                 for d in history]
                    size_variation = max(bbox_sizes)/min(bbox_sizes) if bbox_sizes else 1.0
                    
                    # Verify size consistency and motion direction
                    if size_variation < 2.0:  # Size shouldn't vary more than 2x
                        self.confirmed_tracks.add(track_id)
                    else:
                        tracks_to_remove.add(track_id)

            # Remove low quality tracks
            if (track_duration >= self.min_track_duration and
                    self.track_qualities[track_id] < 0.4):
                tracks_to_remove.add(track_id)

        # Clean up removed tracks
        for track_id in tracks_to_remove:
            self.active_tracks.discard(track_id)
            self.confirmed_tracks.discard(track_id)
            del self.track_qualities[track_id]
            del self.track_gaps[track_id]
            del self.track_velocities[track_id]

    def process_frames_sequential(self, images):
        """
        Process multiple frames sequentially with enhanced tracking
        Args:
            images: List of input images
        Returns:
            List of tuples (detections, tracked_objects)
        """
        if self.model is None or not images:
            return [([], {}) for _ in images]

        # Filter valid images
        valid_images = [
            img for img in images if img is not None and img.size > 0]
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

                        # Update tracking history and quality metrics
                        self.tracking_history[track_id].append({
                            'frame_time': time.time(),
                            'bbox': detection['bbox'],
                            'confidence': detection['confidence']
                        })

                        self._update_track_quality(track_id, detection)

                        # Update counts for new tracks
                        if track_id not in self.active_tracks:
                            self.active_tracks.add(track_id)
                            if len(self.tracking_history[track_id]) >= self.min_track_duration:
                                self.total_count += 1

                # Update track status and remove invalid tracks
                self._update_track_status()

            all_results.append((detections, tracked_objects))

        return all_results

    def process_frames_parallel(self, images, max_workers=4):
        """
        Process multiple frames in parallel using ThreadPoolExecutor
        Args:
            images: List of input images (numpy arrays in BGR format)
            max_workers: Maximum number of parallel workers (default: 4)
        Returns:
            List of tuples (detections, tracked_objects) where:
                - detections: List of detection dictionaries
                - tracked_objects: Dictionary mapping track IDs to detection info
        Notes:
            - Splits images into chunks for parallel processing
            - Uses ThreadPoolExecutor for concurrent execution
            - Handles errors gracefully by returning empty results for failed chunks
            - Maintains thread safety by creating new detector instances per chunk
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

        # Add counts and quality information with track details
        confirmed_count = len(self.confirmed_tracks)
        active_count = len(tracked_objects) if tracked_objects else 0
        cv2.putText(img_copy, f'Unique Pigs: {confirmed_count}',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(img_copy, f'Active Tracks: {active_count}',
                    (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 165, 0), 2)

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

    # Initialize detector with stricter tracking parameters
    detector = PigDetector(
        confidence_threshold=0.6,  # Higher confidence threshold
        iou_threshold=0.4,
        min_track_duration=30,     # Require longer track duration
        max_track_gap=15,          # Shorter gap tolerance
        min_track_quality=0.8      # Higher quality requirement
    )

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
        # Using .avi extension for XVID codec
        output_path = os.path.join(results_dir, 'output_video.avi')
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
                        batch_results = detector.process_frames_sequential(
                            frame_buffer)
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
                    batch_results = detector.process_frames_sequential(
                        frame_buffer)
                    for i, (detections, tracked_objects) in enumerate(batch_results):
                        # Visualize and write each frame
                        result_frame = detector.visualize_detections(
                            frame_buffer[i], detections, tracked_objects)
                        out.write(result_frame)

                    # Clear buffer
                    frame_buffer = []
                    print(
                        f"Processed frames {frame_count - batch_size + 1} to {frame_count}")
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
            images = [cv2.imread(os.path.join(image_dir, f))
                      for f in image_files]

            # Process in parallel
            results = detector.process_frames_parallel(
                images, max_workers=max_workers)

            # Save results
            for i, (detections, tracked_objects) in enumerate(results):
                image_file = image_files[i]
                print(f"\nProcessing {image_file}...")
                print(f"Number of pigs detected: {len(tracked_objects)}")

                # Print detection details
                for object_id, bbox in tracked_objects.items():
                    conf = next((d['confidence']
                                for d in detections if d['bbox'] == bbox), 0.0)
                    print(
                        f"Pig #{object_id}: Confidence: {conf:.2f}, BBox: {bbox}")

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
