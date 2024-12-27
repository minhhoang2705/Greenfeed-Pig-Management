import torch
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import os


class PigDetector:
    def __init__(self, model_path='/home/minhtranh/works/pig_detection/runs/detect/train7/weights/best.pt', confidence_threshold=0.5, iou_threshold=0.4):
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
                model = YOLO('yolov11s.pt')

            # Configure model settings
            model.conf = self.confidence_threshold
            model.iou = self.iou_threshold
            # Enable ByteTrack tracking
            model.tracker = "bytetrack.yaml"
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None

    def detect(self, image):
        """
        Detect pigs in the image
        Args:
            image: Input image (BGR format)
        Returns:
            List of detections, each containing bbox coordinates, confidence, and class
        """
        if self.model is None:
            print("Model not loaded.")
            return []

        if image is None or image.size == 0:
            print("Invalid input image")
            return []

        try:
            # Run inference
            results = self.model(image, verbose=False)
            if not results or len(results) == 0:
                print("No results from model inference")
                return []

            result = results[0]  # Get first result
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
            else:
                print("No detections found in the image")

            return detections
        except Exception as e:
            print(f"Error during detection: {str(e)}")
            return []

    def process_frame(self, image):
        """
        Process a frame with detection and ByteTrack tracking
        Args:
            image: Input image
        Returns:
            Tuple of (detections, tracked_objects)
        """
        # Run inference with tracking
        results = self.model.track(image, persist=True, verbose=False, tracker=self.model.tracker)
        if not results or len(results) == 0:
            return [], {}

        result = results[0]
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

        return detections, tracked_objects

    def visualize_detections(self, image, detections, tracked_objects=None):
        """
        Draw bounding boxes and labels on the image
        Args:
            image: Input image
            detections: List of detections
            tracked_objects: Dictionary of tracked objects with their IDs
        Returns:
            Annotated image
        """
        img_copy = image.copy()

        if tracked_objects is not None:
            # Draw tracked objects with IDs
            for track_id, detection in tracked_objects.items():
                bbox = detection['bbox']
                conf = detection['confidence']

                # Draw bounding box
                cv2.rectangle(img_copy,
                              (bbox[0], bbox[1]),
                              (bbox[2], bbox[3]),
                              (0, 255, 0), 2)

                # Add label with ID and confidence score
                label = f"Pig #{track_id} ({conf:.2f})"
                (label_width, label_height), _ = cv2.getTextSize(
                    label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)

                # Draw label background
                cv2.rectangle(img_copy,
                              (bbox[0], bbox[1] - 25),
                              (bbox[0] + label_width, bbox[1]),
                              (0, 255, 0), -1)

                # Draw label text
                cv2.putText(img_copy, label,
                            (bbox[0], bbox[1] - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        # Add total count
        count = len(tracked_objects) if tracked_objects else len(detections)
        cv2.putText(img_copy, f'Total Pigs: {count}',
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return img_copy


def demo():
    """
    Run a demo of the pig detector on sample images
    """
    # Create results directory
    results_dir = 'results/detections'
    os.makedirs(results_dir, exist_ok=True)

    # Initialize detector
    detector = PigDetector(confidence_threshold=0.3)

    # Process images
    image_dir = 'data/test/images'
    if not os.path.exists(image_dir):
        print(f"Image directory not found: {image_dir}")
        return

    for image_file in sorted(os.listdir(image_dir)):
        if not image_file.lower().endswith(('.png', '.jpg', '.jpeg')):
            continue

        image_path = os.path.join(image_dir, image_file)
        print(f"\nProcessing {image_file}...")

        try:
            # Read image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Could not read image: {image_path}")
                continue

            # Process frame
            detections, tracked_objects = detector.process_frame(image)
            print(f"Number of pigs detected: {len(tracked_objects)}")

            # Print detection details
            for object_id, bbox in tracked_objects.items():
                conf = next((d['confidence']
                            for d in detections if d['bbox'] == bbox), 0.0)
                print(f"Pig #{object_id}: Confidence: {conf:.2f}, BBox: {bbox}")

            # Visualize detections
            result_image = detector.visualize_detections(
                image, detections, tracked_objects)

            # Save result
            output_path = os.path.join(results_dir, f'detected_{image_file}')
            cv2.imwrite(output_path, result_image)
            print(f"Saved detection result to: {output_path}")

        except Exception as e:
            print(f"Error processing {image_file}: {str(e)}")
            continue


if __name__ == '__main__':
    demo()
