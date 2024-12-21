import torch
import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import os

class PigDetector:
    def __init__(self, model_path='weights/best.pt', confidence_threshold=0.5, iou_threshold=0.4):
        """
        Initialize the PigDetector with YOLO model
        Args:
            model_path: Path to the trained YOLO model weights
            confidence_threshold: Minimum confidence threshold for detections
            iou_threshold: IoU threshold for NMS
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.model = self._load_model(model_path)

    def _load_model(self, model_path):
        """
        Load the YOLO model
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
                model = YOLO('yolov8n.pt')
            
            # Configure model settings
            model.conf = self.confidence_threshold
            model.iou = self.iou_threshold
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
        
        try:
            # Run inference
            results = self.model(image, verbose=False)[0]
            detections = []
            
            # Process results
            for r in results.boxes.data.tolist():
                x1, y1, x2, y2, conf, cls = r
                if conf >= self.confidence_threshold:
                    detections.append({
                        'bbox': [int(x1), int(y1), int(x2), int(y2)],
                        'confidence': float(conf),
                        'class': int(cls)
                    })
            
            return detections
        except Exception as e:
            print(f"Error during detection: {str(e)}")
            return []

    def count_pigs(self, image):
        """
        Count the number of pigs in the image
        Args:
            image: Input image
        Returns:
            Number of pigs detected
        """
        detections = self.detect(image)
        return len(detections)

    def visualize_detections(self, image, detections):
        """
        Draw bounding boxes and labels on the image
        Args:
            image: Input image
            detections: List of detections
        Returns:
            Annotated image
        """
        img_copy = image.copy()
        
        for detection in detections:
            bbox = detection['bbox']
            conf = detection['confidence']
            
            # Draw bounding box
            cv2.rectangle(img_copy, 
                         (bbox[0], bbox[1]), 
                         (bbox[2], bbox[3]), 
                         (0, 255, 0), 2)
            
            # Add label with confidence score
            label = f"Pig: {conf:.2f}"
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
        count = len(detections)
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
    image_dir = 'data/images'
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
            
            # Detect pigs
            detections = detector.detect(image)
            print(f"Number of pigs detected: {len(detections)}")
            
            # Print detection details
            for i, det in enumerate(detections, 1):
                print(f"Pig {i}: Confidence: {det['confidence']:.2f}, "
                      f"BBox: {det['bbox']}")
            
            # Visualize detections
            result_image = detector.visualize_detections(image, detections)
            
            # Save result
            output_path = os.path.join(results_dir, f'detected_{image_file}')
            cv2.imwrite(output_path, result_image)
            print(f"Saved detection result to: {output_path}")
            
        except Exception as e:
            print(f"Error processing {image_file}: {str(e)}")
            continue

if __name__ == '__main__':
    demo()
