import torch
import cv2
import numpy as np
from pig_detector import PigDetector
import time
import os

class Deployment:
    def __init__(self, model_path='models/trained_pig_detector.pth', confidence_threshold=0.5, iou_threshold=0.4):
        self.model = self._load_model(model_path)
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device).eval()

    def _load_model(self, model_path):
        try:
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=False)
            model.load_state_dict(torch.load(model_path, map_location=self.device))
            return model
        except Exception as e:
            print(f"Error loading model: {e}")
            return None

    def _preprocess_image(self, image):
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = cv2.resize(image, (640, 640))
        image = image.transpose((2, 0, 1))
        image = np.ascontiguousarray(image)
        image = torch.from_numpy(image).float() / 255.0
        return image.unsqueeze(0).to(self.device)

    def _postprocess_detections(self, detections):
        detections = detections.xyxy[0].cpu().numpy()
        filtered_detections = []
        for *xyxy, conf, cls in detections:
            if conf >= self.confidence_threshold and int(cls) == 0: # Assuming pig is class 0
                filtered_detections.append({
                    'bbox': [int(x) for x in xyxy],
                    'confidence': float(conf)
                })
        return filtered_detections

    def detect(self, image):
        if self.model is None:
            print("Model not loaded.")
            return []
        
        processed_image = self._preprocess_image(image)
        with torch.no_grad():
            detections = self.model(processed_image)
        
        return self._postprocess_detections(detections)

    def count_pigs(self, image):
        detections = self.detect(image)
        return len(detections)

    def run_inference(self, image_path):
        try:
            image = cv2.imread(image_path)
            if image is None:
                raise FileNotFoundError(f"Could not read image file: {image_path}")
        except FileNotFoundError as e:
            print(e)
            return

        detections = self.detect(image)
        pig_count = self.count_pigs(image)

        print(f"Number of pigs detected: {pig_count}")
        for detection in detections:
            print(f"  - Bounding Box: {detection['bbox']}, Confidence: {detection['confidence']}")

        # Draw bounding boxes on the image
        for detection in detections:
            bbox = detection['bbox']
            cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
        
        # Display the image with bounding boxes
        cv2.imshow('Pig Detection', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

if __name__ == '__main__':
    # Example usage
    deployment = Deployment()
    image_path = 'test_image.jpg' # Replace with your image path
    deployment.run_inference(image_path)
