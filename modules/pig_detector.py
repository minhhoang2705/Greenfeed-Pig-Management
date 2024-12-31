import torch
import cv2
import numpy as np
import os

# Fix Qt platform issue
os.environ['QT_QPA_PLATFORM'] = 'offscreen'

class PigDetector:
    # COCO dataset class names
    COCO_CLASSES = {
        16: 'dog',
        17: 'sheep',
        19: 'cow',
        20: 'elephant',
        21: 'bear',
        22: 'zebra'
    }
    def __init__(self, model_path='models/trained_pig_detector.pth', confidence_threshold=0.5, iou_threshold=0.4):
        # Set CUDA device if available
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.model = self._load_model(model_path)
        if self.model is not None:
            self.model.to(self.device).eval()

    def _load_model(self, model_path):
        try:
            print("Loading YOLOv5s model...")
            model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
            # Configure model settings
            model.conf = self.confidence_threshold  # Confidence threshold
            model.iou = self.iou_threshold  # NMS IoU threshold
            model.classes = None  # Detect all classes
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

    def _postprocess_detections(self, results):
        filtered_detections = []
        
        # Access predictions from the Results object
        if hasattr(results, 'pred') and len(results.pred) > 0:
            predictions = results.pred[0]
            
            # Convert predictions to numpy for processing
            if predictions is not None:
                predictions = predictions.cpu().numpy()
                
                for pred in predictions:
                    *xyxy, conf, cls = pred
                    # Look for animals in COCO dataset (including class 16-dog, 17-sheep, 19-cow, etc.)
                    if conf >= self.confidence_threshold and int(cls) in [16, 17, 19, 20, 21, 22]:
                        filtered_detections.append({
                            'bbox': [int(x) for x in xyxy],
                            'confidence': float(conf),
                            'class': int(cls)
                        })
        
        return filtered_detections

    def detect(self, image):
        if self.model is None:
            print("Model not loaded.")
            return []
        
        try:
            # Use the model's built-in preprocessing
            results = self.model(image)
            return self._postprocess_detections(results)
        except Exception as e:
            print(f"Error during detection: {str(e)}")
            return []

    def count_pigs(self, image):
        detections = self.detect(image)
        return len(detections)

if __name__ == '__main__':
    import os
    
    # Create results directory if it doesn't exist
    results_dir = 'results/detections'
    os.makedirs(results_dir, exist_ok=True)
    
    # Example usage with lower confidence threshold for testing
    detector = PigDetector(confidence_threshold=0.3, iou_threshold=0.45)
    
    # Process all images in the data/images directory
    image_dir = 'data/images'
    for image_file in sorted(os.listdir(image_dir)):
        if not image_file.endswith('.png'):
            continue
            
        image_path = os.path.join(image_dir, image_file)
        print(f"\nProcessing {image_file}...")
        
        try:
            image = cv2.imread(image_path)
            if image is None:
                print(f"Could not read image file: {image_path}")
                continue

            # Detect and count pigs
            detections = detector.detect(image)
            pig_count = detector.count_pigs(image)

            # Print results
            print(f"Number of animals detected: {pig_count}")
            for detection in detections:
                class_id = detection.get('class', -1)
                print(f"  - Class: {class_id}, Bounding Box: {detection['bbox']}, Confidence: {detection['confidence']:.2f}")

            # Draw bounding boxes on the image
            for detection in detections:
                bbox = detection['bbox']
                class_id = detection['class']
                class_name = PigDetector.COCO_CLASSES.get(class_id, f'class_{class_id}')
                
                # Different colors for different classes
                color = (
                    (0, 255, 0) if class_id == 19 else  # Green for cow
                    (255, 0, 0) if class_id == 16 else  # Blue for dog
                    (0, 0, 255)                         # Red for others
                )
                
                # Draw bounding box
                cv2.rectangle(image, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                
                # Add class name and confidence score
                label = f"{class_name} {detection['confidence']:.2f}"
                (label_width, label_height), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(image, (bbox[0], bbox[1]-25), (bbox[0]+label_width, bbox[1]), color, -1)
                cv2.putText(image, label, (bbox[0], bbox[1]-5),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
            
            # Save the annotated image
            output_path = os.path.join(results_dir, f'detected_{image_file}')
            cv2.imwrite(output_path, image)
            print(f"Saved detection result to: {output_path}")
                
        except Exception as e:
            print(f"Error processing {image_file}: {str(e)}")
            continue
