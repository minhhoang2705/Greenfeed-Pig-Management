import cv2
import argparse
from modules.pig_detector import PigDetector
from pathlib import Path

def run_demo(image_path=None, video_path=None, camera_index=None, weights_path='weights/best.pt'):
    """
    Run pig detection demo on image, video, or camera feed
    Args:
        image_path: Path to input image
        video_path: Path to input video
        camera_index: Camera device index for live feed
        weights_path: Path to YOLO weights
    """
    # Initialize detector
    detector = PigDetector(
        model_path=weights_path,
        confidence_threshold=0.3,
        iou_threshold=0.45
    )
    
    if image_path:
        # Process single image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not read image: {image_path}")
            return
        
        detections = detector.detect(image)
        result = detector.visualize_detections(image, detections)
        
        # Save and display result
        output_path = f'results/detections/demo_result_{Path(image_path).name}'
        cv2.imwrite(output_path, result)
        print(f"Saved result to: {output_path}")
        
        # Display result
        cv2.imshow('Pig Detection Result', result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    elif video_path or camera_index is not None:
        # Process video or camera feed
        cap = cv2.VideoCapture(camera_index if camera_index is not None else video_path)
        if not cap.isOpened():
            print("Error opening video source")
            return
        
        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # Initialize video writer if processing video file
        if video_path:
            output_path = f'results/detections/demo_result_{Path(video_path).name}'
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame
            detections = detector.detect(frame)
            result = detector.visualize_detections(frame, detections)
            
            # Save frame if processing video
            if video_path:
                out.write(result)
            
            # Display result
            cv2.imshow('Pig Detection', result)
            
            # Break loop on 'q' press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        # Cleanup
        cap.release()
        if video_path:
            out.release()
        cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Pig Detection Demo')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-i', '--image', help='Path to input image')
    group.add_argument('-v', '--video', help='Path to input video')
    group.add_argument('-c', '--camera', type=int, help='Camera device index')
    parser.add_argument('-w', '--weights', default='weights/best.pt',
                        help='Path to YOLO weights file')
    
    args = parser.parse_args()
    
    # Create results directory
    Path('results/detections').mkdir(parents=True, exist_ok=True)
    
    # Run demo
    run_demo(
        image_path=args.image,
        video_path=args.video,
        camera_index=args.camera,
        weights_path=args.weights
    )

if __name__ == '__main__':
    main()
