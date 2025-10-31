import cv2
import argparse
from modules.pig_detector import PigDetector
from pathlib import Path


def run_demo(image_path=None, video_path=None, camera_index=None, weights_path='/home/minhtranh/works/pig_detection/weights/yolo11n.pt'):
    """
    Run pig detection demo on image, video, or camera feed
    Args:
        image_path: Path to input image
        video_path: Path to input video
        camera_index: Camera device index for live feed
        weights_path: Path to YOLO weights
    """
    # Initialize the pig detector with specified model weights and thresholds
    # confidence_threshold: Minimum confidence score for detection (0.7 = 70%)
    # iou_threshold: Intersection over Union threshold for NMS (0.45 = 45%)
    # Initialize detector
    detector = PigDetector(
        model_path=weights_path,
        confidence_threshold=0.7,
        iou_threshold=0.45
    )

    if image_path:
        # Process single image
        image = cv2.imread(image_path)
        if image is None:
            print(f"Could not read image: {image_path}")
            return

        # Process frame with tracking:
        # 1. Detect pigs in the image using YOLO model
        # 2. Track detected pigs across frames (if applicable)
        detections, tracked_objects = detector.process_frame(image)
        
        # Visualize detections by drawing bounding boxes and labels on the image
        result = detector.visualize_detections(
            image, detections, tracked_objects)

        # Save and display result:
        # 1. Save annotated image to results directory
        # 2. Display result in a window until key press
        output_path = f'results/detections/demo_result_{Path(image_path).name}'
        cv2.imwrite(output_path, result)
        print(f"Saved result to: {output_path}")

        # Display result
        cv2.imshow('Pig Detection Result', result)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    elif video_path or camera_index is not None:
        # Process video or camera feed
        cap = cv2.VideoCapture(
            camera_index if camera_index is not None else video_path)
        if not cap.isOpened():
            print("Error opening video source")
            return

        # Get video properties
        frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))

        # Initialize video writer if processing video file:
        # - XVID codec for AVI format
        # - Same resolution and FPS as input video
        if video_path:
            output_path = f'results/detections/demo_result_{Path(video_path).name}'
            fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Codec for AVI format
            out = cv2.VideoWriter(output_path, fourcc, fps,
                                  (frame_width, frame_height))

        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)
                           ) if video_path else 0

        import time
        total_processing_time = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Start timing
            start_time = time.time()

            # Process frame with tracking
            detections, tracked_objects = detector.process_frame(frame)
            result = detector.visualize_detections(
                frame, detections, tracked_objects)

            # Save frame and show progress
            if video_path:
                if not out.write(result):
                    print(f"Warning: Failed to write frame {frame_count}")
                frame_count += 1
                if frame_count % 10 == 0:
                    print(f"Processed frame {frame_count} of {total_frames}")
                
                # Calculate processing time
                frame_time = time.time() - start_time
                total_processing_time += frame_time
                avg_time = total_processing_time / frame_count
                remaining_time = (total_frames - frame_count) * avg_time
                
                print(
                    f"\rProcessing frame {frame_count}/{total_frames} | "
                    f"Frame time: {frame_time:.2f}s | "
                    f"Avg: {avg_time:.2f}s | "
                    f"ETA: {remaining_time:.1f}s", end="", flush=True)
                if frame_count % 10 == 0:
                    print()  # Newline every 10 frames for better visibility

        # Cleanup
        cap.release()
        if video_path:
            out.release()
        cv2.destroyAllWindows()
        print("\nProcessing complete")


def main():
    """
    Main function to handle command line arguments and run the demo
    """
    # Set up argument parser with mutually exclusive options:
    # - Either image, video, or camera input must be specified
    parser = argparse.ArgumentParser(description='Pig Detection Demo')
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('-i', '--image', help='Path to input image')
    group.add_argument('-v', '--video', help='Path to input video')
    group.add_argument('-c', '--camera', type=int, help='Camera device index')
    
    # Optional argument for custom model weights
    parser.add_argument('-w', '--weights', default='weights/best.pt',
                        help='Path to YOLO weights file')

    # Parse command line arguments
    args = parser.parse_args()

    # Create results directory if it doesn't exist
    Path('results/detections').mkdir(parents=True, exist_ok=True)

    # Run demo with parsed arguments
    run_demo(
        image_path=args.image,
        video_path=args.video,
        camera_index=args.camera,
        weights_path=args.weights
    )


if __name__ == '__main__':
    main()
