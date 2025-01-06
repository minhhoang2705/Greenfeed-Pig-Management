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

        # Process frame with tracking
        detections, tracked_objects = detector.process_single_frame(image)
        result = detector.visualize_detections(
            image, detections, tracked_objects)

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
        cap = cv2.VideoCapture(
            camera_index if camera_index is not None else video_path)
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
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            out = cv2.VideoWriter(output_path, fourcc, fps,
                                  (frame_width, frame_height))

        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)
                           ) if video_path else 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame with tracking
            detections, tracked_objects = detector.process_single_frame(frame)
            result = detector.visualize_detections(
                frame, detections, tracked_objects)

            # Save frame and show progress
            if video_path:
                out.write(result)
                frame_count += 1
                print(
                    f"\rProcessing frame {frame_count}/{total_frames}", end="")

        # Cleanup
        cap.release()
        if video_path:
            out.release()
        cv2.destroyAllWindows()
        print("\nProcessing complete")


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
