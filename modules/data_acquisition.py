# Import necessary libraries
import cv2  # OpenCV library for computer vision tasks
import os  # Library for interacting with the operating system
import time  # Library for working with time and dates
from datetime import datetime  # Library for working with dates and times

# Define a class to encapsulate data acquisition functionality
class DataAcquisition:
    # Initialize the class with parameters for output directory, camera ID, and capture interval
    def __init__(self, output_dir='data', camera_id=0, capture_interval=60):
        """
        Initializes the DataAcquisition class.

        Args:
            output_dir (str): The directory where images will be saved. Defaults to 'data'.
            camera_id (int): The ID of the camera to use for data acquisition. Defaults to 0.
            capture_interval (int): The interval in seconds between captures. Defaults to 60.
        """
        self.output_dir = output_dir
        self.camera_id = camera_id
        self.capture_interval = capture_interval
        # Create the output directory if it doesn't exist
        os.makedirs(self.output_dir, exist_ok=True)

    # Start capturing images from the specified camera
    def start_capture(self):
        """
        Starts capturing images from the specified camera.

        Raises:
            IOError: If the camera cannot be opened.
        """
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            raise IOError("Cannot open webcam")

    # Capture a single image from the camera
    def capture_image(self):
        """
        Captures a single image from the camera.

        Returns:
            str: The filename of the captured image.
        """
        ret, frame = self.cap.read()
        if not ret:
            print("Failed to capture frame")
            return None
        
        # Get the current timestamp and format it as YYYYMMDD_HHMMSS
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        # Create a filename for the captured image by combining the output directory, timestamp, and file extension
        filename = os.path.join(self.output_dir, f"image_{timestamp}.jpg")
        # Write the frame to disk as an image file
        cv2.imwrite(filename, frame)
        print(f"Captured image: {filename}")
        return filename

    # Run the data acquisition process until stopped manually
    def run(self):
        """
        Runs the data acquisition process.

        This method starts capturing images from the camera and continues indefinitely until manually stopped.
        """
        self.start_capture()
        try:
            while True:
                self.capture_image()
                time.sleep(self.capture_interval)
        except KeyboardInterrupt:
            print("Data acquisition stopped.")
        finally:
            # Release any system resources acquired by this class
            if self.cap:
                self.cap.release()

# Main entry point for the script
if __name__ == '__main__':
    # Create an instance of the DataAcquisition class
    acquisition = DataAcquisition()
    # Start running the data acquisition process
    acquisition.run()
