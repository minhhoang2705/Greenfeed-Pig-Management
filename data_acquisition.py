import cv2
import os
import time
from datetime import datetime

class DataAcquisition:
    def __init__(self, output_dir='data', camera_id=0, capture_interval=60):
        self.output_dir = output_dir
        self.camera_id = camera_id
        self.capture_interval = capture_interval
        self.cap = None
        os.makedirs(self.output_dir, exist_ok=True)

    def start_capture(self):
        self.cap = cv2.VideoCapture(self.camera_id)
        if not self.cap.isOpened():
            raise IOError("Cannot open webcam")

    def capture_image(self):
        ret, frame = self.cap.read()
        if not ret:
            print("Failed to capture frame")
            return None
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(self.output_dir, f"image_{timestamp}.jpg")
        cv2.imwrite(filename, frame)
        print(f"Captured image: {filename}")
        return filename

    def run(self):
        self.start_capture()
        try:
            while True:
                self.capture_image()
                time.sleep(self.capture_interval)
        except KeyboardInterrupt:
            print("Data acquisition stopped.")
        finally:
            if self.cap:
                self.cap.release()

if __name__ == '__main__':
    acquisition = DataAcquisition()
    acquisition.run()
