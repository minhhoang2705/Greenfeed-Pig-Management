# Pig Detection and Counting System

This project implements an AI-powered system for detecting and counting pigs in a farming environment. It includes modules for data acquisition, data ingestion, model training, and model deployment.

## Project Structure

-   `data_acquisition.py`: Captures images from a camera at regular intervals.
-   `data_ingestion.py`: Loads and preprocesses image data for model training.
-   `training_pipeline.py`: Trains the pig detection model using PyTorch.
-   `deployment.py`: Deploys the trained model for real-time pig detection and counting.
-   `pig_detector.py`: Contains the core pig detection logic using a pre-trained YOLOv5 model.
-   `requirements.txt`: Lists the required Python packages.
-   `models/`: Directory to store trained models.
-   `data/`: Directory to store captured images.
-   `annotations.json`: JSON file containing bounding box annotations for training images.

## Setup

1.  **Install Dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

2.  **Data Acquisition:**
    -   Run `data_acquisition.py` to capture images from a camera.
    ```bash
    python data_acquisition.py
    ```
    -   Captured images will be saved in the `data/` directory.

3.  **Data Annotation:**
    -   Create a JSON file named `annotations.json` in the `pig_detection` directory.
    -   The JSON file should contain annotations for each image in the following format:
        ```json
        {
            "image_name.jpg": [
                {"bbox": [x1, y1, x2, y2]},
                {"bbox": [x1, y1, x2, y2]}
            ],
            "another_image.jpg": [
                {"bbox": [x1, y1, x2, y2]}
            ]
        }
        ```
        -   `x1`, `y1` are the coordinates of the top-left corner of the bounding box.
        -   `x2`, `y2` are the coordinates of the bottom-right corner of the bounding box.

4.  **Model Training:**
    -   Run `training_pipeline.py` to train the model.
    ```bash
    python training_pipeline.py
    ```
    -   The trained model will be saved in the `models/` directory as `trained_pig_detector.pth`.

5.  **Model Deployment:**
    -   Run `deployment.py` to perform inference on a test image.
    ```bash
    python deployment.py
    ```
    -   Replace `test_image.jpg` with the path to your test image.

## Usage

-   **Data Acquisition:** The `data_acquisition.py` script captures images from the specified camera and saves them to the `data/` directory.
-   **Data Ingestion:** The `data_ingestion.py` script loads images and their corresponding bounding box annotations from the `data/` directory and `annotations.json` file, respectively. It prepares the data for model training.
-   **Model Training:** The `training_pipeline.py` script trains the pig detection model using the provided data and saves the trained model to the `models/` directory.
-   **Model Deployment:** The `deployment.py` script loads the trained model and performs inference on a given image, displaying the bounding boxes and the number of detected pigs.

## Notes

-   Ensure that you have a camera connected to your system and that the camera ID is correctly set in `data_acquisition.py`.
-   The `annotations.json` file is required for training the model. You will need to manually annotate the images with bounding boxes.
-   The `training_pipeline.py` script uses transfer learning, freezing all layers except the last one.
-   The `deployment.py` script loads the trained model and performs inference on a given image.
